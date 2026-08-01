"""M7 — LLM co-pilot (Hybrid scope): natural-language narration, OFF the safety path.

A provider-agnostic text generator that turns the already-decided scenario state and RL
decision into human-readable narration. It is strictly additive:

  * It NEVER sets the risk score, the intervention tier, or any safety value — those come
    from the deterministic models (M2-M6). The co-pilot only *phrases* what was decided.
  * Every call has a hard timeout and a fallback: on timeout / error / `provider="off"`
    it returns the deterministic SHAP-based explanation instead. The agentic loop never
    blocks on, or depends on, the LLM.

Providers (config.LLM.provider):
  * "off"    — disabled; callers use the SHAP fallback. (default)
  * "claude" — Anthropic API via the official SDK. Opus 4.8 for synthesis, Haiku 4.5 for
               the cheap per-scenario summaries. Resolves ANTHROPIC_API_KEY from env.
  * "local"  — a llama.cpp OpenAI-compatible server (/v1/chat/completions) via httpx.

For testing, inject `completion_fn(system, prompt, model, timeout_s) -> str` to bypass
any network.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable

from sdiq import config

if TYPE_CHECKING:
    from sdiq.agentic_layer import Decision
    from sdiq.agentic_layer import AgentMemory
    from sdiq.scenario_summary import ScenarioSummary

log = logging.getLogger("sdiq.copilot")

CompletionFn = Callable[[str, str, str, float], str]

_SYSTEM = (
    "You are a co-pilot for a driver-safety system. You receive a structured scenario "
    "assessment that the system has ALREADY analyzed and decided on. Your ONLY job is to "
    "phrase a concise, factual natural-language note for a human. Do NOT invent, change, "
    "or second-guess any risk score, risk level, or intervention — describe only what the "
    "data states. No preamble, no hedging, no markdown. Keep it brief and plain."
)


class LLMCopilot:
    def __init__(self, cfg=None, completion_fn: CompletionFn | None = None) -> None:
        self.cfg = cfg or config.LLM
        self._completion_fn = completion_fn  # injected (tests) or built per provider
        self._client = None                  # lazy anthropic client

    @property
    def enabled(self) -> bool:
        return self._completion_fn is not None or self.cfg.provider in ("claude", "local")

    # -- provider dispatch -------------------------------------------------
    def _complete(self, system: str, prompt: str, model: str) -> str | None:
        """Run one completion with the configured provider; None on any failure."""
        try:
            if self._completion_fn is not None:
                return self._completion_fn(system, prompt, model, self.cfg.timeout_s)
            if self.cfg.provider == "claude":
                return self._complete_claude(system, prompt, model)
            if self.cfg.provider == "local":
                return self._complete_local(system, prompt, model)
            return None  # "off"
        except Exception as exc:                      # timeout / network / SDK / parse
            log.warning("co-pilot completion failed (%s); using SHAP fallback", exc)
            return None

    def _complete_claude(self, system: str, prompt: str, model: str) -> str:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        resp = self._client.with_options(timeout=self.cfg.timeout_s).messages.create(
            model=model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in resp.content if b.type == "text"), "").strip()

    def _complete_local(self, system: str, prompt: str, model: str) -> str:
        import httpx
        r = httpx.post(
            f"{self.cfg.local_base_url.rstrip('/')}/chat/completions",
            json={
                "model": model or self.cfg.local_model,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # -- public narration API (each returns text, with a deterministic fallback) ---
    def summarize_scenario(self, summary: "ScenarioSummary") -> str:
        """One-line NL description of the driving situation (cheap model)."""
        fallback = summary.summary if hasattr(summary, "summary") else str(summary)
        prompt = (
            "Describe this driving scenario in one sentence for the driver.\n"
            + json.dumps(_scenario_payload(summary))
        )
        return self._complete(_SYSTEM, prompt, self.cfg.claude_cheap_model) or fallback

    def explain_intervention(self, summary: "ScenarioSummary",
                             decision: "Decision") -> str:
        """NL rationale for the chosen intervention. Falls back to the SHAP explanation."""
        fallback = decision.explanation
        if decision.tier == 0:
            return fallback
        top = sorted(decision.shap.items(), key=lambda kv: kv[1], reverse=True)[:3]
        prompt = (
            f"The system chose a '{decision.tier_name}' intervention. In one or two "
            "sentences, tell the driver why, grounded ONLY in these decided facts.\n"
            + json.dumps({
                "intervention": decision.tier_name,
                "scenario": _scenario_payload(summary),
                "top_factors": [k for k, _ in top],
            })
        )
        return self._complete(_SYSTEM, prompt, self.cfg.claude_cheap_model) or fallback

    def narrate_memory(self, memory: "AgentMemory") -> str:
        """Short digest of what the agent has learned this/over drives (synthesis model)."""
        entries = list(memory.long) + list(memory.short)
        if not entries:
            return "No driving history recorded yet."
        from collections import Counter
        tiers = Counter(e.tier for e in entries)
        fallback = (f"Reviewed {len(entries)} situations: "
                    + ", ".join(f"{n} {t}" for t, n in tiers.most_common()) + ".")
        prompt = (
            "Summarize what the safety agent has encountered, in 1-2 sentences, from "
            "this tally of past decisions.\n"
            + json.dumps({"count": len(entries), "by_intervention": dict(tiers),
                          "mean_severity": round(sum(e.severity for e in entries) / len(entries), 3)})
        )
        return self._complete(_SYSTEM, prompt, self.cfg.claude_model) or fallback


def _scenario_payload(summary: "ScenarioSummary") -> dict:
    """Compact, decided facts handed to the LLM (never asks it to compute anything)."""
    d = summary.to_dict()
    keep = ("scene_id", "is_night", "is_rain", "n_vru", "env_risk_level",
            "trajectory_risk", "vru_risk", "vru_near_misses", "min_distance", "min_ttc",
            "fused_risk", "combined_safety_score", "band")
    return {k: d[k] for k in keep if k in d}


if __name__ == "__main__":
    # Demo with a fake completion (no network) to show the wiring + fallback.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sdiq.scenario_summary import ScenarioSummarizer
    from sdiq.agentic_layer import AgenticReasoner
    from sdiq.data_loader import iter_nuscenes_scenes

    def fake(system, prompt, model, timeout):
        payload = json.loads(prompt.split("\n", 1)[1])
        s = payload.get("scenario", payload)
        cond = []
        if s.get("is_night"): cond.append("at night")
        if s.get("is_rain"): cond.append("in the rain")
        return (f"[{model}] {s.get('n_vru', 0)} vulnerable road users nearby "
                + (" ".join(cond) or "in clear conditions")
                + f"; risk {s.get('combined_safety_score', 0):.0f}/100.")

    copilot = LLMCopilot(completion_fn=fake)
    summ = ScenarioSummarizer()
    reasoner = AgenticReasoner()
    for sc in list(iter_nuscenes_scenes())[:3]:
        s = summ.summarize(sc)
        d = reasoner.decide(s)
        print(f"\n{sc.scene_id}:")
        print("  summary :", copilot.summarize_scenario(s))
        print("  why     :", copilot.explain_intervention(s, d))
    print("\n  memory  :", copilot.narrate_memory(reasoner.memory))
