# PRISM-AR: Risk-Adaptive AR Interfaces for Vulnerable Road Users

**Background:** Fully autonomous vehicles (FAVs) lack human driver cues, creating a communication gap with pedestrians and other VRUs.  Prior work has explored static external interfaces (LED panels, projections, etc.) and even AR overlays in pedestrian crossing scenarios.  In a VR user study, Pratticò et al. found that AR-based cues (like projected safe/danger zones and stop lines) achieved **state-of-the-art safety and trust**, albeit with higher cognitive load than simpler designs.  Other studies confirm AR’s potential but also highlight design challenges (e.g. field of view, gaze alignment, cultural factors).  Meanwhile, Roy et al. (EIT 2026/ASCE 2027) introduced **PRISM/SafeDriver-IQ**, an agentic multi-model framework that fuses environmental, trajectory, and VRU risk into a continuous 0–100 safety score.  PRISM uses inverse crash-probability modeling (trained on 8+ years of NHTSA CRSS data and Waymo WOMD trajectories) to generate real-time, interpretable safety intelligence.  Crucially, PRISM’s risk output can now drive adaptive feedback, but no existing work has yet connected PRISM-like risk scoring with VRU-facing AR signaling. 

**Objective:** The goal is to build and evaluate **PRISM-AR**, a risk-adaptive AR communication layer for VRUs.  In this framework, PRISM’s continuous safety score and intervention tier (Silent, Advisory, Intervention, Emergency) determine the type, color, and intensity of AR cues presented to pedestrians and cyclists.  For example, low risk (“Silent” tier) yields no or minimal overlay, whereas high risk (“Intervention/Emergency”) triggers red danger zones, flashing no-cross signs, and emergency stop lines.  Unlike prior static designs, PRISM-AR aims to adjust warnings dynamically to current traffic risk, weather, lighting, and VRU proximity.  The research will (1) define the AR cue mappings, (2) create a controlled AR overlay dataset via simulation, and (3) validate that adaptive AR improves safety alignment and reduces under-warning compared to a static AR baseline.

## Data Sources and Driving Scenarios

We will leverage four datasets, all available locally for reproducible research: 

- **CRSS (NHTSA Crash Report Sampling System, 2016–2023):** A national probability sample of police-reported crashes involving all vehicles, pedestrians, and cyclists. CRSS provides environmental and crash-severity data to train the core PRISM risk models. We will use CRSS to extract risk factors (weather, road surface, speed, etc.) and crash labels, as in the SafeDriver-IQ framework. 

- **Waymo Open Motion Dataset (WOMD):** A large AV trajectory dataset. The Motion Dataset contains **103,354** 20-second segments (10 Hz, ~20M frames) of mixed manual/autonomous driving, with 10.8 million tracked objects (vehicles, pedestrians, cyclists) and 3D map data. Waymo covers diverse U.S. cities (San Francisco, Phoenix, Mountain View, Los Angeles, Detroit, Seattle). We will parse ego-vehicle and VRU tracks from Waymo, using it to validate PRISM’s trajectory and VRU risk models, and to generate realistic crossing scenarios. 

- **Argoverse 2 – Motion Forecasting:** 250,000 annotated scenarios for trajectory prediction (with 6-DOF map and rich labels).  Scenes come from six cities, with detailed lane-level maps. We will use a validation split (~24,000 sequences across multiple U.S. cities as in PRISM) to test geographic generalization of PRISM-AR. Argoverse provides pedestrian/cyclist actors and trajectory histories, ideal for simulating VRU crossings in varied environments. 

- **nuScenes (v1.0-mini):** 1000 urban driving scenes (20s each) with full sensor data (6 cameras, 1 LiDAR, 5 radar) and 3D annotations. Critically, nuScenes includes diverse lighting and weather (night, rain), representing adverse conditions. We will use nuScenes (or the mini subset) to stress-test PRISM-AR in low-visibility scenarios.  

**DrivingScene Extraction:** We will build a unified `DrivingScene` format to ingest these datasets. Each scene record contains ego and agent tracks (position, velocity, type, etc.), plus scene attributes (weather, lighting, road condition). For CRSS, we create synthetic “scenes” by sampling typical crash conditions (e.g. wet road, pedestrian involved). For Waymo/Argoverse/nuScenes, we convert trajectory data into ego-centric frames, label the closest pedestrian/cyclist ahead, and compute relevant features (distance, relative speed, TTC). This unified representation feeds into PRISM’s risk models. (All ingestion code will reside in a **`data_ingestion/`** module with dataset-specific loaders.)

## PRISM Risk Scoring

We adopt the PRISM/SafeDriver-IQ architecture as the core risk engine. PRISM combines:  
- **Environmental Risk Model:** Based on CRSS features (weather, lighting, road surface, time-of-day, etc.), trained to predict crash probability.  
- **Trajectory Risk Model:** Learned from Waymo/Argoverse tracks (ego speed, acceleration, obstacle distance, TTC).  
- **VRU Interaction Model:** A specialized model predicting VRU conflict risk (e.g. distance to pedestrian, path overlap).  

These models run in parallel. An **Agentic Reasoning Layer** (a DQN-based policy with short/long-term memory and SHAP explanations) fuses them to assign a continuous safety score (0–100) and a risk tier (Silent/Advisory/Intervention/Emergency) for each time step.  We will use the existing PRISM weights from Roy et al. (2026) as a starting point and fine-tune as needed. The PRISM output for each frame includes: score, tier, and top SHAP factors. This multi-model approach has been shown to produce meaningful, explainable risk intelligence. 

**Risk Output Logging:** For our scenarios, we will log PRISM outputs (risk score, tier) along with the scene data. This creates a time series of risk metrics for each crossing event. SHAP values identify whether, for example, “high speed” or “poor lighting” is driving the current risk. These explanations will guide our AR cue selection (“explanation tag”). 

## AR-VRU Interface Design

We extend the AR interface concepts from Pratticò et al.. In particular, their **“Safe roads extended”** AR design projected onto the road a colored safe/caution/danger zone plus vehicle stop cues. We will use a simplified top-down AR overlay with: 

- **Safe Zone (Green/Amber):** A polygon on the road where crossing is allowed or warned.  
- **Danger Zone (Red):** Indicates the ego-vehicle’s predicted stopping distance and unsafe crossing area.  
- **Stop Line:** A line or marker where the vehicle will stop (optional).  
- **No-Cross Boundary (Flashing Red):** An urgent barrier for emergency.  
- **Pedestrian Indicator:** (Optional) a marking at the VRU location if detected.  
- **Visual Style:** Color and opacity vary by risk tier. For example, “Advisory” might show a translucent amber zone, while “Emergency” shows a bright red barrier.  

These cues draw from AR literature: participants appreciated high visibility and direct mapping to vehicle dynamics. We will define two modes:  
- **Static AR (Baseline):** A fixed interface (like Pratticò’s extended AR) that does *not* depend on risk score.  It might always show a yellow stop line and green safe zone (as a “prediction-only” cue).  
- **Adaptive AR (PRISM-AR):** Cues are chosen by PRISM output. E.g., if PRISM tier is “Silent”, no overlay is shown; if “Advisory”, show a small amber safe zone; if “Intervention”, show a prominent red no-cross line and danger zone; if “Emergency”, flash a red boundary and text. 

The exact mapping will be parameterized (e.g. risk thresholds for color changes) and stored in a config file.  This ensures reproducibility of cue logic. The key hypothesis is that PRISM-AR’s dynamic cue strength will better match true risk: it should give stronger warnings in high-risk scenes and avoid false alarms when risk is low. 

## Dataset Generation (AR Overlay Creation)

We will construct a **controlled AR-VRU scenario dataset** by simulation. The plan is:

1. **Scenario Selection:** Choose a balanced set of ~200 scenarios across risk conditions. For example:  
   - **High Risk/Emergency:** Night + rain + wet roads + high speed + pedestrian suddenly steps in.  
   - **Medium Risk/Intervention:** Dusk + dry road + moderate speed + crossing marked or unmarked crosswalk.  
   - **Low Risk/Advisory:** Daylight + clear + ego braking + pedestrian waiting at curb.  
   - **Minimal Risk/Silent:** No pedestrian or very distant pedestrian.  
   - Ensure examples from Waymo (urban stops, highway exits), Argoverse (city scenarios), and nuScenes (night intersections) are included. Use PRISM to label their risk tier, then select representative frames.  

2. **Simulation Environment:** We will use the **CARLA simulator** to render the scenes. CARLA supports multi-agent traffic, weather control, and custom camera views. We will script each scenario in CARLA: spawning one ego vehicle (following a straight or turning path) and one or more pedestrian/cyclist actors crossing or near the roadway. We will vary lighting (sunny/dawn/dusk/night) and weather (clear/rain/fog). Key parameters: ego speed (e.g., 0–20 m/s), pedestrian behavior (walking or running across lane), and road layout (urban street with sidewalk). We will record a top-down “map view” or an ego-centric view with AR overlay.  

3. **Overlay Rendering:** For each simulation frame (say at 10 Hz), we will generate two paired images:  
   - **Static AR Image:** Render a fixed overlay (stop line, colored zones) on the road, independent of PRISM score. For example, always show a yellow crosswalk projection even if no VRU is present.  
   - **Adaptive PRISM-AR Image:** Render an overlay based on PRISM’s output. If tier=Silent, the overlay is blank (no warning). If tier=Advisory, draw a translucent amber safe zone ahead of the vehicle. If tier=Intervention, add a red “no-cross” boundary and predicted stop line. If Emergency, make the red overlay flash or add exclamation icons.  

   Both images use the same underlying scene but different overlaid cues. We also save a **label file** per frame indicating: PRISM score, tier, top SHAP factor, static cue type, adaptive cue type, and whether the pedestrian was detected by the interface. This forms a CSV annotation.  

4. **Data Format:** The result is a dataset of images (or sequences) with per-frame annotations. Example schema:  

   | scenario_id | timestamp | ego_speed | vru_distance | lighting | weather | PRISM_score | PRISM_tier | static_cue | adaptive_cue | under_warning | lead_time | ... |
   |-------------|-----------|-----------|--------------|----------|---------|-------------|------------|------------|--------------|---------------|-----------|-----|
   Each row corresponds to a frame. Fields like `under_warning` (boolean if adaptive cue is weaker than required) and `lead_time` (seconds between first warning and closest approach) will be computed for evaluation.  

5. **GitHub and Code Structure:** We will create a repository with modules: 
   - `data_ingestion/`: Load CRSS, Waymo, Argoverse, nuScenes into `DrivingScene` objects.  
   - `prism/`: Risk models (environmental RF, trajectory LSTM, VRU detection), fusion policy, SHAP explainer.  
   - `ar_overlay/`: Mapping from PRISM output to overlay commands, CARLA scenario builder and renderer.  
   - `evaluation/`: Scripts to compute metrics (safety, temporal, geometric, cognitive).  
   - `notebooks/`: Sample analyses and figure generation.  

   This modular design ensures Windserf can implement each part iteratively.

## Experimental Protocol

We will perform several validation experiments:

- **A. Static vs Adaptive AR (Main Test):**  Use the paired dataset (static vs PRISM-AR). For each scenario, compare the two modes on metrics such as:
  - *Under-warning Rate:* Fraction of frames where PRISM-AR gives a “safe”/“advisory” cue while static AR gave stronger warning (e.g. PRISM-AR missed a risk). We want PRISM-AR under-warning ≈ 0 (never under-warn in emergencies).  
  - *Over-warning Rate:* Fraction where PRISM-AR warns but static did not. Some over-warning is acceptable, but we check its frequency.  
  - *Warning Lead Time:* How many seconds before the nearest conflict does the first red (Intervention/Emergency) cue appear? We expect PRISM-AR to warn earlier in high-risk scenes.  
  - *Minimum Distance at Crossing:* For crossing events, measure distance/speed of ego when VRU steps off curb. Adaptive AR should reduce risky crossings (increase distance).  
  - *Visual Clutter & Flicker:* Count cue changes per scenario. PRISM-AR should be stable (low flicker) and only as busy as needed, whereas static AR may show unnecessary indicators.  
  - *SHAP Explainability:* Verify that when PRISM-AR raises a warning, the top SHAP factor (e.g. “VRU_proximity” or “darkness”) aligns with the intuitive reason.  

  We expect PRISM-AR to **reduce unsafe indications** in high-risk cases and **avoid false alarms** in low-risk cases, compared to static AR.  We will use paired statistical tests (e.g. McNemar for error rates, paired t-tests or Wilcoxon for lead time) to confirm significance.

- **B. Cross-Dataset Generalization:** Evaluate PRISM-AR separately on Waymo-driven scenarios, Argoverse city scenes, and nuScenes. Check that the tier distribution (silent/advisory/intervention/emergency) matches expected patterns. For example, nuScenes (night/rain) should yield more high-risk cues than clear-day Waymo. We will visualize tier histograms and cue frequencies by dataset.  Consistent performance across datasets will indicate robustness of the approach.

- **C. Adverse-Condition Stress Test:** Focus on lighting/weather extremes. Take scenes with night or heavy rain (from nuScenes or custom CARLA), and check that PRISM-AR escalates appropriately (more red cues) than static AR. Measure average PRISM score under each condition, and ensure AR cues scale with score.

- **D. VRU Proximity Stress Test:** Gradually vary ego-pedestrian distance in scenarios. Plot cue intensity versus ego-VRU distance or TTC. A good system should show monotonically increasing warning severity as distance shrinks. We expect near-zero distance (<3 m) to always trigger Intervention/Emergency in PRISM-AR (consistent with PRISM’s near-miss modeling).  

- **E. Ablation Studies:** Systematically disable components to gauge their impact:  
  - **No VRU Risk Model:** Use only environment+trajectory. We expect PRISM-AR to miss pedestrian-specific cues (e.g. under-warn when VRU appears).  
  - **No Weather Model:** Ignore lighting/weather factors. Adaptive AR should then underreact to night/rain scenes (contrasting with full model).  
  - **No Temporal Smoothing:** Remove any risk memory; measure increase in flicker.  
  - **Static vs Oracle:** Define an oracle interface that knows ground-truth minimum distance and always warns exactly at a threshold. This sets an upper bound on performance (for lead-time and under-warning=0).  

- **F. Human-Subject Proxy (Exploratory):** Though full human trials are beyond the initial scope, we may run a small user study (10–15 participants) in a desktop VR or video-based simulation. Participants would see static vs adaptive AR animations and rate trust or decide when to cross. This would provide qualitative validation (as in ) and check that PRISM-AR cues are interpretable. (We will follow IRB guidelines as needed.)

- **G. Runtime Profiling:** Measure end-to-end latency of PRISM-AR on test hardware. Roy et al. reported ~600 ms on CPU for PRISM. We will measure:  
  - Risk inference time per frame (environment + trajectory + VRU models + policy).  
  - AR overlay rendering time (mapping + graphics).  
  - Overall FPS with optimization (e.g. using GPU/LSTM pre-calc).  
  The goal is ~10 Hz operation. If needed, we will explore optimizations (caching, compiled inference). Timing results and system requirements will be reported.

## Evaluation Metrics

We will compute comprehensive metrics for each scenario and aggregate results:  

- **Safety/Correctness:** Under-warning rate, emergency cue recall, safe-zone violation (static safe-zone displayed when predicted crash). These measure if AR avoids misleading VRUs into danger.  
- **Temporal Performance:** Warning lead time, time difference between first red cue and closest approach. Longer lead time is safer.  
- **Geometric Accuracy:** Stop-line error (difference between predicted and actual stop point), alignment of safe/danger zones with vehicle path.  
- **Visual/Workload:** Cue flicker (tier changes per second), visual clutter (percent of image covered by overlay). Lower clutter is better if safety is maintained.  
- **Cross-Dataset Consistency:** Tier distribution across Waymo/Argoverse/nuScenes should reflect each dataset’s risk profile.  
- **Explainability:** Qualitative check that AR visual elements correspond to SHAP factors (e.g., if “Poor Lighting” drives risk, AR uses high-contrast colors).  

Statistical significance will be tested (paired tests with Bonferroni correction for multiple comparisons).  Wherever possible, we align with metrics used by Pratticò et al. and Tabone et al. for comparability.  For example, decision time and trust ratings (from a user study) could be adapted as future work; currently we focus on objective measures. 

## Dataset and Code Availability

All data sources (CRSS, Waymo WOMD, Argoverse2, nuScenes-mini) will be referenced in the manuscript, and our generated AR-overlay dataset (images and CSV annotations) will be released on GitHub along with code. The repository will include instructions to reproduce the training of PRISM models (from CRSS and Waymo) and to generate AR overlays in CARLA or via provided scripts. We will document the code structure and dependencies (e.g. CARLA version, PyTorch/TensorFlow models, SHAP library). 

## Summary

In summary, this project bridges PRISM’s risk intelligence and pedestrian AR interfaces.  By validating PRISM-AR across multiple public datasets and controlled scenarios, we will demonstrate that **risk-adaptive AR communication** can improve VRU safety signaling.  The structured evaluation (static vs adaptive, cross-dataset, stress tests, ablations) will identify strengths and limitations.  Findings will be contextualized by prior VRU-AR studies: for example, showing that adaptivity retains Pratticò’s high safety performance while addressing their noted drawbacks of static designs.  This work will produce a detailed methodology (with code) enabling peers to reproduce and extend the results, supporting future publications or a TVT journal submission.

**References:** We cite VRU-AR interface studies, open dataset papers, and PRISM/SafeDriver-IQ overviews as key sources informing this plan. All cited text is retained for transparency.