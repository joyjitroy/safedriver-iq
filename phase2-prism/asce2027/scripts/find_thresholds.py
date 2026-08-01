import pandas as pd
import numpy as np
av2 = pd.read_csv(r'C:\paper_results\av2_validation_1000.csv')

# Find actual boundary scores between tiers
print('=== FINDING NATURAL SCORE BOUNDARIES ===')
tiers = ['emergency','intervention','advisory','silent']
for t in tiers:
    sub = av2[av2['intervention_level']==t]['final_safety_score'].sort_values()
    print('{}: n={} | min={:.2f}, p5={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p95={:.2f}, max={:.2f}'.format(
        t, len(sub),
        sub.min(), sub.quantile(0.05), sub.quantile(0.25),
        sub.quantile(0.5), sub.quantile(0.75), sub.quantile(0.95), sub.max()))

# Try thresholds based on actual score gaps
print()
print('=== TRYING THRESHOLDS TO MATCH ACTUAL DISTRIBUTION ===')
# Try: emergency<40, 40<=interv<57, 57<=advisory<73, >=73 silent
for t1, t2, t3 in [(40,57,73),(42,55,72),(40,55,73),(38,55,73),(40,56,73)]:
    s = av2['final_safety_score']
    em = (s < t1).sum()
    iv = ((s >= t1) & (s < t2)).sum()
    ad = ((s >= t2) & (s < t3)).sum()
    si = (s >= t3).sum()
    total = len(av2)
    print('Thresholds {}/{}/{}: em={:.1f}% iv={:.1f}% ad={:.1f}% si={:.1f}%'.format(
        t1, t2, t3,
        em/total*100, iv/total*100, ad/total*100, si/total*100))
print('Target: em=4.6% iv=6.1% ad=76.7% si=12.6%')
