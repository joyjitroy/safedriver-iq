import pandas as pd
import numpy as np

av2 = pd.read_csv(r'C:\paper_results\av2_validation_1000.csv')
s = av2['final_safety_score'].sort_values().values
total = len(s)

# Find score-based percentile thresholds that reproduce original distribution
# Original: emergency=46, intervention=61, advisory=767, silent=126
t1 = s[45]    # 46th value (0-indexed 45) = upper boundary of emergency
t2 = s[106]   # 107th value = upper boundary of intervention
t3 = s[873]   # 874th value = upper boundary of advisory
print('=== BEST-FIT SCORE THRESHOLDS ===')
print('t1 (em/iv): {:.1f}'.format(t1))
print('t2 (iv/ad): {:.1f}'.format(t2))
print('t3 (ad/si): {:.1f}'.format(t3))

# Verify
scores = av2['final_safety_score']
def compute_dist(scores, t1, t2, t3):
    total = len(scores)
    em = (scores < t1).sum()
    iv = ((scores >= t1) & (scores < t2)).sum()
    ad = ((scores >= t2) & (scores < t3)).sum()
    si = (scores >= t3).sum()
    return em, iv, ad, si

print()
print('=== THRESHOLD SENSITIVITY (score-based with best-fit thresholds) ===')
for label, d1, d2, d3 in [
    ('Original ({:.0f}/{:.0f}/{:.0f})'.format(t1, t2, t3), 0, 0, 0),
    ('Shift +5  ({:.0f}/{:.0f}/{:.0f})'.format(t1+5, t2+5, t3+5), 5, 5, 5),
    ('Shift -5  ({:.0f}/{:.0f}/{:.0f})'.format(t1-5, t2-5, t3-5), -5, -5, -5),
]:
    em, iv, ad, si = compute_dist(scores, t1+d1, t2+d2, t3+d3)
    print('{}'.format(label))
    print('  emergency: {} ({:.1f}%)'.format(em, em/total*100))
    print('  intervention: {} ({:.1f}%)'.format(iv, iv/total*100))
    print('  advisory: {} ({:.1f}%)'.format(ad, ad/total*100))
    print('  silent: {} ({:.1f}%)'.format(si, si/total*100))
