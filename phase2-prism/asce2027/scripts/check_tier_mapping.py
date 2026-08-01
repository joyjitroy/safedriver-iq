import pandas as pd
av2 = pd.read_csv(r'C:\paper_results\av2_validation_1000.csv')

print('=== SCORE RANGES BY TIER (final_safety_score) ===')
for t in ['emergency','intervention','advisory','silent']:
    sub = av2[av2['intervention_level']==t]['final_safety_score']
    print('{}: n={}, mean={:.1f}, min={:.1f}, max={:.1f}'.format(
        t, len(sub), sub.mean(), sub.min(), sub.max()))

print()
print('=== CRASH_PROBABILITY RANGES BY TIER ===')
for t in ['emergency','intervention','advisory','silent']:
    sub = av2[av2['intervention_level']==t]['crash_probability']
    print('{}: n={}, mean={:.1f}, min={:.1f}, max={:.1f}'.format(
        t, len(sub), sub.mean(), sub.min(), sub.max()))

# Try thresholding on crash_probability (inverted score)
print()
print('=== THRESHOLD SENSITIVITY ON crash_probability ===')
cp = av2['crash_probability']
total = len(av2)
# If crash_prob thresholds are: >80=emergency, 60-80=intervention, 30-60=advisory, <30=silent
for label, hi_em, hi_iv, hi_ad in [('Original (>80/>60/>30)', 80, 60, 30),
                                     ('Shift+5 (>85/>65/>35)', 85, 65, 35),
                                     ('Shift-5 (>75/>55/>25)', 75, 55, 25)]:
    em = (cp > hi_em).sum()
    iv = ((cp > hi_iv) & (cp <= hi_em)).sum()
    ad = ((cp > hi_ad) & (cp <= hi_iv)).sum()
    si = (cp <= hi_ad).sum()
    print(label)
    print('  emergency: {} ({:.1f}%)'.format(em, em/total*100))
    print('  intervention: {} ({:.1f}%)'.format(iv, iv/total*100))
    print('  advisory: {} ({:.1f}%)'.format(ad, ad/total*100))
    print('  silent: {} ({:.1f}%)'.format(si, si/total*100))
