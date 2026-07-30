import pandas as pd
av2 = pd.read_csv(r'C:\paper_results\av2_validation_1000.csv')

print('=== INTERVENTION_LEVEL DISTRIBUTION ===')
counts = av2['intervention_level'].value_counts()
total = len(av2)
for t in ['emergency','intervention','advisory','silent']:
    pct = round(counts.get(t,0)/total*100,1)
    print('  {}: {} ({}%)'.format(t, counts.get(t,0), pct))

print()
print('=== FINAL_SAFETY_SCORE STATS ===')
print('  mean: {:.1f}'.format(av2['final_safety_score'].mean()))
print('  min:  {:.1f}'.format(av2['final_safety_score'].min()))
print('  max:  {:.1f}'.format(av2['final_safety_score'].max()))

print()
print('=== SCORE DISTRIBUTION BUCKETS (threshold on final_safety_score) ===')
s = av2['final_safety_score']
em = (s < 20).sum()
iv = ((s >= 20) & (s < 40)).sum()
ad = ((s >= 40) & (s < 70)).sum()
si = (s >= 70).sum()
print('  emergency (<20): {} ({:.1f}%)'.format(em, em/total*100))
print('  intervention (20-40): {} ({:.1f}%)'.format(iv, iv/total*100))
print('  advisory (40-70): {} ({:.1f}%)'.format(ad, ad/total*100))
print('  silent (>=70): {} ({:.1f}%)'.format(si, si/total*100))
