import numpy as np
import pandas as pd
import fetchmaker
from scipy.stats import binom_test, binomtest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency 

rottweiler_tl = fetchmaker.get_tail_length('rottweiler')
print(np.mean(rottweiler_tl), np.std(rottweiler_tl))

whippet_rescue = fetchmaker.get_is_rescue('whippet')
num_whippet_rescues = np.count_nonzero(whippet_rescue)
num_whippets = np.size(whippet_rescue)
pval_whippets = binomtest(num_whippet_rescues, num_whippets, 0.08)
print(pval_whippets)


whippet_weight = fetchmaker.get_weight('whippet')
terrier_weight = fetchmaker.get_weight('terrier')
pitbull_weight = fetchmaker.get_weight('pitbull')
weights = np.concatenate([whippet_weight, terrier_weight, pitbull_weight])
weights_labels = ['whippet'] * len(whippet_weight) + ['terrier'] * len(terrier_weight) + ['pitbull'] * len(pitbull_weight)
Ftest, pval_weight = f_oneway(whippet_weight, terrier_weight, pitbull_weight)
weight_result = pairwise_tukeyhsd(weights, weights_labels, 0.05)

if pval_weight < 0.05:
  print('Based on the ANOVA test ran, there is significance to say that the weights are not the same.')
else: 
  print('Based on the ANOVA test ran there is no proof that the weights are substanstially different.')

print(weight_result)


poodle_colors = fetchmaker.get_color('poodle')
shihtzu_colors = fetchmaker.get_color('shihtzu')

color_table = pd.DataFrame({ 
'Poodle' : [np.count_nonzero(poodle_colors == "black"), np.count_nonzero(poodle_colors == "brown"), np.count_nonzero(poodle_colors == "gold"), np.count_nonzero(poodle_colors == "grey"), np.count_nonzero(poodle_colors == "white")],
'Shih_Tzu': [np.count_nonzero(shihtzu_colors == "black"), np.count_nonzero(shihtzu_colors == "brown"), np.count_nonzero(shihtzu_colors == "gold"), np.count_nonzero(shihtzu_colors == "grey"), np.count_nonzero(shihtzu_colors == "white")]})

_, pval_col, _, _ = chi2_contingency(color_table)
print(pval_col)



