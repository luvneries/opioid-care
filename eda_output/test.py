
from itertools import combinations

numerical_cols = ['mentally_unhealthy_days',
'tedsa_frstuse2_7',
 'physically_unhealthy_days',
 'pct_frequent_mental_distress',
 'pct_excessive_drinking',
 'injury_death_rate',
 'tedsa_sub1_7',
 'age_adjusted_mortality',
 'pct_frequent_physical_distress',
 'pct_smokers',
 'tedsd_cbsa_17300',
 'tedsd_cbsa_14540',
 'tedsd_cbsa_21060',
 'pct_insufficient_sleep',
 'preventable_hospitalization_rate',
 'pct_some_college',
 'pct_fair_or_poor_health',
 'pct_unemployed',
 'nmhss_focus_mix_of_mental_health_and_substance_abuse_treatment',
 'length_of_life_pctile_within_state',
 'median_household_income',
 'nmhss_focus_mental_health_treatment',
 'pct_children_in_poverty',
 'income_20th_percentile',
 'pct_children_eligible_free_lunch',
 'pct_diabetic',
 'nmhss_treatfamthrpy',
 'nmhss_langprov_both_staff_and_on_call_interpreter',
 'retaildrug_amphetamine',
 'retaildrug_tapentadol',
 'pct_rural',
 'retaildrug_fentanyl_base'
                 ]
def find_all_combination(col_list):
  all_combinations = []
  for i in range(23, 25):
    for subset in combinations(col_list, i):
        all_combinations.append(list(subset))
  return all_combinations

all_combinations = find_all_combination(numerical_cols)
"""
with open('exp_combinations.txt', 'w') as f:
    for subset in all_combinations:
        f.write("%s\n" % subset)
"""
