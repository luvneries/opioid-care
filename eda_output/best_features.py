#https://blog.morizyun.com/python/library-bigquery-google-cloud.html

import os
import pandas as pd
import numpy as np

from google.cloud import bigquery
from pandas_gbq import read_gbq
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from itertools import combinations

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/pankaj/opioid-care-pkey.json"

project = 'opioid-care'
dataset_name = 'hackathon_opioid'
bq_client = bigquery.Client()

master_df_query = "select * from hackathon_opioid.master_data"
best_100_df_query = "select * from hackathon_opioid.best_features"


master_df = read_gbq(master_df_query, project, dialect='legacy')
best_features_df = read_gbq(best_100_df_query, project, dialect='legacy')

def drop_y(df):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)

df = pd.merge(master_df, best_features_df,  how='inner', on=['state','county'], suffixes=('', '_y'))
drop_y(df)


def preprocessing(data, strategy='mean', scaling='std'):
  from sklearn.preprocessing import Imputer, StandardScaler
  data = Imputer(strategy=strategy).fit_transform(data)
  if scaling=='std':
    data = StandardScaler().fit_transform(data)
  return data


def feature_selection(data, cols):
    X = df[numerical_cols]
    y = df['drug_overdose_mortality_rate']

    y_scaled = StandardScaler().fit_transform(np.array(y).reshape(-1, 1))
    X_scaled = StandardScaler().fit_transform(X)

    # select k best modeling
    model_reduced = SelectKBest(f_regression, k='all').fit(X_scaled, y_scaled)

    # get feature ranking by F-score
    feature_ranking0 = pd.DataFrame()
    feature_ranking0['drivers'] = X.columns
    feature_ranking0['score'] = model_reduced.scores_
    feature_ranking0['T/F'] = model_reduced.get_support()
    feature_ranking = feature_ranking0[feature_ranking0['T/F']==True].sort_values(by='score', ascending=False).reset_index(drop=True).drop('T/F',axis=1)

    return feature_ranking


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
 'nmhss_facnum_11_to_30_facilities',
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

#numerical_cols=numerical_cols[:10]
target = ['drug_overdose_mortality_rate']

def find_all_combination(col_list):
  all_combinations = []
  for i in range(15, 31):
    for subset in combinations(col_list, i):
        all_combinations.append(list(subset))
  return all_combinations


def find_best_variables(all_combinations):
  for i, col_list in enumerate(all_combinations):
    if col_list:
      input_values = df[col_list].values
      labels = df[target].values
      X_train, X_val, y_train, y_val = train_test_split(input_values, labels, random_state=0)
      prep_train = preprocessing(X_train)
      prep_val = preprocessing(X_val)
      model = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10, random_state=0)
      clust_labels = model.fit_predict(np.vstack((prep_train, prep_val)))
      data = pd.DataFrame(clust_labels)
      col_name = 'KMeans_label'
      if col_name in df.columns:
        del df[col_name]
      df.insert(loc=df.shape[1], column=col_name, value=data)

      #Check if our clustering labels marked high risk counties correctly
      high_risk_labeled_counties = df[df['KMeans_label']==2]
      high_death_rate_counties_index = df.drug_overdose_mortality_rate.sort_values(ascending=False)[:len(high_risk_labeled_counties)].index
      high_death_rate_counties = list(df[['state_name','county']].iloc[high_death_rate_counties_index, :]['county'])

      matched_counties = []
      for county in list(high_death_rate_counties):
        if county in list(high_risk_labeled_counties.county):
          matched_counties.append(county)

      accuracy=0
      perf = len(matched_counties)/len(high_risk_labeled_counties)

      if perf > accuracy:
        accuracy = perf
        index = i
        best_subset = col_list
    if i%500==0:
        with open('perf_param_variables.txt', 'w+') as f:
            for item in (index, accuracy, best_subset):
                f.write("%s\n" % item)

  return (index, accuracy, best_subset)


def main():
    all_combinations = find_all_combination(numerical_cols)

    with open('exp_combinations.txt', 'w') as f:
        for subset in all_combinations:
            f.write("%s\n" % subset)
    #print(all_combinations)

    best_variables = find_best_variables(all_combinations)
    with open('best_variables.txt', 'w') as f:
        for item in best_variables:
            f.write("%s\n" % item)


if __name__=="__main__":
    main()
