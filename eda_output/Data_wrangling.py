
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
import google.datalab.storage as storage
import google.datalab.bigquery as bq
from io import BytesIO
import os, sys
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## read datasets

def read_from_storage(object_name, delimiter=','):
  """
  object_name: Full path of file, e.g 'raw-data/Cause-of-Death/CDC_cause_of_death_by_demographics_and_state_20180606.xlsx')
  delimiter: Based on file e.g Default ',' '\t' '|'
  """
  global data, uri
  bucket = storage.Bucket('opioid-care')
  data = bucket.object(object_name)
  uri = data.uri
  %gcs read --object $uri --variable data
  if object_name.endswith('xlsx') or object_name.endswith('xls'):
    return read_from_storage(BytesIO(data))
  return read_from_storage(BytesIO(data), delimiter=delimiter)


# ### from raw datasets

master_df = read_from_storage('eda_output/master_data.csv')
nssats_2016_df = read_from_storage('raw-data/N-SSATS/NSSATSPUF_2016.csv')
mh2016_df = read_from_storage('raw-data/NMHSS/nmhsspuf_2016.csv')
teds_a = read_from_storage('raw-data/TEDS/teds_a_10_15.csv')
teds_d = read_from_storage('raw-data/TEDS/teds_d_2006_2014.csv')
provider_df = read_from_storage('raw-data/Medicare-Provider-Utilization/PartD_Prescriber_PUF_NPI_Drug_15.txt')
retail_drug_2016_df = read_from_storage('eda_output/retail_drug_2016.xlsx')[['DRUG NAME','STATE_CODE','TOTAL GRAMS']]
prescribing_df = read_from_storage('eda_output/Medicare_Part_D_Opioid_Prescribing_Geographic_2016.xlsx')
NSDUH2016_df = read_from_storage('eda_output/NSDUHsaeExcelTabs2016_Features.xlsx')


# ## state transform dicts

state_fip = read_from_storage('eda_output/State_FIP.csv').set_index('FIP')
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'District of Columbia': 'DC',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}


# ## Master County Dataset

master_df['STATE'] = master_df['State'].map(us_state_abbrev)
master_df = master_df.drop('State',axis=1)


# ### N-SSATS -- National Survey of Substance Abuse Treatment Services

nssats_2016_df_clean = nssats_2016_df.dropna(axis='columns')
nssats_2016_df_clean = nssats_2016_df_clean.applymap(str)

binary_col = []
other_col = []
for col in list(nssats_2016_df_clean)[2:]:
    if '0' in nssats_2016_df_clean[col].unique().tolist() and '2' not in nssats_2016_df_clean[col].unique().tolist():
        binary_col += [col]
    else:
        other_col += [col]

nssats_2016_df_clean_binary = nssats_2016_df_clean[binary_col]
nssats_2016_df_clean_binary.replace({'M':'0','D':'0','R':'0'},inplace=True)
nssats_2016_df_clean_binary = nssats_2016_df_clean_binary.applymap(int)
nssats_2016_df_clean_binary['STATE'] = nssats_2016_df_clean['STATE']
nssats_2016_df_clean_binary_pct = nssats_2016_df_clean_binary.groupby('STATE').mean().reset_index()

ownership_df = nssats_2016_df_clean[['STATE','OWNERSHP']]
ownership_df['private_ownership_pct'] = ownership_df['OWNERSHP'].apply(lambda x: 1 if x in ['1','2'] else 0)
ownership_df = ownership_df.groupby('STATE').mean().reset_index()
testing_df = nssats_2016_df_clean[['STATE','TESTING']]
testing_df['testing_service_number'] = testing_df['TESTING'].astype(int)
testing_df = testing_df.groupby('STATE').mean().reset_index()
compsat_df = pd.get_dummies(nssats_2016_df_clean['COMPSAT'],prefix='COMPSAT').applymap(int)
compsat_df['STATE'] = nssats_2016_df_clean['STATE']
compsat_df = compsat_df.groupby('STATE').mean().reset_index()
nssats_2016_df_clean_other_pct = pd.concat([ownership_df, testing_df, compsat_df], axis=1,sort = False)

nssats_2016_df_final =  pd.concat([nssats_2016_df_clean_binary_pct,nssats_2016_df_clean_other_pct.drop('STATE',axis=1)], axis=1,sort = False)
nssats_2016_df_final = nssats_2016_df_final.add_prefix('NSSATS_')
nssats_2016_df_final['STATE'] = nssats_2016_df_final['NSSATS_STATE']
nssats_2016_df_final = nssats_2016_df_final.drop('NSSATS_STATE',axis=1)


# ### N-MHSS -- National Mental Health Services Survey

mh2016_df_clean = mh2016_df.dropna(axis='columns')
mh2016_df_clean = mh2016_df_clean.applymap(str)

binary_col = []
other_col = []
for col in list(mh2016_df_clean)[2:]:
    if 'Yes' in mh2016_df_clean[col].unique().tolist():
        binary_col += [col]
    else:
        other_col += [col]

mh2016_df_clean_binary = mh2016_df_clean[binary_col]
mh2016_df_clean_binary.replace({'-1':'No','-2':'No','-3':'No','-6':'No','-7':'No'},inplace=True)
mh2016_df_clean_binary_transform = pd.DataFrame()
for col in list(mh2016_df_clean_binary):
    mh2016_df_clean_binary_transform[col] = mh2016_df_clean_binary[col].map({'Yes':1,'No':0})
mh2016_df_clean_binary_transform['STATE'] = mh2016_df_clean['LST']
mh2016_df_clean_binary_pct = mh2016_df_clean_binary_transform.groupby('STATE').mean().reset_index()

mh2016_df_clean_other = mh2016_df_clean[other_col]
mh2016_df_clean_other_dummy = pd.get_dummies(mh2016_df_clean_other).applymap(int)
mh2016_df_clean_other_dummy['STATE'] = mh2016_df_clean['LST']
mh2016_df_clean_other_pct = mh2016_df_clean_other_dummy.groupby('STATE').mean().reset_index()

mh2016_df_final =  pd.concat([mh2016_df_clean_binary_pct,mh2016_df_clean_other_pct.drop('STATE',axis=1)], axis=1,sort = False)
mh2016_df_final = mh2016_df_final.add_prefix('NMHSS_')
mh2016_df_final['STATE'] = mh2016_df_final['NMHSS_STATE']
mh2016_df_final = mh2016_df_final.drop('NMHSS_STATE',axis=1)


# ### Retail drug  -- Nintish

retail_drug_2016_clean_df = retail_drug_2016_df.groupby(['DRUG NAME','STATE_CODE']).sum().reset_index()
retail_drug_2016_clean_pivot = retail_drug_2016_clean_df.pivot_table('TOTAL GRAMS', 'STATE_CODE', 'DRUG NAME').reset_index()

# drop col with null value > 0.3
null_lst = retail_drug_2016_clean_pivot.isnull().sum(axis=0).tolist()
keep_idx = []
for i in range(len(null_lst)):
    if null_lst[i]/len(retail_drug_2016_clean_pivot)<0.2:
        keep_idx += [i]

retail_drug_2016_clean_pivot_dropnan = retail_drug_2016_clean_pivot.iloc[:,keep_idx]
retail_drug_2016_clean_pivot_dropnan = retail_drug_2016_clean_pivot_dropnan.add_prefix('RetailDrug_')
retail_drug_2016_clean_pivot_dropnan['STATE'] = retail_drug_2016_clean_pivot_dropnan['RetailDrug_STATE_CODE']
retail_drug_2016_final = retail_drug_2016_clean_pivot_dropnan.drop('RetailDrug_STATE_CODE',axis=1)


# ### TED

state_fip_dict = state_fip.to_dict()['STATE']


# ##### teds_a: 2010 - 2015

teds_a_2015 = teds_a[teds_a['YEAR']==2015]
teds_a_2015_clean = teds_a_2015.applymap(str)
teds_a_2015_clean = teds_a_2015_clean.drop(['STFIPS','CASEID','YEAR','CBSA10','DAYWAIT','ALCDRUG','ALCFLG','COKEFLG',
                                           'DETCRIM','ETHNIC','HALLFLG','IDU','INHFLG','MARFLG','MTHAMFLG','NUMSUBS',
                                           'PCPFLG','PRIMPAY','PSOURCE','SEDHPFLG'],axis=1).reset_index(drop=True)
teds_a_2015_clean_dummy = pd.get_dummies(teds_a_2015_clean).applymap(int)
state_col = teds_a_2015['STFIPS'].map(state_fip_dict).reset_index(drop=True)
teds_a_2015_clean_dummy['STATE'] = state_col
teds_a_2015_final = teds_a_2015_clean_dummy.groupby('STATE').mean().reset_index()
teds_a_2015_final = teds_a_2015_final.add_prefix('TEDSA_')
teds_a_2015_final['STATE'] = teds_a_2015_final['TEDSA_STATE']
teds_a_2015_final = teds_a_2015_final.drop('TEDSA_STATE',axis=1)


# ##### teds_d: 2006 - 2014

teds_d_2014 = teds_d[teds_d['DISYR']==2014].reset_index(drop=True)
teds_d_2014_clean = teds_d_2014.applymap(str)
teds_d_2014_clean = teds_d_2014_clean.drop(['STFIPS','CASEID','DAYWAIT','ALCDRUG','ALCFLG','COKEFLG',
                                           'DETCRIM','ETHNIC','HALLFLG','IDU','INHFLG','MARFLG','MTHAMFLG','NUMSUBS',
                                           'PCPFLG','PRIMPAY','PSOURCE','SEDHPFLG'],axis=1).reset_index(drop=True)
teds_d_2014_clean_dummy = pd.get_dummies(teds_d_2014_clean).applymap(int)
state_col = teds_d_2014['STFIPS'].map(state_fip_dict).reset_index(drop=True)
teds_d_2014_clean_dummy['STATE'] = state_col
teds_d_2014_final = teds_d_2014_clean_dummy.groupby('STATE').mean().reset_index()
teds_d_2014_final = teds_d_2014_final.add_prefix('TEDSD_')
teds_d_2014_final['STATE'] = teds_d_2014_final['TEDSD_STATE']
teds_d_2014_final = teds_d_2014_final.drop('TEDSD_STATE',axis=1)


# ### Medicare Provider Unilization

num_col = [10, 11, 12, 13, 14, 15, 17, 19, 20]
col_keep = ['nppes_provider_state']
for i in num_col:
    col_keep += [list(provider_df)[int(i)-1]]
provider_num_df = provider_df[col_keep ]

# drop col with null value > 0.3
null_lst = provider_num_df.isnull().sum(axis=0).tolist()
keep_idx = []
for i in range(len(null_lst)):
    if null_lst[i]/len(provider_num_df)<0.3:
        keep_idx += [i]
provider_num_dropnan_df = provider_num_df.iloc[:,keep_idx]

provider_state_final = provider_num_dropnan_df.groupby('nppes_provider_state').mean().reset_index()
provider_state_final = provider_state_final.add_prefix('Medical_Provider_')
provider_state_final['STATE'] = provider_state_final['Medical_Provider_nppes_provider_state']
provider_state_final = provider_state_final.drop('Medical_Provider_nppes_provider_state',axis=1)


# ### NSDUH -- NATIONAL SURVEY ON DRUG USE AND HEALTH --- Ronald

NSDUH2016_df = NSDUH2016_df.add_prefix('NSDUH_')
NSDUH2016_df['STATE'] = NSDUH2016_df['NSDUH_State'].map(us_state_abbrev)
NSDUH2016_df = NSDUH2016_df[4:]
NSDUH2016_final = NSDUH2016_df.drop('NSDUH_State',axis=1)


# ## Opioid prescribing  by County
# provider_df




# # Data Wrangling -- 11 tables

# In[71]:


final_df = pd.merge(master_df, prescribing_df, on=['STATE','County'], how='left')
final_df = pd.merge(final_df,nssats_2016_df_final, on='STATE',how='left')
final_df = pd.merge(final_df, mh2016_df_final, on='STATE',how='left')
final_df = pd.merge(final_df, NSDUH2016_final, on='STATE',how='left')
final_df = pd.merge(final_df, provider_state_final, on='STATE',how='left')
final_df = pd.merge(final_df, teds_a_2015_final, on='STATE',how='left')
final_df = pd.merge(final_df, teds_d_2014_final, on='STATE',how='left')
final_df = pd.merge(final_df, retail_drug_2016_final, on='STATE',how='left')


# ## Imputation NaN

def kmeans_missing(X, n_clusters, max_iter=10):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters, n_jobs=-1)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat

features_df = final_df.drop(['STATE','County'],axis=1)
labels, centroids, features_imp_df = kmeans_missing(features_df.values, 10, max_iter=10)

final_imp_df = pd.DataFrame(features_imp_df,columns=list(final_df)[2:])
final_imp_df = pd.concat([final_df.loc[:,['County','STATE']],final_imp_df],axis=1)


# ## Feature Ranking


X = final_imp_df.drop(['Drug_Overdose_Mortality_Rate','STATE','County'],axis=1)
y = final_imp_df['Drug_Overdose_Mortality_Rate']
y_scaled = StandardScaler().fit_transform(np.array(y).reshape(-1, 1))
X_scaled = StandardScaler().fit_transform(X)

# select k best modeling
model_reduced = SelectKBest(f_regression, k=100).fit(X_scaled, y_scaled)

# get feature ranking by F-score
feature_ranking0 = pd.DataFrame()
feature_ranking0['drivers'] = X.columns
feature_ranking0['score'] = model_reduced.scores_
feature_ranking0['T/F'] = model_reduced.get_support()
feature_ranking = feature_ranking0[feature_ranking0['T/F']==True].sort_values(by='score', ascending=False).reset_index(drop=True).drop('T/F',axis=1)

# get reduced dataset
feature_selected = X.columns[model_reduced.get_support()]
X_reduced = X[feature_selected]
df_new = X_reduced.copy()
df_new['Drug_Overdose_Mortality_Rate'] = y
df_new[['STATE','County']] = final_imp_df[['STATE','County']]


# ## export

final_df.to_csv('eda_output/final_df.csv',index=False)
!gsutil cp 'eda_output/final_df.csv' 'gs://opioid-care/eda_output/final_df.csv'

df_new.to_csv('eda_output/100Best_features_dataset.csv',index=False)
!gsutil cp 'eda_output/100Best_features_dataset.csv' 'gs://opioid-care/eda_output/100Best_features_dataset.csv'

feature_ranking.to_csv('eda_output/100Best_feature_ranking.csv',index=False)
!gsutil cp 'eda_output/100Best_feature_ranking.csv.csv' 'gs://opioid-care/eda_output/100Best_feature_ranking.csv.csv'
