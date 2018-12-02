#!/usr/bin/env python

# author: Pankaj Sharma
# 11/14/2018

# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'regression'

# list of all the columns (header) of the input data file(s)
HEADER = ['state',
 'mentally_unhealthy_days',
 'physically_unhealthy_days',
 'pct_frequent_mental_distress',
 'pct_excessive_drinking',
 'injury_death_rate',
 'age_adjusted_mortality',
 'pct_frequent_physical_distress',
 'pct_smokers',
 'pct_insufficient_sleep',
 'preventable_hospitalization_rate',
 'pct_some_college',
 'pct_fair_or_poor_health',
 'pct_unemployed',
 'length_of_life_pctile_within_state',
 'median_household_income',
 'pct_physically_inactive',
 'length_of_life_quartile_within_state',
 'pct_children_in_poverty',
 'income_20th_percentile',
 'pct_children_eligible_free_lunch',
 'pct_diabetic',
 'drug_overdose_mortality_rate']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [['NA'],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0]]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['mentally_unhealthy_days',
 'physically_unhealthy_days',
 'pct_frequent_mental_distress',
 'pct_excessive_drinking',
 'injury_death_rate',
 'age_adjusted_mortality',
 'pct_frequent_physical_distress',
 'pct_smokers',
 'pct_insufficient_sleep',
 'preventable_hospitalization_rate',
 'pct_some_college',
 'pct_fair_or_poor_health',
 'pct_unemployed',
 'length_of_life_pctile_within_state',
 'median_household_income',
 'pct_physically_inactive',
 'length_of_life_quartile_within_state',
 'pct_children_in_poverty',
 'income_20th_percentile',
 'pct_children_eligible_free_lunch',
 'pct_diabetic']

# numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}	#{'CHAS': 2}

# categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'state': ['ID', 'IA', 'WV', 'NE', 'CA', 'MO', 'SD', 'MI', 'TX', 'NC', 'MS', 'KY', 'IN', 'TN', 'OR', 'IL', 'DE', 'AL', 'GA', 'MT', 'MN', 'WY', 'NY', 'CO', 'WI', 'VT', 'LA', 'NJ', 'HI', 'OH', 'SC', 'MD', 'NV', 'OK', 'VA', 'FL', 'WA', 'PA', 'ND', 'UT', 'AR', 'NM', 'NH', 'KS', 'AZ', 'ME', 'MA', 'RI', 'AK', 'CT']}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# list of all the categorical feature names
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys()) \
 \
# list of all the input feature names to be used in the model
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# the column include the weight of each record
WEIGHT_COLUMN_NAME = None

# target feature name (response or class variable)
TARGET_NAME = 'drug_overdose_mortality_rate'

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = ['state',
 'mentally_unhealthy_days',
 'physically_unhealthy_days',
 'pct_frequent_mental_distress',
 'pct_excessive_drinking',
 'injury_death_rate',
 'age_adjusted_mortality',
 'pct_frequent_physical_distress',
 'pct_smokers',
 'pct_insufficient_sleep',
 'preventable_hospitalization_rate',
 'pct_some_college',
 'pct_fair_or_poor_health',
 'pct_unemployed',
 'length_of_life_pctile_within_state',
 'median_household_income',
 'pct_physically_inactive',
 'length_of_life_quartile_within_state',
 'pct_children_in_poverty',
 'income_20th_percentile',
 'pct_children_eligible_free_lunch',
 'pct_diabetic']

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [['NA'],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0]]
