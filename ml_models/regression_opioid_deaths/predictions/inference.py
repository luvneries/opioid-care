
# coding: utf-8

# In[1]:


# change these to try this notebook out
BUCKET = 'opioid-care'
PROJECT = 'opioid-care'
REGION = 'us-central1'


# In[2]:


import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


# In[3]:


#get_ipython().run_line_magic('bash', '')
#gcloud config set project $PROJECT
#gcloud config set compute/region $REGION


# In[1]:


#import pandas as pd
import time
import re

#pd.options.display.max_columns=35


# <h2> Using the model to predict </h2>
# <p>
# Send a JSON request to the endpoint of the service to make it predict death rate

# In[2]:


#test_data = pd.read_json('data/test-data.json')
#test_data


# In[1]:


request_data_json = [{"state":"MO","mentally_unhealthy_days":3.7,"physically_unhealthy_days":4.0,"pct_frequent_mental_distress":11.6,"pct_excessive_drinking":19.0,"injury_death_rate":52.88948031,"age_adjusted_mortality":304.2,"pct_frequent_physical_distress":12.0,"pct_smokers":19.5,"pct_insufficient_sleep":29.3,"preventable_hospitalization_rate":41.4,"pct_some_college":79.7502029,"pct_fair_or_poor_health":15.8,"pct_unemployed":4.12557695,"length_of_life_pctile_within_state":7.0,"median_household_income":50305.0,"pct_physically_inactive":19.5,"length_of_life_quartile_within_state":1.0,"pct_children_in_poverty":17.7,"income_20th_percentile":17667.0,"pct_children_eligible_free_lunch":30.74736162,"pct_diabetic":7.9},]


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

PROJECT = 'opioid-care'  # Change to GCP project where the Cloud ML Engine model is deployed
MODEL_NAME = 'drugdeaths'  # Change to the deployed Cloud ML Engine model
MODEL_VERSION = "v1"  # If None, the default version will be used

def predict(instances):
    """ Use a deployed model to Cloud ML Engine to perform prediction

    Args:
        instances: list of json, csv, or tf.example objects, based on the serving function called
    Returns:
        response - dictionary. If no error, response will include an item with 'predictions' key
    """

    credentials = GoogleCredentials.get_application_default()

    service = discovery.build('ml', 'v1', credentials=credentials)
    model_url = 'projects/{}/models/{}'.format(PROJECT, MODEL_NAME)

    if MODEL_VERSION is not None:
        model_url += '/versions/{}'.format(MODEL_VERSION)

    request_data = {
        'instances': instances
    }

    response = service.projects().predict(
        body=request_data,
        name=model_url
    ).execute()

    output = response.values()
    print(request_data)
    return output


# In[3]:


print(predict(request_data_json))



