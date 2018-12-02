#!/bin/bash

REGION="us-central1"
BUCKET="opioid-care" # change to your bucket name

MODEL_NAME="drugdeaths" # change to your estimator name
MODEL_VERSION="v1" # change to your model version

MODEL_BINARIES=$(gsutil ls gs://${BUCKET}/ml_models/trained_ml_models/${MODEL_NAME}/export/estimator | tail -1)
TEST_JSON=gs://${BUCKET}/test_input/new-data.json

gsutil ls ${MODEL_BINARIES}

# delete model version
gcloud ml-engine versions delete ${MODEL_VERSION} --model=${MODEL_NAME}

# delete model
gcloud ml-engine ml_models delete ${MODEL_NAME}

# deploy model to GCP
gcloud ml-engine ml_models create ${MODEL_NAME} --regions=${REGION}

# deploy model version
gcloud ml-engine versions create ${MODEL_VERSION} --model=${MODEL_NAME} --origin=${MODEL_BINARIES} --runtime-version=1.8

# invoke deployed model to make prediction given new data instances
gcloud ml-engine predict --model=${MODEL_NAME} --version=${MODEL_VERSION} --json-instances=${TEST_JSON}
