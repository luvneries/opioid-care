#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION="us-central1"
TIER="STANDARD_1" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="opioid-care" # change to your bucket name

MODEL_NAME="drugdeaths" # change to your model name

MODULE_PATH=~/ml_models/regression_opioid_deaths	# module directory path to separate from other modules

# PACKAGE_PATH variable can be a GCS location to a zipped package or module location at local.
# Change args accordingly in submit command e.g --packages (if GCS location) else --package-path

#PACKAGE_PATH=${MODULE_PATH}/trainer
PACKAGE_PATH=gs://opioid-care/ml_models/regression_opioid_deaths/package.zip

TRAIN_FILES=gs://${BUCKET}/ml_models/regression_opioid_deaths/data/train-data-*.csv
EVAL_FILES=gs://${BUCKET}/ml_models/regression_opioid_deaths/data/eval-data*.csv
MODEL_DIR=gs://${BUCKET}/ml_models/trained_ml_models/${MODEL_NAME}
STATS_FILE=gs://${BUCKET}/ml_models/trained_ml_models/data/stats.json

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}


gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --runtime-version=1.8 \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --packages=${PACKAGE_PATH}  \
        -- \
        --train-files=${TRAIN_FILES} \
	--feature-stats-file=${STATS_FILE} \
        --eval-files=${EVAL_FILES} \
	--train-steps=10000


# add --reuse-job-dir to resume training

