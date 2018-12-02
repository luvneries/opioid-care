#!/bin/bash

echo "Training local ML model"

MODULE=~/ml_models/regression_opioid_deaths
MODEL_NAME="drugdeaths" # change to your model name

PACKAGE_PATH=${MODULE}/trainer
TRAIN_FILES=${MODULE}/data/train-data-*.csv
VALID_FILES=${MODULE}/data/eval-data.csv
STATS_FILE=${MODULE}/data/stats.json

MODEL_DIR=~/ml_models/trained_ml_models/${MODEL_NAME}
TEST_DATA=${MODULE}/data/new-data.json

gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
	--job-dir=${MODEL_DIR} \
        -- \
        --train-files=${TRAIN_FILES} \
        --num-epochs=1 \
	--feature-stats-file=${STATS_FILE} \
        --train-batch-size=500 \
        --eval-files=${VALID_FILES} \
        --eval-batch-size=500 \
        --learning-rate=0.001 \
        --hidden-units="128,40,40" \
        --layer-sizes-scale-factor=0.5 \
        --num-layers=2  


${ls ${MODEL_DIR}/export/estimator}
MODEL_LOCATION=${MODEL_DIR}/export/estimator/$(ls ${MODEL_DIR}/export/estimator | tail -1)
echo ${MODEL_LOCATION}
${ls ${MODEL_LOCATION}}

# invoke trained model to make prediction given new data instances
gcloud ml-engine local predict --model-dir=${MODEL_LOCATION} --json-instances=${TEST_DATA}
