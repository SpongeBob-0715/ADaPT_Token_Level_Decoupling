#!/bin/bash

set -x

MODEL_PATH=xxxx  # replace it with your local file path

WANDB_DISABLE_SERVICE=true python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=your_experiment_name || true
