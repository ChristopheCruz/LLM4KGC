#!/bin/bash

python fine-tune/fine_tune_llms.py \
        --train_source webnlg_data/train_val_data \
        --train_target webnlg_data/train_val_data \
        --pred_file_source webnlg_data/train_val_data \
        --pred_file_target webnlg_data/train_val_data \		
        --base_model_name "mistral"