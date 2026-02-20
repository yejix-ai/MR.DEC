#!/bin/bash

DEMO_FILE="/path/to/3.1_after_step1_fix/mimic_admission_demo.csv"
ICD_FILE="/path/to/3.1_after_step1_fix/mimic_hosp_icd_new.csv"
LAB_FILE="/path/to/3.1_after_step1_fix/mimic_hosp_lab_filtered.csv"
MED_FILE="/path/to/3.1_after_step1_fix/mimic_hosp_med_filtered.csv"
CXR_PRETRAINED="/path/to/pretrained_weight/eva_x_base_patch16_merged520k_mim.pt"

python main.py \
    --demo_file "$DEMO_FILE" \
    --icd_file "$ICD_FILE" \
    --lab_file "$LAB_FILE" \
    --med_file "$MED_FILE" \
    --cxr_model_type eva_x_base \
    --cxr_pretrained_path "$CXR_PRETRAINED" \
    --clinical_bert_model answerdotai/ModernBERT-base \
    --balance_data \
    --max_days 10    \
    --d_model 368 \
    --nhead 8 \
    --num_layers 4 \
    --dropout 0.4 \
    --lr 1e-4 \
    --num_epochs 100 \
    --eval_every 1 \
    --l2_wd 1e-3 \
    --pos_weight 1.0 \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --save_dir ./save/end_to_end/test \
    --gpu_id 0 \
    --rand_seed 123 \
    --wandb_enabled \
    --modality multimodal \
    --wandb_run_name "multi_cls_bce_only_reg" \
    --enable_explainability \
    --feature_mode cls_only \
    --do_train \
    --model_name decoder \
    --remove_duplication \
    --loss_option bce