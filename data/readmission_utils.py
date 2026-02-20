import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb
from collections import defaultdict, Counter
import cv2
import torch.nn.functional as F


def get_readmission_label_mimic_flexible_fast(df_demo, max_seq_len=None, unpaired=False):
    """
    [수정됨] StudyDate 형식을 명확히 지정하여 Day 0 쏠림 현상 해결
    
    Args:
        df_demo: Demo dataframe
        max_seq_len: Maximum sequence length
        unpaired: If True, include all days (CXR + EHR only days). If False, only CXR days.
    """
    df = df_demo.copy()

    # 1. 입퇴원 시간 변환 (표준 포맷이므로 자동 파싱)
    for col in ['admittime', 'dischtime']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. [수정] StudyDate 변환 (MIMIC-CXR의 YYYYMMDD 포맷 대응)
    # 먼저 format='%Y%m%d'로 시도하고, 실패하면(NaT) 일반 파싱 시도 (Hybrid 방식)
    df['StudyDate_temp'] = pd.to_datetime(df['StudyDate'], format='%Y%m%d', errors='coerce')
    
    # 혹시 포맷이 섞여있어 NaT가 된 경우를 위해 일반 파싱으로 채우기
    mask_nat = df['StudyDate_temp'].isna()
    if mask_nat.sum() > 0:
        df.loc[mask_nat, 'StudyDate_temp'] = pd.to_datetime(df.loc[mask_nat, 'StudyDate'], errors='coerce')
    
    df['StudyDate'] = df['StudyDate_temp']
    df.drop(columns=['StudyDate_temp'], inplace=True)

    # 3. 정렬
    df = df.sort_values(by=['subject_id', 'hadm_id', 'StudyDate', 'StudyTime'])

    # 결과 컨테이너
    labels_list = []
    node_included_files = {}
    label_splits = []
    time_deltas = {}
    total_stay = {}
    time_idxs = {}

    grouped = df.groupby(['subject_id', 'hadm_id', 'admittime', 'dischtime', 'splits', 'readmitted_within_30days'], sort=False)

    for (subject_id, hadm_id, admit_dt, discharge_dt, split, label_raw), group in tqdm(grouped, desc="Processing Admissions"):
        
        curr_name = str(subject_id) + "_" + str(hadm_id)
        
        # 날짜 계산 (.floor('d') 사용)
        admit_date_only = admit_dt.floor('d') 
        disch_date_only = discharge_dt.floor('d')
        
        # Hospital Stay
        hospital_stay_days = (disch_date_only - admit_date_only).days + 1
        total_stay[curr_name] = hospital_stay_days
        
        if unpaired:
            # UNPAIRED MODE: 모든 day 포함 (CXR이 없는 day도)
            # 각 day별로 CXR이 있는지 확인
            all_days_files = []
            all_days_indices = []
            
            # CXR이 있는 day 매핑
            cxr_day_map = {}
            for _, row in group.iterrows():
                study_date_only = pd.to_datetime(row['StudyDate']).floor('d')
                day_idx = (study_date_only - admit_date_only).days
                day_idx = max(0, day_idx)  # 음수 방지
                
                if day_idx not in cxr_day_map:
                    cxr_day_map[day_idx] = []
                cxr_day_map[day_idx].append(row['image_path'])
            
            # 모든 day에 대해 레코드 생성
            for day_num in range(hospital_stay_days):
                if day_num in cxr_day_map:
                    # CXR이 있는 day: 각 CXR마다 레코드 생성
                    for img_path in cxr_day_map[day_num]:
                        all_days_files.append(img_path)
                        all_days_indices.append(day_num)
                else:
                    # CXR이 없는 day: None으로 표시
                    all_days_files.append(None)
                    all_days_indices.append(day_num)
            
            if max_seq_len is not None and len(all_days_files) > max_seq_len:
                all_days_files = all_days_files[-max_seq_len:]
                all_days_indices = all_days_indices[-max_seq_len:]
            
            node_included_files[curr_name] = all_days_files
            time_idxs[curr_name] = np.array(all_days_indices)
            
            # Time Delta 계산
            if len(all_days_indices) > 1:
                deltas = np.diff(all_days_indices, prepend=all_days_indices[0])
                safe_stay = hospital_stay_days if hospital_stay_days > 0 else 1
                time_deltas[curr_name] = deltas / safe_stay
            else:
                time_deltas[curr_name] = np.array([0.0])
                
        else:
            # PAIRED MODE: CXR이 있는 day만 포함
            if max_seq_len is not None:
                group = group.tail(max_seq_len)

            files = group['image_path'].tolist()
            node_included_files[curr_name] = files

            study_dates_only = group['StudyDate'].dt.floor('d')

            # Time Index
            days_from_admit = (study_dates_only - admit_date_only).dt.days.values
            days_from_admit = np.maximum(days_from_admit, 0)
            time_idxs[curr_name] = days_from_admit

            # Time Delta
            if len(files) > 0:
                deltas = study_dates_only.diff().dt.days.fillna(0).values
                safe_stay = hospital_stay_days if hospital_stay_days > 0 else 1
                time_deltas[curr_name] = deltas / safe_stay
            else:
                time_deltas[curr_name] = np.array([])
        
        labels_list.append(1 if str(label_raw).lower() == "true" else 0)
        label_splits.append(split)

    return (
        np.array(labels_list),
        node_included_files,
        label_splits,
        time_deltas,
        total_stay,
        time_idxs,
    )
    
    
def merge_subwords_and_scores(tokens, scores, method='max'):
    """
    BERT/Byte-level/SentencePiece 토큰을 단어 단위로 병합하고 점수를 집계합니다.
    - WordPiece: "##" 접두사는 이전 단어에 이어 붙임
    - Byte-level BPE: "Ġ" (RoBERTa류) / "▁" (SentencePiece류)는 "새 단어 시작" 신호
    """
    specials = {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}
    merged_words, merged_scores = [], []

    current_word = ""
    current_score_list = []

    for token, score in zip(tokens, scores):
        if token in specials:
            continue

        is_new_word = False
        clean = token

        if clean.startswith("##"):
            clean = clean[2:]
            is_new_word = False
        elif clean.startswith("Ġ") or clean.startswith("▁"):
            clean = clean[1:]
            is_new_word = True
        else:
            # No prefix: continuation unless we don't have a current word yet
            is_new_word = (current_word == "")

        if is_new_word:
            if current_word:
                final_score = max(current_score_list) if method == 'max' else sum(current_score_list) / len(current_score_list)
                merged_words.append(current_word)
                merged_scores.append(final_score)
            current_word = clean
            current_score_list = [score]
        else:
            # continuation of current word
            current_word += clean
            current_score_list.append(score)

    if current_word:
        final_score = max(current_score_list) if method == 'max' else sum(current_score_list) / len(current_score_list)
        merged_words.append(current_word)
        merged_scores.append(final_score)

    return merged_words, merged_scores

def explain_admission_end_to_end(
    model, cxr_processor, ehr_processor,
    sample_data, df_demo, df_icd, df_lab, df_med, device,
    max_days=30, modality="multimodal"
):
    model.eval()

    hadm_id = sample_data['hadm_id']
    img_paths = sample_data['img_paths']
    day_indices = sample_data['day_indices']

    if not img_paths: return None

    # BERT 임베딩 레이어 찾기 (text 또는 multimodal일 때만)
    bert_model = None
    embed_layer = None
    if modality in ["multimodal", "text"] and ehr_processor:
        bert_model = ehr_processor.model
        if hasattr(bert_model, 'embeddings'):
            embed_layer = bert_model.embeddings
        elif hasattr(bert_model, 'bert') and hasattr(bert_model.bert, 'embeddings'):
            embed_layer = bert_model.bert.embeddings
        elif hasattr(bert_model, 'distilbert') and hasattr(bert_model.distilbert, 'embeddings'):
            embed_layer = bert_model.distilbert.embeddings

    with torch.set_grad_enabled(True):
        seq_len = min(sample_data['num_days'], max_days)
        
        processed_img_feats = []
        processed_ehr_feats = []
        raw_images_for_viz = {} 
        input_embeddings_list = [] 
        day_to_path = {d: p for p, d in zip(img_paths, day_indices)}
        img_tensors_map = {}
        cls_tokens_map = {}  # Image CLS 토큰 저장 (gradient 추출용)
        ehr_cls_tokens_map = {}  # EHR CLS 토큰 저장 (gradient 추출용)
        img_patch_features_map = {}  # Image 패치별 feature 저장 (CLS gradient 분배용)
        ehr_token_features_map = {}  # EHR 토큰별 feature 저장 (CLS gradient 분배용)
        ehr_cls_features_map = {}  # EHR CLS feature 저장 (backward 전 값, attention weight 계산용) 

        try:
            df_adm = df_demo[df_demo['hadm_id'] == hadm_id].iloc[0]
            
            for t in range(seq_len):
                # EHR Feature (text 또는 multimodal일 때만)
                if modality in ["multimodal", "text"] and ehr_processor:
                    text = ehr_processor.create_daily_ehr_text(
                        df_adm, df_icd, df_lab, df_med, hadm_id, t
                    )
                    inputs = ehr_processor.tokenizer(
                        text, padding='max_length', truncation=True, max_length=512, return_tensors='pt'
                    ).to(device)
                    
                    if embed_layer:
                        input_ids = inputs['input_ids']
                        full_emb = embed_layer(input_ids)
                        full_emb.retain_grad() 
                        input_embeddings_list.append((full_emb, input_ids, text))
                        bert_out = bert_model(inputs_embeds=full_emb, attention_mask=inputs['attention_mask'])
                    else:
                        bert_out = bert_model(**inputs)
                        input_embeddings_list.append((None, inputs['input_ids'], text))

                    if hasattr(bert_out, 'last_hidden_state'):
                        ehr_feat_day = bert_out.last_hidden_state[:, 0, :]
                        # [CLS gradient 분배용] 모든 토큰 feature 저장 (detach하여 저장)
                        token_features = bert_out.last_hidden_state.detach()  # (B, seq_len, dim)
                        ehr_token_features_map[t] = token_features
                    else:
                        ehr_feat_day = bert_out[0][:, 0, :]
                        # [CLS gradient 분배용] 모든 토큰 feature 저장 (detach하여 저장)
                        token_features = bert_out[0].detach()  # (B, seq_len, dim)
                        ehr_token_features_map[t] = token_features
                    
                    # [핵심 수정] EHR CLS 토큰에 retain_grad() 추가 (Image와 동일한 방식)
                    ehr_feat_day.retain_grad()
                    ehr_cls_tokens_map[t] = ehr_feat_day
                    # [CLS feature 저장] backward 전의 CLS feature 값 저장 (attention weight 계산용)
                    ehr_cls_features_map[t] = ehr_feat_day.detach().clone()  # backward 전 값 보존
                    processed_ehr_feats.append(ehr_feat_day)
                else:
                    # EHR 사용 안 함 - zero padding
                    ehr_feat_day = torch.zeros(1, 768).to(device)  # Default dim
                    processed_ehr_feats.append(ehr_feat_day)

                # Image Feature (image 또는 multimodal일 때만)
                if modality in ["multimodal", "image"] and cxr_processor:
                    if t in day_to_path:
                        raw_img = Image.open(day_to_path[t]).convert('RGB')
                        raw_images_for_viz[t] = raw_img
                        # HuggingFace 모델은 image_processor 사용, 그렇지 않으면 transforms 사용
                        if getattr(cxr_processor, "is_huggingface", False) and hasattr(cxr_processor, "image_processor"):
                            inputs = cxr_processor.image_processor(raw_img, return_tensors="pt").to(device)
                            img_tensor = inputs["pixel_values"]
                        elif hasattr(cxr_processor, "transforms") and cxr_processor.transforms is not None:
                            img_tensor = cxr_processor.transforms(raw_img).unsqueeze(0).to(device)
                        else:
                            img_tensor = torch.zeros(1, 3, 224, 224).to(device)
                        img_tensor.requires_grad_(True)
                        img_tensors_map[t] = img_tensor
                        
                        # 모델 출력에서 CLS/pooled 표현 선택
                        outputs = cxr_processor.encoder(pixel_values=img_tensor) if getattr(cxr_processor, "is_huggingface", False) else cxr_processor.encoder.forward_features(img_tensor)
                        if getattr(cxr_processor, "is_huggingface", False):
                            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                                feat_img = outputs.pooler_output
                                # [CLS gradient 분배용] 모든 패치 feature 저장 (last_hidden_state 사용)
                                if hasattr(outputs, "last_hidden_state"):
                                    patch_features = outputs.last_hidden_state  # (B, seq_len, dim)
                                    img_patch_features_map[t] = patch_features
                            elif hasattr(outputs, "last_hidden_state"):
                                feat_img = outputs.last_hidden_state[:, 0, :]
                                # [CLS gradient 분배용] 모든 패치 feature 저장
                                patch_features = outputs.last_hidden_state  # (B, seq_len, dim)
                                img_patch_features_map[t] = patch_features
                            else:
                                feat_img = outputs[0][:, 0, :]
                                patch_features = outputs[0]  # (B, seq_len, dim)
                                img_patch_features_map[t] = patch_features
                        else:
                            feat_img = cxr_processor.encoder.forward_head(outputs, pre_logits=True)
                            # [CLS gradient 분배용] 모든 패치 feature 저장 (forward_features 출력 사용)
                            if hasattr(outputs, 'shape') and len(outputs.shape) >= 2:
                                # outputs가 (B, seq_len, dim) 형태인 경우
                                patch_features = outputs  # (B, seq_len, dim)
                                img_patch_features_map[t] = patch_features
                        
                        # [핵심 수정] CLS 토큰에 retain_grad() 추가하여 멀티모달 모델에서 흐른 gradient 추출 가능하게
                        feat_img.retain_grad()
                        cls_tokens_map[t] = feat_img
                    else:
                        feat_img = torch.zeros(1, cxr_processor.feat_dim).to(device)
                    processed_img_feats.append(feat_img)
                else:
                    # Image 사용 안 함 - zero padding
                    feat_img = torch.zeros(1, 768).to(device)  # Default dim
                    processed_img_feats.append(feat_img)
                
        except Exception as e:
            print(f"Error prep explain inputs: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Transformer Forward - modality에 따라 처리
        stack_img = torch.stack(processed_img_feats, dim=0) 
        stack_ehr = torch.stack(processed_ehr_feats, dim=0) 
        
        max_dim = max(stack_img.shape[-1], stack_ehr.shape[-1])
        if stack_img.shape[-1] < max_dim:
            pad = torch.zeros(seq_len, 1, max_dim - stack_img.shape[-1]).to(device)
            stack_img = torch.cat([stack_img, pad], dim=-1)
        if stack_ehr.shape[-1] < max_dim:
            pad = torch.zeros(seq_len, 1, max_dim - stack_ehr.shape[-1]).to(device)
            stack_ehr = torch.cat([stack_ehr, pad], dim=-1)

        # Modality에 따라 combined_feat 생성
        if modality == "multimodal":
            combined_feat = torch.stack([stack_img, stack_ehr], dim=2).permute(1, 0, 2, 3)  # (1, seq, 2, dim)
        elif modality == "image":
            combined_feat = stack_img.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
        else:  # text
            combined_feat = stack_ehr.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
            
        mask = torch.ones(1, seq_len, dtype=torch.bool).to(device)

        day_outputs = model(combined_feat, mask)
        
        if isinstance(day_outputs, list): day_outputs = day_outputs[-1]
        if day_outputs.dim() == 3: final_logit = day_outputs[0, -1, 0]
        else: final_logit = day_outputs[0]
        
        # Backward
        model.zero_grad()
        if cxr_processor and hasattr(cxr_processor, 'encoder'): 
            cxr_processor.encoder.zero_grad()
        if bert_model is not None:
            bert_model.zero_grad()

        # 1단계: 멀티모달 모델 backward로 CLS gradient 계산
        final_logit.backward(retain_graph=False)
        
        # 2단계: CLS gradient를 사용해서 encoder → img backprop
        # 각 day별로 CLS를 output으로 하는 forward를 다시 수행하고 backward
        for t in cls_tokens_map:
            if t in img_tensors_map and cls_tokens_map[t].grad is not None:
                cls_grad = cls_tokens_map[t].grad  # 멀티모달 모델에서 계산된 CLS gradient
                img_tensor = img_tensors_map[t]
                
                # [핵심] CLS를 output으로 하는 forward를 다시 수행
                # encoder는 freeze되어 있지만, forward는 가능하고 backward도 input까지는 가능
                if getattr(cxr_processor, "is_huggingface", False):
                    outputs = cxr_processor.encoder(pixel_values=img_tensor)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        cls_output = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        cls_output = outputs.last_hidden_state[:, 0, :]
                    else:
                        cls_output = outputs[0][:, 0, :]
                else:
                    outputs = cxr_processor.encoder.forward_features(img_tensor)
                    cls_output = cxr_processor.encoder.forward_head(outputs, pre_logits=True)
                
                # CLS gradient를 사용해서 encoder → img backprop
                # 이렇게 하면 "CLS gradient → encoder → img" 경로의 gradient가 계산됨
                img_grad_from_cls = torch.autograd.grad(
                    outputs=cls_output,
                    inputs=img_tensor,
                    grad_outputs=cls_grad,
                    retain_graph=False,
                    only_inputs=True,
                    create_graph=False
                )[0]
                
                # img_tensor.grad에 저장 (기존 gradient가 있으면 더하기)
                if img_tensor.grad is not None:
                    img_tensor.grad = img_tensor.grad + img_grad_from_cls
                else:
                    img_tensor.grad = img_grad_from_cls
        
        # EHR도 동일한 방식으로 처리
        for t in ehr_cls_tokens_map:
            if t < len(input_embeddings_list):
                emb, input_ids, text = input_embeddings_list[t]
                if emb is not None and ehr_cls_tokens_map[t].grad is not None:
                    ehr_cls_grad = ehr_cls_tokens_map[t].grad
                    
                    # CLS를 output으로 하는 forward를 다시 수행
                    if embed_layer:
                        # embed_layer를 사용하는 경우: full_emb를 다시 계산
                        full_emb = embed_layer(input_ids)
                        attention_mask = torch.ones_like(input_ids)
                        bert_out = bert_model(inputs_embeds=full_emb, attention_mask=attention_mask)
                    else:
                        # embed_layer가 없는 경우: tokenizer로 다시 입력 생성
                        inputs = ehr_processor.tokenizer(
                            text,
                            padding='max_length', truncation=True, max_length=512, return_tensors='pt'
                        ).to(device)
                        bert_out = bert_model(**inputs)
                    
                    if hasattr(bert_out, 'last_hidden_state'):
                        cls_output = bert_out.last_hidden_state[:, 0, :]
                    else:
                        cls_output = bert_out[0][:, 0, :]
                    
                    # CLS gradient를 사용해서 BERT → embedding backprop
                    # emb는 이미 input_embeddings_list에 저장된 embedding
                    if embed_layer:
                        # embed_layer를 사용하는 경우: full_emb에 대한 gradient 계산
                        emb_grad_from_cls = torch.autograd.grad(
                            outputs=cls_output,
                            inputs=full_emb,
                            grad_outputs=ehr_cls_grad,
                            retain_graph=False,
                            only_inputs=True,
                            create_graph=False
                        )[0]
                        
                        # emb.grad에 저장
                        if emb.grad is not None:
                            emb.grad = emb.grad + emb_grad_from_cls
                        else:
                            emb.grad = emb_grad_from_cls
                    # else: embed_layer가 없으면 emb.grad는 backward()로 이미 계산됨

        # Visualization Logic
        viz_results = {}
        day_scores = [] 
        
        for t in range(seq_len):
            score = 0
            if t in img_tensors_map and img_tensors_map[t].grad is not None:
                score += img_tensors_map[t].grad.abs().sum().item()
            if t < len(input_embeddings_list):
                emb, _, _ = input_embeddings_list[t]
                if emb is not None and emb.grad is not None:
                    score += emb.grad.abs().sum().item()
            day_scores.append((t, score))
        
        day_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_days = day_scores[:5] 
        
        # Global Keywords (후처리된 단어 단위로 집계)
        global_token_scores = defaultdict(float)
        for t, (emb, input_ids, _) in enumerate(input_embeddings_list):
            if emb is not None and emb.grad is not None:
                # [수정 1] .detach() 추가
                token_grads = (emb.grad * emb).sum(dim=-1).abs().squeeze(0).detach().cpu().numpy()
                
                tokens = ehr_processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                # [후처리 적용] BPE 토큰을 단어 단위로 병합
                clean_words, clean_scores = merge_subwords_and_scores(tokens, token_grads, method='max')
                
                # 병합된 단어 단위로 점수 집계
                for word, score in zip(clean_words, clean_scores):
                    if word not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                        global_token_scores[word] += float(score)
        
        sorted_global_tokens = sorted(global_token_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        table_data = [[t, float(s)] for t, s in sorted_global_tokens]
        viz_results['global_top_keywords'] = wandb.plot.bar(
            wandb.Table(data=table_data, columns=["Token", "Aggregated Score"]),
            "Token", "Aggregated Score", title="Global Top-10 Keywords"
        )

        for rank, (day_idx, day_score) in enumerate(top_k_days):
            day_prefix = f"day_{day_idx}"
            
            if day_idx in img_tensors_map:
                target_tensor = img_tensors_map[day_idx]
                
                # [CLS gradient 활용] CLS gradient 정보를 가중치로 사용
                cls_grad_weight = 1.0
                if day_idx in cls_tokens_map and cls_tokens_map[day_idx].grad is not None:
                    cls_grad = cls_tokens_map[day_idx].grad
                    cls_grad_weight = cls_grad.abs().mean().item()  # CLS gradient 크기를 가중치로 사용
                    # 정규화 (0~1 범위로)
                    if cls_grad_weight > 0:
                        # 다른 day들의 CLS gradient와 비교해서 상대적 가중치 계산
                        all_cls_grads = [cls_tokens_map[t].grad.abs().mean().item() 
                                       for t in cls_tokens_map if cls_tokens_map[t].grad is not None]
                        if len(all_cls_grads) > 0:
                            max_cls_grad = max(all_cls_grads)
                            cls_grad_weight = cls_grad_weight / (max_cls_grad + 1e-8)
                
                if target_tensor.grad is not None and target_tensor.grad.abs().max() > 1e-6:
                    # img_tensor.grad가 있는 경우: 기존 방식 사용
                    grad = target_tensor.grad[0].cpu().detach()
                    saliency = grad.abs().max(dim=0)[0].numpy()
                    
                    # [CLS gradient 분배] cls_only 모드: CLS gradient를 패치별로 분배
                    if day_idx in cls_tokens_map and cls_tokens_map[day_idx].grad is not None and day_idx in img_patch_features_map:
                        cls_grad = cls_tokens_map[day_idx].grad  # (1, dim)
                        cls_feat = cls_tokens_map[day_idx]  # (1, dim)
                        patch_features = img_patch_features_map[day_idx]  # (1, num_patches, dim)
                        
                        # 각 패치 feature와 CLS feature의 유사도 계산 (코사인 유사도)
                        # patch_features: (1, num_patches, dim), cls_feat: (1, dim)
                        cls_feat_norm = cls_feat / (cls_feat.norm(dim=-1, keepdim=True) + 1e-8)  # (1, dim)
                        patch_features_norm = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)  # (1, num_patches, dim)
                        
                        # 코사인 유사도: (1, num_patches, dim) @ (1, dim, 1) -> (1, num_patches, 1) -> (1, num_patches)
                        attention_weights = (patch_features_norm @ cls_feat_norm.unsqueeze(-1)).squeeze(-1)  # (1, num_patches)
                        attention_weights = torch.softmax(attention_weights * 10, dim=-1)  # Temperature scaling으로 sharpening
                        
                        # CLS gradient를 각 패치로 분배 (attention weight 사용)
                        cls_grad_magnitude = cls_grad.abs().mean().item()  # CLS gradient 크기
                        patch_weights = attention_weights[0].cpu().detach().numpy()  # (num_patches,)
                        
                        # saliency map에 패치별 가중치 적용
                        # saliency는 (H, W) 형태이므로, 패치 수에 맞게 reshape 필요
                        # 일반적으로 Vision Transformer는 H*W/16 또는 H*W/14 패치를 가짐
                        num_patches = patch_weights.shape[0]
                        saliency_h, saliency_w = saliency.shape
                        
                        # 패치별 가중치를 saliency map 크기에 맞게 upsampling
                        if num_patches > 0:
                            # 패치 그리드 크기 추정 (대략적으로)
                            patch_grid_size = int(np.sqrt(num_patches))
                            if patch_grid_size * patch_grid_size == num_patches:
                                # 정사각형 그리드인 경우
                                patch_weights_2d = patch_weights.reshape(patch_grid_size, patch_grid_size)
                                # saliency map 크기로 upsampling
                                patch_weights_2d_resized = cv2.resize(patch_weights_2d, (saliency_w, saliency_h), interpolation=cv2.INTER_LINEAR)
                                # 패치별 가중치를 saliency에 곱하기
                                saliency = saliency * patch_weights_2d_resized
                            else:
                                # 정사각형이 아닌 경우: 평균 가중치 사용
                                avg_weight = patch_weights.mean()
                                saliency = saliency * avg_weight
                    else:
                        # 패치 feature가 없는 경우: 기존 방식 (상수 가중치)
                        saliency = saliency * cls_grad_weight
                    
                    # [개선] Percentile normalization으로 작은 gradient도 잘 보이게
                    if saliency.max() > 0:
                        p5, p95 = np.percentile(saliency[saliency > 0], [5, 95]) if (saliency > 0).any() else (0, saliency.max())
                        if p95 > p5:
                            saliency = np.clip((saliency - p5) / (p95 - p5), 0, 1)
                        else:
                            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                    else:
                        saliency = np.zeros_like(saliency)
                else:
                    # img_tensor.grad가 None이거나 매우 작은 경우: CLS gradient 기반 uniform 히트맵
                    # 주의: 이 방법은 이미지 내 위치 정보를 제공하지 않음 (day-level 중요도만)
                    raw_img = raw_images_for_viz[day_idx]
                    saliency = np.ones((raw_img.size[1], raw_img.size[0]), dtype=np.float32) * cls_grad_weight
                    raise Exception("img_tensor.grad is None, using CLS gradient weight only")
                    # CLS gradient가 크면 전체 이미지에 uniform하게 적용
                    # 하지만 실제로 어느 부분이 중요한지는 알 수 없음
                
                raw_img = raw_images_for_viz[day_idx]
                saliency = cv2.resize(saliency, raw_img.size)
                # 더 부드러운 히트맵을 위해 여러 번 blur 적용 및 더 큰 kernel 사용
                # 1차 blur: 큰 kernel로 전체적인 smoothing
                saliency = cv2.GaussianBlur(saliency, (51, 51), 15)
                # 2차 blur: 추가 smoothing
                saliency = cv2.GaussianBlur(saliency, (31, 31), 10)
                # 히트맵 생성 (더 부드러운 colormap 사용)
                heatmap = cm.jet(saliency)[:, :, :3]
                heatmap = (heatmap * 255).astype(np.uint8)
                # 더 부드러운 오버레이: 원본 이미지를 더 많이 보이게 (0.75:0.25)
                overlay = cv2.addWeighted(np.array(raw_img), 0.75, heatmap, 0.25, 0)
                
                # Gradient 통계 로깅 (디버깅용)
                if target_tensor.grad is not None:
                    # CPU로 이동 후 detach (안전장치: 여러 번 호출해도 안전)
                    grad_abs = target_tensor.grad.abs()
                    if grad_abs.device.type != 'cpu':
                        grad_abs = grad_abs.cpu()
                    grad_abs = grad_abs.detach()
                    grad_stats = {
                        'max': float(grad_abs.max().item()),
                        'mean': float(grad_abs.mean().item()),
                        'p99': float(np.percentile(grad_abs.numpy().flatten(), 99)),
                        'nonzero_frac': float((grad_abs > 1e-6).float().mean().item()),
                        'cls_grad_weight': float(cls_grad_weight)  # CLS gradient 가중치 추가
                    }
                else:
                    grad_stats = {
                        'max': 0.0,
                        'mean': 0.0,
                        'p99': 0.0,
                        'nonzero_frac': 0.0,
                        'cls_grad_weight': float(cls_grad_weight),  # CLS gradient 가중치
                        'note': 'img_tensor.grad is None, using CLS gradient weight only'
                    }
                viz_results[f"{day_prefix}_cxr_grad_stats"] = grad_stats
                
                viz_results[f"{day_prefix}_cxr"] = wandb.Image(
                    overlay, caption=f"Day {day_idx} (Rank {rank+1}, img_grad_max={grad_stats.get('max', 0):.4f}, cls_weight={cls_grad_weight:.4f})"
                )

            if day_idx < len(input_embeddings_list):
                emb, input_ids, _ = input_embeddings_list[day_idx]
                if emb is not None and emb.grad is not None:
                    # [EHR CLS gradient weight 적용] 이미지와 동일한 로직
                    ehr_cls_grad_weight = 1.0
                    if day_idx in ehr_cls_tokens_map and ehr_cls_tokens_map[day_idx].grad is not None:
                        ehr_cls_grad = ehr_cls_tokens_map[day_idx].grad
                        ehr_cls_grad_weight = ehr_cls_grad.abs().mean().item()
                        # 정규화 (0~1 범위로)
                        if ehr_cls_grad_weight > 0:
                            # 다른 day들의 EHR CLS gradient와 비교해서 상대적 가중치 계산
                            all_ehr_cls_grads = [ehr_cls_tokens_map[t].grad.abs().mean().item() 
                                                for t in ehr_cls_tokens_map if ehr_cls_tokens_map[t].grad is not None]
                            if len(all_ehr_cls_grads) > 0:
                                max_ehr_cls_grad = max(all_ehr_cls_grads)
                                ehr_cls_grad_weight = ehr_cls_grad_weight / (max_ehr_cls_grad + 1e-8)
                    
                    # [수정 2] .detach() 추가
                    token_grads = (emb.grad * emb).sum(dim=-1).abs().squeeze(0).detach().cpu().numpy()
                    
                    # [CLS gradient 분배] cls_only 모드: CLS gradient를 토큰별로 분배
                    # 조건 확인: ehr_cls_tokens_map에 day_idx가 있고, gradient가 있으며, ehr_token_features_map에도 있어야 함
                    has_cls_grad = day_idx in ehr_cls_tokens_map and ehr_cls_tokens_map[day_idx].grad is not None
                    has_token_features = day_idx in ehr_token_features_map
                    
                    if has_cls_grad and has_token_features and day_idx in ehr_cls_features_map:
                        ehr_cls_grad = ehr_cls_tokens_map[day_idx].grad  # (1, dim)
                        # CLS feature는 backward 전에 저장된 값 사용 (이미지와 동일한 방식)
                        ehr_cls_feat = ehr_cls_features_map[day_idx]  # (1, dim) - backward 전 값
                        token_features = ehr_token_features_map[day_idx]  # (1, seq_len, dim) - 이미 detach된 값
                        
                        # 각 토큰 feature와 CLS feature의 유사도 계산 (코사인 유사도)
                        # 이미지와 동일한 방식으로 GPU에서 직접 계산
                        ehr_cls_feat_norm = ehr_cls_feat / (ehr_cls_feat.norm(dim=-1, keepdim=True) + 1e-8)  # (1, dim)
                        token_features_norm = token_features / (token_features.norm(dim=-1, keepdim=True) + 1e-8)  # (1, seq_len, dim)
                        
                        # 코사인 유사도: (1, seq_len, dim) @ (1, dim, 1) -> (1, seq_len, 1) -> (1, seq_len)
                        attention_weights = (token_features_norm @ ehr_cls_feat_norm.unsqueeze(-1)).squeeze(-1)  # (1, seq_len)
                        attention_weights = torch.softmax(attention_weights * 10, dim=-1)  # Temperature scaling으로 sharpening
                        
                        # CLS gradient를 각 토큰으로 분배 (attention weight 사용)
                        token_weights = attention_weights[0].cpu().detach().numpy()  # (seq_len,)
                        
                        # token_grads에 토큰별 가중치 적용
                        # token_grads는 (seq_len,) 형태이므로 직접 곱하기
                        if len(token_weights) == len(token_grads):
                            token_grads = token_grads * token_weights
                        else:
                            # 길이가 다른 경우: 평균 가중치 사용
                            avg_weight = float(token_weights.mean())
                            token_grads = token_grads * avg_weight
                    else:
                        # 토큰 feature가 없는 경우: 기존 방식 (상수 가중치)
                        token_grads = token_grads * ehr_cls_grad_weight
                    
                    # [개선] Percentile normalization 적용 (이미지와 동일한 방식)
                    # attention weight를 곱한 후 값이 너무 작아질 수 있으므로 normalization 필요
                    if len(token_grads) > 0 and token_grads.max() > 0:
                        token_grads_array = np.array(token_grads)
                        if (token_grads_array > 0).any():
                            # 이미지와 동일한 percentile normalization (5-95%)
                            p5, p95 = np.percentile(token_grads_array[token_grads_array > 0], [5, 95])
                            if p95 > p5:
                                token_grads = np.clip((token_grads_array - p5) / (p95 - p5 + 1e-8), 0, 1)
                            else:
                                # 모든 값이 같거나 p95 == p5인 경우: min-max normalization
                                token_grads = (token_grads_array - token_grads_array.min()) / (token_grads_array.max() - token_grads_array.min() + 1e-8)
                        else:
                            token_grads = np.zeros_like(token_grads_array)
                    
                    tokens = ehr_processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                    clean_words, clean_scores = merge_subwords_and_scores(tokens, token_grads, method='max')
                    html_code = get_colored_text_html(clean_words, clean_scores)
                    viz_results[f"{day_prefix}_ehr_html"] = wandb.Html(html_code, inject=False)

        return viz_results
    
def get_colored_text_html(words, scores):
    """Generate HTML for text coloring based on importance scores."""
    if len(scores) > 0:
        max_score = max(scores)
        if max_score > 0:
            # Percentile normalization으로 더 부드러운 하이라이트 (이미지와 동일한 로직)
            scores_array = np.array(scores)
            if (scores_array > 0).any():
                p5, p95 = np.percentile(scores_array[scores_array > 0], [5, 95])
                if p95 > p5:
                    norm_scores = np.clip((scores_array - p5) / (p95 - p5 + 1e-8), 0, 1)
                    norm_scores = norm_scores.tolist()
                else:
                    norm_scores = [s / max_score for s in scores]
            else:
                norm_scores = [s / max_score for s in scores]
        else:
            norm_scores = [0] * len(scores)
    else:
        norm_scores = []
        
    html_parts = ['<div style="font-family: monospace; line-height: 1.5;">']
    
    for word, score in zip(words, norm_scores):
        if word in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        # 더 부드러운 하이라이트: alpha 값을 조정 (normalization 후이므로 더 높은 값 사용)
        alpha = score * 0.6  # normalization 후이므로 더 높은 alpha 값 사용
        # 더 부드러운 색상: 빨간색 대신 주황색 계열 사용
        bg_color = f"rgba(255, 165, 0, {alpha:.2f})"
        html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px; border-radius: 4px; transition: background-color 0.2s;">{word}</span>')
        
    html_parts.append('</div>')
    return " ".join(html_parts)


def explain_admission_end_to_end_all_features(
    model, cxr_processor, ehr_processor, 
    sample_data, df_demo, df_icd, df_lab, df_med, device,
    max_days=2, max_seq_len_modality=512, modality="multimodal"
):
    """
    v5용 시각화 함수: All Features 버전
    - 각 modality가 (seq_len, dim) 형태
    - max_days는 보통 2 (first_last) 또는 더 많은 날짜
    """
    model.eval()
    
    hadm_id = sample_data['hadm_id']
    img_paths = sample_data['img_paths']
    day_indices = sample_data['day_indices']
    
    if not img_paths: 
        return None

    # BERT 임베딩 레이어 찾기
    bert_model = ehr_processor.model if ehr_processor else None
    embed_layer = None
    
    if bert_model:
        if hasattr(bert_model, 'embeddings'):
            embed_layer = bert_model.embeddings
        elif hasattr(bert_model, 'bert') and hasattr(bert_model.bert, 'embeddings'):
            embed_layer = bert_model.bert.embeddings
        elif hasattr(bert_model, 'distilbert') and hasattr(bert_model.distilbert, 'embeddings'):
            embed_layer = bert_model.distilbert.embeddings

    with torch.set_grad_enabled(True):
        seq_len = min(sample_data['num_days'], max_days)
        
        # 준비할 데이터
        day_features_list = []
        day_token_masks = []
        raw_images_for_viz = {}
        img_tensors_map = {}
        input_embeddings_list = []
        
        try:
            df_adm = df_demo[df_demo['hadm_id'] == hadm_id].iloc[0]
            
            for t in range(seq_len):
                day_num = day_indices[t] + 1
                img_path = img_paths[t]
                
                # 1. Image Feature
                if modality in ["multimodal", "image"] and cxr_processor:
                    raw_img = Image.open(img_path).convert('RGB')
                    raw_images_for_viz[t] = raw_img
                    
                    # Align preprocessing with training: use HF image_processor if available
                    if getattr(cxr_processor, "is_huggingface", False) and hasattr(cxr_processor, "image_processor"):
                        inputs = cxr_processor.image_processor(raw_img, return_tensors="pt").to(device)
                        img_tensor = inputs["pixel_values"]
                    elif hasattr(cxr_processor, 'transforms') and cxr_processor.transforms is not None:
                        img_tensor = cxr_processor.transforms(raw_img).unsqueeze(0).to(device)
                    else:
                        img_tensor = torch.zeros(1, 3, 224, 224).to(device)
                    
                    img_tensor.requires_grad_(True)
                    img_tensors_map[t] = img_tensor
                    
                    # All features 추출
                    if getattr(cxr_processor, "is_huggingface", False):
                        # HuggingFace 모델: pixel_values를 사용하여 forward
                        outputs = cxr_processor.encoder(pixel_values=img_tensor)
                        if hasattr(outputs, "last_hidden_state"):
                            feat_raw = outputs.last_hidden_state  # (B, seq_len, dim)
                        else:
                            feat_raw = outputs[0]  # (B, seq_len, dim)
                    else:
                        # 일반 모델: forward_features 사용
                        feat_raw = cxr_processor.encoder.forward_features(img_tensor)  # (B, seq, dim)
                    
                    if feat_raw.dim() == 2:
                        feat_raw = feat_raw.unsqueeze(1)
                    img_feat = feat_raw  # (1, seq, dim)
                else:
                    img_feat = None
                
                # 2. EHR Feature  
                if modality in ["multimodal", "text"] and ehr_processor:
                    text = ehr_processor.create_daily_ehr_text(
                        df_adm, df_icd, df_lab, df_med, hadm_id, day_num
                    )
                    inputs = ehr_processor.tokenizer(
                        text, padding='max_length', truncation=True, 
                        max_length=max_seq_len_modality, return_tensors='pt'
                    ).to(device)
                    
                    if embed_layer:
                        input_ids = inputs['input_ids']
                        full_emb = embed_layer(input_ids)
                        full_emb.retain_grad()
                        input_embeddings_list.append((full_emb, input_ids, text))
                        bert_out = bert_model(inputs_embeds=full_emb, attention_mask=inputs['attention_mask'])
                    else:
                        bert_out = bert_model(**inputs)
                        input_embeddings_list.append((None, inputs['input_ids'], text))
                    
                    if hasattr(bert_out, 'last_hidden_state'):
                        ehr_feat = bert_out.last_hidden_state  # (1, seq, dim)
                    else:
                        ehr_feat = bert_out[0]  # (1, seq, dim)
                else:
                    ehr_feat = None
                
                # 3. Combine (패딩 포함)
                feat_dim = 0
                if img_feat is not None:
                    feat_dim = max(feat_dim, img_feat.shape[-1])
                if ehr_feat is not None:
                    feat_dim = max(feat_dim, ehr_feat.shape[-1])
                
                # Padding to (1, n_modalities, max_seq_len_modality, feat_dim)
                # IMPORTANT: use differentiable ops (F.pad / cat) instead of slice assignment,
                # otherwise gradients to img_tensor/embeddings can get disconnected.
                def _pad_seq_and_dim(x, target_seq, target_dim):
                    """
                    x: (1, seq, dim)
                    returns: (1, target_seq, target_dim)
                    """
                    if x is None:
                        return None
                    seq = x.shape[1]
                    dim = x.shape[2]
                    # Pad dim last
                    if dim < target_dim:
                        x = F.pad(x, (0, target_dim - dim, 0, 0, 0, 0))
                    else:
                        x = x[:, :, :target_dim]
                    # Pad / truncate seq
                    if seq < target_seq:
                        x = F.pad(x, (0, 0, 0, target_seq - seq, 0, 0))
                    else:
                        x = x[:, :target_seq, :]
                    return x

                if modality == "multimodal":
                    # Build token-level masks for each modality
                    img_len = min(img_feat.shape[1], max_seq_len_modality) if img_feat is not None else 0
                    ehr_len = min(ehr_feat.shape[1], max_seq_len_modality) if ehr_feat is not None else 0
                    img_mask = torch.zeros(max_seq_len_modality, dtype=torch.bool, device=device)
                    ehr_mask = torch.zeros(max_seq_len_modality, dtype=torch.bool, device=device)
                    if img_len > 0:
                        img_mask[:img_len] = True
                    if ehr_len > 0:
                        ehr_mask[:ehr_len] = True

                    padded_img = _pad_seq_and_dim(img_feat, max_seq_len_modality, feat_dim)
                    padded_ehr = _pad_seq_and_dim(ehr_feat, max_seq_len_modality, feat_dim)
                    if padded_img is None:
                        padded_img = torch.zeros(1, max_seq_len_modality, feat_dim, device=device)
                    if padded_ehr is None:
                        padded_ehr = torch.zeros(1, max_seq_len_modality, feat_dim, device=device)

                    combined = torch.stack([padded_img, padded_ehr], dim=1)  # (1, 2, S, D)
                    token_mask = torch.stack([img_mask, ehr_mask], dim=0).unsqueeze(0)  # (1, 2, S)
                    
                elif modality == "image":
                    img_len = min(img_feat.shape[1], max_seq_len_modality) if img_feat is not None else 0
                    img_mask = torch.zeros(max_seq_len_modality, dtype=torch.bool, device=device)
                    if img_len > 0:
                        img_mask[:img_len] = True

                    padded_img = _pad_seq_and_dim(img_feat, max_seq_len_modality, feat_dim)
                    if padded_img is None:
                        padded_img = torch.zeros(1, max_seq_len_modality, feat_dim, device=device)
                    combined = padded_img.unsqueeze(1)  # (1, 1, S, D)
                    token_mask = img_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S)
                    
                else:  # text
                    ehr_len = min(ehr_feat.shape[1], max_seq_len_modality) if ehr_feat is not None else 0
                    ehr_mask = torch.zeros(max_seq_len_modality, dtype=torch.bool, device=device)
                    if ehr_len > 0:
                        ehr_mask[:ehr_len] = True

                    padded_ehr = _pad_seq_and_dim(ehr_feat, max_seq_len_modality, feat_dim)
                    if padded_ehr is None:
                        padded_ehr = torch.zeros(1, max_seq_len_modality, feat_dim, device=device)
                    combined = padded_ehr.unsqueeze(1)  # (1, 1, S, D)
                    token_mask = ehr_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S)
                
                day_features_list.append(combined)
                day_token_masks.append(token_mask)
                
        except Exception as e:
            print(f"Error in explain_admission_all_features: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Model Forward
        # combined_feat: (1, num_days, n_modalities, seq_len, dim)
        combined_feat = torch.cat(day_features_list, dim=0).unsqueeze(0)  # (1, days, n_mod, seq, dim)
        # Token-level mask: (1, days, n_mod, seq)
        # day_token_masks: List[(1, M, S)] length=days
        # -> (days, M, S) -> (1, days, M, S)
        mask = torch.cat(day_token_masks, dim=0).unsqueeze(0)

        day_outputs = model(combined_feat, mask)
        
        if isinstance(day_outputs, list): 
            day_outputs = day_outputs[-1]
        if day_outputs.dim() > 1:
            final_logit = day_outputs[0, -1] if day_outputs.dim() == 2 else day_outputs[0]
        else:
            final_logit = day_outputs[0]
        
        # Backward
        model.zero_grad()
        if cxr_processor and hasattr(cxr_processor, 'encoder'):
            cxr_processor.encoder.zero_grad()
        if bert_model:
            bert_model.zero_grad()
        final_logit.backward()

        # Visualization
        viz_results = {}
        day_scores = []
        
        # Day importance
        for t in range(seq_len):
            score = 0
            if t in img_tensors_map and img_tensors_map[t].grad is not None:
                score += img_tensors_map[t].grad.abs().sum().item()
            if t < len(input_embeddings_list):
                emb, _, _ = input_embeddings_list[t]
                if emb is not None and emb.grad is not None:
                    score += emb.grad.abs().sum().item()
            day_scores.append((t, score))
        
        day_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Global Keywords (후처리된 단어 단위로 집계)
        global_token_scores = defaultdict(float)
        for t, (emb, input_ids, _) in enumerate(input_embeddings_list):
            if emb is not None and emb.grad is not None:
                token_grads = (emb.grad * emb).sum(dim=-1).abs().squeeze(0).detach().cpu().numpy()
                tokens = ehr_processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                # [후처리 적용] BPE 토큰을 단어 단위로 병합
                clean_words, clean_scores = merge_subwords_and_scores(tokens, token_grads, method='max')
                
                # 병합된 단어 단위로 점수 집계
                for word, score in zip(clean_words, clean_scores):
                    if word not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                        global_token_scores[word] += float(score)
        
        sorted_global_tokens = sorted(global_token_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_global_tokens:
            table_data = [[t, float(s)] for t, s in sorted_global_tokens]
            viz_results['global_top_keywords'] = wandb.plot.bar(
                wandb.Table(data=table_data, columns=["Token", "Aggregated Score"]),
                "Token", "Aggregated Score", title="Global Top-10 Keywords"
            )

        # Per-day visualization
        # [all_features 모드용] Day-level 가중치 계산 (CLS gradient 대신 day의 전체 gradient 합계 사용)
        all_day_scores = [score for _, score in day_scores]
        max_day_score = max(all_day_scores) if all_day_scores else 1.0
        
        for rank, (day_idx, day_score) in enumerate(day_scores[:min(5, len(day_scores))]):
            day_prefix = f"day_{day_indices[day_idx]}"
            
            # [all_features 모드용] Day-level 가중치 (CLS gradient weight 대신)
            day_grad_weight = day_score / (max_day_score + 1e-8) if max_day_score > 0 else 1.0
            
            # Image heatmap
            if day_idx in img_tensors_map:
                target_tensor = img_tensors_map[day_idx]
                if target_tensor.grad is not None:
                    grad = target_tensor.grad[0].cpu().detach()
                    saliency = grad.abs().max(dim=0)[0].numpy()
                    
                    # [Day-level 가중치 적용] all_features 모드에서는 day의 전체 gradient 합계를 가중치로 사용
                    saliency = saliency * day_grad_weight
                    
                    # Robust normalize (avoid "almost all black/blue" maps when gradients are spiky/tiny)
                    lo = np.percentile(saliency, 5)
                    hi = np.percentile(saliency, 99)
                    if hi > lo:
                        saliency = np.clip((saliency - lo) / (hi - lo), 0.0, 1.0)
                    
                    raw_img = raw_images_for_viz[day_idx]
                    saliency = cv2.resize(saliency, raw_img.size)
                    # 더 부드러운 히트맵을 위해 여러 번 blur 적용 및 더 큰 kernel 사용
                    # 1차 blur: 큰 kernel로 전체적인 smoothing
                    saliency = cv2.GaussianBlur(saliency, (51, 51), 15)
                    # 2차 blur: 추가 smoothing
                    saliency = cv2.GaussianBlur(saliency, (31, 31), 10)
                    # 히트맵 생성 (더 부드러운 colormap 사용)
                    heatmap = cm.jet(saliency)[:, :, :3]
                    heatmap = (heatmap * 255).astype(np.uint8)
                    # 더 부드러운 오버레이: 원본 이미지를 더 많이 보이게 (0.75:0.25)
                    overlay = cv2.addWeighted(np.array(raw_img), 0.75, heatmap, 0.25, 0)
                    
                    viz_results[f"{day_prefix}_cxr"] = wandb.Image(
                        overlay, caption=f"Day {day_indices[day_idx]} (Rank {rank+1}, day_weight={day_grad_weight:.4f})"
                    )
                    
                    # Debug stats for gradients to diagnose "almost zero heatmap"
                    # CPU로 이동 후 detach (안전장치: 여러 번 호출해도 안전)
                    g_abs = grad.abs()
                    if g_abs.device.type != 'cpu':
                        g_abs = g_abs.cpu()
                    g_abs = g_abs.detach()
                    viz_results[f"{day_prefix}_cxr_grad_stats"] = {
                        "max": float(g_abs.max().item()),
                        "mean": float(g_abs.mean().item()),
                        "p99": float(np.percentile(g_abs.numpy().flatten(), 99)),
                        "nonzero_frac(>1e-6)": float((g_abs > 1e-6).float().mean().item())
                    }

            # Text visualization
            if day_idx < len(input_embeddings_list):
                emb, input_ids, _ = input_embeddings_list[day_idx]
                if emb is not None and emb.grad is not None:
                    token_grads = (emb.grad * emb).sum(dim=-1).abs().squeeze(0).detach().cpu().numpy()
                    
                    # [Day-level 가중치 적용] all_features 모드에서는 day의 전체 gradient 합계를 가중치로 사용
                    token_grads = token_grads * day_grad_weight
                    
                    tokens = ehr_processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                    clean_words, clean_scores = merge_subwords_and_scores(tokens, token_grads, method='max')
                    html_code = get_colored_text_html(clean_words, clean_scores)
                    viz_results[f"{day_prefix}_ehr_html"] = wandb.Html(html_code, inject=False)

        return viz_results
