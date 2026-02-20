import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import sys
import argparse
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Project root = MR.DEC (so imports resolve: online_processor, model, data, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from online_processor import (OnlineCXRProcessor, OnlineClinicalBERTProcessor,
                              OnlineCXRProcessorAll, OnlineClinicalBERTProcessorAll
)
from data.readmission_utils import (
    get_readmission_label_mimic_flexible_fast, explain_admission_end_to_end,
    explain_admission_end_to_end_all_features
)
from model.autoregressive_transformer import (
    ReadmissionEncoder,
    AutoregressiveTransformer,
    CrossAttention,
    MLP,
    supervised_contrastive_loss,
    compute_day_wise_loss
)

from group_balanced_sampler import GroupBalancedBatchSampler
import utils
import pickle
import hashlib
import json

def get_feature_cache_key(args, split_name, admission_ids):
    """Build cache key from encoder and dataset config."""
    cache_config = {
        'cxr_model_type': args.cxr_model_type,
        'cxr_pretrained_path': os.path.basename(args.cxr_pretrained_path) if args.cxr_pretrained_path else None,
        'clinical_bert_model': args.clinical_bert_model,
        'feature_mode': args.feature_mode,
        'max_days': args.max_days,
        'modality': args.modality,
        'max_seq_len_modality': args.max_seq_len_modality,
        'remove_duplication': args.remove_duplication,
        'unpaired': args.unpaired,
        'balance_data': getattr(args, 'balance_data', False),
        'drop_image_ratio': getattr(args, 'drop_image_ratio', 0.0),
        'drop_ehr_ratio': getattr(args, 'drop_ehr_ratio', 0.0),
        'rand_seed': args.rand_seed,
        'split_name': split_name,
        'num_samples': len(admission_ids),
        'admission_ids_hash': hashlib.md5('_'.join(sorted(admission_ids)).encode()).hexdigest()[:16]
    }
    config_str = json.dumps(cache_config, sort_keys=True)
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    return cache_hash, cache_config


def get_cache_filepath(cache_dir, cache_hash, split_name):
    return os.path.join(cache_dir, f"features_{split_name}_{cache_hash}.pkl")


def save_features_to_cache(features, cache_filepath, cache_config):
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
    cache_data = {'features': features, 'config': cache_config, 'version': '1.0'}
    with open(cache_filepath, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_size_mb = os.path.getsize(cache_filepath) / (1024 * 1024)
    print(f"  Cache saved: {cache_filepath} ({cache_size_mb:.1f} MB)")


def load_features_from_cache(cache_filepath):
    if not os.path.exists(cache_filepath):
        return None
    
    try:
        with open(cache_filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        cache_size_mb = os.path.getsize(cache_filepath) / (1024 * 1024)
        print(f"  Cache loaded: {cache_filepath} ({cache_size_mb:.1f} MB)")
        return cache_data['features']
    except Exception as e:
        print(f"  Failed to load cache: {e}")
        return None


def run_extractor_with_cache(
    args, split_name, admission_ids, labels, admission_files, time_idxs,
    cxr_processor, ehr_processor, df_demo, df_icd, df_lab, df_med,
    feature_batch_size, logger=None
):
    """Load features from cache if available; otherwise extract and optionally save."""
    use_cache = getattr(args, 'use_feature_cache', False)
    cache_dir = getattr(args, 'feature_cache_dir', './cache/features')
    
    if use_cache:
        cache_hash, cache_config = get_feature_cache_key(args, split_name, admission_ids)
        cache_filepath = get_cache_filepath(cache_dir, cache_hash, split_name)
        
        print(f"\n[Feature Cache] Split: {split_name}")
        print(f"  Cache key: {cache_hash}")
        cached_features = load_features_from_cache(cache_filepath)
        if cached_features is not None:
            if logger:
                logger.info(f"Loaded {split_name} features from cache: {cache_filepath}")
            return cached_features
        print(f"  Cache miss, extracting features...")
    features = run_extractor_once(
        admission_ids, labels,
        admission_files, time_idxs, cxr_processor, ehr_processor, 
        df_demo, df_icd, df_lab, df_med, args.max_days, 
        desc=f"Ext {split_name}", modality=args.modality,
        max_seq_len_modality=args.max_seq_len_modality, 
        feature_mode=args.feature_mode,
        remove_duplication=args.remove_duplication,
        drop_image_ratio=args.drop_image_ratio,
        drop_ehr_ratio=args.drop_ehr_ratio,
        unpaired=args.unpaired,
        batch_size=feature_batch_size
    )
    if use_cache:
        save_features_to_cache(features, cache_filepath, cache_config)
        if logger:
            logger.info(f"Saved {split_name} features to cache: {cache_filepath}")
    
    return features


def run_extractor_once(
    admission_ids, labels, admission_files, time_idxs, 
    cxr_processor, ehr_processor, 
    df_demo, df_icd, df_lab, df_med, 
    max_days, desc="Extracting Features",
    modality="multimodal",
    max_seq_len_modality=512,
    feature_mode="all_features",
    remove_duplication=False,
    drop_image_ratio=0.0,
    drop_ehr_ratio=0.0,
    unpaired=False,
    batch_size=256
):
    img_dim = cxr_processor.feat_dim if cxr_processor else 0
    ehr_dim = ehr_processor.model.config.hidden_size if ehr_processor else 0
    feat_dim = max(img_dim, ehr_dim)
    np.random.seed(42)

    print(f"[Pre-indexing] Grouping DataFrames by hadm_id...", flush=True)
    df_icd_grouped = {hadm_id: group for hadm_id, group in df_icd.groupby('hadm_id')}
    df_lab_grouped = {hadm_id: group for hadm_id, group in df_lab.groupby('hadm_id')}
    df_med_grouped = {hadm_id: group for hadm_id, group in df_med.groupby('hadm_id')}
    df_demo_indexed = df_demo.set_index('hadm_id')
    empty_icd = df_icd.iloc[:0]
    empty_lab = df_lab.iloc[:0]
    empty_med = df_med.iloc[:0]
    print(f"[Pre-indexing] Done. ICD: {len(df_icd_grouped)}, Lab: {len(df_lab_grouped)}, Med: {len(df_med_grouped)} groups", flush=True)

    print(f"[Phase 1] Collecting image paths and EHR texts...", flush=True)
    all_img_paths = []
    all_img_metadata = []
    all_ehr_texts = []
    all_ehr_metadata = []
    admission_metadata = []
    
    for i, admission_id in enumerate(tqdm(admission_ids, desc="Processing Admissions")):
        hadm_id = int(admission_id.split('_')[1])
        img_paths = admission_files[admission_id]
        img_days = time_idxs[admission_id]
        
        admission_icds = df_icd_grouped.get(hadm_id, empty_icd)
        admission_labs = df_lab_grouped.get(hadm_id, empty_lab)
        admission_meds = df_med_grouped.get(hadm_id, empty_med)
        if 'SUPER_GROUP' in df_icd.columns and len(admission_icds) > 0:
            group_counts = Counter(admission_icds['SUPER_GROUP'].dropna())
            super_group = group_counts.most_common(1)[0][0] if len(group_counts) > 0 else 'Unknown'
        else:
            super_group = 'Unknown'
        
        day_to_img_path = {d: p for p, d in zip(img_paths, img_days)}
        sorted_days = sorted(list(day_to_img_path.keys()))
        
        if feature_mode == "all_features":
            if len(sorted_days) == 1:
                target_days = [sorted_days[0]]
            else:
                target_days = [sorted_days[0], sorted_days[-1]]
        else:
            if unpaired:
                hospital_stay = max(time_idxs[admission_id]) + 1 if len(time_idxs[admission_id]) > 0 else 1
                target_days = list(range(hospital_stay))[-max_days:]
            else:
                target_days = sorted_days[-max_days:] if len(sorted_days) > max_days else sorted_days
        
        try:
            row = df_demo_indexed.loc[hadm_id]
        except KeyError:
            row = df_demo[df_demo['hadm_id'] == hadm_id].iloc[0]
        day_infos = []
        for day_idx, t_day in enumerate(target_days):
            img_path = day_to_img_path.get(t_day, None)
            day_num = t_day + 1
            if modality in ["multimodal", "image"]:
                drop_image = (img_path is not None) and (drop_image_ratio > 0.0) and (np.random.rand() < drop_image_ratio)
            else:
                drop_image = True
            if modality in ["multimodal", "text"]:
                drop_ehr = (drop_ehr_ratio > 0.0) and (np.random.rand() < drop_ehr_ratio)
            else:
                drop_ehr = True
            if modality in ["multimodal", "image"] and not drop_image and img_path is not None:
                all_img_paths.append(img_path)
                all_img_metadata.append((i, day_idx, img_path))
            if modality in ["multimodal", "text"] and not drop_ehr:
                daily_text = ehr_processor.create_daily_ehr_text(row, admission_icds, admission_labs, admission_meds, hadm_id, day_num, remove_duplication)
                all_ehr_texts.append(daily_text)
                all_ehr_metadata.append((i, day_idx))
            
            day_infos.append({
                'day_idx': day_idx,
                't_day': t_day,
                'img_path': img_path,
                'drop_image': drop_image,
                'drop_ehr': drop_ehr
            })
        
        admission_metadata.append({
            'admission_id': admission_id,
            'hadm_id': hadm_id,
            'label': labels[i],
            'super_group': super_group,
            'target_days': target_days,
            'day_to_img_path': day_to_img_path,
            'day_infos': day_infos
        })
    print(f"[Phase 2] Extracting features (batch_size={batch_size})...", flush=True)
    all_img_feats_dict = {}
    if len(all_img_paths) > 0 and cxr_processor:
        print(f"  Extracting CXR features for {len(all_img_paths)} images...", flush=True)
        img_feats_dict_batch, _ = cxr_processor.extract_features(all_img_paths, batch_size=batch_size)
        for (adm_idx, day_idx, img_path) in all_img_metadata:
            if img_path in img_feats_dict_batch:
                all_img_feats_dict[(adm_idx, day_idx)] = img_feats_dict_batch[img_path]
        print(f"  CXR features extracted: {len(all_img_feats_dict)}", flush=True)
    all_ehr_feats_dict = {}
    ehr_batch_size = min(batch_size, 64)
    if len(all_ehr_texts) > 0 and ehr_processor:
        print(f"  Extracting EHR features for {len(all_ehr_texts)} texts (batch_size={ehr_batch_size})...", flush=True)
        ehr_feats_batch = ehr_processor.extract_features(all_ehr_texts, max_length=max_seq_len_modality, batch_size=ehr_batch_size)
        for (adm_idx, day_idx), feat in zip(all_ehr_metadata, ehr_feats_batch):
            all_ehr_feats_dict[(adm_idx, day_idx)] = feat
        print(f"  EHR features extracted: {len(all_ehr_feats_dict)}", flush=True)
    print(f"[Phase 3] Constructing feature data for {len(admission_ids)} admissions...", flush=True)
    extracted_data = []
    
    for i, meta in enumerate(tqdm(admission_metadata, desc=desc)):
        admission_id = meta['admission_id']
        hadm_id = meta['hadm_id']
        label = meta['label']
        super_group = meta['super_group']
        target_days = meta['target_days']
        day_to_img_path = meta['day_to_img_path']
        day_infos = meta['day_infos']
        
        day_list = []
        seq_lens_list = []
        valid_paths = []
        valid_days = []
        
        for day_info in day_infos:
            day_idx = day_info['day_idx']
            t_day = day_info['t_day']
            img_path = day_info['img_path']
            drop_image = day_info['drop_image']
            drop_ehr = day_info['drop_ehr']
            
            if feature_mode == "all_features":
                modality_feats = []
                modality_lens = []
                if modality in ["multimodal", "image"]:
                    if not drop_image and (i, day_idx) in all_img_feats_dict:
                        raw_img_feat = all_img_feats_dict[(i, day_idx)]
                        v_len_img = min(raw_img_feat.shape[0], max_seq_len_modality)
                        img_feat = np.zeros((v_len_img, feat_dim), dtype=np.float32)
                        img_feat[:, :img_dim] = raw_img_feat[:v_len_img]
                        modality_feats.append(img_feat)
                        modality_lens.append(v_len_img)
                    else:
                        empty_img = np.zeros((0, feat_dim), dtype=np.float32)
                        modality_feats.append(empty_img)
                        modality_lens.append(0)
                if modality in ["multimodal", "text"]:
                    if not drop_ehr and (i, day_idx) in all_ehr_feats_dict:
                        raw_ehr_feat = all_ehr_feats_dict[(i, day_idx)]
                        if raw_ehr_feat.ndim == 1:
                            raw_ehr_feat = raw_ehr_feat[np.newaxis, :]
                        v_len_ehr = min(raw_ehr_feat.shape[0], max_seq_len_modality)
                        ehr_feat = np.zeros((v_len_ehr, feat_dim), dtype=np.float32)
                        ehr_feat[:, :ehr_dim] = raw_ehr_feat[:v_len_ehr]
                        modality_feats.append(ehr_feat)
                        modality_lens.append(v_len_ehr)
                    else:
                        empty_ehr = np.zeros((0, feat_dim), dtype=np.float32)
                        modality_feats.append(empty_ehr)
                        modality_lens.append(0)
                
                day_list.append(modality_feats)
                seq_lens_list.append(modality_lens)
                    
            else:
                has_cxr = img_path is not None
                if unpaired and modality == "multimodal":
                    if has_cxr and not drop_image and (i, day_idx) in all_img_feats_dict:
                        feat_img = np.zeros(feat_dim, dtype=np.float32)
                        raw_img_feat = all_img_feats_dict[(i, day_idx)]
                        if raw_img_feat.ndim > 1:
                            feat_img[:img_dim] = raw_img_feat[0] if raw_img_feat.shape[0] > 0 else 0
                        else:
                            feat_img[:img_dim] = raw_img_feat
                        day_list.append(feat_img)
                        seq_lens_list.append(0)
                    if not drop_ehr and (i, day_idx) in all_ehr_feats_dict:
                        feat_ehr = np.zeros(feat_dim, dtype=np.float32)
                        raw_ehr_feat = all_ehr_feats_dict[(i, day_idx)]
                        if raw_ehr_feat.ndim == 1:
                            feat_ehr[:ehr_dim] = raw_ehr_feat
                        else:
                            feat_ehr[:ehr_dim] = raw_ehr_feat[0] if raw_ehr_feat.shape[0] > 0 else 0
                        day_list.append(feat_ehr)
                        seq_lens_list.append(1)
                else:
                    feat_img = np.zeros(feat_dim, dtype=np.float32)
                    feat_ehr = np.zeros(feat_dim, dtype=np.float32)

                    if modality in ["multimodal", "image"] and not drop_image and (i, day_idx) in all_img_feats_dict:
                        raw_img_feat = all_img_feats_dict[(i, day_idx)]
                        if raw_img_feat.ndim > 1:
                            feat_img[:img_dim] = raw_img_feat[0] if raw_img_feat.shape[0] > 0 else 0
                        else:
                            feat_img[:img_dim] = raw_img_feat
                        
                    if modality in ["multimodal", "text"] and not drop_ehr and (i, day_idx) in all_ehr_feats_dict:
                        raw_ehr_feat = all_ehr_feats_dict[(i, day_idx)]
                        if raw_ehr_feat.ndim == 1:
                            feat_ehr[:ehr_dim] = raw_ehr_feat
                        else:
                            feat_ehr[:ehr_dim] = raw_ehr_feat[0] if raw_ehr_feat.shape[0] > 0 else 0
                    
                    if modality == "multimodal":
                        combined_feat = np.stack([feat_img, feat_ehr], axis=0)
                    elif modality == "image":
                        combined_feat = np.expand_dims(feat_img, axis=0)
                    elif modality == "text":
                        combined_feat = np.expand_dims(feat_ehr, axis=0)
                    
                    day_list.append(combined_feat)
            
            valid_paths.append(img_path)
            valid_days.append(t_day)
        if feature_mode == "all_features":
            extracted_data.append({
                'day_features': day_list,
                'seq_lens': seq_lens_list,
                'num_days': len(day_list),
                'feat_dim': feat_dim,
                'label': torch.FloatTensor([label]),
                'admission_id': admission_id,
                'img_paths': valid_paths,
                'day_indices': valid_days, 
                'hadm_id': hadm_id,
                'super_group': super_group
            })
        else:
            if len(day_list) > 0:
                if unpaired and modality == "multimodal":
                    features = torch.FloatTensor(np.stack(day_list))
                    modality_ids = torch.LongTensor(seq_lens_list)
                    extracted_data.append({
                        'features': features,
                        'num_tokens': len(day_list),
                        'modality_ids': modality_ids,
                        'is_unpaired': True,
                        'label': torch.FloatTensor([label]),
                        'admission_id': admission_id,
                        'img_paths': valid_paths,
                        'day_indices': valid_days, 
                        'hadm_id': hadm_id,
                        'super_group': super_group
                    })
                else:
                    features = torch.FloatTensor(np.stack(day_list))
                    extracted_data.append({
                        'features': features,
                        'num_days': len(day_list),
                        'is_unpaired': False,
                        'label': torch.FloatTensor([label]),
                        'admission_id': admission_id,
                        'img_paths': valid_paths,
                        'day_indices': valid_days, 
                        'hadm_id': hadm_id,
                        'super_group': super_group
                    })

    return extracted_data

class FeatureDataset(Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def collate_fn_all_features(batch, max_days=2, max_seq_len=512):
    B = len(batch)
    n_mod = len(batch[0]['day_features'][0])
    feat_dim = batch[0]['feat_dim']
    max_seq_in_batch = 0
    for item in batch:
        for day_lens in item['seq_lens']:
            for seq_len in day_lens:
                max_seq_in_batch = max(max_seq_in_batch, seq_len)
    max_seq_in_batch = max(1, min(max_seq_in_batch, max_seq_len))
    batch_features = torch.zeros(B, max_days, n_mod, max_seq_in_batch, feat_dim)
    mask = torch.zeros(B, max_days, n_mod, max_seq_in_batch, dtype=torch.bool)
    labels = []
    super_groups = []
    for i, item in enumerate(batch):
        num_days = min(item['num_days'], max_days)
        labels.append(item['label'])
        super_groups.append(item['super_group'])
        for d in range(num_days):
            day_feats = item['day_features'][d]
            day_lens = item['seq_lens'][d]
            for m in range(n_mod):
                feat = day_feats[m]
                actual_len = min(day_lens[m], max_seq_in_batch)
                batch_features[i, d, m, :actual_len, :] = torch.FloatTensor(feat[:actual_len])
                mask[i, d, m, :actual_len] = True
    return {
        'features': batch_features,
        'mask': mask,
        'labels': torch.stack(labels),
        'super_groups': super_groups
    }

def collate_fn_cls_only(batch, max_days=30):
    B = len(batch)
    if batch[0].get('is_unpaired', False):
        dim = batch[0]['features'].shape[1]
        max_seq_len = max_days * 2
        batch_features = torch.zeros(B, max_seq_len, dim)
        batch_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)
        batch_modality_ids = torch.zeros(B, max_seq_len, dtype=torch.long)
        labels = []
        super_groups = []
        for i, item in enumerate(batch):
            L = min(item['num_tokens'], max_seq_len)
            batch_features[i, :L] = item['features'][:L]
            batch_mask[i, :L] = True
            batch_modality_ids[i, :L] = item['modality_ids'][:L]
            labels.append(item['label'])
            super_groups.append(item['super_group'])
        return {
            'features': batch_features,
            'mask': batch_mask,
            'modality_ids': batch_modality_ids,
            'labels': torch.stack(labels),
            'super_groups': super_groups,
            'is_unpaired': True
        }
    n_mod, dim = batch[0]['features'].shape[1], batch[0]['features'].shape[2]
    batch_features = torch.zeros(B, max_days, n_mod, dim)
    batch_mask = torch.zeros(B, max_days, dtype=torch.bool)
    labels = []
    super_groups = []
    for i, item in enumerate(batch):
        L = min(item['num_days'], max_days)
        batch_features[i, :L] = item['features'][:L]
        batch_mask[i, :L] = True
        labels.append(item['label'])
        super_groups.append(item['super_group'])
    return {
        'features': batch_features,
        'mask': batch_mask,
        'labels': torch.stack(labels),
        'super_groups': super_groups,
        'is_unpaired': False
    }

def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_preds = (np.array(y_probs) >= threshold).astype(int)
    y_true = np.array(y_true)
    
    return {
        'acc': accuracy_score(y_true, y_preds),
        'f1': f1_score(y_true, y_preds, zero_division=0),
        'precision': precision_score(y_true, y_preds, zero_division=0),
        'recall': recall_score(y_true, y_preds, zero_division=0),
        'auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    }


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, feature_mode="cls_only", 
                do_contrast=False, contrastive_weight=0.5, contrastive_temperature=0.07, loss_option="bce"):
    model.train()
    epoch_loss = 0
    epoch_contrast_loss = 0
    epoch_bce_loss = 0
    all_probs, all_labels = [], []
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device).squeeze(-1)
        mask = batch['mask'].to(device)
        is_unpaired = batch.get('is_unpaired', False)
        if feature_mode == "all_features":
            logits, contrastive_emb = model(features, mask)
        else:
            logits, contrastive_emb = model(features, mask, is_unpaired=is_unpaired)
        if loss_option == "bce":
            bce_loss, _ = compute_day_wise_loss(logits, labels, mask, criterion)
            total_loss = bce_loss
            epoch_loss += bce_loss.item()
            epoch_bce_loss += bce_loss.item()
        elif loss_option == "contrast":
            assert do_contrast and 'super_groups' in batch and contrastive_emb is not None, \
                "Contrast loss requires do_contrast=True, super_groups in batch, and valid contrastive embeddings"
                
            unique_groups = list(set(batch['super_groups']))
            group_to_idx = {g: i for i, g in enumerate(unique_groups)}
            group_indices = torch.tensor([group_to_idx[g] for g in batch['super_groups']],
                                        dtype=torch.long, device=device)
            contrast_loss = supervised_contrastive_loss(
                contrastive_emb, group_indices, labels, temperature=contrastive_temperature
            )
            total_loss = contrast_loss
            epoch_loss += contrast_loss.item()
            epoch_contrast_loss += contrast_loss.item()
        elif loss_option == "mix":
            bce_loss, _ = compute_day_wise_loss(logits, labels, mask, criterion)
            assert do_contrast and 'super_groups' in batch and contrastive_emb is not None, \
                "Mix loss requires do_contrast=True, super_groups in batch, and valid contrastive embeddings"
            unique_groups = list(set(batch['super_groups']))
            group_to_idx = {g: i for i, g in enumerate(unique_groups)}
            group_indices = torch.tensor([group_to_idx[g] for g in batch['super_groups']],
                                        dtype=torch.long, device=device)
            contrast_loss = supervised_contrastive_loss(
                contrastive_emb, group_indices, labels, temperature=contrastive_temperature
            )
            total_loss = bce_loss + contrastive_weight * contrast_loss
            
            epoch_loss += total_loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_contrast_loss += contrast_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        final_logit = logits
        if final_logit.dim() > 1:
            final_logit = final_logit.squeeze(-1)
        all_probs.extend(torch.sigmoid(final_logit).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_probs)
    metrics['loss'] = epoch_loss / len(dataloader)
    
    if do_contrast and loss_option in ["contrast", "mix"]:
        metrics['contrast_loss'] = epoch_contrast_loss / len(dataloader)
        if loss_option == "mix":
            metrics['bce_loss'] = epoch_bce_loss / len(dataloader)
    
    return metrics

def evaluate(model, dataloader, criterion, device, split_name, feature_mode="cls_only",
             do_contrast=False, contrastive_temperature=0.07, loss_option="bce", contrastive_weight=0.5):
    model.eval()
    total_loss = 0
    total_contrast_loss = 0
    total_bce_loss = 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {split_name}"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device).squeeze(-1)
            mask = batch['mask'].to(device)
            is_unpaired = batch.get('is_unpaired', False)
            if feature_mode == "all_features":
                logits, contrastive_emb = model(features, mask)
            else:
                logits, contrastive_emb = model(features, mask, is_unpaired=is_unpaired)
            bce_loss, _ = compute_day_wise_loss(logits, labels, mask, criterion)
            total_bce_loss += bce_loss.item()
            
            if loss_option == "bce":
                total_loss += bce_loss.item()
                
            elif loss_option in ["contrast", "mix"]:
                if do_contrast and 'super_groups' in batch and contrastive_emb is not None:
                    unique_groups = list(set(batch['super_groups']))
                    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
                    group_indices = torch.tensor([group_to_idx[g] for g in batch['super_groups']], 
                                                dtype=torch.long, device=device)
                    
                    contrast_loss = supervised_contrastive_loss(
                        contrastive_emb,
                        group_indices,
                        labels,
                        temperature=contrastive_temperature
                    )
                    total_contrast_loss += contrast_loss.item()
                    
                    if loss_option == "contrast":
                        total_loss += contrast_loss.item()
                    else:
                        total_loss += bce_loss.item() + contrastive_weight * contrast_loss.item()
                else:
                    total_loss += bce_loss.item()
            final_logit = logits
            if final_logit.dim() > 1:
                final_logit = final_logit.squeeze(-1)
            all_probs.extend(torch.sigmoid(final_logit).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    metrics = calculate_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    if do_contrast and loss_option in ["contrast", "mix"]:
        metrics['contrast_loss'] = total_contrast_loss / len(dataloader)
        if loss_option == "mix":
            metrics['bce_loss'] = total_bce_loss / len(dataloader)
    if wandb.run is not None:
        best_auc_key = f"{split_name}/Best_AUC"
        current_auc = metrics.get('auc', 0)
        prev_best_auc = wandb.run.summary.get(best_auc_key, 0)
        if current_auc > prev_best_auc:
            wandb.run.summary[best_auc_key] = current_auc
        best_f1_key = f"{split_name}/Best_F1"
        current_f1 = metrics.get('f1', 0)
        prev_best_f1 = wandb.run.summary.get(best_f1_key, 0)
        if current_f1 > prev_best_f1:
            wandb.run.summary[best_f1_key] = current_f1
        min_loss_key = f"{split_name}/Min_Loss"
        current_loss = metrics.get('loss', float('inf'))
        prev_min_loss = wandb.run.summary.get(min_loss_key, float('inf'))
        if current_loss < prev_min_loss:
            wandb.run.summary[min_loss_key] = current_loss

    return metrics

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    utils.seed_torch(seed=args.rand_seed)
    args.save_dir = utils.get_save_dir(args.save_dir, training=args.do_train)
    
    if args.wandb_enabled:
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    logger = utils.get_logger(args.save_dir, "train")
    
    df_demo = pd.read_csv(args.demo_file)

    if getattr(args, "export_val_lists_only", False):
        labels_all, admission_files, _, _, _, _time_idxs = get_readmission_label_mimic_flexible_fast(
            df_demo, max_seq_len=None, unpaired=args.unpaired
        )
        admission_ids = list(admission_files.keys())
        excluded_hadm_ids = {25161730, 24707962}
        filtered_indices = []
        for i, admission_id in enumerate(admission_ids):
            hadm_id = int(admission_id.split('_')[1])
            if hadm_id not in excluded_hadm_ids:
                filtered_indices.append(i)
        admission_ids = [admission_ids[i] for i in filtered_indices]
        labels_all = [labels_all[i] for i in filtered_indices]
        if args.filtered_admission_file:
            with open(args.filtered_admission_file.replace('.txt', '_train_admission_ids.txt'), 'r') as f:
                filtered_train_ids = set(line.strip() for line in f)
            with open(args.filtered_admission_file.replace('.txt', '_val_admission_ids.txt'), 'r') as f:
                filtered_val_ids = set(line.strip() for line in f)
            train_idx = [i for i, aid in enumerate(admission_ids) if aid in filtered_train_ids]
            val_idx = [i for i, aid in enumerate(admission_ids) if aid in filtered_val_ids]
        else:
            all_indices = np.arange(len(admission_ids))
            np.random.shuffle(all_indices)
            split = int(len(all_indices) * 0.9)
            train_idx = all_indices[:split].tolist()
            val_idx = all_indices[split:].tolist()
        if args.toy_sample_ratio is not None and args.toy_sample_ratio < 1.0:
            num_train_samples = max(1, int(len(train_idx) * args.toy_sample_ratio))
            num_val_samples = max(1, int(len(val_idx) * args.toy_sample_ratio))
            np.random.shuffle(train_idx)
            np.random.shuffle(val_idx)
            train_idx = train_idx[:num_train_samples]
            val_idx = val_idx[:num_val_samples]
        out_dir = getattr(args, "val_list_out_dir", ".")
        os.makedirs(out_dir, exist_ok=True)

        # 1) imbalance
        val_admission_ids_imbalance = [admission_ids[i] for i in val_idx]
        imbalance_path = os.path.join(out_dir, "imbalance_val_list.txt")
        with open(imbalance_path, "w") as f:
            for aid in val_admission_ids_imbalance:
                f.write(aid + "\n")
        _rng_state = np.random.get_state()
        try:
            val_labels_original = [labels_all[i] for i in val_idx]
            val_pos_indices = [val_idx[i] for i, l in enumerate(val_labels_original) if l == 1]
            val_neg_indices = [val_idx[i] for i, l in enumerate(val_labels_original) if l == 0]
            if len(val_neg_indices) > len(val_pos_indices):
                val_neg_indices = np.random.choice(val_neg_indices, len(val_pos_indices), replace=False).tolist()
            val_idx_balance = val_pos_indices + val_neg_indices
            np.random.shuffle(val_idx_balance)
        finally:
            np.random.set_state(_rng_state)

        val_admission_ids_balance = [admission_ids[i] for i in val_idx_balance]
        balance_path = os.path.join(out_dir, "balance_val_list.txt")
        with open(balance_path, "w") as f:
            for aid in val_admission_ids_balance:
                f.write(aid + "\n")

        logger.info(
            f"Exported val id lists -> {imbalance_path} (n={len(val_admission_ids_imbalance)}), "
            f"{balance_path} (n={len(val_admission_ids_balance)})"
        )
        return
    df_icd = pd.read_csv(args.icd_file, low_memory=False)
    df_lab = pd.read_csv(args.lab_file, dtype={"flag": str}, low_memory=False)
    df_med = pd.read_csv(args.med_file, low_memory=False)
    
    cxr_processor = None
    ehr_processor = None
    img_feat_dim = 0
    ehr_feat_dim = 0
    
    if args.feature_mode == "all_features":
        if args.modality in ["multimodal", "image"]:
            cxr_processor = OnlineCXRProcessorAll(args.cxr_model_type, args.cxr_pretrained_path, device, return_all_features=True)
            img_feat_dim = cxr_processor.feat_dim
        if args.modality in ["multimodal", "text"]:
            ehr_processor = OnlineClinicalBERTProcessorAll(args.clinical_bert_model, device, return_all_features=True)
            ehr_feat_dim = ehr_processor.model.config.hidden_size
    else:
        if args.modality in ["multimodal", "image"]:
            cxr_processor = OnlineCXRProcessor(args.cxr_model_type, args.cxr_pretrained_path, device)
            img_feat_dim = cxr_processor.feat_dim
        if args.modality in ["multimodal", "text"]:
            ehr_processor = OnlineClinicalBERTProcessor(args.clinical_bert_model, device)
            ehr_feat_dim = ehr_processor.model.config.hidden_size
    
    max_feat_dim = max(img_feat_dim, ehr_feat_dim)
    labels_all, admission_files, _, _, _, time_idxs = get_readmission_label_mimic_flexible_fast(df_demo, max_seq_len=None, unpaired=args.unpaired)
    admission_ids = list(admission_files.keys())
    excluded_hadm_ids = {25161730, 24707962}
    filtered_indices = [i for i, aid in enumerate(admission_ids) if int(aid.split('_')[1]) not in excluded_hadm_ids]
    admission_ids = [admission_ids[i] for i in filtered_indices]
    labels_all = [labels_all[i] for i in filtered_indices]
    if args.filtered_admission_file:
        with open(args.filtered_admission_file.replace('.txt', '_train_admission_ids.txt'), 'r') as f:
            filtered_train_ids = set(line.strip() for line in f)
        with open(args.filtered_admission_file.replace('.txt', '_val_admission_ids.txt'), 'r') as f:
            filtered_val_ids = set(line.strip() for line in f)
        train_idx = [i for i, aid in enumerate(admission_ids) if aid in filtered_train_ids]
        val_idx = [i for i, aid in enumerate(admission_ids) if aid in filtered_val_ids]
    else:
        all_indices = np.arange(len(admission_ids))
        np.random.shuffle(all_indices)
        split = int(len(all_indices) * 0.9)
        train_idx = all_indices[:split].tolist()
        val_idx = all_indices[split:].tolist()
    if args.toy_sample_ratio is not None and args.toy_sample_ratio < 1.0:
        logger.info(f"Applying toy sampling: using {args.toy_sample_ratio*100:.1f}% of data")
        num_train_samples = max(1, int(len(train_idx) * args.toy_sample_ratio))
        num_val_samples = max(1, int(len(val_idx) * args.toy_sample_ratio))
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        train_idx = train_idx[:num_train_samples]
        val_idx = val_idx[:num_val_samples]
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    val_idx_imbalance = val_idx.copy()
    if args.balance_data:
        train_labels_original = [labels_all[i] for i in train_idx]
        pos_indices = [train_idx[i] for i, l in enumerate(train_labels_original) if l == 1]
        neg_indices = [train_idx[i] for i, l in enumerate(train_labels_original) if l == 0]
        
        if len(neg_indices) > len(pos_indices):
            neg_indices = np.random.choice(neg_indices, len(pos_indices), replace=False).tolist()
            train_idx = pos_indices + neg_indices
            np.random.shuffle(train_idx)
            
        val_labels_original = [labels_all[i] for i in val_idx]
        val_pos_indices = [val_idx[i] for i, l in enumerate(val_labels_original) if l == 1]
        val_neg_indices = [val_idx[i] for i, l in enumerate(val_labels_original) if l == 0]
        
        if len(val_neg_indices) > len(val_pos_indices):
            val_neg_indices = np.random.choice(val_neg_indices, len(val_pos_indices), replace=False).tolist()
            val_idx = val_pos_indices + val_neg_indices
            np.random.shuffle(val_idx)
    if getattr(args, "export_val_lists", False):
        out_dir = getattr(args, "val_list_out_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        val_admission_ids_imbalance = [admission_ids[i] for i in val_idx_imbalance]
        imbalance_path = os.path.join(out_dir, "imbalance_val_list.txt")
        with open(imbalance_path, "w") as f:
            for aid in val_admission_ids_imbalance:
                f.write(aid + "\n")
        if args.balance_data:
            val_idx_balance_for_export = val_idx
        else:
            _rng_state = np.random.get_state()
            try:
                val_labels_original = [labels_all[i] for i in val_idx_imbalance]
                val_pos_indices = [val_idx_imbalance[i] for i, l in enumerate(val_labels_original) if l == 1]
                val_neg_indices = [val_idx_imbalance[i] for i, l in enumerate(val_labels_original) if l == 0]
                if len(val_neg_indices) > len(val_pos_indices):
                    val_neg_indices = np.random.choice(val_neg_indices, len(val_pos_indices), replace=False).tolist()
                val_idx_balance_for_export = val_pos_indices + val_neg_indices
                np.random.shuffle(val_idx_balance_for_export)
            finally:
                np.random.set_state(_rng_state)

        val_admission_ids_balance = [admission_ids[i] for i in val_idx_balance_for_export]
        balance_path = os.path.join(out_dir, "balance_val_list.txt")
        with open(balance_path, "w") as f:
            for aid in val_admission_ids_balance:
                f.write(aid + "\n")

        logger.info(
            f"Exported val id lists -> {imbalance_path} (n={len(val_admission_ids_imbalance)}), "
            f"{balance_path} (n={len(val_admission_ids_balance)})"
        )

        if getattr(args, "export_val_lists_only", False):
            logger.info("export_val_lists_only enabled; exiting after exporting val lists.")
            return
    if args.drop_image_ratio > 0.0 or args.drop_ehr_ratio > 0.0:
        logger.info(f"Modality drop: Image={args.drop_image_ratio*100:.1f}%, EHR={args.drop_ehr_ratio*100:.1f}%")
    feature_batch_size = getattr(args, 'feature_extraction_batch_size', 256)
    if getattr(args, 'use_feature_cache', False):
        logger.info(f"Feature cache: {args.feature_cache_dir}")
        if getattr(args, 'clear_feature_cache', False):
            import shutil
            if os.path.exists(args.feature_cache_dir):
                shutil.rmtree(args.feature_cache_dir)
                logger.info(f"Cleared feature cache: {args.feature_cache_dir}")
    train_admission_ids = [admission_ids[i] for i in train_idx]
    train_labels = [labels_all[i] for i in train_idx]
    val_admission_ids = [admission_ids[i] for i in val_idx]
    val_labels = [labels_all[i] for i in val_idx]
    train_feats = run_extractor_with_cache(
        args, "train", train_admission_ids, train_labels, 
        admission_files, time_idxs, cxr_processor, ehr_processor, 
        df_demo, df_icd, df_lab, df_med, feature_batch_size, logger
    )
    
    val_feats = run_extractor_with_cache(
        args, "val", val_admission_ids, val_labels,
        admission_files, time_idxs, cxr_processor, ehr_processor,
        df_demo, df_icd, df_lab, df_med, feature_batch_size, logger
    )
    
    if args.feature_mode == "all_features":
        collate_func = lambda x: collate_fn_all_features(x, max_days=args.max_days, max_seq_len=args.max_seq_len_modality)
    else:
        collate_func = lambda x: collate_fn_cls_only(x, max_days=args.max_days)
    
    if args.modality == "multimodal":
        input_channels = 2
    else:
        input_channels = 1
    seq_len = args.max_seq_len_modality if args.feature_mode == "all_features" else 1
    logger.info(f"Building {args.model_name} (mode={args.feature_mode}, MaxDays={args.max_days}, SeqLen={seq_len})...")

    if args.model_name == "decoder":
        model = AutoregressiveTransformer(
            day_feat_dim=max_feat_dim,
            ehr_feat_dim=ehr_feat_dim, 
            img_feat_dim=img_feat_dim,
            d_model=args.d_model, 
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout, 
            max_days=args.max_days,
            input_channels=input_channels,
            max_seq_len_per_modality=seq_len,
            enable_contrastive=args.do_contrast,
            contrastive_dim=args.contrastive_dim,
            contrastive_tau=args.contrastive_temperature,
        ).to(device)

    elif args.model_name == "encoder":    
        model = ReadmissionEncoder(
            day_feat_dim=max_feat_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_days=args.max_days,
            input_channels=input_channels,
            max_seq_len_per_modality=seq_len,
            enable_contrastive=args.do_contrast,
            contrastive_dim=args.contrastive_dim,
            contrastive_tau=args.contrastive_temperature,
        ).to(device)

    elif args.model_name == "cross-attention":
        model = CrossAttention(
            ehr_feat_dim=ehr_feat_dim, 
            img_feat_dim=img_feat_dim,
            d_model=args.d_model, 
            nhead=args.nhead, 
            num_layers=args.num_layers,
            dropout=args.dropout, 
            max_days=args.max_days,
            input_channels=input_channels,
            max_seq_len_per_modality=seq_len,
            enable_contrastive=args.do_contrast,
            contrastive_dim=args.contrastive_dim,
            contrastive_tau=args.contrastive_temperature,
        ).to(device)
    elif args.model_name == "mlp":
        mlp_input_dim = max_feat_dim * args.max_days * input_channels
        model = MLP(
            input_dim=mlp_input_dim,
            dropout=args.dropout,
            enable_contrastive=args.do_contrast,
            contrastive_dim=args.contrastive_dim,
            contrastive_tau=args.contrastive_temperature,
        ).to(device)
    train_dataset = FeatureDataset(train_feats)
    val_dataset = FeatureDataset(val_feats)
    
    if args.do_contrast and args.loss_option in ["contrast", "mix"]:
        logger.info("Using GroupBalancedBatchSampler for contrastive learning")
        train_sampler = GroupBalancedBatchSampler(
            train_dataset, 
            batch_size=args.train_batch_size, 
            samples_per_group=2,
            drop_last=False,
            shuffle=True,
            use_batch_balance=args.use_batch_balance,
            verbose=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_func)
        val_sampler = GroupBalancedBatchSampler(
            val_dataset, 
            batch_size=args.test_batch_size, 
            samples_per_group=2,
            drop_last=False,
            shuffle=False,
            use_batch_balance=args.use_batch_balance,
            verbose=True
        )
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_func)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_func)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_func)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.pos_weight).to(device))
    saver = utils.CheckpointSaver(save_dir=args.save_dir, metric_name="auc", maximize_metric=True, log=logger)
    estimated_steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * estimated_steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.1))
    cosine_steps = max(1, total_steps - warmup_steps)
    logger.info(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, cosine_steps={cosine_steps}")
    class CustomLRScheduler:
        def __init__(self, optimizer, warmup_steps, cosine_steps, 
                     base_lr, warmup_start_factor, min_lr_factor):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.cosine_steps = cosine_steps
            self.base_lr = base_lr
            self.warmup_start_lr = base_lr * warmup_start_factor
            self.min_lr = base_lr * min_lr_factor
            self.step_count = 0
            
        def step(self):
            self.step_count += 1
            if self.step_count <= self.warmup_steps:
                progress = self.step_count / self.warmup_steps
                lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            else:
                cosine_step = self.step_count - self.warmup_steps
                progress = cosine_step / self.cosine_steps
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    warmup_start_factor = 1e-4
    min_lr_factor = 1e-5
    
    scheduler = CustomLRScheduler(
        optimizer, warmup_steps, cosine_steps,
        args.lr, warmup_start_factor, min_lr_factor
    )
    
    logger.info(f"LR schedule: warmup {warmup_start_factor*args.lr:.2e} -> {args.lr:.2e}, then cosine {args.lr:.2e} -> {args.lr*min_lr_factor:.2e}")

    if args.do_train:
        best_val_auc = 0.0
        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = getattr(args, "early_stop_patience", None)
        loss_factor = getattr(args, "early_stop_loss_factor", None)
        best_train_metrics = {
            'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'loss': float('inf')
        }
        best_val_metrics = {
            'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'loss': float('inf')
        }

        for epoch in range(1, args.num_epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, 
                feature_mode=args.feature_mode,
                do_contrast=args.do_contrast,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
                loss_option=args.loss_option
            )
            if train_metrics.get('auc', 0) > best_train_metrics['auc']:
                best_train_metrics['auc'] = train_metrics.get('auc', 0)
            if train_metrics.get('f1', 0) > best_train_metrics['f1']:
                best_train_metrics['f1'] = train_metrics.get('f1', 0)
            if train_metrics.get('precision', 0) > best_train_metrics['precision']:
                best_train_metrics['precision'] = train_metrics.get('precision', 0)
            if train_metrics.get('recall', 0) > best_train_metrics['recall']:
                best_train_metrics['recall'] = train_metrics.get('recall', 0)
            if train_metrics.get('loss', float('inf')) < best_train_metrics['loss']:
                best_train_metrics['loss'] = train_metrics.get('loss', float('inf'))
            
            log_dict = {
                "epoch": epoch, "lr": optimizer.param_groups[0]['lr'],
                "train/loss": train_metrics['loss'], "train/auc": train_metrics['auc']
            }
            
            if args.do_contrast and 'contrast_loss' in train_metrics:
                log_dict["train/contrast_loss"] = train_metrics['contrast_loss']
            if args.loss_option == "mix" and 'bce_loss' in train_metrics:
                log_dict["train/bce_loss"] = train_metrics['bce_loss']
            
            if epoch % args.eval_every == 0:
                val_metrics = evaluate(
                    model, val_loader, criterion, device, "VAL", 
                    feature_mode=args.feature_mode,
                    do_contrast=args.do_contrast,
                    contrastive_temperature=args.contrastive_temperature,
                    loss_option=args.loss_option,
                    contrastive_weight=args.contrastive_weight
                )
                saver.save(epoch, model, optimizer, val_metrics['auc'])
                log_dict.update({
                    "val/loss": val_metrics['loss'], 
                    "val/auc": val_metrics['auc'],
                    "val/f1": val_metrics['f1'],
                    "val/precision": val_metrics['precision'],
                    "val/recall": val_metrics['recall'],
                    "val/acc": val_metrics['acc']
                })
                if args.do_contrast and 'contrast_loss' in val_metrics:
                    log_dict["val/contrast_loss"] = val_metrics['contrast_loss']
                if args.loss_option == "mix" and 'bce_loss' in val_metrics:
                    log_dict["val/bce_loss"] = val_metrics['bce_loss']
                logger.info(f"Epoch {epoch} | Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
                if val_metrics.get('auc', 0) > best_val_metrics['auc']:
                    best_val_metrics['auc'] = val_metrics.get('auc', 0)
                if val_metrics.get('f1', 0) > best_val_metrics['f1']:
                    best_val_metrics['f1'] = val_metrics.get('f1', 0)
                if val_metrics.get('precision', 0) > best_val_metrics['precision']:
                    best_val_metrics['precision'] = val_metrics.get('precision', 0)
                if val_metrics.get('recall', 0) > best_val_metrics['recall']:
                    best_val_metrics['recall'] = val_metrics.get('recall', 0)
                if val_metrics.get('loss', float('inf')) < best_val_metrics['loss']:
                    best_val_metrics['loss'] = val_metrics.get('loss', float('inf'))
                current_auc = val_metrics.get('auc', 0.0)
                current_loss = val_metrics.get('loss', float("inf"))
                if current_auc > best_val_auc + 1e-4:
                    best_val_auc = current_auc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if current_loss < best_val_loss:
                    best_val_loss = current_loss
                loss_diverged = False
                if loss_factor is not None:
                    if not np.isfinite(current_loss) or current_loss > best_val_loss * loss_factor:
                        loss_diverged = True
                        logger.info(
                            f"Early stopping (loss divergence) at epoch {epoch}: "
                            f"Val loss={current_loss:.4f}, best_val_loss={best_val_loss:.4f}, factor={loss_factor}"
                        )
                if (patience is not None and epochs_no_improve >= patience) or loss_diverged:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best Val AUC={best_val_auc:.4f}, best Val loss={best_val_loss:.4f}, "
                        f"patience={patience}, loss_factor={loss_factor})"
                    )
                    break
            
            if args.wandb_enabled: wandb.log(log_dict)
        if args.wandb_enabled:
            wandb.run.summary.update({
                'best_train/auc': best_train_metrics['auc'],
                'best_train/f1': best_train_metrics['f1'],
                'best_train/precision': best_train_metrics['precision'],
                'best_train/recall': best_train_metrics['recall'],
                'best_train/loss': best_train_metrics['loss'],
                'best_val/auc': best_val_metrics['auc'],
                'best_val/f1': best_val_metrics['f1'],
                'best_val/precision': best_val_metrics['precision'],
                'best_val/recall': best_val_metrics['recall'],
                'best_val/loss': best_val_metrics['loss'],
            })
            best_metrics_table = wandb.Table(
                columns=["Split", "AUC", "F1", "Precision", "Recall", "Loss"],
                data=[
                    ["Train", 
                     f"{best_train_metrics['auc']:.4f}",
                     f"{best_train_metrics['f1']:.4f}",
                     f"{best_train_metrics['precision']:.4f}",
                     f"{best_train_metrics['recall']:.4f}",
                     f"{best_train_metrics['loss']:.4f}"],
                    ["Val",
                     f"{best_val_metrics['auc']:.4f}",
                     f"{best_val_metrics['f1']:.4f}",
                     f"{best_val_metrics['precision']:.4f}",
                     f"{best_val_metrics['recall']:.4f}",
                     f"{best_val_metrics['loss']:.4f}"]
                ]
            )
            wandb.log({"best_metrics_table": best_metrics_table})
            
            logger.info("Best metrics saved to wandb:")
            logger.info(f"  Train - AUC: {best_train_metrics['auc']:.4f}, F1: {best_train_metrics['f1']:.4f}, "
                       f"Precision: {best_train_metrics['precision']:.4f}, Recall: {best_train_metrics['recall']:.4f}, "
                       f"Loss: {best_train_metrics['loss']:.4f}")
            logger.info(f"  Val   - AUC: {best_val_metrics['auc']:.4f}, F1: {best_val_metrics['f1']:.4f}, "
                       f"Precision: {best_val_metrics['precision']:.4f}, Recall: {best_val_metrics['recall']:.4f}, "
                       f"Loss: {best_val_metrics['loss']:.4f}")
    
    if args.enable_explainability:
        model = utils.load_model_checkpoint(os.path.join(args.save_dir, "best.pth.tar"), model).to(device)
        metrics = evaluate(model, val_loader, criterion, device, "TEST", 
                          feature_mode=args.feature_mode,
                          max_days=args.max_days,
                          max_seq_len=args.max_seq_len_modality,
                          cxr_processor=cxr_processor,
                          ehr_processor=ehr_processor,
                          df_icd=df_icd,
                          df_lab=df_lab,
                          df_med=df_med,
                          modality=args.modality,
                          max_seq_len_modality=args.max_seq_len_modality,
                          feat_dim=max_feat_dim,
                          img_dim=img_feat_dim,
                          ehr_dim=ehr_feat_dim,
                          drop_image_ratio=args.drop_image_ratio,
                          drop_ehr_ratio=args.drop_ehr_ratio,
                          unpaired=args.unpaired,
                          remove_duplication=args.remove_duplication,
                          feature_extraction_batch_size=feature_extraction_batch_size)
        logger.info(f"Final Test AUC: {metrics['auc']:.4f}")
        
        explain_func = explain_admission_end_to_end_all_features if args.feature_mode == "all_features" else explain_admission_end_to_end
        
        for idx, sample in enumerate(tqdm(val_loader.dataset.data_list[:args.num_explain_samples], desc="Generating Viz")):
            viz = explain_func(
                model, cxr_processor, ehr_processor, sample, df_demo, df_icd, df_lab, df_med, device,
                max_days=args.max_days, modality=args.modality,
                **({"max_seq_len_modality": args.max_seq_len_modality} if args.feature_mode == "all_features" else {})
            )
            if args.wandb_enabled and viz:
                wandb.log({f"explain/sample_{idx}_{k}": v for k, v in viz.items()})

    if args.wandb_enabled: wandb.finish()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, required=True)
    parser.add_argument("--icd_file", type=str, required=True)
    parser.add_argument("--lab_file", type=str, required=True)
    parser.add_argument("--med_file", type=str, required=True)
    parser.add_argument("--cxr_pretrained_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_days", type=int, default=2) 
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len_modality", type=int, default=512) 
    parser.add_argument("--cxr_model_type", type=str, default='eva_x_base')
    parser.add_argument("--clinical_bert_model", type=str, default='nazyrova/clinicalBERT')
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./save/readmit_cross_attn_all")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--rand_seed", type=int, default=123)
    parser.add_argument("--l2_wd", type=float, default=1e-5)
    parser.add_argument("--pos_weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--balance_data", action="store_true")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="overfitting_challenge")
    parser.add_argument("--wandb_run_name", type=str, default="run_v1")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "image", "text"])
    parser.add_argument("--feature_mode", type=str, default="all_features", choices=["all_features", "cls_only"])
    parser.add_argument("--feature_extraction_batch_size", type=int, default=512, help="Batch size for feature extraction (CXR processing)")
    parser.add_argument("--filtered_admission_file", type=str, default=None)
    parser.add_argument("--enable_explainability", action="store_true")
    parser.add_argument("--num_explain_samples", type=int, default=10)
    parser.add_argument("--model_name", type=str, required=True, default="decoder", choices=["decoder", "encoder", "cross-attention", "mlp"])
    parser.add_argument("--remove_duplication", action="store_true")
    parser.add_argument("--toy_sample_ratio", type=float, default=None, help="Use only this fraction of data for quick testing (e.g., 0.05 for 5%%)")
    parser.add_argument("--drop_image_ratio", type=float, default=0.0, help="Drop image modality with this probability for each day (0.0-1.0)")
    parser.add_argument("--drop_ehr_ratio", type=float, default=0.0, help="Drop EHR modality with this probability for each day (0.0-1.0)")
    parser.add_argument("--unpaired", action="store_true", help="Include all days (CXR + EHR-only days). If False, only CXR days.")

    # Validation list export options
    parser.add_argument("--export_val_lists", action="store_true",
                        help="Export val admission_id lists to txt (balance_val_list.txt / imbalance_val_list.txt)")
    parser.add_argument("--val_list_out_dir", type=str, default=".",
                        help="Output directory for exported val list txt files")
    parser.add_argument("--export_val_lists_only", action="store_true",
                        help="Only export val lists and exit (skips training/extraction)")
    
    # Contrastive Learning Arguments
    parser.add_argument("--do_contrast", action="store_true", help="Enable contrastive learning with SUPER_GROUP")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="Weight for contrastive loss (lambda)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss")
    parser.add_argument("--contrastive_dim", type=int, default=128, help="Projection head output dimension")
    parser.add_argument("--loss_option", default="bce", choices=["bce", "contrast", "mix"])
    parser.add_argument("--use_batch_balance", action="store_true", help="Use batch balancing in GroupBalancedBatchSampler")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (in eval steps)")
    parser.add_argument("--early_stop_loss_factor", type=float, default=2.5, help="Stop if val loss exceeds best loss by this factor")
    
    # Feature caching options
    parser.add_argument("--use_feature_cache", action="store_true", 
                        help="Cache extracted features to disk (skip re-extraction if encoder settings unchanged)")
    parser.add_argument("--feature_cache_dir", type=str, default="./cache/features",
                        help="Directory to store cached features")
    parser.add_argument("--clear_feature_cache", action="store_true",
                        help="Clear existing cache before running (forces re-extraction)")

    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())
