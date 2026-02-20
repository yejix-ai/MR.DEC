import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import sys
import os
from timm.data.transforms_factory import create_transform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OnlineCXRProcessor:    
    def __init__(self, model_type='eva_x_base', pretrained_path=None, device='cuda'):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # cls 전용 프로세서: 기본은 CLS 출력만 사용
        self.return_all_features = False
        
        self.model_configs = {
            'eva_x_tiny': {'feat_dim': 192},
            'eva_x_small': {'feat_dim': 384},
            'eva_x_base': {'feat_dim': 768},
            'microsoft/rad-dino': {'feat_dim': 768},  # RAD-DINO v2 (ViT-B/14)
            'microsoft/rad-dino-maira-2': {'feat_dim': 768}  # DINOv2-based model
        }
        self.feat_dim = self.model_configs.get(model_type, {'feat_dim': 768})['feat_dim']
        
        self._load_model(pretrained_path)
        self._setup_transforms()
    
    def _load_model(self, pretrained_path):
        if self.model_type == 'eva_x_tiny':
            from cxr.eva_x import eva_x_tiny_patch16
            self.encoder = eva_x_tiny_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'eva_x_small':
            from cxr.eva_x import eva_x_small_patch16
            self.encoder = eva_x_small_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'eva_x_base':
            from cxr.eva_x import eva_x_base_patch16
            self.encoder = eva_x_base_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'microsoft/rad-dino-maira-2':
            # HuggingFace model
            from transformers import AutoImageProcessor, AutoModel
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_type)
            self.encoder = AutoModel.from_pretrained(self.model_type)
            if pretrained_path and os.path.exists(pretrained_path):
                # Load additional weights if provided and file exists
                try:
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    self.encoder.load_state_dict(state_dict, strict=False)
                    print(f"Loaded additional weights from {pretrained_path}")
                except Exception as e:
                    print(f"Warning: Failed to load pretrained weights from {pretrained_path}: {e}")
            elif pretrained_path:
                print(f"Warning: Pretrained path {pretrained_path} does not exist. Using default HuggingFace weights.")
            self.is_huggingface = True
        else:
            # Try to load as HuggingFace model by default
            try:
                from transformers import AutoImageProcessor, AutoModel
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_type)
                self.encoder = AutoModel.from_pretrained(self.model_type)
                if pretrained_path and os.path.exists(pretrained_path):
                    # Load additional weights if provided and file exists
                    try:
                        state_dict = torch.load(pretrained_path, map_location='cpu')
                        self.encoder.load_state_dict(state_dict, strict=False)
                        print(f"Loaded additional weights from {pretrained_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load pretrained weights from {pretrained_path}: {e}")
                elif pretrained_path:
                    print(f"Warning: Pretrained path {pretrained_path} does not exist. Using default HuggingFace weights.")
                self.is_huggingface = True
                print(f"Loaded {self.model_type} as HuggingFace model")
            except Exception as e:
                raise ValueError(f"Unknown model type: {self.model_type}. Error: {e}")
        
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
    
    def _setup_transforms(self):
        """Setup image transforms"""
        # HuggingFace models use their own image processor
        if hasattr(self, 'is_huggingface') and self.is_huggingface:
            # Transform is handled by image_processor
            self.transforms = None
        else:
            self.transforms = create_transform(
                input_size=(3, 224, 224),
                is_training=False,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
                interpolation='bicubic',
                crop_pct=0.9
            )
    
    def extract_features(self, image_paths, batch_size=32):
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        features_dict = {}
        failed_images = []
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        print(f"  [CXR] Device: {self.device}, Total batches: {total_batches}, batch_size: {batch_size}")
        
        def load_image(img_path):
            """이미지 로드 (병렬 처리용)"""
            try:
                img = Image.open(img_path).convert('RGB')
                return (img_path, img, None)
            except Exception as e:
                return (img_path, None, str(e))
        
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(image_paths), batch_size)):
                batch_start = time.time()
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                valid_paths = []
                
                # 병렬 이미지 로딩 (8 workers)
                io_start = time.time()
                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(load_image, batch_paths))
                io_time = time.time() - io_start
                
                # 결과 처리
                for img_path, img, error in results:
                    if error:
                        print(f"Failed to load {img_path}: {error}")
                        failed_images.append(img_path)
                    elif img is not None:
                        if self.is_huggingface:
                            batch_images.append(img)
                        else:
                            img_tensor = self.transforms(img)
                            batch_images.append(img_tensor)
                        valid_paths.append(img_path)
                
                if not batch_images:
                    continue
                
                try:
                    gpu_start = time.time()
                    if self.is_huggingface:
                        # Process with HuggingFace image processor
                        inputs = self.image_processor(batch_images, return_tensors="pt").to(self.device)
                        outputs = self.encoder(**inputs)
                        # return_all_features=True -> 전체 토큰 시퀀스, False -> CLS
                        if self.return_all_features:
                            if hasattr(outputs, 'last_hidden_state'):
                                features = outputs.last_hidden_state  # (B, L, D) includes CLS
                            else:
                                raise ValueError("last_hidden_state not found in outputs (HuggingFace vision model)")
                        else:
                            if hasattr(outputs, 'last_hidden_state'):
                                features = outputs.last_hidden_state[:, 0, :]  # CLS
                            else:
                                features = outputs[0][:, 0, :]
                        features = features.detach().cpu().numpy()
                    else:
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        x = self.encoder.forward_features(batch_tensor)
                        features = self.encoder.forward_head(x, pre_logits=True)
                        features = features.detach().cpu().numpy()
                    gpu_time = time.time() - gpu_start
                    
                    # Store features
                    for path, feat in zip(valid_paths, features):
                        features_dict[path] = feat
                    
                    # 진행률 출력 (매 10배치 또는 첫 배치)
                    if batch_idx % 10 == 0 or batch_idx == 0:
                        total_time = time.time() - batch_start
                        print(f"    Batch {batch_idx+1}/{total_batches} | IO: {io_time:.2f}s, GPU: {gpu_time:.2f}s, Total: {total_time:.2f}s", flush=True)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_images.extend(valid_paths)
        
        return features_dict, failed_images

class OnlineClinicalBERTProcessor:
    
    SUBGROUPS_EXCLUDED = [
        "Z00-Z13", "Z14-Z15", "Z16-Z16", "Z17-Z17", "Z18-Z18", "Z19-Z19",
        "Z20-Z29", "Z30-Z39", "Z40-Z53", "Z55-Z65", "Z66-Z66", "Z67-Z67",
        "Z68-Z68", "Z69-Z76", "Z77-Z99",
    ]
    
    def __init__(self, model_name='nazyrova/clinicalBERT', device='cuda'):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        print(f"Initializing {model_name}...")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_dim = 768
        
    def extract_features(self, text, max_length=256, debug_print=False, batch_size=64):
        """
        텍스트 또는 텍스트 리스트에서 feature 추출
        - text가 str이면 단일 처리
        - text가 list면 배치 처리
        """
        if isinstance(text, str):
            # 단일 텍스트 처리
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features = features.cpu().numpy()
            return features
        else:
            # 배치 처리
            all_features = []
            for i in range(0, len(text), batch_size):
                batch_texts = text[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]  # CLS token
                    all_features.append(features.cpu().numpy())
            
            return np.concatenate(all_features, axis=0) if len(all_features) > 0 else np.array([])
    
    def create_daily_ehr_text(self, row, df_icd, df_lab, df_med, admit_id, day_num, duplication=True):
        """일별 EHR 데이터를 ClinicalBERT용 텍스트로 변환 (ehr_v2.py와 동일한 방식)"""
        text_parts = []
        
        # Demographics
        age = row.get('age', 'unknown')
        gender = row.get('gender', 'unknown')
        ethnicity = row.get('ethnicity', 'unknown')
        text_parts.append(f"Patient is a {age} year old {gender} {ethnicity}")
        
        curr_icd = df_icd[df_icd["hadm_id"] == admit_id]
        if len(curr_icd) > 0:
            icd_descs = []
            for _, row in curr_icd.iterrows():
                subgroup = row.get("SUBGROUP", "")
                desc = row.get("SUBGROUP_DESC", "")
                # SUBGROUP으로 제외 여부 판단, SUBGROUP_DESC를 텍스트로 사용
                if isinstance(desc, str) and isinstance(subgroup, str) and subgroup not in self.SUBGROUPS_EXCLUDED:
                    icd_descs.append(desc)
            if icd_descs:
                text_parts.append(f"Diagnosed with: {', '.join(icd_descs)}")
        
        # Laboratory
        if "charttime" in df_lab.columns:
            curr_lab = df_lab[(df_lab["hadm_id"] == admit_id) & 
                             (df_lab["Day_Number"] == float(day_num))]
            if len(curr_lab) > 0:
                abnormal_labs = curr_lab[curr_lab["flag"] == "abnormal"]["label_fluid"].tolist()
                abnormal_labs = [lab for lab in abnormal_labs if isinstance(lab, str)]
                if abnormal_labs:
                    text_parts.append(f"Abnormal laboratory findings: {', '.join(abnormal_labs)}")
        
        # Medications
        if "charttime" in df_med.columns or "starttime" in df_med.columns:
            curr_med = df_med[(df_med["hadm_id"] == admit_id) & 
                             (df_med["Day_Number"] == float(day_num))]
            if len(curr_med) > 0:
                meds = curr_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].tolist()
                meds = [med for med in meds if isinstance(med, str)]
                if meds:
                    text_parts.append(f"Prescribed medications: {', '.join(meds)}")
        
        # 최종 문장 생성
        final_text = f"Hospital day {day_num}. {'. '.join(text_parts)}."
        
        # duplication이 True면 진단명/약물명/검사명 단위로 중복 제거
        if duplication:
            # "Diagnosed with:", "Prescribed medications:", "Abnormal laboratory findings:" 뒤의 항목들 중복 제거
            import re
            
            # 각 섹션별로 처리
            patterns = [
                (r'(Diagnosed with: )([^.]+)', r'\1'),
                (r'(Prescribed medications: )([^.]+)', r'\1'),
                (r'(Abnormal laboratory findings: )([^.]+)', r'\1')
            ]
            
            for pattern, prefix in patterns:
                match = re.search(pattern, final_text)
                if match:
                    full_match = match.group(0)
                    prefix_text = match.group(1)
                    items_text = match.group(2)
                    
                    # 쉼표로 구분된 항목들 추출
                    items = [item.strip() for item in items_text.split(',')]
                    # 중복 제거 (순서 유지, 하나는 남김)
                    unique_items = list(dict.fromkeys(items))
                    
                    # 재구성
                    new_text = prefix_text + ', '.join(unique_items)
                    final_text = final_text.replace(full_match, new_text)
        
        return final_text
    

    def generate_features_for_admission(self, df_demo, df_icd, df_lab, df_med, hadm_id, save_text=False, save_dir=None):

        df_adm = df_demo[df_demo['hadm_id'] == hadm_id]
        if len(df_adm) == 0:
            raise ValueError(f"Admission {hadm_id} not found in demographics")
        
        row = df_adm.iloc[0]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        
        dt_range = pd.date_range(
            start=pd.to_datetime(admit_dt).date(),
            end=pd.to_datetime(discharge_dt).date()
        )
        
        if len(dt_range) <= 1:
            # Single day admission - create one entry
            dt_range = [pd.to_datetime(admit_dt).date()]
        
        daily_features = []
        daily_texts = []  # Store texts for saving
        
        for day_idx, dt in enumerate(dt_range):
            day_num = (dt if isinstance(dt, pd.Timestamp) else pd.Timestamp(dt)).date()
            day_num = (day_num - pd.to_datetime(admit_dt).date()).days + 1
            

            daily_text = self.create_daily_ehr_text(
                row, df_icd, df_lab, df_med, hadm_id, day_num
            )
            
            daily_texts.append(daily_text)
            
            features = self.extract_features(daily_text, max_length=512)
            daily_features.append(features)  # Keep (1, 768) shape like ehr_v2.py
        
        # Save texts if requested
        if save_text and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            subject_id = row.get('subject_id', 'unknown')
            text_file = os.path.join(save_dir, f"{subject_id}_{hadm_id}_ehr_text.txt")
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Admission ID: {hadm_id} (Subject: {subject_id}) ===\n")
                f.write(f"Admit Date: {admit_dt}\n")
                f.write(f"Discharge Date: {discharge_dt}\n")
                f.write(f"Total Days: {len(daily_texts)}\n")
                f.write("="*80 + "\n\n")
                
                for day_idx, text in enumerate(daily_texts, 1):
                    f.write(f"[Day {day_idx}]\n")
                    f.write(f"{text}\n")
                    f.write("-"*80 + "\n\n")
        
        # Return features
        return np.stack(daily_features) if len(daily_features) > 0 else None

class OnlineCXRProcessorAll:    
    def __init__(self, model_type='eva_x_base', pretrained_path=None, device='cuda', return_all_features=True):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.return_all_features = return_all_features
        
        self.model_configs = {
            'eva_x_tiny': {'feat_dim': 192},
            'eva_x_small': {'feat_dim': 384},
            'eva_x_base': {'feat_dim': 768},
            'microsoft/rad-dino': {'feat_dim': 768},  # RAD-DINO v2 (ViT-B/14)
            'microsoft/rad-dino-maira-2': {'feat_dim': 768}  # DINOv2-based model
        }
        # model_type이 dict에 없으면 기본값 768 사용
        self.feat_dim = self.model_configs.get(model_type, {'feat_dim': 768})['feat_dim']
        
        self._load_model(pretrained_path)
        self._setup_transforms()
    
    def _load_model(self, pretrained_path):
        if self.model_type == 'eva_x_tiny':
            from cxr.eva_x import eva_x_tiny_patch16
            self.encoder = eva_x_tiny_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'eva_x_small':
            from cxr.eva_x import eva_x_small_patch16
            self.encoder = eva_x_small_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'eva_x_base':
            from cxr.eva_x import eva_x_base_patch16
            self.encoder = eva_x_base_patch16(pretrained=pretrained_path)
            self.is_huggingface = False
        elif self.model_type == 'microsoft/rad-dino-maira-2':
            # HuggingFace model
            from transformers import AutoImageProcessor, AutoModel
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_type)
            self.encoder = AutoModel.from_pretrained(self.model_type)
            if pretrained_path and os.path.exists(pretrained_path):
                # Load additional weights if provided and file exists
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.encoder.load_state_dict(state_dict, strict=False)
            elif pretrained_path:
                # Warn if pretrained_path is provided but file doesn't exist
                print(f"Warning: Pretrained weight file not found at {pretrained_path}. Using HuggingFace model weights only.")
            self.is_huggingface = True
        else:
            # Try to load as HuggingFace model by default
            try:
                from transformers import AutoImageProcessor, AutoModel
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_type)
                self.encoder = AutoModel.from_pretrained(self.model_type)
                if pretrained_path and os.path.exists(pretrained_path):
                    # Load additional weights if provided and file exists
                    try:
                        state_dict = torch.load(pretrained_path, map_location='cpu')
                        self.encoder.load_state_dict(state_dict, strict=False)
                        print(f"Loaded additional weights from {pretrained_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load pretrained weights from {pretrained_path}: {e}")
                elif pretrained_path:
                    print(f"Warning: Pretrained path {pretrained_path} does not exist. Using default HuggingFace weights.")
                self.is_huggingface = True
                print(f"Loaded {self.model_type} as HuggingFace model")
            except Exception as e:
                raise ValueError(f"Unknown model type: {self.model_type}. Error: {e}")
            
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
    def _setup_transforms(self):
        # HuggingFace models use their own image processor
        if hasattr(self, 'is_huggingface') and self.is_huggingface:
            # Transform is handled by image_processor
            self.transforms = None
        else:
            self.transforms = create_transform(
                input_size=(3, 224, 224),
                is_training=False,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
                interpolation='bicubic',
                crop_pct=0.9
            )
    
    def extract_features(self, image_paths, batch_size=32):
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        features_dict = {}
        failed_images = []
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        print(f"  [CXR-All] Device: {self.device}, Total batches: {total_batches}, batch_size: {batch_size}")
        
        def load_image(img_path):
            """이미지 로드 (병렬 처리용)"""
            try:
                img = Image.open(img_path).convert('RGB')
                return (img_path, img, None)
            except Exception as e:
                return (img_path, None, str(e))
        
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(image_paths), batch_size)):
                batch_start = time.time()
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                valid_paths = []
                
                # 병렬 이미지 로딩 (8 workers)
                io_start = time.time()
                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(load_image, batch_paths))
                io_time = time.time() - io_start
                
                # 결과 처리
                for img_path, img, error in results:
                    if error:
                        print(f"Failed to load {img_path}: {error}")
                        failed_images.append(img_path)
                    elif img is not None:
                        if self.is_huggingface:
                            batch_images.append(img)
                        else:
                            img_tensor = self.transforms(img)
                            batch_images.append(img_tensor)
                        valid_paths.append(img_path)
                
                if not batch_images:
                    continue
                
                try:
                    gpu_start = time.time()
                    if self.is_huggingface:
                        # Process with HuggingFace image processor
                        inputs = self.image_processor(batch_images, return_tensors="pt").to(self.device)
                        outputs = self.encoder(**inputs)
                        
                        # return_all_features=True -> 전체 토큰 시퀀스, False -> CLS
                        if self.return_all_features:
                            if hasattr(outputs, 'last_hidden_state'):
                                features = outputs.last_hidden_state  # (B, L, D) includes CLS
                            else:
                                features = outputs[0]
                        else:
                            if hasattr(outputs, 'last_hidden_state'):
                                features = outputs.last_hidden_state[:, 0, :]  # CLS
                            else:
                                features = outputs[0][:, 0, :]
                    else:
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        
                        # [수정] return_all_features 옵션 적용
                        if self.return_all_features:
                            # (Batch, Num_Patches, Dim)
                            features = self.encoder.forward_features(batch_tensor)
                        else:
                            # (Batch, Dim) - CLS Pooling
                            x = self.encoder.forward_features(batch_tensor)
                            features = self.encoder.forward_head(x, pre_logits=True)
                        
                    features = features.detach().cpu().numpy()
                    gpu_time = time.time() - gpu_start
                    
                    for path, feat in zip(valid_paths, features):
                        features_dict[path] = feat
                    
                    # 진행률 출력 (매 10배치 또는 첫 배치)
                    if batch_idx % 10 == 0 or batch_idx == 0:
                        total_time = time.time() - batch_start
                        print(f"    Batch {batch_idx+1}/{total_batches} | IO: {io_time:.2f}s, GPU: {gpu_time:.2f}s, Total: {total_time:.2f}s", flush=True)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_images.extend(valid_paths)
        
        return features_dict, failed_images


class OnlineClinicalBERTProcessorAll:
    
    SUBGROUPS_EXCLUDED = [
        "Z00-Z13", "Z14-Z15", "Z16-Z16", "Z17-Z17", "Z18-Z18", "Z19-Z19",
        "Z20-Z29", "Z30-Z39", "Z40-Z53", "Z55-Z65", "Z66-Z66", "Z67-Z67",
        "Z68-Z68", "Z69-Z76", "Z77-Z99",
    ]
    
    def __init__(self, model_name='nazyrova/clinicalBERT', device='cuda', return_all_features=True):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            return_all_features: True면 (Batch, Seq, Dim), False면 (Batch, Dim) 반환
        """
        print(f"Initializing {model_name}...")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.return_all_features = return_all_features
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_dim = 768
        
        
    def extract_features(self, text, max_length=512, debug_print=False, batch_size=64):
        """
        텍스트 또는 텍스트 리스트에서 feature 추출
        - text가 str이면 단일 처리
        - text가 list면 배치 처리
        """
        if isinstance(text, str):
            # 단일 텍스트 처리
            if self.model_name == "answerdotai/ModernBERT-base":
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
            else:
                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.return_all_features:
                    features = outputs.last_hidden_state
                else:
                    features = outputs.last_hidden_state[:, 0, :]
                features = features.cpu().numpy()
            return features
        else:
            # 배치 처리
            all_features = []
            for i in range(0, len(text), batch_size):
                batch_texts = text[i:i+batch_size]
                
                if self.model_name == "answerdotai/ModernBERT-base":
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        batch_texts,
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    if self.return_all_features:
                        features = outputs.last_hidden_state  # (B, Seq, Dim)
                    else:
                        features = outputs.last_hidden_state[:, 0, :]  # (B, Dim)
                    all_features.append(features.cpu().numpy())
            
            return np.concatenate(all_features, axis=0) if len(all_features) > 0 else np.array([])
    
    def create_daily_ehr_text(self, row, df_icd, df_lab, df_med, admit_id, day_num, remove_duplication=True):
        """
        일별 EHR 데이터를 ClinicalBERT용 텍스트로 변환
        [수정사항] float64 타입의 Day_Number를 int로 강제 변환하여 매칭 오류 해결
        """
        text_parts = []
        
        # 1. Demographics
        age = row.get('age', 'unknown')
        gender = row.get('gender', 'unknown')
        ethnicity = row.get('ethnicity', 'unknown')
        text_parts.append(f"Patient is a {age} year old {gender} {ethnicity}")
        
        # 2. ICD Diagnoses
        curr_icd = df_icd[df_icd["hadm_id"] == admit_id]
        if len(curr_icd) > 0:
            icd_descs = []
            for _, r in curr_icd.iterrows():
                subgroup = r.get("SUBGROUP", "")
                desc = r.get("SUBGROUP_DESC", "")
                if isinstance(desc, str) and isinstance(subgroup, str) and subgroup not in self.SUBGROUPS_EXCLUDED:
                    icd_descs.append(desc)
            if icd_descs:
                text_parts.append(f"Diagnosed with: {', '.join(icd_descs)}")
        
        # 3. Laboratory Findings (수정됨)
        # 먼저 ID로 필터링 (속도 향상 및 범위 축소)
        lab_subset = df_lab[df_lab["hadm_id"] == admit_id]
        
        if not lab_subset.empty and "Day_Number" in lab_subset.columns:
            # [핵심 수정] Day_Number 컬럼을 int로 바꿔서 비교 (2.0 -> 2)
            curr_lab = lab_subset[lab_subset["Day_Number"].astype(int) == int(day_num)]
            
            if len(curr_lab) > 0:
                abnormal_labs = curr_lab[curr_lab["flag"] == "abnormal"]["label_fluid"].tolist()
                abnormal_labs = [str(lab) for lab in abnormal_labs if pd.notnull(lab)]
                abnormal_labs = list(set(abnormal_labs)) # 중복 제거
                if abnormal_labs:
                    text_parts.append(f"Abnormal laboratory findings: {', '.join(abnormal_labs)}")
        
        # 4. Medications (수정됨)
        med_subset = df_med[df_med["hadm_id"] == admit_id]
        
        if not med_subset.empty and "Day_Number" in med_subset.columns:
            # [핵심 수정] Day_Number 컬럼을 int로 바꿔서 비교
            curr_med = med_subset[med_subset["Day_Number"].astype(int) == int(day_num)]
            
            if len(curr_med) > 0:
                meds = curr_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].tolist()
                meds = [str(med) for med in meds if pd.notnull(med)]
                meds = list(set(meds)) # 중복 제거
                if meds:
                    text_parts.append(f"Prescribed medications: {', '.join(meds)}")
        
        # 최종 문장 생성
        final_text = f"Hospital day {day_num}. {'. '.join(text_parts)}."
        
        # remove_duplication이 True면 진단명/약물명/검사명 단위로 중복 제거
        if remove_duplication:
            # "Diagnosed with:", "Prescribed medications:", "Abnormal laboratory findings:" 뒤의 항목들 중복 제거
            import re
            
            # 각 섹션별로 처리
            patterns = [
                (r'(Diagnosed with: )([^.]+)', r'\1'),
                (r'(Prescribed medications: )([^.]+)', r'\1'),
                (r'(Abnormal laboratory findings: )([^.]+)', r'\1')
            ]
            
            for pattern, prefix in patterns:
                match = re.search(pattern, final_text)
                if match:
                    full_match = match.group(0)
                    prefix_text = match.group(1)
                    items_text = match.group(2)
                    
                    # 쉼표로 구분된 항목들 추출
                    items = [item.strip() for item in items_text.split(',')]
                    # 중복 제거 (순서 유지, 하나는 남김)
                    unique_items = list(dict.fromkeys(items))
                    
                    # 재구성
                    new_text = prefix_text + ', '.join(unique_items)
                    final_text = final_text.replace(full_match, new_text)
        
        return final_text
    
    def generate_features_for_admission(self, df_demo, df_icd, df_lab, df_med, hadm_id):
        # First/Last Day 실험을 위해 main.py에서 별도로 호출하는 경우가 많으므로
        # 이 메소드는 '전체 입원 기간'을 생성하는 기본 로직을 유지하되,
        # extract_features의 변경된 반환 차원을 따르도록 둡니다.
        
        df_adm = df_demo[df_demo['hadm_id'] == hadm_id]
        if len(df_adm) == 0:
            raise ValueError(f"Admission {hadm_id} not found in demographics")
        
        row = df_adm.iloc[0]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        
        dt_range = pd.date_range(
            start=pd.to_datetime(admit_dt).date(),
            end=pd.to_datetime(discharge_dt).date()
        )
        
        if len(dt_range) <= 1:
            dt_range = [pd.to_datetime(admit_dt).date()]
        
        daily_features = []
        for day_idx, dt in enumerate(dt_range):
            day_num = (dt if isinstance(dt, pd.Timestamp) else pd.Timestamp(dt)).date()
            day_num = (day_num - pd.to_datetime(admit_dt).date()).days + 1
            
            daily_text = self.create_daily_ehr_text(
                row, df_icd, df_lab, df_med, hadm_id, day_num
            )
            
            # (1, Seq_Len, Dim) or (1, Dim)
            features = self.extract_features(daily_text, max_length=512)
            daily_features.append(features)
        
        # 결과 Shape: (Total_Days, 1, Seq_Len, Dim) or (Total_Days, 1, Dim)
        return np.stack(daily_features)


# -----------------------------------------------------------------------------
# 2. Explainability Logic (End-to-End Gradient)
# -----------------------------------------------------------------------------
def get_colored_text_html(words, scores):
    """
    Generate HTML for text coloring based on importance scores.
    """
    # Normalize scores for visualization (0 to 1)
    if len(scores) > 0:
        max_score = max(scores)
        if max_score > 0:
            norm_scores = [s / max_score for s in scores]
        else:
            norm_scores = [0] * len(scores)
    else:
        norm_scores = []
        
    html_parts = ['<div style="font-family: monospace; line-height: 1.5;">']
    
    for word, score in zip(words, norm_scores):
        # Skip special tokens for cleaner view
        if word in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        # Background color: Red with alpha based on score
        # R=255, G=255*(1-score), B=255*(1-score) gives Red intensity
        alpha = score * 0.5  # Max opacity 0.5 for readability
        bg_color = f"rgba(255, 0, 0, {alpha:.2f})"
        html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px; border-radius: 3px;">{word}</span>')
        
    html_parts.append('</div>')
    return " ".join(html_parts)

def explain_single_sample(
    model, cxr_processor, ehr_processor, 
    sample_data, df_demo, df_icd, df_lab, df_med, device
):
    """
    Re-run forward pass from raw inputs to capture gradients for a single sample.
    """
    model.eval()
    cxr_processor.encoder.eval() 
    ehr_processor.model.eval()
    
    # Unpack sample info
    hadm_id = sample_data['hadm_id']
    img_paths = sample_data['img_paths']
    day_indices = sample_data['day_indices']
    label = sample_data['label'].item()
    
    # We will focus on the LAST day/event available for explanation to see immediate readmission cause
    target_idx = -1 
    target_img_path = img_paths[target_idx]
    target_day_idx = day_indices[target_idx]
    
    # ---------------------------------------------------------
    # A. Re-create Inputs with Gradient Tracking
    # ---------------------------------------------------------
    
    # 1. Image Forward (with Gradient)
    try:
        raw_img = Image.open(target_img_path).convert('RGB')
        img_tensor = cxr_processor.transforms(raw_img).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True) # Important: Enable gradient on input image
        
        # Manually call encoder to keep graph connected
        # Note: calling internal layers directly based on online_processor.py structure
        cxr_feat_raw = cxr_processor.encoder.forward_features(img_tensor)
        cxr_feat = cxr_processor.encoder.forward_head(cxr_feat_raw, pre_logits=True) # (1, 768)
    except Exception as e:
        print(f"Explainability Error (Image): {e}")
        return None, None

    # 2. Text Forward (with Gradient)
    try:
        # Re-generate text
        df_adm = df_demo[df_demo['hadm_id'] == hadm_id].iloc[0]
        text = ehr_processor.create_daily_ehr_text(
            df_adm, df_icd, df_lab, df_med, hadm_id, target_day_idx
        )
        
        # Tokenize
        inputs = ehr_processor.tokenizer(
            text, padding='max_length', truncation=True, max_length=512, return_tensors='pt'
        ).to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get Embeddings & Hook Gradient
        # Access embedding layer directly to enable grad on embeddings
        embeddings = ehr_processor.model.embeddings(input_ids)
        embeddings.retain_grad() # Hook gradient here
        
        # Pass embeddings to BERT model
        # Note: BERT models usually accept 'inputs_embeds'
        bert_outputs = ehr_processor.model(
            inputs_embeds=embeddings, 
            attention_mask=attention_mask
        )
        ehr_feat = bert_outputs.last_hidden_state[:, 0, :] # CLS token (1, 768)
        
    except Exception as e:
        print(f"Explainability Error (Text): {e}")
        return None, None

    # ---------------------------------------------------------
    # B. Transformer Forward & Backward
    # ---------------------------------------------------------
    
    # Combine (Just one day for this visualization simplicity, or reconstruct full sequence)
    # To properly visualize the contribution of THIS day, we simulate a 1-day sequence
    # or we could feed the pre-computed features for other days. 
    # For simplicity and clarity, let's treat this as a single-step prediction to find local importance.
    
    # Shape: (B=1, Days=1, Modalities=2, Dim)
    combined_feat = torch.stack([cxr_feat, ehr_feat], dim=1).unsqueeze(0) 
    
    # Forward Transformer
    # mask is all True for this single day
    mask = torch.ones(1, 1, dtype=torch.bool).to(device)
    
    # Enable gradients for the whole model
    model.zero_grad()
    cxr_processor.encoder.zero_grad()
    ehr_processor.model.zero_grad()
    
    day_outputs = model(combined_feat, mask)
    final_logit = day_outputs[-1] # Prediction
    prob = torch.sigmoid(final_logit)
    
    # Backward
    # We want to explain "Why is this probability high?" (or low)
    # Typically we backprop the logit of the target class (1 for Readmission)
    target_score = final_logit # Logit for class 1
    target_score.backward()
    
    # ---------------------------------------------------------
    # C. Generate Visualizations
    # ---------------------------------------------------------
    
    viz_results = {}
    
    # 1. Image Heatmap (Saliency Map)
    # Gradient w.r.t Input Image
    if img_tensor.grad is not None:
        grad = img_tensor.grad[0].cpu().detach() # (3, 224, 224)
        # Take max magnitude across channels
        saliency, _ = torch.max(grad.abs(), dim=0) # (224, 224)
        saliency = saliency.numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Apply slight blur to make it look like a heatmap (common practice for saliency)
        saliency = cv2.GaussianBlur(saliency, (11, 11), 0)
        saliency = (saliency / saliency.max() * 255).astype(np.uint8)
        
        heatmap_colored = cm.jet(saliency / 255.0)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        original_img_np = np.array(raw_img.resize((224, 224)))
        overlay = cv2.addWeighted(original_img_np, 0.6, heatmap_colored, 0.4, 0)
        
        viz_results['cxr_image'] = wandb.Image(
            overlay, 
            caption=f"CXR Attention (GT:{label}, Pred:{prob.item():.2f})"
        )
    
    # 2. Text Importance (Input x Gradient)
    if embeddings.grad is not None:
        # (Input x Gradient) is a standard attribution method
        # embeddings: (1, seq_len, dim), grad: (1, seq_len, dim)
        grad = embeddings.grad[0].cpu().detach()
        emb = embeddings[0].cpu().detach()
        
        # Dot product -> Sum across dimension -> Score per token
        # Shape: (seq_len,)
        token_importance = (grad * emb).sum(dim=-1).abs()
        token_importance = token_importance.numpy()
        
        # Get Tokens
        tokens = ehr_processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Generate HTML
        html_viz = get_colored_text_html(tokens, token_importance)
        viz_results['ehr_html'] = wandb.Html(html_viz, inject=False)
        
        # Top-K words text summary
        top_k_indices = np.argsort(token_importance)[-5:][::-1]
        top_words = [(tokens[i], float(token_importance[i])) for i in top_k_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
        viz_results['ehr_top_words'] = str(top_words)

    return viz_results
