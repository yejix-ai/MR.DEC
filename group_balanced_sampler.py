import torch
from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict, Counter
import numpy as np

class GroupBalancedBatchSampler(Sampler):
    """
    Contrastive Learning을 위한 Batch Sampler.
    
    공통 기능:
        - 각 배치에 (SUPER_GROUP, label) 조합이 같은 샘플을 최소 2개(samples_per_group)씩 포함시킵니다.
    
    모드 설정 (use_batch_balance):
        - True: Super Group의 실제 분포 비율(TARGET_GROUP_RATIOS)에 따라 확률적으로 그룹을 선택합니다.
        - False: 모든 Super Group을 균등한 확률(Uniform)로 선택합니다.
    """
    
    # 14개 그룹에 대한 재구성된 비율
    TARGET_GROUP_RATIOS = {
        'Circulatory': 0.195,
        'Endocrine & Metabolic': 0.131,
        'Symptoms & Signs': 0.084,
        'Respiratory': 0.079,
        'Trauma & Poisoning': 0.076,
        'Digestive': 0.072,
        'Genitourinary': 0.059,
        'Factors & Services': 0.052,
        'Blood & Immune': 0.052,
        'Mental Disorders': 0.050,
        'Nervous System': 0.045,
        'Musculoskeletal & Skin': 0.045,
        'Infectious': 0.037,
        'Neoplasms': 0.025
    }

    def __init__(self, dataset, batch_size, samples_per_group=2, drop_last=False, shuffle=True, 
                 use_batch_balance=True, verbose=True):
        """
        Args:
            use_batch_balance (bool): True면 비율 기반 샘플링, False면 균등 샘플링
            verbose (bool): True면 배치 통계 출력
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_group = samples_per_group
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.use_batch_balance = use_batch_balance # 모드 선택 플래그
        self.verbose = verbose
        
        # 데이터 정리
        self.group_label_to_indices = defaultdict(list)
        self.super_group_map = {} 
        
        # 제외되거나 알 수 없는 그룹 처리
        unknown_groups = set()
        
        for idx, item in enumerate(dataset.data_list):
            super_group = item.get('super_group', 'Unknown')
            
            # 비율 사전에 없으면 스킵 (True/False 모드 모두 동일하게 적용)
            if super_group not in self.TARGET_GROUP_RATIOS:
                continue

            label = item['label'].item() if torch.is_tensor(item['label']) else item['label']
            if isinstance(label, (list, np.ndarray)):
                label = label[0]
            
            key = (super_group, int(label))
            self.group_label_to_indices[key].append(idx)
            self.super_group_map[key] = super_group
        
        self.group_label_keys = list(self.group_label_to_indices.keys())
        
        print(f"[GroupBalancedBatchSampler] Initialized.")
        print(f"  - Mode: {'Weighted Balance (Ratio)' if self.use_batch_balance else 'Uniform Balance'}")
        print(f"  - Valid combinations: {len(self.group_label_keys)}")
        
        # 통계 출력
        for key in sorted(self.group_label_keys, key=lambda x: (x[1], x[0])):
            group, label = key
            count = len(self.group_label_to_indices[key])
            print(f"    * ({group}, label={label}): {count} samples")

    def __iter__(self):
        # 1. 인덱스 셔플링
        group_label_pools = {}
        for key in self.group_label_keys:
            pool = self.group_label_to_indices[key].copy()
            if self.shuffle:
                random.shuffle(pool)
            group_label_pools[key] = pool
        
        positions = {key: 0 for key in self.group_label_keys}
        batches = []
        
        while True:
            # 2. 현재 샘플이 남아있는(Available) 조합 찾기 (최소 samples_per_group 이상인 것만)
            available_keys = []
            for key in self.group_label_keys:
                remaining = len(group_label_pools[key]) - positions[key]
                if remaining >= self.samples_per_group:
                    available_keys.append(key)
            
            if not available_keys:
                break
            
            # 배치에 들어갈 그룹(Key)의 개수 계산
            num_groups_in_batch = self.batch_size // self.samples_per_group
            num_select = min(len(available_keys), num_groups_in_batch)
            
            # 3. 그룹 선택 (여기가 핵심 분기점)
            if self.use_batch_balance:
                # [CASE A] 비율 기반 확률적 선택 (Weighted Random)
                probs = []
                for key in available_keys:
                    s_group = self.super_group_map[key]
                    weight = self.TARGET_GROUP_RATIOS.get(s_group, 0.01)
                    probs.append(weight)
                
                # 정규화
                probs = np.array(probs)
                if probs.sum() == 0: probs = np.ones(len(probs)) / len(probs)
                else: probs = probs / probs.sum()
                
                if self.shuffle:
                    selected_indices = np.random.choice(len(available_keys), size=num_select, replace=False, p=probs)
                    selected_keys = [available_keys[i] for i in selected_indices]
                else:
                    selected_keys = available_keys[:num_select] # 셔플 안하면 그냥 앞에서부터
                    
            else:
                # [CASE B] 균등 선택 (Uniform Random) - 기존 min 2 방식
                # 모든 그룹이 뽑힐 확률이 동일함 (데이터 많은 그룹은 나중에 몰려 나옴)
                if self.shuffle:
                    selected_keys = random.sample(available_keys, num_select)
                else:
                    selected_keys = available_keys[:num_select]

            # 4. 선택된 키에서 데이터 추출 (공통 로직)
            batch = []
            for key in selected_keys:
                start_pos = positions[key]
                # 여기서 무조건 samples_per_group(2개) 만큼 자름 -> 최소 개수 보장
                indices = group_label_pools[key][start_pos:start_pos + self.samples_per_group]
                batch.extend(indices)
                positions[key] += self.samples_per_group
            
            # 5. 빈 공간 채우기 (자투리 공간)
            # Positive pair 보장을 위해 각 그룹에서 최소 2개씩 가져오도록 함
            if len(batch) < self.batch_size:
                # 빈 공간 채울 때는 모든 그룹에서 자유롭게 채우되, 각 그룹에서 최소 2개씩 가져옴
                # 먼저 available_keys에서, 그 다음 모든 group_label_keys에서
                all_keys_to_fill = list(available_keys) + [k for k in self.group_label_keys if k not in available_keys]
                for key in all_keys_to_fill:
                    if len(batch) >= self.batch_size: break
                    rem_in_pool = len(group_label_pools[key]) - positions[key]
                    if rem_in_pool >= self.samples_per_group:
                        # 최소 2개 이상 남아있으면 2개씩 가져옴 (positive pair 보장)
                        take = min(self.batch_size - len(batch), rem_in_pool)
                        # take가 1개면 다음 그룹으로 넘어감 (positive pair 보장을 위해)
                        if take < self.samples_per_group:
                            continue
                        start_pos = positions[key]
                        batch.extend(group_label_pools[key][start_pos:start_pos+take])
                        positions[key] += take
                    elif rem_in_pool > 0 and len(batch) + rem_in_pool <= self.batch_size:
                        # 남은 샘플이 1개이고, 배치 크기를 초과하지 않으면 가져옴
                        # (이 경우는 positive pair가 없을 수 있지만, 배치 크기를 맞추기 위해)
                        start_pos = positions[key]
                        batch.extend(group_label_pools[key][start_pos:start_pos+rem_in_pool])
                        positions[key] += rem_in_pool

            if len(batch) > 0:
                if self.shuffle: random.shuffle(batch)
                batches.append(batch)
        
        # 6. 남은 데이터 처리
        if not self.drop_last:
            remaining_indices = []
            for key in self.group_label_keys:
                start = positions[key]
                remaining_indices.extend(group_label_pools[key][start:])
            
            if remaining_indices:
                if self.shuffle: random.shuffle(remaining_indices)
                for i in range(0, len(remaining_indices), self.batch_size):
                    batch = remaining_indices[i:i + self.batch_size]
                    if len(batch) > 0:
                        batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
            
        # 통계 출력 및 Yield
        for i, batch in enumerate(batches):
            if self.verbose:
                self._print_batch_stats(i, batch)
            yield batch

    def _print_batch_stats(self, batch_idx, batch_indices):
        """배치 통계 출력"""
        groups = []
        labels = []
        
        for idx in batch_indices:
            item = self.dataset.data_list[idx]
            groups.append(item.get('super_group', 'Unknown'))
            l = item['label']
            if torch.is_tensor(l): l = l.item()
            elif isinstance(l, (list, np.ndarray)): l = l[0]
            labels.append(int(l))
            
        group_counts = Counter(groups)
        label_counts = Counter(labels)
        total = len(batch_indices)
        
        # 배치 크기 확인
        expected_size = self.batch_size
        if total != expected_size:
            print(f"\n⚠️  [Batch {batch_idx}] Size: {total} (Expected: {expected_size}, Diff: {expected_size - total})")
        else:
            print(f"\n🏷️  [Batch {batch_idx}] Size: {total}")
        neg = label_counts.get(0, 0)
        pos = label_counts.get(1, 0)
        print(f"   └── Label Dist: Neg(0)={neg} ({neg/total*100:.1f}%), Pos(1)={pos} ({pos/total*100:.1f}%)")
        print(f"   └── Group Dist:")
        sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
        for g, c in sorted_groups:
            print(f"       - {g}: {c} ({c/total*100:.1f}%)")

    def __len__(self):
        total = sum(len(v) for v in self.group_label_to_indices.values())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size