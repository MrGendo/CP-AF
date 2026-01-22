import torch
from torch.utils.data import Sampler
import numpy as np
import collections

class MPerClassSampler(Sampler):
    """
    Custom sampler to ensure each batch contains M samples from K different classes.
    Fixed version.
    """
    def __init__(self, labels, m, batch_size):
        self.labels = np.array(labels)
        self.m = m
        self.batch_size = batch_size
        if self.batch_size % self.m != 0:
            raise ValueError("batch_size must be a multiple of m.")
        
        self.n_classes_per_batch = self.batch_size // self.m
        
        self.labels_to_indices = collections.defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.labels_to_indices[label].append(idx)
        
        self.unique_labels = sorted(self.labels_to_indices.keys())
        
        # Shuffle indices for each class at the beginning of each epoch in __iter__
        self.num_samples = len(self.labels)
        
    def __iter__(self):
        # Shuffle indices for each class at the start of each epoch
        for label in self.unique_labels:
            np.random.shuffle(self.labels_to_indices[label])
            
        used_indices_count = collections.defaultdict(int)
        num_batches = self.num_samples // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # 【修正】正确地从所有类别中随机选择 n_classes_per_batch 个
            if self.n_classes_per_batch <= len(self.unique_labels):
                batch_classes = np.random.choice(self.unique_labels, self.n_classes_per_batch, replace=False)
            else: # If batch size is larger than num_classes * m
                batch_classes = np.random.choice(self.unique_labels, len(self.unique_labels), replace=False)

            for class_label in batch_classes:
                indices_for_class = self.labels_to_indices[class_label]
                start_idx = used_indices_count[class_label]
                
                # Take m samples
                class_batch_indices = list(indices_for_class[start_idx : start_idx + self.m])
                
                # If not enough samples, wrap around by reshuffling and starting from 0
                if len(class_batch_indices) < self.m:
                    np.random.shuffle(self.labels_to_indices[class_label])
                    remaining_needed = self.m - len(class_batch_indices)
                    class_batch_indices.extend(self.labels_to_indices[class_label][:remaining_needed])
                    used_indices_count[class_label] = remaining_needed
                else:
                    used_indices_count[class_label] += self.m
                
                batch_indices.extend(class_batch_indices)
            
            yield batch_indices

    def __len__(self):
        # 【修正】返回正确的批次数
        return self.num_samples // self.batch_size