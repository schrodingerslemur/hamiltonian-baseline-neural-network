import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HamiltonianDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        t = self.data.iloc[idx, 0]
        q = self.data.iloc[idx, 1:3].values.astype(np.float32)
        p = self.data.iloc[idx, 3:5].values.astype(np.float32)
        H = self.data.iloc[idx, 5].astype(np.float32)
        pq = np.concatenate((q,p))
        return torch.tensor(pq, dtype=torch.float32), torch.tensor(H, dtype=torch.float32)
    
class HamiltonianFiniteDataset(HamiltonianDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file)
    
    def __getitem__(self, idx):
        if idx >= len(self.data) - 1:
            raise IndexError("Index out of bounds: idx + 1 exceeds data length")
        
        t = self.data.iloc[idx, 0]
        q = self.data.iloc[idx, 1:3].values.astype(np.float32)
        q_dt = self.data.iloc[idx+1, 1:3].values.astype(np.float32)
        q_t = q[:len(q_dt)]

        p = self.data.iloc[idx, 3:5].values.astype(np.float32)
        p_dt = self.data.iloc[idx+1, 3:5].values.astype(np.float32)
        p_t = p[:len(p_dt)]

        pq = np.concatenate((q_t, p_t))
        q_pt = np.concatenate((q_t, p_dt))
        qt_p = np.concatenate((q_dt, p_t))

        return torch.tensor(pq, dtype=torch.float32), torch.tensor(qt_p, dtype=torch.float32), torch.tensor(q_pt, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data) - 1  # to prevent out of bounds


