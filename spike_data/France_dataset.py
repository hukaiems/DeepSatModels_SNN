import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd  
import numpy as np   
import pickle        
import os
# --- CONSTANTS ---
# label remap
REMAP_DICT = {0: 20,
              1: 0, 2: 1, 3: 2,
              4: 20, 5: 20,
              6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11,
              15: 20, 16: 20, 17: 20, 18: 20, 19: 20,
              20: 12,
              21: 20, 22: 20, 23: 20, 24: 20,
              25: 13,
              26: 20, 27: 20, 28: 20, 29: 20, 30: 20, 31: 20, 32: 20,
              33: 14, 34: 15, 35: 16, 36: 17,
              37: 20, 38: 20, 39: 20, 40: 20, 41: 20, 42: 20, 43: 20, 44: 20, 45: 20, 46: 20, 47: 20,
              48: 18,
              49: 20, 50: 20, 51: 20, 52: 20, 53: 20, 54: 20, 55: 20, 56: 20, 57: 20, 58: 20, 59: 20, 60: 20,
              61: 20, 62: 20, 63: 20, 64: 20, 65: 20, 66: 20, 67: 20, 68: 20, 69: 20, 70: 20, 71: 20, 72: 20,
              73: 19,
              74: 20, 75: 20, 76: 20, 77: 20, 78: 20, 79: 20, 80: 20, 81: 20, 82: 20, 83: 20, 84: 20, 85: 20,
              86: 20, 87: 20, 88: 20, 89: 20, 90: 20, 91: 20, 92: 20, 93: 20, 94: 20, 95: 20, 96: 20, 97: 20,
              98: 20, 99: 20, 100: 20, 101: 20, 102: 20, 103: 20, 104: 20, 105: 20, 106: 20, 107: 20,
              108: 20, 109: 20, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20, 116: 20, 117: 20,
              118: 20, 119: 20, 120: 20, 121: 20, 122: 20, 123: 20, 124: 20, 125: 20, 126: 20, 127: 20,
              128: 20, 129: 20, 130: 20, 131: 20, 132: 20, 133: 20, 134: 20, 135: 20, 136: 20, 137: 20,
              138: 20, 139: 20, 140: 20, 141: 20, 142: 20, 143: 20, 144: 20, 145: 20, 146: 20, 147: 20,
              148: 20, 149: 20, 150: 20, 151: 20, 152: 20, 153: 20, 154: 20, 155: 20, 156: 20, 157: 20,
              158: 20, 159: 20, 160: 20, 161: 20, 162: 20, 163: 20, 164: 20, 165: 20, 166: 20, 167: 20
              }

# Normalization stats
NORM_MEAN = torch.tensor([2035.6698, 1791.9031, 1671.6442, 1588.1144, 1926.0026, 3002.9922, 3493.6042, 3417.2434, 3783.7788, 1375.4257, 135.2576, 2335.4624, 1482.3265
])
NORM_STD = torch.tensor([
1637.0688, 1681.7963, 1574.4857, 1779.6345, 1725.1960, 1530.0372, 1535.4208, 1517.7098, 1524.0082, 1133.7241, 369.9772, 1127.3473, 1025.9170
])


class FranceDataset:
    def __init__(self, csv_path, root_dir, max_seq_len=30, mode='train'):
        """
        Args:
            csv_path (str): Path to the .csv file
            root_dir (str): Folder 2016 - 18
            max_seq_len (int): The time length of each batch.
        """

        self.df = pd.read_csv(csv_path, header=None)
        self.root_dir = root_dir
        self.mode = mode
        self.max_seq_len = max_seq_len

        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                           'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. Find file path
        relative_path = self.df.iloc[idx, 0]
        full_path = os.path.join(self.root_dir, relative_path)

        with open(full_path, 'rb') as f:



            
            sample = pickle.load(f)
        
        # -----------------------------------------------------------
        # PART A: RESIZE & STACK IMAGES
        # -----------------------------------------------------------
        processed_bands = []
        for b_name in self.band_names:
            band = torch.tensor(sample[b_name], dtype=torch.float32)

            # Upsample if needed ( up to 48x48)

            if band.shape[-1] != 48:
                # interpolate lib works with 4D - (B, C, H, W)
                band = band.unsqueeze(1)  # fake channels
                band = F.interpolate(band, size=(48, 48), mode='bilinear', align_corners=False) # False align_corners to keep the space as correct in geometry
                band = band.squeeze(1)
            
            processed_bands.append(band)

        x = torch.stack(processed_bands, dim=1) # x.shape is (T, C, H, W)

        # -----------------------------------------------------------
        # PART B: NORMALIZE
        # -----------------------------------------------------------

        mean = NORM_MEAN.view(1, 13, 1, 1)
        std = NORM_STD.view(1, 13, 1, 1)
        x = (x - mean) / std


        # -----------------------------------------------------------
        # PART C: PROCESS LABELS (Remap + Resize)
        # -----------------------------------------------------------
        label_raw = sample['labels']
        label_remapped = np.full_like(label_raw, 20) # create the same tensor fill with 20

        for k, v in REMAP_DICT.items():
            label_remapped[label_raw == k] = v
        
        y = torch.from_numpy(label_remapped).long() # create tensor wrapper for numpy arr

        if y.shape[-1] != 48:
            y = y.unsqueeze(0).unsqueeze(0).float()
            y = F.interpolate(y, size=(48, 48), mode='nearest')
            y = y.squeeze().long() # remove all dim with size 1
        

        # -----------------------------------------------------------
        # PART D: TIME PADDING (Cut or Pad to max_seq_len)
        # -----------------------------------------------------------
        # x shape is (T, 13, 48, 48)
        current_len = x.shape[0]
        target_len = self.max_seq_len

        # Get dates 
        dates = torch.tensor(sample['doy'], dtype=torch.long)
        
        if current_len >= target_len:
            # too long case
            x = x[:target_len]
            dates = dates[:target_len]
        else:
            # too short case
            pad_amount = target_len - current_len
            new_x = torch.zeros(target_len, 13, 48, 48)
            new_x[:current_len] = x
            x = new_x

            # pad dates
            new_dates = torch.zeros(target_len, dtype=torch.long)
            new_dates[:current_len] = dates
            dates = new_dates

        return x, y, dates