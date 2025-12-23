# --- 1. ADD ALL IMPORTS AT THE TOP ---
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import pickle
import random 

PASTIS_CLASSES = [
    "0: Background", "1: Meadow", "2: Soft Winter Wheat", "3: Corn", 
    "4: Winter Barley", "5: Winter Rapeseed", "6: Spring Barley", "7: Sunflower", 
    "8: Grapevine", "9: Beet", "10: Winter Triticale", "11: Winter Durum Wheat", 
    "12: Fruits/Veg/Flowers", "13: Potatoes", "14: Leguminous Fodder", "15: Soybeans", 
    "16: Orchard", "17: Mixed Cereal", "18: Sorghum", "19: Void Label"
]

# initilize cut or pad function
class CutOrPad:
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
    
    def __call__(self, tensor):
        # calculate the dif
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        
        # too short
        if diff > 0:
            # create an a shape for the tensor base on the diff
            pad_shape = [diff] + list(tensor.shape[1:])
            # now fill zeros into the shape
            padding = torch.zeros(pad_shape, dtype=tensor.dtype)
            # now stack it on the origin tensor
            tensor = torch.cat((tensor, padding), dim=0 ) #concatenate them on time dimension. every other dim has to be the same
        elif diff < 0:
            tensor = tensor[:self.max_seq_len] # left out those redundant value.

        return tensor 


# initilize the dataset handling
class PastisDataset:
    # dunder init contains 3 things root dir + data_paths and seq len for a batch
    def __init__(self, data, root_dir, max_seq_len, mode='train'):
        if isinstance(data, str):
            self.data_paths = pd.read_csv(data, header=None)
        
        elif isinstance(data, pd.DataFrame):
            self.data_paths = data
        
        elif isinstance(data, list):
            self.data_paths = pd.DataFrame(data)

        self.root_dir = root_dir
        self.transform = CutOrPad(max_seq_len=max_seq_len)
        self.mode = mode
        # In __init__
        self.mean = torch.tensor([
            1165.9398193359375, 1375.6534423828125, 1429.2191162109375, 1764.798828125,
            2719.273193359375, 3063.61181640625, 3205.90185546875, 3319.109619140625,
            2422.904296875, 1639.370361328125
        ]).view(1, -1, 1, 1).float()

        self.std = torch.tensor([
            1942.6156005859375, 1881.9234619140625, 1959.3798828125, 1867.2239990234375,
            1754.5850830078125, 1769.4046630859375, 1784.860595703125, 1767.7100830078125,
            1458.963623046875, 1299.2833251953125
        ]).view(1, -1, 1, 1).float()

    
    # dunder return len for the DataLoader to know and run
    def __len__(self):
        return len(self.data_paths)

    # to run the pastis dataset
    def __getitem__(self, idx: int):
        # 1. it construct the path from root and data_paths
        pkl_path = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0]) #remember self because these are class vars
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f) #load all the dict into sample var

        # 2. now take those key inside the sample out
        img_tensor = torch.tensor(sample['img'], dtype=torch.float32) #syntax is torch.float32 - torch.type
        doy_tensor = torch.tensor(sample['doy'], dtype=torch.long) #just date so long is enough
        label_tensor = torch.tensor(sample['labels'], dtype=torch.long)

        # 3. Padding and cutting for the sequence, helps the time length in the batch is equal so it can be stack.
        x_sequence = self.transform(img_tensor)
        date_indices = self.transform(doy_tensor)

        # 4. Apply transformation
        x_sequence, label_tensor = self.apply_transforms(x_sequence, label_tensor)

        return {
            'sequence': x_sequence,
            'dates': date_indices,
            'labels': label_tensor
        }
    
    # transformation data
    def apply_transforms(self, x, y):
        # 1. Normalization - Formula (X - Mean) / Std so activation wont be explode like LIF.
        x = (x.float() - self.mean) / self.std

        # 2. Geometric Augmentation (Train only)
        if self.mode == 'train':

            # 50 percent chance get flipping
            # Horizontal flip
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])

            # Vertical flip
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])
            
            # 90 degree rotation
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[-2, -1])
                y = torch.rot90(y, k, dims=[-2, -1])
            
            # Noise injection (Robustness)
            if random.random() < 0.2:
                noise = torch.randn_like(x) * 0.05 # create tensor "x" shape fill with bell curver fvalues
                x += noise
            
        return x, y
            

            