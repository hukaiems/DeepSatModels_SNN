# --- 1. ADD ALL IMPORTS AT THE TOP ---
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import pickle

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
    def __init__(self, csv_file, root_dir, max_seq_len):
        self.data_paths = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = CutOrPad(max_seq_len=max_seq_len)
    
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

        return {
            'sequence': x_sequence,
            'dates': date_indices,
            'labels': label_tensor
        }

