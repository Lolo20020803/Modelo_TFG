from torch.utils.data import Dataset
import os
import torch
import random

class LidarDatasetCache(Dataset):
    def __init__(self, cache_folder,mode='train', file_list=None):
        self.cache_folder = cache_folder
        self.mode = mode
        if file_list is not None:
            self.cache_files = file_list
        else:
            self.cache_files = sorted([f for f in os.listdir(cache_folder) if f.endswith(".pt")])

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_folder, self.cache_files[idx])
        data = torch.load(cache_path, map_location='cpu') 
        
        lidar_input = data['input'].float()   
        target = data['target'].float()       
        if self.mode == 'train':
            #Flipeamos alguna imagenes
            if random.random() > 0.5: 
                lidar_input = torch.flip(lidar_input, dims=[2])
                target = torch.flip(target, dims=[2])
                lidar_input[2, :, :] = -lidar_input[2, :, :]
                target[2, :, :] = -target[2, :, :]
                target[7, :, :] = -target[7, :, :]

            shift = random.randint(-3, 3) 
            if shift != 0:
                # 'roll' mueve los datos circularmente
                lidar_input = torch.roll(lidar_input, shifts=shift, dims=1)
                target = torch.roll(target, shifts=shift, dims=1)
                if shift > 0: # Desplazamiento hacia abajo, limpiar la parte de arriba
                    lidar_input[:, :shift, :] = 0
                    target[:, :shift, :] = 0
                elif shift < 0: # Desplazamiento hacia arriba, limpiar la parte de abajo
                    lidar_input[:, shift:, :] = 0
                    target[:, shift:, :] = 0
        return lidar_input, target
    