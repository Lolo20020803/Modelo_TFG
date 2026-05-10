import torch
from torch.utils.data import Dataset
import os
import numpy as np
import open3d as o3d
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.TFG_functions import project_to_range_view_torch

class LidarInferenceDataset(Dataset):
    def __init__(self, pcd_folder, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_range=100.0):
        """
        Dataset que carga PCDs y devuelve directamente la Proyección Range View.
        """
        self.pcd_folder = pcd_folder
        self.pcd_files = sorted([f for f in os.listdir(pcd_folder) if f.endswith(".pcd")])
        
        # Guardamos los parámetros de proyección
        self.H = H
        self.W = W
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.max_range = max_range

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, idx):
        # 1. Cargar archivo PCD
        pcd_path = os.path.join(self.pcd_folder, self.pcd_files[idx])
        pcd = o3d.io.read_point_cloud(pcd_path)
        points_np = np.asarray(pcd.points)

        # 2. Convertir a Tensor
        points_tensor = torch.tensor(points_np, dtype=torch.float32)

        # 3. Proyectar usando TU función
        # La función devuelve: range_image, points_tensor, (u, v)
        range_image, _, _ = project_to_range_view_torch(
            points_tensor, 
            H=self.H, 
            W=self.W, 
            fov_up=self.fov_up, 
            fov_down=self.fov_down, 
            max_range=self.max_range,
            device='cpu' 
        )

        # range_image tiene shape (5, H, W) lista para el modelo
        return range_image, self.pcd_files[idx]
    
class LidarDatasetRaw(Dataset):
    def __init__(self, pcd_folder, label_folder):
        self.pcd_folder = pcd_folder
        self.label_folder = label_folder
        self.pcd_files = sorted([f for f in os.listdir(pcd_folder) if f.endswith(".pcd")])

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, idx):
        # 1. Carga Puntos (CPU)
        pcd_path = os.path.join(self.pcd_folder, self.pcd_files[idx])
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points) # Array numpy (N, 3)

        # 2. Ruta de labels (la procesaremos después porque necesitamos la proyección)
        label_path = os.path.join(self.label_folder, self.pcd_files[idx].replace(".pcd", ".txt"))

        # Devolvemos tensores crudos. NO convertimos a .to('cuda') aquí para evitar errores con multiproceso
        return torch.tensor(points, dtype=torch.float32), label_path
def collate_fn_raw(batch):
    """
    Permite agrupar datos de tamaño variable (nubes de puntos) en un batch.
    """
    points_list = []
    labels_paths = []
    
    for points, label_path in batch:
        points_list.append(points)
        labels_paths.append(label_path)
        
    return points_list, labels_paths