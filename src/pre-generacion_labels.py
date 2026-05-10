import torch
import open3d as o3d
import numpy as np
import os
import time
from tqdm import tqdm 

from TFG_functions import project_to_range_view_torch, create_target_from_labels
from Dataset_Raw import LidarDatasetRaw
dataset_folder  = os.path.join(os.getcwd(), "../CosasTFG/Datasets")
pcd_folder = os.path.join(dataset_folder, "LCAS_20160523_1200_1218_pcd")
labels_folder = os.path.join(dataset_folder, "LCAS_20160523_1200_1218_labels")
CACHE_FOLDER = os.path.join(dataset_folder, "PROCESSED_CACHE_PT")

# Constantes
H, W = 16, 1024          
FOV_UP = 15.0            
FOV_DOWN = -15.0         
MAX_RANGE = 100.0        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_and_cache():
    """Ejecuta el pipeline híbrido y guarda los tensores procesados."""
    
    if os.path.exists(CACHE_FOLDER):
        print(f"ATENCIÓN: Usando carpeta de caché: {CACHE_FOLDER}")
        print("Asegúrate de haber borrado los archivos antiguos si cambiaste la configuración.")
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    print(f"Dispositivo de procesamiento: {DEVICE}")
    print(f"Configuración: H={H}, FOV=[{FOV_UP}, {FOV_DOWN}]")
    
    dataset = LidarDatasetRaw(pcd_folder, labels_folder)
    start_time = time.time()
    
    for idx in tqdm(range(len(dataset)), desc="Pre-procesando y Caché"):
        
        # --- FASE 1: Obtener datos crudos ---
        points_raw, label_path = dataset[idx] 
        
        pcd_name = dataset.pcd_files[idx].replace(".pcd", "")
        cache_path = os.path.join(CACHE_FOLDER, f"{pcd_name}.pt")
        
        if os.path.exists(cache_path):
            continue

        # --- FASE 2: Proyección en GPU ---
        points_gpu = points_raw.to(DEVICE)
        
        # PASAMOS EXPLÍCITAMENTE LOS PARÁMETROS AQUÍ
        range_image_gpu, points_in_img_gpu, (u_gpu, v_gpu) = project_to_range_view_torch(
            points_gpu, 
            H=H, W=W, 
            fov_up=FOV_UP, 
            fov_down=FOV_DOWN, 
            max_range=MAX_RANGE,
            device=DEVICE
        )
        
        # --- FASE 3: Generar Targets en CPU ---
        u_cpu = u_gpu.cpu().numpy()
        v_cpu = v_gpu.cpu().numpy()
        points_in_img_cpu = points_in_img_gpu.cpu().numpy()
        
        # TAMBIÉN AQUÍ PASAMOS LOS PARÁMETROS
        target_np = create_target_from_labels(
            label_path, 
            points_in_img_cpu, 
            (u_cpu, v_cpu), 
            H=H, W=W,
            fov_up=FOV_UP, 
            fov_down=FOV_DOWN
        )
        
        target_tensor = torch.tensor(target_np, dtype=torch.float32, device=DEVICE)
        
        # --- FASE 4: Guardar en Caché ---
        torch.save({
            'input': range_image_gpu.cpu(),   
            'target': target_tensor.cpu()     
        }, cache_path)
        
    duration = time.time() - start_time
    print(f"\n✅ Pre-procesamiento completado en {duration:.2f} segundos.")
    print(f"Archivos guardados en: {CACHE_FOLDER}")

if __name__ == '__main__':
    preprocess_and_cache()