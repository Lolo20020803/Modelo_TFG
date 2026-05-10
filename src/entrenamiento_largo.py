import shutil
import os
import torch
import torch.multiprocessing as mp
import time
import random 
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# Importaciones propias
from utils import limpiar_memoria_gpu, guardar_resultados_entrenamiento
from Dataset_Cached import LidarDatasetCache
from TFG_functions import detection_loss, evaluate_validation
from Modelo import LaserNet_LiDAR



def main():
    dataset_folder  = os.path.join(os.getcwd(),"../CosasTFG/Datasets")
    CACHE_FOLDER = os.path.join(dataset_folder, "PROCESSED_CACHE_PT")
    MODEL_FOLDER = os.path.join(os.getcwd(), "Modelos")
    VAL_COPY_FOLDER = os.path.join(os.getcwd(), "VALIDATION_SUBSET")
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # Hiperparámetros 
    BATCH_SIZE = 64
    EPOCHS = 150            
    LEARNING_RATE = 1e-4
    PATIENCE = 15          
    
    torch.set_num_threads(1)
    limpiar_memoria_gpu()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando entrenamiento largo en dispositivo: {device}")

    if torch.cuda.is_available() and mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass 
    
    # --- 1. CARGA---
    print("Escaneando directorio de caché y mezclando aleatoriamente...")
    all_files = sorted([f for f in os.listdir(CACHE_FOLDER) if f.endswith(".pt")])
    
    # Mezcla determinista para reproducibilidad
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    print(f" - Train: {len(train_files)} muestras (Con Data Augmentation)")
    print(f" - Val:   {len(val_files)} muestras (Sin Augmentation)")


    if os.path.exists(VAL_COPY_FOLDER):
        shutil.rmtree(VAL_COPY_FOLDER)
    os.makedirs(VAL_COPY_FOLDER, exist_ok=True)

    for i, file_name in enumerate(val_files):
        src_path = os.path.join(CACHE_FOLDER, file_name)
        dst_path = os.path.join(VAL_COPY_FOLDER, file_name)
        shutil.copy2(src_path, dst_path)
        
        if i % 1000 == 0 and i > 0:
            print(f"   Copiados {i}/{len(val_files)}...")
            
    print("Copia de seguridad de validacion completada.")

    # --- 2. DATASETS ---
    train_dataset = LidarDatasetCache(CACHE_FOLDER, file_list=train_files, mode='train')
    val_dataset = LidarDatasetCache(CACHE_FOLDER, file_list=val_files, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # --- MODELO, OPTIMIZADOR Y SCHEDULER ---
    channels_config = [32, 64, 128]  
    model = LaserNet_LiDAR(lidar_in_channels=5, num_out_channels=9, deep_aggregation_num_channels=channels_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(device=device) 


    # --- NOMBRES DE ARCHIVOS ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_path = os.path.join(MODEL_FOLDER, f'detector_BEST_{timestamp}.pth')
    last_model_path = os.path.join(MODEL_FOLDER, f'detector_LAST_{timestamp}.pth')

    # --- BUCLE DE ENTRENAMIENTO ---
    print(f"Comienzo del bucle de entrenamiento ({EPOCHS} epocas)...")    

    training_loss_history = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss_epoch = 0
            
            # Bucle de entrenamiento
            for i, (lidar_batch, target_batch) in enumerate(train_loader):
                lidar_batch = lidar_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device):
                    output = model(lidar_batch)
                    loss = detection_loss(output, target_batch) 
                
                if not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss_epoch += loss.item() # Variable corregida
                else:
                    print(f"Warning: NaN loss en batch {i}")

            # --- FIN DE ÉPOCA ---
            avg_train_loss = train_loss_epoch / len(train_loader)
            training_loss_history.append(avg_train_loss)

            # Validación
            avg_val_loss = evaluate_validation(model, val_loader, device)
            
            # Scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.2e}")

            # --- CHECKPOINTING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"   Nuevo record! Modelo guardado.")
            else:
                early_stopping_counter += 1
                print(f"   Sin mejora ({early_stopping_counter}/{PATIENCE})")
            
            # Guardar siempre el último
            torch.save(model.state_dict(), last_model_path)
            
            # --- EARLY STOPPING ---
            if early_stopping_counter >= PATIENCE:
                print(f"Early Stopping activado en epoca {epoch+1}.")
                break

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente. Guardando resultados parciales...")

    execution_time = time.time() - start_time

    config_str = f"Channels: {str(channels_config)} | 80/20 Split"    

    guardar_resultados_entrenamiento(
        execution_time=execution_time,
        training_loss=training_loss_history, 
        batch_size=BATCH_SIZE,
        epochs=epoch+1, 
        learning_rate=LEARNING_RATE,
        model_config=config_str,
        ruta_csv="tabla_comparativa_modelos.csv"
    )

    print("\n========================================")
    print(f"Entrenamiento finalizado.")
    print(f"Tiempo total: {execution_time/60:.2f} min")
    print(f"Mejor Loss (Val): {best_val_loss:.6f}")
    print(f"Modelo Guardado: {best_model_path}")
    print("========================================")

if __name__ == '__main__':
    main()