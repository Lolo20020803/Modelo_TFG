import csv
import os
from datetime import datetime
import gc
import torch
import matplotlib.pyplot as plt
import statistics
import numpy as np

def guardar_resultados_entrenamiento(execution_time, training_loss, batch_size, epochs, learning_rate, 
                                     model_config="", ruta_csv="resultados_entrenamiento.csv"):
    """
    Guarda en un CSV todas las pérdidas por época y la media final.
    Crea el archivo si no existe.
    """
    try:
        fieldnames = [
            "Fecha", "ID_Exp", "Tiempo(s)", "Final_Loss", "Min_Loss", "Mean_Loss",
            "Batch_Size", "Epochs", "LR", "Config_Modelo", "Historial_Loss"
        ]
        crear_cabecera = not os.path.exists(ruta_csv)
        loss_final = training_loss[-1] if training_loss else 0
        loss_min = min(training_loss) if training_loss else 0
        loss_mean = statistics.mean(training_loss) if training_loss else 0

        fecha_obj = datetime.now()
        fecha_str = fecha_obj.strftime("%Y-%m-%d %H:%M:%S")
        id_exp = fecha_obj.strftime("%d%H%M") 

        datos = {
            "Fecha": fecha_str,
            "ID_Exp": id_exp,
            "Tiempo(s)": round(execution_time, 2),
            "Final_Loss": f"{loss_final:.6f}",  
            "Min_Loss": f"{loss_min:.6f}",      
            "Mean_Loss": f"{loss_mean:.6f}",
            "Batch_Size": batch_size,
            "Epochs": epochs,
            "LR": learning_rate,
            "Config_Modelo": model_config,      
            "Historial_Loss": str(training_loss) 
        }


        with open(ruta_csv, mode="a", newline="", encoding='utf-8') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=fieldnames, delimiter=';')
            if crear_cabecera:
                escritor.writeheader()
            escritor.writerow(datos)
            
        print(f"Resultados guardados correctamente en '{ruta_csv}'")

    except Exception as e:
        print(f"Error al guardar resultados en CSV: {e}")


def limpiar_memoria_gpu():
    for var in ['model', 'optimizer', 'input_tensor', 'target_tensor', 'output', 'loss']:
        if var in globals():
            del globals()[var]
    
    gc.collect()
    
    torch.cuda.empty_cache()
    print("VRAM liberada.")

def visualizar_batch(CACHE_FOLDER):
    files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith(".pt")]
    if not files:
        print("No hay archivos en caché.")
        return

    # Cargar un archivo al azar que sepamos que tiene gente (o iterar hasta encontrar uno)
    found = False
    for f in files:
        data = torch.load(os.path.join(CACHE_FOLDER, f))
        target = data['target']
        
        # Verificar si hay algún objeto en el target
        if target[0, :, :].max() > 0.5:
            print(f"Visualizando: {f} (Contiene objetos)")
            input_tensor = data['input']
            found = True
            break
    
    if not found:
        print("Generación de etiquetas fallanda.")
        return

    # Input: (5, H, W). El canal 0 es Rango (profundidad)
    # Target: (9, H, W). El canal 0 es Heatmap de confianza
    
    range_img = input_tensor[0, :, :].numpy()
    confidence_mask = target[0, :, :].numpy()

    plt.figure(figsize=(15, 6))
    
    # 1. Imagen de Rango
    plt.subplot(2, 1, 1)
    plt.title("Input: Range Image (Depth)")
    plt.imshow(range_img, cmap='jet')
    plt.axis('off')

    # 2. Target (Ground Truth)
    plt.subplot(2, 1, 2)
    plt.title("Target: Confidence Mask")
    plt.imshow(confidence_mask, cmap='gray')
    
    # Superponer para ver alineación
    plt.imshow(range_img, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
def decode_boxes(pred_tensor, input_tensor, threshold=0.5, apply_sigmoid=True):
    """
    Decodifica los tensores de salida del modelo a cajas 3D.
    INCLUYE: Filtro para evitar cajas con dimensiones 0 (ruido del Ground Truth).
    """
    # 1. Obtener mapa de probabilidad
    if apply_sigmoid:
        prob_map = torch.sigmoid(pred_tensor[0])
    else:
        prob_map = pred_tensor[0] 

    # 2. Filtrar por umbral de confianza
    mask = prob_map > threshold
    if not mask.any(): return []
    
    v_indices, u_indices = torch.where(mask)
    boxes = []
    MAX_RANGE = 80.0  # El rango máximo que definimos para tu LiDAR
    
    # Pasamos a CPU una sola vez para iterar rápido si estamos en GPU
    if v_indices.device.type == 'cuda':
        v_indices = v_indices.cpu()
        u_indices = u_indices.cpu()
        pred_tensor = pred_tensor.cpu()
        prob_map = prob_map.cpu()
        input_tensor = input_tensor.cpu()

    for v, u in zip(v_indices, u_indices):
        # A) Recuperar dimensiones (Canales 4, 5, 6 del output)
        w, l, h = pred_tensor[4:7, v, u].tolist()

        # --- FILTRO CRÍTICO QUE AÑADIMOS ---
        # Si la caja es diminuta, es un error del padding del Ground Truth
        if w < 0.1 or l < 0.1: 
            continue 
        # -----------------------------------

        # B) Recuperar coordenadas absolutas del pixel (del input proyectado)
        px = input_tensor[1, v, u].item() * MAX_RANGE
        py = input_tensor[2, v, u].item() * MAX_RANGE
        pz = input_tensor[3, v, u].item() * MAX_RANGE
        
        # C) Recuperar offsets/deltas (Canales 1, 2, 3 del output)
        dx, dy, dz = pred_tensor[1:4, v, u].tolist()
        
        # D) Recuperar ángulo (Canales 7, 8 -> sin, cos)
        sin_t, cos_t = pred_tensor[7:9, v, u].tolist()
        angle = np.arctan2(sin_t, cos_t)
        
        score = prob_map[v, u].item()
        
        boxes.append({
            'center': [px + dx, py + dy, pz + dz], # Centro final = Pixel + Offset
            'dims': [l, w, h], 
            'angle': angle,
            'score': score
        })
        
    return boxes