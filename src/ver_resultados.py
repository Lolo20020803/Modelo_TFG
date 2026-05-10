import os
import random
import torch
import matplotlib.pyplot as plt

from Modelo import LaserNet_LiDAR
from utils import decode_boxes

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
MODEL_PATH = "./Modelos/detector_BEST_2026-05-09_20-08-05.pth" 
VAL_FOLDER = "VALIDATION_SUBSET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.6834

# ==========================================
# 2. CARGA DEL MODELO
# ==========================================
model = LaserNet_LiDAR(lidar_in_channels=5, num_out_channels=9, deep_aggregation_num_channels=[32, 64, 128]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==========================================
# 3. BÚSQUEDA Y EXTRACCIÓN (PREDICCIÓN VS REAL)
# ==========================================
val_files = [f for f in os.listdir(VAL_FOLDER) if f.endswith(".pt")]
random.shuffle(val_files)

print("Buscando un frame con peatones reales en validación...")

sample_file, lidar_input, output, target = None, None, None, None

for file in val_files:
    sample_path = os.path.join(VAL_FOLDER, file)
    data = torch.load(sample_path, map_location=DEVICE)
    target = data['target'].unsqueeze(0) # (1, 9, H, W)
    
    if target[0, 0, :, :].max() > 0.5:
        sample_file = file
        print(f"✅ Frame encontrado: {sample_file}")
        
        lidar_input = data['input'].unsqueeze(0).to(DEVICE) 
        
        with torch.inference_mode():
            output = model(lidar_input)
        break

if sample_file is None:
    print("❌ No se encontraron peatones.")
    exit()

# Extraer Cajas Predichas
cajas_pred = decode_boxes(output[0], lidar_input[0], threshold=THRESHOLD, apply_sigmoid=True)

# Extraer Cajas Reales (Ground Truth) buscando los picos exactos (valor 1.0) de tu gaussiana
v_gt, u_gt = torch.where(target[0, 0, :, :] == 1.0)

print("\n--- COMPARATIVA DE DIMENSIONES (MÉTricas 3D) ---")
print(f"Detectados {len(cajas_pred)} peatones (Predicción) vs {len(u_gt)} peatones (Real).")

# Imprimir información de la primera caja predicha si existe
if len(cajas_pred) > 0:
    print(f"\n[PREDICCIÓN 1] -> Dims (l, w, h): {cajas_pred[0]['dims'][0]:.2f}, {cajas_pred[0]['dims'][1]:.2f}, {cajas_pred[0]['dims'][2]:.2f} | Confianza: {cajas_pred[0]['score']:.2f}")

# Imprimir información de la primera caja real si existe
if len(u_gt) > 0:
    v, u = v_gt[0].item(), u_gt[0].item()
    w, l, h = target[0, 4:7, v, u].tolist()
    print(f"[ETIQUETA REAL 1] -> Dims (l, w, h): {l:.2f}, {w:.2f}, {h:.2f} | Centro en pixel: (u={u}, v={v})")

# ==========================================
# 4. VISUALIZACIÓN 2D ESTIRADA
# ==========================================
range_img = lidar_input[0, 0, :, :].cpu().numpy() 
prob_map_pred = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
prob_map_gt = target[0, 0, :, :].cpu().numpy()

# 3 filas ahora: Predicción, Real, y la superposición
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Fila 1: Heatmap Predicho
axes[0].set_title(f"Heatmap PREDICCIÓN (Threshold: {THRESHOLD})")
im0 = axes[0].imshow(prob_map_pred, cmap='hot', aspect='auto')
fig.colorbar(im0, ax=axes[0])

# Fila 2: Heatmap Ground Truth (Etiqueta)
axes[1].set_title("Heatmap ETIQUETA REAL (Gaussiana)")
im1 = axes[1].imshow(prob_map_gt, cmap='hot', aspect='auto')
fig.colorbar(im1, ax=axes[1])

# Fila 3: Imagen LiDAR con ambas marcas
axes[2].set_title("Range Image con Comparativa de Centros")
axes[2].imshow(range_img, cmap='jet', aspect='auto')

# Dibujar marcas de la predicción (Cruces Blancas)
v_pred, u_pred = torch.where(torch.sigmoid(output[0, 0]) > THRESHOLD)
if len(u_pred) > 0:
    axes[2].scatter(u_pred.cpu(), v_pred.cpu(), c='white', marker='x', s=100, linewidths=2, label="Predicción")

# Dibujar marcas del Ground Truth (Círculos Verdes)
if len(u_gt) > 0:
    axes[2].scatter(u_gt.cpu(), v_gt.cpu(), c='lime', marker='o', s=80, facecolors='none', linewidths=2, label="Real (Etiqueta)")

axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()