import os
import torch
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler # Importar esto



def evaluate_validation(model, val_loader, device):
    """Calcula la loss media en el conjunto de validación sin entrenar"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lidar_batch, target_batch in val_loader:
            lidar_batch = lidar_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            
            with torch.amp.autocast(device):
                output = model(lidar_batch)
                loss = detection_loss(output, target_batch)
            
            if not torch.isnan(loss):
                val_loss += loss.item()
    
    return val_loss / len(val_loader)


def penalty_reduced_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Focal Loss optimizada para heatmaps continuos (tipo CenterNet).
    """
    # Aplicar sigmoide para asegurar probabilidades entre 0 y 1
    pred = torch.sigmoid(pred)
    # Clamp para evitar inestabilidad numérica en el logaritmo
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    
    # Índices exactos del centro (donde la gaussiana es 1) y del resto
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    # Aquí está la clave: reducir la penalización de los falsos positivos 
    # si caen en la falda de la gaussiana del ground truth.
    neg_weights = torch.pow(1 - target, beta)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
        
    return loss

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Dibuja una mancha gaussiana en el heatmap (in-place).
    center: (x, y) -> (u, v) en coordenadas de imagen
    radius: radio de la mancha
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0], heatmap.shape[1]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # Usamos np.maximum para preservar el pico más alto si se solapan
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        
    return heatmap

def project_to_range_view_torch(points_tensor, H=16, W=1024, fov_up=15.0, fov_down=-15.0, max_range=100.0, device='cuda'):
    """
    Versión optimizada en GPU de project_to_range_view.
    
    Args:
        points_tensor (torch.Tensor): Tensor de puntos (N, 3).
        H, W (int): Dimensiones de la imagen.
        device (str): Dispositivo ('cuda' o 'cpu').
    """
    if points_tensor.device.type != device:
        points_tensor = points_tensor.to(device)

    x, y, z = points_tensor[:, 0], points_tensor[:, 1], points_tensor[:, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2)

    # 2. Filtrado de puntos válidos
    valid_mask = (r > 0.1) & (r < max_range)
    points_tensor = points_tensor[valid_mask]
    x, y, z, r = x[valid_mask], y[valid_mask], z[valid_mask], r[valid_mask]

    # 3. Proyección Esférica (Trigonometría en GPU)
    yaw = -torch.atan2(y, x)
    # torch.clamp es el equivalente a np.clip
    pitch = torch.asin(torch.clamp(z / (r + 1e-8), -1.0, 1.0))

    fov_up_rad = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    fov_rad = fov_up_rad - fov_down_rad

    # Calcular coordenadas UV
    u = (0.5 * (1 - yaw / np.pi) * W).to(torch.int32)
    v = ((1 - (pitch - fov_down_rad) / fov_rad) * H).to(torch.int32)

    # 4. Filtrado dentro de la imagen
    in_image_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, r, points_tensor = u[in_image_mask], v[in_image_mask], r[in_image_mask], points_tensor[in_image_mask]

    # 5. ORDENAMIENTO POR PROFUNDIDAD
    order = torch.argsort(r, descending=True)
    
    u_sorted = u[order].long() # Indices necesitan ser long para indexar
    v_sorted = v[order].long()
    r_sorted = r[order]
    points_sorted = points_tensor[order]

    # Crear imagen de rango (H, W, 5) en GPU
    range_image = torch.zeros((H, W, 5), dtype=torch.float32, device=device)

    # 6. Asignación avanzada (Scatter/Index Put)
    # PyTorch permite indexación directa con tensores de índices
    range_image[v_sorted, u_sorted, 0] = r_sorted / max_range
    range_image[v_sorted, u_sorted, 1:4] = points_sorted[:, :3]
    range_image[v_sorted, u_sorted, 4] = 1.0

    # 7. Normalización final
    range_image[:, :, 0] = (range_image[:, :, 0] - 0.5) / 0.5
    range_image[:, :, 1:4] /= max_range


    range_image[v_sorted, u_sorted, 4] = 1.0 

    range_image = range_image.permute(2, 0, 1)

    return range_image, points_tensor, (u, v)


def train_model_gpu(model, dataloader, optimizer, criterion, device="cuda", num_epochs=10):
    model.to(device)
    scaler = torch.amp.GradScaler(device=device) 
    print(f"Iniciando entrenamiento en {device}...")
    
    loss_list = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (lidar_batch, target_batch) in enumerate(dataloader):
            lidar_batch = lidar_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device): # Contexto de precisión mixta
                # 1. Forward Pass (Obligatorio)
                output = model(lidar_batch)
                
                # 2. Cálculo de Loss (SOLO UNA VEZ y dentro del contexto)
                loss = criterion(output, target_batch)
            
            # 3. Backpropagation con escalado
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item() 
            else:
                print(f"Warning: NaN loss en batch {i}")

        avg_loss = epoch_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss Media: {avg_loss:.6f}")
        
    return loss_list


def focal_loss(pred, target, alpha=0.25, gamma=2.0,reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def detection_loss(prediction, target, lambda_box=1.5):
    """
    Pérdida actualizada.
    """
    # 1. Pérdida del Heatmap (Canal 0) usando la versión Penalty-Reduced
    loss_conf = penalty_reduced_focal_loss(
        prediction[:, 0:1, :, :], 
        target[:, 0:1, :, :]
    )
    
    # 2. Pérdida de Regresión 
    regression_mask = target[:, 4, :, :] > 0.0
    
    if regression_mask.any():
        pred_reg = prediction[:, 1:, :, :].permute(0, 2, 3, 1)[regression_mask]
        tgt_reg  = target[:, 1:, :, :].permute(0, 2, 3, 1)[regression_mask]

        offset_loss = F.smooth_l1_loss(pred_reg[:, 0:3], tgt_reg[:, 0:3])
        dim_loss    = F.smooth_l1_loss(pred_reg[:, 3:6], tgt_reg[:, 3:6])
        orient_loss = F.smooth_l1_loss(pred_reg[:, 6:8], tgt_reg[:, 6:8])

        loss_box = offset_loss + 1.0 * dim_loss + 0.5 * orient_loss
        total_loss = loss_conf + lambda_box * loss_box
    else:
        total_loss = loss_conf

    return total_loss


def create_target_from_labels(labels_path, cloud_points_in_image, image_coords, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
   
    target_tensor = np.zeros((9, H, W), dtype=np.float32)
    p_obj_mask = np.zeros((H, W), dtype=np.float32)
    u_all, v_all = image_coords

    if not os.path.exists(labels_path):
      return target_tensor

    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] not in ["pedestrian", "group"]: continue

            centroid = np.array([list(map(float, parts[1:4]))])
            min_b = np.array(list(map(float, parts[4:7])))
            max_b = np.array(list(map(float, parts[7:10])))
            dims = max_b - min_b

            # Proyección de esquinas para saber el tamaño en 2D
            corners = np.array([[min_b[0], min_b[1], min_b[2]], [max_b[0], min_b[1], min_b[2]],
                                [min_b[0], max_b[1], min_b[2]], [max_b[0], max_b[1], min_b[2]],
                                [min_b[0], min_b[1], max_b[2]], [max_b[0], min_b[1], max_b[2]],
                                [min_b[0], max_b[1], max_b[2]], [max_b[0], max_b[1], max_b[2]]])

            r_corners = np.linalg.norm(corners, axis=1)
            yaw_corners = -np.arctan2(corners[:, 1], corners[:, 0])
            pitch_corners = np.arcsin(np.clip(corners[:, 2] / (r_corners + 1e-8), -1.0, 1.0))

            fov_up_rad, fov_down_rad = np.deg2rad(fov_up), np.deg2rad(fov_down)
            fov_rad = fov_up_rad - fov_down_rad

            u_corners = (0.5 * (1 - yaw_corners / np.pi) * W).astype(np.int32)
            v_corners = ((1 - (pitch_corners - fov_down_rad) / fov_rad) * H).astype(np.int32)

            valid_corners = (u_corners >= 0) & (u_corners < W) & (v_corners >= 0) & (v_corners < H)
            
            # --- 1. DIBUJAR GAUSSIANAS ---
            if np.sum(valid_corners) > 0:
                # Bounding box 2D aproximado
                u_corners_valid = u_corners[valid_corners]
                v_corners_valid = v_corners[valid_corners]

                u_min, u_max = np.min(u_corners_valid), np.max(u_corners_valid)
                v_min, v_max = np.min(v_corners_valid), np.max(v_corners_valid)
                
               # Si el ancho de la caja es más de la mitad de la imagen, cruzó el límite
                if (u_max - u_min) > (W / 2):
                    # Pasamos las coordenadas altas (ej. 1020) a la zona negativa (ej. -4)
                    u_shifted = np.where(u_corners_valid > W/2, u_corners_valid - W, u_corners_valid)
                    u_min_s, u_max_s = np.min(u_shifted), np.max(u_shifted)
                    
                    center_u = int((u_min_s + u_max_s) // 2)
                    center_u = center_u % W # Devolver el centro al rango [0, W-1]
                    w_radius = (u_max_s - u_min_s) / 2
                else:
                    center_u = (u_min + u_max) // 2
                    w_radius = (u_max - u_min) / 2
                
                center_v = (v_min + v_max) // 2
                h_radius = (v_max - v_min) / 2
                
                radius = int(max(2, min(h_radius, w_radius)))

                draw_gaussian(p_obj_mask, (center_u, center_v), radius)

            # --- 2. ASIGNAR REGRESIÓN (Con filtro de fantasmas) ---
            if len(cloud_points_in_image) > 0:
                distances = cdist(centroid, cloud_points_in_image)
                min_dist_val = np.min(distances)
                
                # --- FIX: FILTRO DE FANTASMAS ---
                # Si el LiDAR pasa a más de 1m del centroide real, ignoramos la caja.
                # Evita que el modelo aprenda a predecir personas en el aire vacío.
                if min_dist_val > 1.0: 
                    continue

                nearest_point_idx = np.argmin(distances)
                u_c, v_c = u_all[nearest_point_idx], v_all[nearest_point_idx]

                if 0 <= u_c < W and 0 <= v_c < H:
                    target_tensor[1:4, v_c, u_c] = centroid - cloud_points_in_image[nearest_point_idx]
                    target_tensor[4:7, v_c, u_c] = dims
                    target_tensor[7:9, v_c, u_c] = [0.0, 1.0] # (TODO: Usar yaw real si es posible)

    target_tensor[0, :, :] = p_obj_mask
    return target_tensor