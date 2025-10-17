# Guía de Entrenamiento del Modelo EPP Detector

Esta guía completa cubre todo el proceso de entrenamiento del modelo YOLOv8 para detección de Equipos de Protección Personal (EPP) en faenas mineras chilenas.

## Tabla de Contenidos

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación de Dependencias](#instalación-de-dependencias)
3. [Adquisición y Preparación del Dataset](#adquisición-y-preparación-del-dataset)
4. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
5. [Exportación a ONNX](#exportación-a-onnx)
6. [Evaluación del Modelo](#evaluación-del-modelo)
7. [Monitoreo del Entrenamiento](#monitoreo-del-entrenamiento)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Requisitos del Sistema

### Hardware Mínimo

**Para Entrenamiento:**
- CPU: 4+ cores
- RAM: 16GB+
- GPU: NVIDIA con 8GB+ VRAM (recomendado: T4, V100, A100)
- Almacenamiento: 50GB+ libre

**Para Inferencia (API):**
- CPU: 2+ cores
- RAM: 4GB+
- GPU: Opcional (mejora velocidad)
- Almacenamiento: 10GB+

### Software

- **Sistema Operativo**: Ubuntu 20.04+, macOS 12+, Windows 10+
- **Python**: 3.10 o superior
- **CUDA**: 11.8+ (para entrenamiento con GPU)
- **cuDNN**: 8.9+ (recomendado)

### Verificación de GPU

```bash
# Verificar GPU NVIDIA
nvidia-smi

# Verificar CUDA en PyTorch
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

---

## Instalación de Dependencias

### 1. Crear Entorno Virtual

```bash
# Opción A: venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Opción B: conda
conda create -n epp-detector python=3.10
conda activate epp-detector
```

### 2. Instalar PyTorch con CUDA

```bash
# Para GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Para CPU solamente (no recomendado para entrenamiento)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Instalar Dependencias de Entrenamiento

```bash
# Instalar todas las dependencias
pip install -r requirements-training.txt

# Verificar instalación
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

---

## Adquisición y Preparación del Dataset

### Opción 1: Dataset de Roboflow (Recomendado para Inicio)

Roboflow proporciona datasets públicos de detección de EPP para construcción que pueden ser adaptados para minería.

**Datasets Recomendados:**
1. **Hard Hat Workers Detection** (~5000 imágenes)
   - URL: https://universe.roboflow.com/
   - Buscar: "hard hat detection" o "construction safety"
   - Formato: YOLOv8

2. **Construction Site Safety Dataset**
   - Incluye: cascos, chalecos, personas
   - Pre-etiquetado en formato YOLO

**Descarga desde Roboflow:**

```bash
# Opción A: Descarga directa desde web
# 1. Ir a Roboflow Universe
# 2. Seleccionar dataset
# 3. Descargar en formato YOLOv8
# 4. Extraer a data/roboflow/

# Opción B: Usar API de Roboflow (requiere cuenta gratuita)
pip install roboflow
python data/scripts/download_roboflow.py --output-dir data/roboflow
```

### Opción 2: Dataset Personalizado (Minería Chilena)

Para mejor rendimiento en faenas mineras chilenas, se recomienda crear un dataset personalizado:

**Fuentes de Imágenes:**
- Cámaras de seguridad de faenas mineras
- Inspecciones de seguridad documentadas
- Videos de inducción de seguridad (extraer frames)
- Simulacros de emergencia

**Herramientas de Anotación:**
- **LabelImg**: Simple y efectivo para YOLO
- **CVAT**: Plataforma web colaborativa
- **Roboflow**: Incluye herramientas de anotación

**Instalación de LabelImg:**

```bash
pip install labelImg
labelImg  # Iniciar interfaz gráfica
```

**Proceso de Anotación:**

1. Organizar imágenes en carpeta
2. Iniciar LabelImg y configurar formato YOLO
3. Anotar cada objeto con su clase:
   - `0`: hardhat (casco puesto - cumple)
   - `1`: head (cabeza sin casco - violación)
   - `2`: person (persona completa)
4. Guardar anotaciones (.txt por cada imagen)

**Recomendaciones de Anotación:**
- Mínimo: 1000-2000 imágenes para resultados aceptables
- Óptimo: 5000+ imágenes para producción
- Balance de clases: 40% hardhat, 30% person, 30% head
- Diversidad: múltiples ángulos, distancias, iluminación
- Contexto minero: subterránea, rajo abierto, condiciones de polvo

### Preparación del Dataset

Una vez descargado o anotado el dataset, prepararlo con el script:

```bash
# Estructura esperada del dataset fuente:
# source_dataset/
#   images/
#     image1.jpg
#     image2.jpg
#   labels/
#     image1.txt
#     image2.txt

# Preparar dataset con split 70/20/10
python scripts/prepare_dataset.py \
  --source /path/to/source_dataset \
  --output data \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1

# Estructura resultante:
# data/
#   images/
#     train/
#     val/
#     test/
#   labels/
#     train/
#     val/
#     test/
#   epp_dataset.yaml
#   train_statistics.txt
#   val_statistics.txt
#   test_statistics.txt
```

**Validar Dataset:**

```bash
# Revisar estadísticas
cat data/train_statistics.txt
cat data/val_statistics.txt

# Verificar que hay suficientes imágenes
ls data/images/train/ | wc -l  # Debería ser > 500
ls data/images/val/ | wc -l    # Debería ser > 100
```

---

## Entrenamiento del Modelo

### Configuración Básica

El archivo `configs/training_config.yaml` contiene todos los hiperparámetros. Los valores por defecto están optimizados para minería, pero puedes ajustarlos:

```yaml
# Tamaño del modelo (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model: yolov8n.pt  # Nano - más rápido, menor accuracy

# Épocas de entrenamiento
epochs: 100  # Aumentar a 150-200 si dataset es pequeño

# Batch size (ajustar según GPU)
batch: 16  # T4: 16, V100: 32, CPU: 4

# Learning rate
lr0: 0.001  # AdamW default
```

### Entrenar Modelo

**Entrenamiento Básico:**

```bash
python scripts/train_model.py \
  --config configs/training_config.yaml \
  --epochs 100 \
  --batch 16
```

**Entrenamiento con GPU Específica:**

```bash
# Usar GPU 0
python scripts/train_model.py \
  --config configs/training_config.yaml \
  --device cuda:0

# Multi-GPU (GPUs 0,1,2,3)
python scripts/train_model.py \
  --config configs/training_config.yaml \
  --device 0,1,2,3
```

**Reanudar Entrenamiento:**

```bash
# Reanudar desde último checkpoint
python scripts/train_model.py \
  --config configs/training_config.yaml \
  --resume \
  --checkpoint runs/train/epp_detector/weights/last.pt
```

**Entrenamiento en CPU (Lento):**

```bash
# Solo para testing o datasets muy pequeños
python scripts/train_model.py \
  --config configs/training_config.yaml \
  --device cpu \
  --batch 4 \
  --epochs 10
```

### Tiempo de Entrenamiento Estimado

| Hardware | Dataset Size | Epochs | Tiempo Estimado |
|----------|--------------|--------|-----------------|
| T4 GPU   | 2000 imgs    | 100    | 2-3 horas       |
| V100 GPU | 2000 imgs    | 100    | 1-1.5 horas     |
| A100 GPU | 2000 imgs    | 100    | 0.5-1 hora      |
| CPU      | 2000 imgs    | 100    | 20-30 horas     |

### Salida del Entrenamiento

El script guarda los resultados en `runs/train/epp_detector/`:

```
runs/train/epp_detector/
├── weights/
│   ├── best.pt          # Mejor modelo (mayor mAP)
│   └── last.pt          # Último checkpoint
├── results.png          # Gráficas de pérdida y métricas
├── confusion_matrix.png # Matriz de confusión
├── F1_curve.png         # Curva F1
├── P_curve.png          # Curva de precisión
├── R_curve.png          # Curva de recall
└── args.yaml            # Configuración usada
```

El mejor modelo también se copia automáticamente a:
```
models/yolov8n_epp.pt
```

---

## Exportación a ONNX

La API de epp-detector usa formato ONNX para inferencia optimizada.

### Exportar Modelo Entrenado

```bash
# Exportar a ONNX con configuración por defecto
python scripts/export_model.py \
  --weights models/yolov8n_epp.pt

# Exportar con FP16 (más rápido, ligeramente menor accuracy)
python scripts/export_model.py \
  --weights models/yolov8n_epp.pt \
  --half

# Exportar a múltiples formatos
python scripts/export_model.py \
  --weights models/yolov8n_epp.pt \
  --formats onnx torchscript

# Custom output path
python scripts/export_model.py \
  --weights runs/train/epp_detector/weights/best.pt \
  --output models/yolov8n_epp_v2.onnx
```

### Validar Exportación

El script valida automáticamente el modelo ONNX:

```
INFO - ONNX export successful: models/yolov8n_epp.onnx
INFO - Validating ONNX model: models/yolov8n_epp.onnx
INFO - ONNX model structure is valid
INFO - Input: images [1, 3, 640, 640]
INFO - Output: output0 [1, 84, 8400]
INFO - Available providers: CUDAExecutionProvider, CPUExecutionProvider
INFO - Dummy inference successful
INFO - ONNX model validation passed

Model Size Comparison:
  PyTorch (.pt):  6.23 MB
  ONNX:           6.15 MB
  Size reduction: 1.3%
```

### Configurar API para Usar Modelo

Actualizar `.env`:

```bash
MODEL_PATH=models/yolov8n_epp.onnx
MODEL_TYPE=onnx
CONFIDENCE_THRESHOLD=0.5
```

---

## Evaluación del Modelo

### Evaluación Completa

```bash
python scripts/evaluate_model.py \
  --weights models/yolov8n_epp.pt \
  --data data/epp_dataset.yaml \
  --visualize \
  --num-vis 50
```

### Métricas de Evaluación

El script genera un reporte completo:

```
==================== MODEL EVALUATION REPORT ====================

Overall Metrics:
  mAP@0.5:      0.8742
  mAP@0.5:0.95: 0.6534

Per-Class Metrics:

  hardhat:
    Precision:    0.9123
    Recall:       0.8876
    AP@0.5:       0.9012
    AP@0.5:0.95:  0.6789

  head:
    Precision:    0.8654
    Recall:       0.9234
    AP@0.5:       0.8965
    AP@0.5:0.95:  0.6543

  person:
    Precision:    0.8876
    Recall:       0.8543
    AP@0.5:       0.8249
    AP@0.5:0.95:  0.6270

Inference Speed:
  Mean:   23.45 ms
  Median: 22.89 ms
  Std:    2.34 ms
  FPS:    42.6

==================================================================
```

### Benchmark de Velocidad

```bash
# Benchmark solo velocidad (sin validación)
python scripts/evaluate_model.py \
  --weights models/yolov8n_epp.onnx \
  --benchmark-only \
  --image-dir data/images/test \
  --num-benchmark 200
```

### Criterios de Éxito

Para deployment en producción, el modelo debe cumplir:

| Métrica | Target | Crítico |
|---------|--------|---------|
| mAP@0.5 | > 0.85 | > 0.80 |
| Precision (hardhat) | > 0.90 | > 0.85 |
| Recall (head) | > 0.85 | > 0.80 |
| Velocidad (T4) | < 50ms | < 100ms |
| FPS (T4) | > 20 | > 10 |

**Justificación:**
- **Precision alta en hardhat**: Minimizar falsos positivos (trabajador marcado como sin casco cuando sí lo tiene)
- **Recall alto en head**: Crítico detectar todas las violaciones (trabajador sin casco debe ser detectado)
- **Velocidad**: Real-time capability para monitoreo continuo

---

## Monitoreo del Entrenamiento

### TensorBoard

YOLO genera automáticamente logs de TensorBoard:

```bash
# Iniciar TensorBoard
tensorboard --logdir runs/train

# Abrir en navegador
# http://localhost:6006
```

**Métricas a Monitorear:**

1. **Pérdidas (Losses)**:
   - `train/box_loss`: Pérdida de regresión de bounding boxes
   - `train/cls_loss`: Pérdida de clasificación
   - `val/box_loss`: Validación de boxes
   - `val/cls_loss`: Validación de clasificación

2. **Métricas (Metrics)**:
   - `metrics/precision`: Precisión en validación
   - `metrics/recall`: Recall en validación
   - `metrics/mAP50`: mAP @ IoU=0.5
   - `metrics/mAP50-95`: mAP promedio IoU 0.5-0.95

### Gráficas Generadas

YOLO genera automáticamente:

- `results.png`: Pérdidas y métricas por época
- `confusion_matrix.png`: Matriz de confusión normalizada
- `F1_curve.png`: Curva F1 vs confidence threshold
- `P_curve.png`: Curva de precisión
- `R_curve.png`: Curva de recall
- `PR_curve.png`: Curva precisión-recall

### Early Stopping

El entrenamiento se detiene automáticamente si no hay mejora:

```yaml
# En training_config.yaml
patience: 20  # Detener si no mejora en 20 épocas
```

---

## Troubleshooting

### Problema: CUDA Out of Memory

**Síntomas:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
```

**Soluciones:**

1. Reducir batch size:
```bash
python scripts/train_model.py --config configs/training_config.yaml --batch 8
```

2. Reducir tamaño de imagen:
```yaml
# En training_config.yaml
imgsz: 416  # En lugar de 640
```

3. Usar modelo más pequeño:
```yaml
model: yolov8n.pt  # Nano en lugar de small/medium
```

4. Liberar memoria GPU:
```bash
nvidia-smi
# Matar procesos que usen GPU
kill -9 <PID>
```

### Problema: Training Muy Lento

**Síntomas:**
- < 1 imagen/segundo
- GPU utilization < 50%

**Soluciones:**

1. Verificar que GPU está siendo usada:
```bash
nvidia-smi
# Debe mostrar proceso Python usando GPU
```

2. Aumentar workers:
```yaml
workers: 8  # En training_config.yaml
```

3. Usar AMP (Automatic Mixed Precision):
```yaml
amp: true  # Ya activado por defecto
```

4. Verificar CUDA/cuDNN:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problema: Overfitting

**Síntomas:**
- train_loss << val_loss
- mAP en train > mAP en val por gran margen

**Soluciones:**

1. Aumentar data augmentation:
```yaml
mosaic: 1.0
mixup: 0.15
hsv_v: 0.6
```

2. Agregar más datos de entrenamiento

3. Usar early stopping:
```yaml
patience: 15
```

4. Reducir complejidad del modelo (usar yolov8n en lugar de yolov8s)

### Problema: Underfitting

**Síntomas:**
- train_loss y val_loss altos
- mAP < 0.5

**Soluciones:**

1. Entrenar más épocas:
```bash
python scripts/train_model.py --epochs 200
```

2. Aumentar learning rate:
```yaml
lr0: 0.002
```

3. Usar modelo más grande:
```yaml
model: yolov8s.pt  # En lugar de yolov8n.pt
```

4. Verificar calidad de anotaciones

### Problema: Clase Desbalanceada

**Síntomas:**
- Una clase tiene AP mucho menor que otras
- Confusion matrix muestra bias hacia clase mayoritaria

**Soluciones:**

1. Agregar más ejemplos de clase minoritaria

2. Usar class weights:
```python
# TODO: Implementar en train_model.py
```

3. Aumentar augmentation para clase minoritaria

4. Usar técnicas de oversampling

---

## Best Practices

### 1. Desarrollo Iterativo

**Fase 1: Baseline (1-2 días)**
- Usar dataset pequeño (500 imágenes)
- Entrenar yolov8n por 50 épocas
- Validar que pipeline funciona
- Target: mAP > 0.6

**Fase 2: Mejora (1 semana)**
- Dataset completo (2000+ imágenes)
- Entrenar yolov8n por 100 épocas
- Ajustar hyperparámetros
- Target: mAP > 0.8

**Fase 3: Optimización (2 semanas)**
- Agregar datos específicos de minería chilena
- Fine-tuning en condiciones difíciles
- Probar yolov8s si necesita más accuracy
- Target: mAP > 0.85, velocidad < 50ms

### 2. Versionado de Modelos

Usar nomenclatura clara:

```
models/
├── yolov8n_epp_v1.0_roboflow.pt      # Baseline Roboflow
├── yolov8n_epp_v1.1_finetuned.pt     # Fine-tuned en minería
├── yolov8n_epp_v2.0_final.pt         # Producción
└── yolov8n_epp_v2.0_final.onnx       # Deployment
```

### 3. Documentación de Experimentos

Mantener log de experimentos:

```markdown
## Experimento 2024-01-15: Baseline Roboflow

**Dataset**: Roboflow Hard Hat Workers (4821 imágenes)
**Config**: yolov8n, 100 epochs, batch 16
**Resultados**:
- mAP@0.5: 0.84
- Precision: 0.88
- Recall: 0.82
- Velocidad: 18ms (T4 GPU)

**Observaciones**:
- Buena detección en condiciones normales
- Falla en condiciones de baja luz
- Confunde cascos amarillos con fondos brillantes

**Next Steps**:
- Agregar imágenes con baja iluminación
- Aumentar HSV augmentation
```

### 4. Testing en Condiciones Reales

Antes de deployment:

1. Probar en imágenes de faenas reales no vistas
2. Verificar en diferentes turnos (día/noche)
3. Probar en diferentes minas (subterránea/rajo abierto)
4. Validar con supervisores de seguridad

### 5. Monitoreo Post-Deployment

Implementar:
- Logging de predicciones
- Revisión manual de casos edge
- Reentrenamiento periódico con nuevos datos
- A/B testing de versiones

---

## Recursos Adicionales

### Documentación Oficial

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)

### Datasets Públicos

- [Roboflow Universe](https://universe.roboflow.com/)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [COCO Dataset](https://cocodataset.org/)

### Papers de Referencia

- YOLOv8: [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- Object Detection: [Papers With Code](https://paperswithcode.com/task/object-detection)

### Comunidad

- [Ultralytics Discord](https://discord.gg/ultralytics)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow - YOLO](https://stackoverflow.com/questions/tagged/yolo)

---

## Soporte

Para problemas específicos del proyecto epp-detector:

1. Revisar [README.md](../README.md)
2. Consultar [CLAUDE.md](../CLAUDE.md) para arquitectura
3. Abrir issue en GitHub

---

**Última Actualización**: 2024-10-16
**Versión**: 1.0
**Autor**: Bastián Berríos
