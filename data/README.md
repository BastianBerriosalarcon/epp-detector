# Documentación de Datasets

Información completa sobre los datasets utilizados para entrenar y evaluar el modelo de detección de cascos.

## Tabla de Contenidos

1. [Dataset Base: Roboflow](#dataset-base-roboflow)
2. [Dataset Chile (Futuro)](#dataset-chile-futuro)
3. [Clases Detectadas](#clases-detectadas)
4. [Splits de Datos](#splits-de-datos)
5. [Data Augmentation](#data-augmentation)
6. [Estadísticas](#estadísticas)
7. [Uso](#uso)

---

## Dataset Base: Roboflow

### Hard Hat Workers Dataset

**Fuente:** [Roboflow Universe - Hard Hat Workers](https://universe.roboflow.com/roboflow-universe-projects/hard-hat-workers)

**Características:**
- **Proyecto**: hard-hat-workers
- **Workspace**: roboflow-universe-projects
- **Versión**: 2
- **Formato**: YOLOv8 (formato nativo de Ultralytics)
- **Tamaño estimado**: ~5,000 imágenes
- **Licencia**: CC BY 4.0 (uso libre con atribución)

### Contenido del Dataset

**Escenarios incluidos:**
- Sitios de construcción
- Fábricas industriales
- Instalaciones de manufactura
- Ambientes exteriores e interiores
- Múltiples condiciones de iluminación

**Distribución geográfica:**
- Estados Unidos: ~60%
- Europa: ~25%
- Asia: ~10%
- Otros: ~5%

**Calidad de anotaciones:**
- Anotaciones manuales verificadas
- Bounding boxes ajustados con precisión
- Validación de calidad multi-ronda
- Consistencia entre anotadores > 95%

### Limitaciones para Contexto Chileno

Si bien es un dataset de alta calidad, presenta limitaciones para aplicación en minería chilena:

1. **Contexto:** Principalmente construcción, no minería
2. **Equipamiento:** Cascos tipo construcción (no mineros certificados)
3. **Condiciones ambientales:** No incluye desierto, alta radiación UV
4. **Uniformes:** No incluye colores corporativos de mineras chilenas
5. **Señalética:** Ausencia de señalética SERNAGEOMIN

**Estrategia:** Usar como dataset base para transfer learning, luego fine-tuning con datos locales (ver `docs/chile_context.md`).

---

## Dataset Chile (Futuro)

### Objetivo

Crear dataset complementario con imágenes específicas de la industria minera chilena para fine-tuning localizado.

### Especificaciones Target

**Tamaño:**
- **Fase 1 (MVP)**: 500-1,000 imágenes
- **Fase 2**: 2,000-5,000 imágenes
- **Fase 3**: Recolección continua en producción

**Contenido objetivo:**
- Faenas mineras reales (con permisos)
- Cascos mineros certificados Chile
- Condiciones del desierto de Atacama
- Distancias típicas de cámaras industriales (10-30m)
- Uniformes corporativos chilenos
- Señalética local

**Fuentes planificadas:**
- Colaboración con mineras (CODELCO, BHP, etc.)
- Videos públicos de seguridad (SERNAGEOMIN, ACHS)
- Crowdsourcing controlado con trabajadores

### Estado Actual

**Estado:** Pendiente

**Próximos pasos:**
1. Definir protocolo de captura de datos
2. Obtener permisos y autorizaciones
3. Desarrollar pipeline de anotación
4. Validar calidad de anotaciones
5. Integrar con pipeline de training

**Ver:** `docs/chile_context.md` para estrategia completa.

---

## Clases Detectadas

El modelo detecta **3 clases** de objetos:

### 1. `hardhat` (Casco)

**Descripción:** Casco de seguridad correctamente colocado en la cabeza de una persona.

**Características:**
- Casco visible y reconocible
- Posicionado correctamente en la cabeza
- Puede ser de cualquier color (blanco, amarillo, naranja, azul, rojo)
- Incluye cascos con o sin accesorios (visera, lámpara, orejeras)

**Importancia:** Clase objetivo principal. Detectar presencia = cumplimiento.

### 2. `head` (Cabeza sin casco)

**Descripción:** Cabeza humana visible sin casco de protección.

**Características:**
- Cabeza claramente visible
- SIN casco de seguridad
- Puede incluir otros accesorios (gorro, pelo, etc.)

**Importancia:** Indica incumplimiento de normas de seguridad.

### 3. `person` (Persona)

**Descripción:** Persona completa o parcial en la escena.

**Características:**
- Cuerpo humano visible (completo o parcial)
- Utilizado como contexto adicional
- Ayuda a distinguir cabezas de otros objetos

**Importancia:** Contexto para validación de detecciones.

### Mapeo de IDs

```yaml
# data.yaml
names:
  0: hardhat
  1: head
  2: person
```

---

## Splits de Datos

Los datos se dividen en tres conjuntos para training, validación y testing.

### Distribución Estándar

**Training Set (70-75%):**
- Usado para entrenar el modelo
- Mayor cantidad de imágenes
- Incluye data augmentation

**Validation Set (15-20%):**
- Evaluación durante training
- Early stopping basado en val loss
- NO se usa para training
- SIN data augmentation (solo resize)

**Test Set (10-15%):**
- Evaluación final del modelo
- NUNCA visto durante training
- Reporte de métricas oficiales
- SIN data augmentation

### Splits del Dataset Roboflow

Los splits exactos se definen en `data.yaml` después de descargar el dataset:

```yaml
# Ejemplo de data.yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['hardhat', 'head', 'person']
```

**Verificar splits:**
```bash
# Después de descargar
cat data/roboflow/data.yaml

# Contar imágenes por split
ls data/roboflow/train/images | wc -l
ls data/roboflow/valid/images | wc -l
ls data/roboflow/test/images | wc -l
```

---

## Data Augmentation

Durante el entrenamiento se aplican transformaciones para mejorar generalización y robustez.

### Augmentations Aplicadas

**Transformaciones geométricas:**
- **Mosaic (1.0)**: Combina 4 imágenes en una mosaico
- **MixUp (0.0)**: Mezcla dos imágenes (deshabilitado por defecto)
- **Translation (0.1)**: Desplazamiento ±10%
- **Scale (0.5)**: Escalado 0.5x - 1.5x
- **Rotation (0.0°)**: Sin rotación (vertical importante para cascos)
- **Shear (0.0°)**: Sin shearing
- **Perspective (0.0)**: Sin transformación perspectiva
- **Flip horizontal (0.5)**: 50% probabilidad de flip izquierda-derecha
- **Flip vertical (0.0)**: Sin flip vertical (arriba-abajo no realista)

**Transformaciones de color:**
- **HSV-Hue (0.015)**: Variación de tono ±1.5%
- **HSV-Saturation (0.7)**: Variación saturación ±70%
- **HSV-Value (0.4)**: Variación brillo ±40%

### Configuración en Training Script

```python
# scripts/train_gcp.py
train_config = {
    # ... otros parámetros
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
}
```

### Justificación

**Por qué no rotación:**
- Cascos tienen orientación vertical natural
- Rotar imagen crea ejemplos no realistas
- Puede confundir al modelo

**Por qué HSV fuerte:**
- Simula diferentes condiciones de luz (día, noche, sombra)
- Cascos de diferentes colores y desgaste
- Importante para generalización

**Por qué Mosaic:**
- Aumenta diversidad de contextos
- Mejora detección de objetos pequeños
- Estándar en YOLOv8

---

## Estadísticas

### Dataset Base Roboflow

**Tamaño estimado** (se actualizará después de descarga):

| Split      | Imágenes | Labels | Objetos/Imagen |
|------------|----------|--------|----------------|
| Train      | ~3,500   | ~3,500 | 2.5            |
| Validation | ~750     | ~750   | 2.5            |
| Test       | ~750     | ~750   | 2.5            |
| **Total**  | **~5,000** | **~5,000** | **2.5** |

**Distribución de clases** (estimado):

| Clase    | Count  | Porcentaje |
|----------|--------|------------|
| hardhat  | ~7,000 | 55%        |
| head     | ~3,500 | 28%        |
| person   | ~2,000 | 17%        |
| **Total**| **~12,500** | **100%** |

**Estadísticas de bounding boxes:**

| Métrica       | Valor    |
|---------------|----------|
| Ancho medio   | 12-15%   |
| Alto medio    | 15-20%   |
| Área media    | 2-3%     |
| IoU promedio  | 0.85+    |

### Actualizar Estadísticas

Después de descargar el dataset, ejecutar:

```bash
# Generar estadísticas reales
python scripts/analyze_dataset.py --dataset-path data/roboflow
```

**TODO:** Agregar script de análisis de dataset.

---

## Uso

### Descargar Dataset

```bash
# Configurar API key en .env
echo "ROBOFLOW_API_KEY=tu_key_aqui" > .env

# Descargar dataset
python data/scripts/download_roboflow.py --output-dir ./data/roboflow

# Verificar descarga
ls -lh data/roboflow/
cat data/roboflow/data.yaml
```

### Estructura de Directorios

Después de la descarga:

```
data/roboflow/
├── data.yaml              # Configuración del dataset
├── README.roboflow.txt    # Info de Roboflow
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt     # Formato YOLO
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Formato de Labels (YOLO)

Cada archivo `.txt` contiene una línea por objeto:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Valores normalizados [0, 1]:**

```txt
0 0.5 0.3 0.15 0.2    # hardhat en centro superior
1 0.7 0.6 0.12 0.18   # head sin casco a la derecha
2 0.5 0.5 0.4 0.8     # person completa
```

### Entrenar con Dataset

```bash
# Training básico
python scripts/train_gcp.py \
    --dataset-path ./data/roboflow \
    --epochs 100 \
    --batch-size 16

# Training avanzado
python scripts/train_gcp.py \
    --dataset-path ./data/roboflow \
    --epochs 100 \
    --batch-size 16 \
    --model yolov8n.pt \
    --img-size 640 \
    --export-onnx
```

---

## Referencias

- **Roboflow Dataset**: https://universe.roboflow.com/roboflow-universe-projects/hard-hat-workers
- **YOLOv8 Data Format**: https://docs.ultralytics.com/datasets/detect/
- **Data Augmentation**: https://docs.ultralytics.com/modes/train/#augmentation
- **Transfer Learning Strategy**: Ver `docs/chile_context.md`

---

## Contacto

Para preguntas sobre los datasets:
- Abrir issue en GitHub
- Contactar al mantenedor del proyecto

**Última actualización:** 2025-01-15
