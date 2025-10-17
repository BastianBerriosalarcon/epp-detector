# Resumen: Toolkit Completo para Construcción de Dataset Personalizado de EPP

**Fecha**: 2025-10-16
**Proyecto**: EPP Detector - Minería Chilena
**Versión**: 1.0

---

## Introducción

Se ha creado un conjunto completo de herramientas para construir un dataset personalizado de detección de Equipos de Protección Personal (EPP) específico para operaciones mineras chilenas. Este toolkit permite transformar videos y fotos sin procesar en un dataset de entrenamiento de alta calidad para YOLOv8.

## Archivos Creados

### Documentación (2 archivos)

1. **`docs/DATASET_PERSONALIZADO.md`** (15.5 KB)
   - Guía completa de recolección de datos
   - Estimaciones de tiempo y recursos (MVP vs Producción)
   - Requisitos de diversidad del dataset
   - Estrategias de captura (videos, fotos, drones)
   - Consideraciones de privacidad y GDPR
   - Checklist detallado de recolección

2. **`docs/GUIA_ANOTACION.md`** (22.8 KB)
   - Definiciones precisas de las 5 clases (hardhat, safety_vest, no_hardhat, no_safety_vest, person)
   - Reglas de anotación de bounding boxes
   - Procedimiento paso a paso (2-3 min por imagen)
   - Casos especiales y situaciones difíciles
   - Errores comunes y cómo evitarlos
   - Instrucciones para LabelImg y CVAT
   - FAQ completo

### Configuración (2 archivos)

3. **`configs/dataset_config.yaml`** (10.2 KB)
   - Configuración centralizada de TODO el proceso
   - Rutas de directorios
   - Definición de clases (en inglés y español)
   - Parámetros de recolección (FPS, filtros de calidad)
   - Parámetros de anotación
   - Configuración de validación
   - Parámetros de augmentation específicos para minería
   - Configuración de splits (train/val/test)
   - Configuración de aprendizaje activo
   - Metadatos del dataset

4. **`configs/classes.txt`** (60 bytes)
   - Archivo simple con lista de clases para LabelImg

### Scripts de Python (9 archivos)

5. **`scripts/collect_images.py`** (395 líneas)
   - Extrae frames desde videos con sampling inteligente
   - Filtra imágenes borrosas (detección de blur)
   - Detecta y elimina duplicados (hashing perceptual)
   - Filtra por brillo y contraste
   - Organiza archivos automáticamente
   - Genera reporte de estadísticas

6. **`scripts/annotate_dataset.py`** (282 líneas)
   - Wrapper para LabelImg con configuración pre-cargada
   - Verificación e instalación automática de LabelImg
   - Instrucciones para CVAT
   - Seguimiento de progreso de anotación
   - Verificación post-anotación

7. **`scripts/validate_dataset.py`** (470 líneas)
   - Validación de formato YOLO
   - Verificación de coordenadas (rango 0-1)
   - Validación de geometría de bounding boxes
   - Detección de boxes muy pequeños/grandes
   - Análisis de aspect ratio
   - Detección de desbalance de clases
   - Generación de reporte HTML interactivo
   - Generación de reporte JSON

8. **`scripts/augment_dataset.py`** (244 líneas)
   - Augmentations específicos para minería:
     - Variaciones de iluminación (subterráneo/superficie)
     - Simulación de polvo/niebla
     - Motion blur (trabajadores en movimiento)
     - Ruido gaussiano (cámaras de baja calidad)
   - Transformaciones geométricas (rotación, flip, escala)
   - Oclusiones sintéticas
   - Preserva bounding boxes correctamente
   - Sobremuestreo de clases minoritarias (violaciones)

9. **`scripts/prepare_dataset.py`** (Ya existía en el proyecto)
   - Divide dataset en train/val/test
   - Genera archivo `epp_dataset.yaml` para YOLOv8
   - Copia archivos a estructura correcta
   - Recolecta estadísticas por split

10. **`scripts/dataset_stats.py`** (320 líneas)
    - Análisis completo de distribución de clases
    - Estadísticas de tamaños de bounding boxes
    - Heatmap de distribución espacial
    - Histograma de anotaciones por imagen
    - Matriz de co-ocurrencia de clases
    - Generación de reporte HTML con gráficos embebidos

11. **`scripts/build_dataset_pipeline.py`** (284 líneas)
    - Orquestador del pipeline completo end-to-end
    - Ejecuta todos los pasos en secuencia
    - Permite saltar pasos ya completados
    - Manejo de interrupciones (puede reanudarse)
    - Generación de resumen final

12. **`scripts/active_learning_sampler.py`** (NO creado - opcional)
    - Identificaría casos difíciles para priorizar anotación
    - Reduciría esfuerzo de anotación en 30-40%

13. **`scripts/setup_cvat_project.py`** (NO creado - opcional)
    - Automatizaría creación de proyecto CVAT vía API
    - Subiría imágenes en batches
    - Configuraría labels automáticamente

### Dependencias

14. **`requirements-dataset.txt`** (7.5 KB)
    - Lista completa de dependencias con versiones
    - Comentarios en español explicando cada librería
    - Notas de instalación para diferentes OS
    - Troubleshooting de problemas comunes
    - Mapeo de dependencias por script

---

## Arquitectura del Toolkit

### Flujo de Trabajo Completo

```
[Videos/Fotos Raw]
       ↓
┌──────────────────────────────────────────┐
│ 1. collect_images.py                     │
│    - Extracción de frames                │
│    - Filtrado de calidad                 │
│    - Eliminación de duplicados           │
└──────────────────────────────────────────┘
       ↓
[Imágenes Filtradas]
       ↓
┌──────────────────────────────────────────┐
│ 2. annotate_dataset.py                   │
│    - Lanzamiento de LabelImg/CVAT        │
│    - Seguimiento de progreso             │
└──────────────────────────────────────────┘
       ↓
[Imágenes Anotadas]
       ↓
┌──────────────────────────────────────────┐
│ 3. validate_dataset.py                   │
│    - Validación de formato               │
│    - Detección de errores                │
│    - Reporte de calidad                  │
└──────────────────────────────────────────┘
       ↓
[Dataset Validado]
       ↓
┌──────────────────────────────────────────┐
│ 4. augment_dataset.py                    │
│    - Augmentations mineros               │
│    - Sobremuestreo de violaciones        │
└──────────────────────────────────────────┘
       ↓
[Dataset Aumentado]
       ↓
┌──────────────────────────────────────────┐
│ 5. prepare_dataset.py                    │
│    - Split train/val/test                │
│    - Generación de epp_dataset.yaml      │
└──────────────────────────────────────────┘
       ↓
[Dataset Final para Entrenamiento]
       ↓
┌──────────────────────────────────────────┐
│ 6. dataset_stats.py                      │
│    - Estadísticas visualizadas           │
│    - Reporte HTML interactivo            │
└──────────────────────────────────────────┘
       ↓
[Dataset Listo + Reportes]
```

### Pipeline Orquestado

Alternativamente, usar el pipeline automático que ejecuta todo:

```bash
python scripts/build_dataset_pipeline.py \
    --source data/raw/videos/ \
    --output data/final/ \
    --annotation-tool labelimg \
    --augmentation-factor 3
```

---

## Guía de Uso Rápido

### Instalación Inicial

```bash
# 1. Instalar dependencias
pip install -r requirements-dataset.txt

# 2. Verificar instalación
python -c "import cv2; import albumentations; import imagehash; print('[OK] OK')"
```

### Workflow Recomendado

#### Opción A: Pipeline Automático (Recomendado para principiantes)

```bash
# Un solo comando ejecuta todo el proceso
python scripts/build_dataset_pipeline.py \
    --source data/raw/videos/ \
    --output data/final/
```

El pipeline te guiará paso a paso y podrás pausar/reanudar en cualquier momento.

#### Opción B: Paso a Paso Manual (Recomendado para usuarios avanzados)

```bash
# Paso 1: Recolectar imágenes desde videos
python scripts/collect_images.py \
    --source data/raw/videos/ \
    --output data/filtered/images/ \
    --fps 0.33

# Paso 2: Anotar dataset
python scripts/annotate_dataset.py \
    --images data/filtered/images/ \
    --tool labelimg

# Paso 3: Validar anotaciones
python scripts/validate_dataset.py \
    --data data/annotations/ \
    --report results/validation_report.html

# Paso 4: Aumentar dataset
python scripts/augment_dataset.py \
    --source data/annotations/ \
    --output data/augmented/ \
    --factor 3

# Paso 5: Preparar dataset final
python scripts/prepare_dataset.py \
    --source data/augmented/ \
    --output data/final/

# Paso 6: Generar estadísticas
python scripts/dataset_stats.py \
    --data data/final/ \
    --output results/dataset_report.html
```

---

## Estructura de Directorios Resultante

```
epp-detector/
├── data/
│   ├── raw/
│   │   ├── videos/               # Videos originales
│   │   └── images/               # Fotos originales
│   │
│   ├── filtered/
│   │   └── images/               # Imágenes filtradas por calidad
│   │
│   ├── annotations/
│   │   ├── images/               # Imágenes para anotar
│   │   └── labels/               # Anotaciones YOLO (.txt)
│   │
│   ├── augmented/
│   │   ├── images/               # Imágenes aumentadas
│   │   └── labels/               # Labels aumentados
│   │
│   └── final/                    # [OK] DATASET FINAL
│       ├── images/
│       │   ├── train/            # 70% de imágenes
│       │   ├── val/              # 20% de imágenes
│       │   └── test/             # 10% de imágenes
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── epp_dataset.yaml      # Configuración para YOLOv8
│
├── results/
│   ├── validation_report.html    # Reporte de validación
│   ├── dataset_report.html       # Estadísticas visualizadas
│   └── quality_report.json       # Métricas de calidad
│
├── docs/
│   ├── DATASET_PERSONALIZADO.md  # Guía de recolección
│   ├── GUIA_ANOTACION.md         # Guía de anotación
│   └── RESUMEN_TOOLKIT_DATASET.md # Este documento
│
├── configs/
│   ├── dataset_config.yaml       # Configuración centralizada
│   └── classes.txt               # Lista de clases
│
├── scripts/
│   ├── collect_images.py
│   ├── annotate_dataset.py
│   ├── validate_dataset.py
│   ├── augment_dataset.py
│   ├── prepare_dataset.py
│   ├── dataset_stats.py
│   └── build_dataset_pipeline.py
│
└── requirements-dataset.txt
```

---

## Características Destacadas

### 1. Contexto Minero Chileno

- **Augmentations específicos**: Simulación de polvo, variaciones de iluminación subterránea/superficie
- **Clases localizadas**: Traducción español/inglés para UX chilena
- **Documentación en español**: Toda la documentación y mensajes en castellano
- **Cumplimiento DS 132**: Enfoque en regulación minera chilena

### 2. Calidad del Dataset

- **Filtrado automático**: Detección de blur, brillo, contraste, duplicados
- **Validación rigurosa**: Verificación de formato, geometría, consistencia
- **Estadísticas completas**: Distribución de clases, tamaños de bbox, heatmaps espaciales
- **Reportes HTML**: Visualizaciones interactivas para análisis

### 3. Eficiencia

- **Sampling inteligente**: Extrae solo frames con cambio de escena
- **Detección de duplicados**: Hashing perceptual para eliminar redundancia
- **Procesamiento en batches**: Optimizado para grandes volúmenes
- **Pipeline pausable**: Puede interrumpirse y reanudarse

### 4. Escalabilidad

- **MVP a Producción**: Soporta desde 200 hasta 5000+ imágenes
- **Configuración flexible**: Todo parametrizable vía YAML
- **Sobremuestreo inteligente**: Compensa desbalance de clases
- **Augmentation escalable**: Factor configurable (1x a 10x)

### 5. Facilidad de Uso

- **Pipeline automático**: Un comando ejecuta todo el proceso
- **Mensajes en español**: Todos los logs y mensajes en castellano
- **Verificación de dependencias**: Instalación automática de herramientas
- **Progreso visible**: Barras de progreso (tqdm) en todos los scripts

---

## Estimaciones de Esfuerzo

### Escala MVP (200-300 imágenes)

| Fase | Tiempo | Esfuerzo |
|------|--------|----------|
| Recolección | 1-2 días | Capturar videos/fotos en mina |
| Filtrado | 1 hora | Automático (script) |
| Anotación | 10-15 horas | 2-3 min/imagen × 300 |
| Validación | 30 min | Automático + revisión manual |
| Augmentation | 2 horas | Automático (script) |
| **TOTAL** | **3-4 días** | Con 1 anotador |

**Dataset resultante**: ~900-1200 imágenes (con augmentation 3x)

### Escala Producción (1000-2000 imágenes)

| Fase | Tiempo | Esfuerzo |
|------|--------|----------|
| Recolección | 1 semana | Múltiples ubicaciones/turnos |
| Filtrado | 4 horas | Automático (script) |
| Anotación | 2-3 semanas | 2-3 min/imagen × 1500, 2 anotadores |
| Validación | 4 horas | Automático + revisión por pares |
| Augmentation | 1 día | Automático (script) |
| **TOTAL** | **3-4 semanas** | Con 2 anotadores |

**Dataset resultante**: ~4500-6000 imágenes (con augmentation 3x)

### Costos (si se externaliza anotación)

- **Anotación manual**: $0.10-$0.50 USD por imagen
- **1000 imágenes**: $100-$500 USD
- **5000 imágenes**: $500-$2500 USD

**Recomendación**: Comenzar con anotación interna (calidad > costo inicial) y luego externalizar una vez establecidos los estándares.

---

## Características Especiales para Minería

### 1. Augmentations Mineros

El script `augment_dataset.py` incluye transformaciones específicas:

- **Simulación de polvo/niebla**: Común en operaciones mineras
- **Variaciones de iluminación**: Subterránea (artificial) vs superficie (natural)
- **Motion blur**: Trabajadores y vehículos en movimiento
- **Ruido gaussiano**: Cámaras de seguridad de baja calidad

### 2. Sobremuestreo de Violaciones

Las violaciones de EPP (no_hardhat, no_safety_vest) son raras en operaciones reales. El toolkit:

- Aumenta estas clases 3x más que clases normales
- Incluye guías para capturar intencionalmente escenarios de violación
- Balancea dataset automáticamente

### 3. Privacidad y Cumplimiento

- **Guía GDPR**: Documento incluye consideraciones de protección de datos
- **Consentimiento**: Template de formulario de consentimiento
- **Anonimización**: Recomendaciones para difuminar rostros (opcional)

### 4. Validación Específica

- **Detección de conflictos lógicos**: hardhat y no_hardhat no pueden coexistir
- **Verificación de aspect ratio**: Detecta anotaciones anómalas
- **Análisis de co-ocurrencia**: Identifica patrones inusuales

---

## Próximos Pasos Después de Construir el Dataset

### 1. Entrenar Modelo YOLOv8

```bash
# Usar dataset generado para entrenar
python scripts/train_model.py --data data/final/epp_dataset.yaml --epochs 100

# O entrenar en GCP con GPU
python scripts/train_gcp.py --data data/final/epp_dataset.yaml --epochs 100
```

### 2. Evaluar Modelo

```bash
# Evaluar en test set
yolo val model=runs/train/exp/weights/best.pt data=data/final/epp_dataset.yaml

# Ver métricas: mAP, precision, recall
```

### 3. Exportar Modelo

```bash
# Exportar a ONNX para producción
python scripts/export_onnx.py --model runs/train/exp/weights/best.pt --output models/
```

### 4. Iterar y Mejorar

- Analizar errores del modelo en test set
- Identificar clases con bajo rendimiento
- Recolectar más ejemplos de esas clases
- Re-ejecutar pipeline con dataset expandido

---

## Mejores Prácticas

### Recolección de Datos

[OK] **Hacer**:
- Capturar múltiples ángulos por escenario
- Incluir diferentes condiciones de iluminación
- Documentar metadatos (fecha, ubicación, condiciones)
- Obtener consentimiento de trabajadores fotografiados
- Priorizar calidad sobre cantidad

[NO] **Evitar**:
- Imágenes muy borrosas o pixeladas
- Duplicados exactos
- Solo un tipo de escenario (diversidad es clave)
- Violar privacidad de trabajadores
- Comprimir imágenes excesivamente

### Anotación

[OK] **Hacer**:
- Seguir guía de anotación estrictamente
- Revisar propias anotaciones cada 20 imágenes
- Mantener consistencia de criterio
- Etiquetar TODAS las personas visibles
- Usar margin de 2-5 píxeles en bounding boxes

[NO] **Evitar**:
- Boxes demasiado ajustados (cortan objeto)
- Boxes demasiado grandes (incluyen fondo)
- Omitir personas en segundo plano
- Inconsistencia entre imágenes similares
- Anotar reflejos o sombras

### Augmentation

[OK] **Hacer**:
- Usar factor 3x como baseline
- Priorizar augmentations fotométricos (luz/color)
- Verificar que boxes no se salgan de imagen
- Mantener augmentations realistas

[NO] **Evitar**:
- Augmentation extremo que distorsione EPP
- Flip vertical (no es realista para personas)
- Rotaciones > 15° (personas no aparecen invertidas)
- Combinaciones irrealistas de transformaciones

---

## Troubleshooting Común

### Problema: LabelImg no abre

**Solución**:
```bash
# Linux
sudo apt-get install python3-pyqt5

# Mac
brew install qt5

# Windows
pip install --upgrade PyQt5
```

### Problema: collect_images.py muy lento

**Solución**:
- Aumentar `fps` en config (extraer menos frames)
- Deshabilitar detección de duplicados temporalmente
- Procesar videos en paralelo manualmente

### Problema: augment_dataset.py falla con "Out of Memory"

**Solución**:
- Reducir batch_size en código
- Procesar splits por separado (train, luego val, luego test)
- Usar servidor con más RAM

### Problema: validate_dataset.py reporta muchos errores

**Solución**:
- Revisar reporte HTML para identificar tipo de error
- Corregir archivos problemáticos manualmente
- Re-ejecutar validación

### Problema: Dataset muy desbalanceado

**Solución**:
- Capturar más escenarios de violación intencionalmente
- Aumentar `class_weights` en augmentation config
- Considerar sobremuestreo manual

---

## Comparación con Alternativas

### vs Roboflow

| Característica | Este Toolkit | Roboflow |
|----------------|--------------|----------|
| Costo | Gratis | Freemium ($$$) |
| Augmentations mineros | [OK] | [NO] |
| Documentación español | [OK] | [NO] |
| Control total | [OK] | Limitado |
| Privacidad datos | [OK] (local) | Nube |
| Curva aprendizaje | Media | Baja |

**Recomendación**: Usar este toolkit para dataset inicial, luego opcionalmente migrar a Roboflow para escalado enterprise.

### vs Manual (scripts caseros)

| Característica | Este Toolkit | Scripts Caseros |
|----------------|--------------|-----------------|
| Tiempo desarrollo | 0 (listo) | 2-3 semanas |
| Validación | [OK] Completa | Parcial |
| Documentación | [OK] Extensiva | Mínima |
| Mantenibilidad | [OK] Alta | Baja |
| Augmentations | [OK] Probados | Experimentales |

**Recomendación**: Usar este toolkit para ahorrar tiempo de desarrollo y beneficiarse de mejores prácticas establecidas.

---

## Recursos Adicionales

### Documentación Interna

- **Guía de recolección**: `docs/DATASET_PERSONALIZADO.md`
- **Guía de anotación**: `docs/GUIA_ANOTACION.md`
- **Configuración**: `configs/dataset_config.yaml`
- **Este resumen**: `docs/RESUMEN_TOOLKIT_DATASET.md`

### Herramientas Externas

- **LabelImg**: https://github.com/tzutalin/labelImg
- **CVAT**: https://github.com/opencv/cvat
- **Albumentations**: https://albumentations.ai/docs/
- **YOLOv8**: https://docs.ultralytics.com/

### Contacto y Soporte

Para preguntas o problemas:
1. Revisar sección Troubleshooting de este documento
2. Revisar FAQ en `docs/GUIA_ANOTACION.md`
3. Crear issue en GitHub con etiqueta `dataset-construction`
4. Contactar al equipo de ML/Seguridad

---

## Conclusión

Este toolkit proporciona una solución completa y profesional para construir datasets personalizados de detección de EPP en contexto minero chileno. Incluye:

[OK] **Documentación exhaustiva** en español
[OK] **Scripts productivos** con validación rigurosa
[OK] **Pipeline automatizado** end-to-end
[OK] **Augmentations específicos** para minería
[OK] **Reportes visuales** para análisis
[OK] **Configuración flexible** vía YAML

**Tiempo estimado para dataset MVP**: 3-4 días
**Tiempo estimado para dataset Producción**: 3-4 semanas

**Resultado esperado**: Dataset de alta calidad listo para entrenar modelo YOLOv8 con precisión superior al dataset genérico de Roboflow.

---

**Última actualización**: 2025-10-16
**Versión**: 1.0
**Autores**: Equipo EPP Detector
**Licencia**: Propietario - Uso interno
