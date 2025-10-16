# Model Card: Hard Hat Detection YOLOv8

## Información del Modelo

### Detalles

- **Nombre del modelo:** Hard Hat Detection YOLOv8n
- **Versión:** 1.0.0 (Baseline)
- **Fecha:** 2025-01-15
- **Arquitectura:** YOLOv8 Nano
- **Framework:** Ultralytics YOLOv8
- **Formato:** PyTorch (.pt), ONNX (.onnx)
- **Autor:** Bastian Berrios
- **Licencia:** MIT

### Descripción

Modelo de detección de objetos basado en YOLOv8n para identificar el uso correcto de cascos de seguridad en entornos industriales. Detecta tres clases: cascos (hardhat), cabezas sin casco (head), y personas (person).

**Propósito:** Sistema de monitoreo automatizado de cumplimiento de equipos de protección personal (EPP) en la industria minera chilena, alineado con DS 132 (Reglamento de Seguridad Minera).

---

## Uso Previsto

### Casos de Uso Principales

1. **Monitoreo de Seguridad en Tiempo Real**
   - Detección automática de uso de cascos en cámaras de seguridad
   - Alertas en tiempo real a supervisores
   - Dashboard de cumplimiento

2. **Auditoría y Compliance**
   - Registro histórico de cumplimiento EPP
   - Evidencia objetiva para fiscalizaciones SERNAGEOMIN
   - Reportes de cumplimiento normativo

3. **Análisis de Seguridad**
   - Identificación de zonas de riesgo
   - Patrones de incumplimiento
   - Métricas para mejora continua

### Usuarios Objetivo

- **Supervisores de seguridad:** Monitoreo en tiempo real
- **Jefes de operaciones:** Dashboard de cumplimiento
- **Equipo de compliance:** Reportes y auditorías
- **Ingenieros ML:** Fine-tuning y mejora continua

### Ambientes de Aplicación

**Primario:**
- Faenas mineras (superficie y subterránea)
- Plantas de procesamiento
- Áreas de mantenimiento

**Secundario:**
- Construcción
- Manufactura industrial
- Puertos y logística

---

## Datos de Entrenamiento

### Dataset Base

**Nombre:** Hard Hat Workers (Roboflow Universe)

**Características:**
- **Tamaño:** ~5,000 imágenes
- **Fuente:** Roboflow Universe
- **Distribución geográfica:** Internacional (US, Europa, Asia)
- **Escenarios:** Construcción, industria general
- **Anotaciones:** Manuales, verificadas

**Clases:**
1. `hardhat`: Casco de seguridad colocado correctamente
2. `head`: Cabeza visible sin casco
3. `person`: Persona completa o parcial

### Splits de Datos

| Split      | Tamaño    | Uso                           |
|------------|-----------|-------------------------------|
| Train      | ~3,500    | Entrenamiento del modelo      |
| Validation | ~750      | Early stopping, tuning        |
| Test       | ~750      | Evaluación final              |

### Preprocesamiento

- Redimensionamiento: 640x640 píxeles
- Normalización: [0, 1]
- Formato: RGB

### Augmentación

- Mosaic (1.0)
- Variación HSV (Hue: 0.015, Sat: 0.7, Val: 0.4)
- Flip horizontal (0.5)
- Translation (0.1)
- Scale (0.5)

**Justificación:** Mejorar generalización a diferentes condiciones de iluminación, colores de cascos, y ángulos de cámara.

### Limitaciones del Dataset

- No incluye condiciones específicas de minería chilena
- Ausencia de cascos mineros certificados locales
- No incluye ambientes de desierto (Atacama)
- No incluye señalética SERNAGEOMIN

**Mitigación:** Planeado fine-tuning con dataset local (ver `docs/chile_context.md`).

---

## Métricas de Desempeño

### Métricas Objetivo

**TODO:** Completar después del entrenamiento inicial.

| Métrica          | Valor Target | Valor Actual | Estado |
|------------------|--------------|--------------|--------|
| mAP50            | > 0.85       | TBD          | -      |
| mAP50-95         | > 0.65       | TBD          | -      |
| Precision        | > 0.90       | TBD          | -      |
| Recall           | > 0.85       | TBD          | -      |
| F1-Score         | > 0.87       | TBD          | -      |
| Latencia (GPU)   | < 50ms       | TBD          | -      |
| Latencia (CPU)   | < 200ms      | TBD          | -      |

### Métricas por Clase

**TODO:** Completar después del entrenamiento inicial.

| Clase    | Precision | Recall | mAP50 |
|----------|-----------|--------|-------|
| hardhat  | TBD       | TBD    | TBD   |
| head     | TBD       | TBD    | TBD   |
| person   | TBD       | TBD    | TBD   |

### Configuración de Evaluación

- **Dataset:** Test set (nunca visto en training)
- **IoU threshold:** 0.5 (mAP50), 0.5-0.95 (mAP50-95)
- **Confidence threshold:** 0.25 (default)
- **NMS threshold:** 0.45 (Non-Maximum Suppression)

---

## Consideraciones Éticas

### Privacidad

**Riesgos:**
- Captura de rostros en imágenes de cámaras de seguridad
- Identificación potencial de trabajadores

**Mitigaciones:**
- Modelo NO entrenado para reconocimiento facial
- Detección basada en cascos, no identidad personal
- Recomendación: Anonimizar rostros en deployment (blur automático)
- Cumplimiento Ley 19.628 (Protección de Datos Personales - Chile)
- Política de retención de datos clara y limitada

**Buenas prácticas:**
- Informar a trabajadores sobre monitoreo automatizado
- Acceso restringido a grabaciones
- No usar para evaluación de desempeño individual (solo compliance EPP)

### Sesgo y Equidad

**Riesgos potenciales:**
- Menor precisión en ciertos tipos de cascos (colores, marcas)
- Posible sesgo geográfico (dataset internacional)
- Variación de rendimiento según demografía no intencionada

**Mitigaciones:**
- Validación en múltiples tipos de cascos
- Fine-tuning con datos locales chilenos
- Monitoreo de métricas desagregadas por subgrupos
- Revisión periódica de falsos negativos

**Compromiso:**
- Modelo es herramienta de apoyo, NO reemplazo de supervisión humana
- Decisiones finales de seguridad son humanas
- Transparencia en limitaciones del sistema

### Impacto Social

**Positivo:**
- Reducción de accidentes laborales
- Cumplimiento normativo mejorado
- Cultura de seguridad reforzada

**Negativo (potencial):**
- Percepción de vigilancia excesiva
- Falsa sensación de seguridad (si se confía 100% en automatización)

**Mitigación:**
- Comunicación clara del propósito (seguridad, no vigilancia)
- Involucrar a trabajadores en implementación
- Enfoque en educación y prevención, no solo detección

---

## Limitaciones

### Técnicas

1. **Oclusión:** Dificultad para detectar cascos parcialmente ocultos
2. **Distancia:** Rendimiento degradado para objetos muy lejanos (>30m estimado)
3. **Condiciones extremas:** Niebla densa, lluvia intensa, polvo puede afectar detección
4. **Ángulos:** Vista cenital o muy lateral puede reducir precisión
5. **Objetos similares:** Potencial confusión con otros objetos redondeados

### De Datos

1. **Generalización:** Entrenado con datos internacionales, no específicos de Chile
2. **Diversidad limitada:** No incluye todos los modelos de cascos mineros
3. **Escenarios faltantes:** Ambientes subterráneos, tuneles, condiciones de poca luz

### De Deployment

1. **Latencia:** Requiere GPU para inferencia real-time en múltiples cámaras
2. **Conectividad:** Necesita conexión estable para sistema centralizado
3. **Hardware:** Cámaras de calidad mínima requerida (720p+, 15fps+)

### De Contexto

1. **Regulación específica:** Requiere validación legal para uso en compliance formal
2. **Falsos positivos:** Pueden generar alertas innecesarias
3. **Falsos negativos:** Crítico en seguridad - pueden pasar incumplimientos desapercibidos

**Mitigación clave:** Sistema debe ser complemento, no reemplazo de supervisión humana.

---

## Especificaciones Técnicas

### Arquitectura

**Modelo base:** YOLOv8n (Nano variant)

**Características:**
- Parámetros: ~3.2M
- Tamaño modelo: ~6 MB (.pt), ~12 MB (.onnx)
- FLOPs: ~8.7 G
- Capas: 225

**Backbone:** CSPDarknet con C2f modules
**Neck:** PAN (Path Aggregation Network)
**Head:** Decoupled detection head

### Hiperparámetros de Entrenamiento

**TODO:** Actualizar con valores reales después de training.

```yaml
# Configuración de entrenamiento
epochs: 100
batch_size: 16
img_size: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01  # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
patience: 10  # Early stopping

# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
```

### Requisitos de Inferencia

**Mínimos:**
- CPU: 4 cores, 8GB RAM
- Latencia: ~200ms por imagen

**Recomendados:**
- GPU: NVIDIA T4 o superior, 4GB VRAM
- Latencia: <50ms por imagen
- Throughput: 20+ FPS

**Optimizaciones:**
- Formato ONNX para 30-50% speed-up
- TensorRT para máximo rendimiento en producción
- Batch inference para múltiples cámaras

---

## Versionamiento

### Historial de Versiones

#### v1.0.0 (Baseline) - 2025-01-15
- Modelo inicial entrenado en dataset Roboflow
- YOLOv8n arquitectura
- 100 épocas, early stopping
- Formato: PyTorch y ONNX
- **Estado:** Desarrollo inicial

#### v1.1.0 (Planned) - TBD
- Fine-tuning con dataset Chile (500-1K imágenes)
- Mejora en condiciones locales
- Validación en faena piloto

#### v2.0.0 (Planned) - TBD
- Modelo productivo con dataset completo (2K+ imágenes Chile)
- Validación en múltiples faenas
- Certificación para uso en compliance formal

---

## Cómo Citar

Si utilizas este modelo en investigación o producción, por favor citar:

```bibtex
@software{hardhat_detection_yolov8,
  author = {Berrios, Bastian},
  title = {Hard Hat Detection YOLOv8 for Chilean Mining Industry},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/tu-usuario/hardhat-detection}
}
```

---

## Referencias

### Modelo y Framework

- **YOLOv8:** Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- **Dataset:** Roboflow Universe - Hard Hat Workers. https://universe.roboflow.com/roboflow-universe-projects/hard-hat-workers

### Regulación

- **DS 132:** Decreto Supremo 132, Reglamento de Seguridad Minera, Chile.
- **Ley 16.744:** Seguro contra Accidentes del Trabajo y Enfermedades Profesionales.
- **SERNAGEOMIN:** Servicio Nacional de Geología y Minería de Chile.

### Contexto Técnico

- **Transfer Learning:** Ver `docs/chile_context.md`
- **Training Pipeline:** Ver `scripts/train_gcp.py`
- **Deployment:** Ver `docs/deployment.md`

---

## Contacto y Soporte

**Desarrollador:** Bastian Berrios

**Repositorio:** https://github.com/tu-usuario/hardhat-detection

**Issues:** https://github.com/tu-usuario/hardhat-detection/issues

**Documentación completa:** Ver `/docs` directory

---

## Licencia

Este modelo se distribuye bajo licencia MIT. Ver `LICENSE` para más detalles.

**Dataset licencia:** CC BY 4.0 (Roboflow)

**Uso comercial:** Permitido con atribución apropiada.

---

**Última actualización:** 2025-01-15

**Estado:** Modelo baseline en desarrollo. Pendiente entrenamiento y evaluación completa.
