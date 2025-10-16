# Contexto Chileno: Detección de Cascos en Minería

## Tabla de Contenidos

1. [Problema de Negocio](#problema-de-negocio)
2. [Marco Regulatorio Chileno](#marco-regulatorio-chileno)
3. [Estrategia de Datos para Chile](#estrategia-de-datos-para-chile)
4. [Fuentes de Datos Locales](#fuentes-de-datos-locales)
5. [Approach de Fine-tuning](#approach-de-fine-tuning)
6. [Valor Diferenciador](#valor-diferenciador)
7. [Siguientes Pasos](#siguientes-pasos)

---

## Problema de Negocio

### Contexto de la Industria Minera en Chile

Chile es el mayor productor de cobre del mundo, representando aproximadamente el 28% de la producción global. La minería aporta:

- **~10% del PIB nacional**
- **~50% de las exportaciones**
- **Más de 250,000 empleos directos**
- **~700,000 empleos indirectos**

La seguridad laboral es **crítica** tanto por:
1. **Aspecto humano**: Proteger la vida de los trabajadores
2. **Aspecto legal**: Cumplimiento normativo estricto
3. **Aspecto económico**: Multas, paralizaciones, daño reputacional

### Estadísticas de Accidentes

Según SERNAGEOMIN (Servicio Nacional de Geología y Minería):

- **2023**: 13 fatalidades en minería
- **2022**: 23 fatalidades
- **Tendencia**: Reducción pero aún crítico
- **Principales causas**: Caídas de objetos, golpes, aplastamientos

**Impacto del uso incorrecto de EPP (Equipos de Protección Personal):**
- ~40% de accidentes graves relacionados con no uso o mal uso de EPP
- Multas promedio: $10M-$50M CLP por incumplimiento grave
- Paralización de operaciones: Hasta 30 días

### Oportunidad Tecnológica

**Desafíos actuales:**
- Inspección manual de EPP es **inconsistente**
- Supervisores no pueden estar en todas partes simultáneamente
- Registro manual de incidencias es **lento e incompleto**
- Auditorías solo capturan **snapshot puntual**

**Solución propuesta:**
Sistema automatizado de detección de uso de cascos mediante:
- Cámaras de seguridad existentes (ya desplegadas)
- Deep Learning (YOLOv8) para detección en tiempo real
- Alertas automáticas a supervisores
- Dashboard de cumplimiento y analytics

---

## Marco Regulatorio Chileno

### DS 132: Reglamento de Seguridad Minera

**Decreto Supremo N°132** (actualizado 2024) del Ministerio de Minería establece:

**Artículo 153** - Elementos de Protección Personal:
> "Todo trabajador que ingrese a faenas mineras debe utilizar obligatoriamente casco de seguridad certificado, calzado de seguridad, lentes de protección cuando corresponda, y demás EPP según evaluación de riesgos."

**Artículo 154** - Supervisión de EPP:
> "El empleador debe implementar sistemas de control y supervisión del uso correcto de EPP, incluyendo mecanismos de registro y verificación."

**Sanciones por incumplimiento (Art. 158):**
- **Leves**: Multa 1-50 UTM (~$600K-$30M CLP)
- **Graves**: Multa 51-500 UTM + paralización parcial
- **Gravísimas**: Multa 501-2000 UTM + paralización total
- **Reincidencia**: Duplicación de multas + cierre temporal

### Ley 16.744: Seguro contra Accidentes del Trabajo

Establece obligaciones del empleador:
- Prevenir accidentes y enfermedades laborales
- Implementar sistemas de gestión de seguridad
- **Demostrar diligencia debida** en prevención

**Implicancia legal:**
Un sistema automatizado de monitoreo demuestra **diligencia debida activa**, reduciendo responsabilidad legal en caso de accidentes.

### SERNAGEOMIN: Fiscalización y Auditorías

El Servicio Nacional de Geología y Minería fiscaliza:

- **Inspecciones no anunciadas**: Verificación in situ de cumplimiento EPP
- **Auditorías documentales**: Revisión de registros de seguridad
- **Investigación de accidentes**: Análisis de causas raíz

**Valor del sistema propuesto:**
- Registro automático y permanente de cumplimiento
- Evidencia objetiva en auditorías
- Trazabilidad completa de incidencias
- Datos históricos para mejora continua

---

## Estrategia de Datos para Chile

### Dataset Base: Roboflow "Hard Hat Workers"

**Características:**
- **~5,000 imágenes** etiquetadas
- **Clases**: hardhat, head, person
- **Procedencia**: Internacional (US, Europa, Asia)
- **Escenarios**: Construcción, industria general

**Ventajas:**
- Calidad alta de anotaciones
- Diversidad de condiciones (luz, ángulos, distancias)
- Pretrained base sólida

**Limitaciones para Chile:**
- No incluye contexto minero específico
- Cascos tipo construcción (no mineros)
- Ausencia de condiciones del norte de Chile (desierto, alta radiación)
- No incluye señalética o equipamiento chileno

### Dataset Chile: Fine-tuning Localizado

**Objetivo:**
Mejorar rendimiento en condiciones específicas de minería chilena mediante transfer learning.

**Características objetivo:**
- Imágenes de faenas mineras reales (con permisos)
- Cascos mineros certificados Chile (colores corporativos)
- Condiciones lumínicas del desierto de Atacama
- Distancias típicas de cámaras industriales
- Integración con uniformes corporativos chilenos

**Tamaño estimado:**
- **Fase 1 (MVP)**: 500-1,000 imágenes adicionales
- **Fase 2 (Producción)**: 2,000-5,000 imágenes
- **Fase 3 (Optimización)**: Recolección continua en producción

---

## Fuentes de Datos Locales

### Opción 1: Colaboración con Mineras

**Empresas objetivo:**
- CODELCO (estatal, mayor productora cobre mundo)
- BHP (Minera Escondida)
- Antofagasta Minerals
- Anglo American

**Approach:**
1. Presentar proyecto como **caso piloto de innovación**
2. Enfatizar beneficios de seguridad y compliance
3. Solicitar acceso a footage de cámaras (anonimizado)
4. Ofrecer modelo entrenado de vuelta como beneficio

**Desafíos:**
- Procesos burocráticos largos
- Confidencialidad de operaciones
- Requiere contactos en áreas de seguridad/innovación

### Opción 2: Datos Públicos y Simulados

**Fuentes públicas:**
- **YouTube**: Videos de seguridad minera (ej: SERNAGEOMIN, ACHS)
- **Flickr/Wikimedia**: Fotos de faenas bajo licencias abiertas
- **Sitios corporativos**: Material de capacitación público

**Captura de frames:**
```python
# Ejemplo: Extraer frames de video YouTube
import cv2
from pytube import YouTube

video_url = "https://youtube.com/watch?v=MINING_SAFETY_VIDEO"
yt = YouTube(video_url)
stream = yt.streams.first()
stream.download(filename="safety_video.mp4")

# Extraer 1 frame cada 30 (1 por segundo aprox)
cap = cv2.VideoCapture("safety_video.mp4")
# ... procesamiento
```

**Generación sintética:**
- Modificar dataset base con filtros (simular desierto)
- Data augmentation agresiva (luz, contraste, saturación)
- No reemplaza datos reales pero complementa

### Opción 3: Crowdsourcing Controlado

**Plataforma propia:**
1. Crear app web simple de captura
2. Reclutar trabajadores mineros (con compensación)
3. Solicitar fotos de cascos en contextos variados
4. Validación manual de calidad

**Consideraciones legales:**
- Consentimiento informado claro
- Anonimización de rostros (GDPR/Ley 19.628)
- Compensación justa por tiempo

---

## Approach de Fine-tuning

### Transfer Learning desde Dataset Base

**Estrategia recomendada:**

1. **Fase 1: Baseline (Dataset Roboflow)**
   - Entrenar YOLOv8n en dataset completo Roboflow
   - 100 épocas, early stopping patience=10
   - Objetivo: mAP50 > 0.85

2. **Fase 2: Fine-tuning Chile (500-1K imágenes)**
   - **Freeze backbone**: Congelar primeras 15 capas
   - Entrenar solo detection head (últimas 10 capas)
   - Learning rate reducido (lr=0.001)
   - 50 épocas adicionales
   - Objetivo: Mejora 5-10% en subset chileno

3. **Fase 3: Full Fine-tuning (2K+ imágenes)**
   - Descongelar todas las capas
   - Learning rate muy bajo (lr=0.0001)
   - 100 épocas con strong augmentation
   - Objetivo: mAP50 > 0.90 en condiciones chilenas

**Hiperparámetros clave:**
```python
# Fine-tuning config
config = {
    "freeze": 15,  # Primeras 15 capas congeladas
    "lr0": 0.001,  # LR reducido
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 7.5,  # Box loss gain
    "cls": 0.5,  # Class loss gain (menos importante)
    "mosaic": 1.0,  # Strong augmentation
    "mixup": 0.1,
}
```

### Validación de Mejora

**Métricas clave:**
1. **mAP50**: Mean Average Precision @ IoU=0.5
2. **Precision**: Minimizar falsos positivos
3. **Recall**: Minimizar falsos negativos (crítico en seguridad)
4. **Latencia**: <100ms para real-time

**Test set específico Chile:**
- Separar 200-300 imágenes chilenas para test
- **NO usar en training ni validation**
- Evaluar modelo base vs. fine-tuned
- Reportar mejora estadísticamente significativa

---

## Valor Diferenciador

### Para el Portfolio Profesional

Este proyecto demuestra:

1. **Visión de Negocio**
   - Identificación de problema real con impacto económico
   - Comprensión del marco regulatorio local
   - Propuesta de valor clara y cuantificable

2. **Expertise Técnico**
   - State-of-the-art ML (YOLOv8, ONNX)
   - Transfer learning y fine-tuning
   - Deployment production-ready (FastAPI, Docker, K8s)
   - MLOps (MLflow, CI/CD, monitoring)

3. **Contexto Local**
   - Adaptación a regulación chilena
   - Comprensión industria minera nacional
   - Estrategia de datos localizada

4. **End-to-End Ownership**
   - Desde investigación inicial hasta deployment
   - Documentación exhaustiva
   - Código production-quality

### Diferenciación vs. Proyectos Genéricos

**Proyecto genérico:**
> "Detección de objetos con YOLO"

**Este proyecto:**
> "Sistema de monitoreo automatizado de EPP para cumplimiento DS 132 en minería chilena, con transfer learning localizado y deployment cloud-native"

### Escalabilidad del Concepto

El mismo approach aplica a:
- **Construcción**: DS 594 (Reglamento Sanitario Construcción)
- **Industria**: Detectar otros EPP (chalecos, guantes)
- **Retail**: Control de aforo, distanciamiento
- **Agricultura**: Uso de protección en fumigación

**Valor para empleadores:**
- Capacidad de adaptar ML a contexto local
- Comprensión de compliance y regulación
- Visión de producto (no solo técnica)

---

## Siguientes Pasos

### Roadmap del Proyecto

**Fase Actual: MVP Internacional**
- [x] Dataset base Roboflow
- [x] Training pipeline GCP
- [x] API básica FastAPI
- [ ] Evaluación en test set
- [ ] Deploy básico local

**Fase 2: Localización Chile (Q1 2025)**
- [ ] Recolectar 500 imágenes Chile
- [ ] Anotar con Roboflow
- [ ] Fine-tuning localizado
- [ ] Evaluación comparativa
- [ ] Documentar mejoras

**Fase 3: Piloto Real (Q2 2025)**
- [ ] Contactar 1-2 mineras
- [ ] Integrar con cámaras existentes
- [ ] Dashboard de monitoreo
- [ ] Validación en campo
- [ ] Recolectar feedback

**Fase 4: Productización (Q3 2025)**
- [ ] Escalamiento K8s
- [ ] Monitoreo y alertas
- [ ] SLA y reliability
- [ ] Documentación operacional
- [ ] Modelo de negocio

### KPIs de Éxito

**Técnicos:**
- mAP50 > 0.90 en dataset Chile
- Latencia < 100ms (P95)
- Uptime > 99.5%
- False negative rate < 2% (crítico seguridad)

**Negocio:**
- Reducción 30%+ tiempo supervisión manual
- ROI positivo vs. multas evitadas
- Adopción 1+ faena piloto
- Feedback positivo usuarios finales

---

## Conclusión

Este proyecto no es solo "otro detector de objetos", sino una **solución de negocio contextualizada** para un problema real de la industria chilena. La combinación de:

- **ML state-of-the-art**
- **Compliance regulatorio**
- **Transfer learning localizado**
- **Deployment production-ready**

...lo convierte en un proyecto portfolio **altamente diferenciador** para posiciones de ML Engineer, Applied Scientist o Tech Lead en Chile.

El enfoque demuestra capacidad de:
1. Identificar oportunidades de ML en industria tradicional
2. Navegar regulación y compliance
3. Adaptar soluciones globales a contexto local
4. Ejecutar end-to-end desde investigación a producción

**Valor final:** No solo demuestras que puedes entrenar modelos, sino que puedes **crear productos de ML que resuelven problemas reales de negocio**.

---

**Referencias:**
- DS 132 Reglamento Seguridad Minera: https://www.sernageomin.cl
- Ley 16.744: https://www.bcn.cl/leychile
- Estadísticas SERNAGEOMIN: https://www.sernageomin.cl/estadisticas-de-seguridad-minera/
- CODELCO Innovación: https://www.codelco.com/innovacion
