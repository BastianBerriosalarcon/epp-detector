# Detector de EPP para Minería Chilena

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Objetivo

Sistema de detección automática de Equipos de Protección Personal (EPP) mediante visión por computadora, diseñado específicamente para la industria minera chilena. El sistema identifica en tiempo real:

- **Casco de seguridad** (minería)
- **Chaleco reflectante** (alta visibilidad)
- **Zapatos de seguridad** (puntera de acero)

**Problema:** Los accidentes laborales en minería chilena representan el 8.5% del total nacional (SUSESO 2023). El uso incorrecto o ausencia de EPP es una de las principales causas de lesiones graves y fatalidades. La supervisión manual es inconsistente y costosa.

**Solución:** Automatización de la detección de EPP mediante YOLOv8, desplegada en edge devices y sistemas de video vigilancia existentes.

---

## 🇨🇱 Diferenciador: Contexto Chile

Este proyecto NO es un detector genérico de EPP. Está diseñado para:

1. **Cumplimiento DS 132**: Reglamento de Seguridad Minera (actualizado 2024)
2. **Condiciones locales**: Polvo de cobre, iluminación subterránea, equipos amarillos/naranjos
3. **Dataset chileno**: Imágenes reales de faenas mineras chilenas (División Andina, El Teniente, minas del norte)
4. **Integración con normativa**: Alertas configurables según estándares SERNAGEOMIN

**Fine-tuning específico** en datos de:
- Minería subterránea (baja iluminación)
- Rajo abierto (alto contraste, polvo)
- Equipos y uniformes de empresas chilenas (Codelco, Antofagasta Minerals, etc.)

---

## Arquitectura

```
[TODO: Diagrama de arquitectura - Iteración 3]

Input (Video/Imagen) → YOLOv8 (ONNX Runtime) → Post-processing →
  ├─ API REST (FastAPI)
  ├─ Alertas (Missing EPP)
  └─ Dashboard (Streamlit)
```

**Componentes:**
- **Inference Engine**: YOLOv8n optimizado (ONNX) para edge deployment
- **API**: FastAPI con endpoints de detección batch/streaming
- **Frontend**: Streamlit para visualización y validación de alertas
- **Storage**: GCP Cloud Storage para imágenes y modelos
- **Monitoring**: MLflow para tracking de métricas de modelo

---

## Performance

| Métrica | Objetivo | Actual |
|---------|----------|--------|
| **mAP@0.5** | ≥ 0.85 | TBD |
| **Precision** | ≥ 0.90 | TBD |
| **Recall** | ≥ 0.88 | TBD |
| **Inference (CPU)** | < 100ms | TBD |
| **Inference (GPU)** | < 30ms | TBD |
| **False Positives** | < 5% | TBD |

**Hardware target:**
- NVIDIA Jetson Nano (edge)
- CPU Intel i5+ (fallback)
- Cloud: GCP Compute Engine (N1-standard-2)

---

## Quick Start

### 1. Setup del entorno

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/epp-detector.git
cd epp-detector

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Solo para desarrollo
```

### 2. Entrenamiento (TODO - Iteración 4)

```bash
# Descargar dataset
python scripts/download_dataset.py

# Entrenar YOLOv8
python scripts/train.py --config configs/yolov8n.yaml
```

### 3. Inference

```bash
# API REST
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Streamlit Dashboard
streamlit run streamlit_app/app.py
```

### 4. Docker

```bash
# Build
docker build -t epp-detector:latest .

# Run API
docker run -p 8000:8000 epp-detector:latest

# Run con GPU
docker run --gpus all -p 8000:8000 epp-detector:latest
```

---

## Tech Stack

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **Detección** | YOLOv8 (Ultralytics) | SOTA en real-time object detection, <30ms inference |
| **Runtime** | ONNX Runtime | Portabilidad multi-plataforma, optimización CPU/GPU |
| **API** | FastAPI | Async, auto-docs, validación con Pydantic |
| **Frontend** | Streamlit | Prototipado rápido, ideal para demos ML |
| **Cloud** | Google Cloud Platform | Cloud Storage + Compute Engine |
| **MLOps** | MLflow | Tracking de experimentos y registro de modelos |
| **CI/CD** | GitHub Actions | Testing automático, build Docker |
| **Containerización** | Docker | Reproducibilidad, deployment simplificado |

---

## Datasets

### 1. Dataset Base (Roboflow)
- **Fuente**: [Construction Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **Clases**: Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest, Person
- **Imágenes**: ~5,000 anotadas (YOLO format)

### 2. Dataset Chile (Propio) 🇨🇱
- **Fuente**: Imágenes de faenas mineras chilenas (colaboración empresas)
- **Objetivo**: 1,000+ imágenes adicionales
- **Clases específicas**:
  - Casco amarillo/blanco (estándar minería)
  - Chaleco naranja reflectante
  - Zapatos de seguridad (visible en planos medios)
- **Condiciones**: Subterránea, rajo abierto, diferentes iluminaciones

**TODO**: Crear script de anotación semi-automática con modelo pre-entrenado

---

## Roadmap

### Iteración 1: Setup ✅
- [x] Estructura de directorios
- [x] .gitignore, .dockerignore
- [x] README básico

### Iteración 2: Core Files ✅
- [x] requirements.txt / requirements-dev.txt
- [x] README profesional
- [x] Configuración base

### Iteración 3: Inference Engine (En progreso)
- [ ] Módulo de detección YOLOv8
- [ ] Descarga de modelo pre-entrenado
- [ ] Pipeline de pre/post-processing
- [ ] Tests unitarios

### Iteración 4: API REST
- [ ] FastAPI con endpoints `/detect` y `/health`
- [ ] Manejo de imágenes (upload/URL)
- [ ] Respuesta JSON con bounding boxes
- [ ] Documentación OpenAPI

### Iteración 5: Training Pipeline
- [ ] Script de descarga de dataset
- [ ] Configuración YAML para YOLOv8
- [ ] Script de entrenamiento con MLflow
- [ ] Validación y métricas

### Iteración 6: Streamlit Dashboard
- [ ] Upload de imágenes
- [ ] Visualización de detecciones
- [ ] Estadísticas de EPP
- [ ] Filtros por clase

### Iteración 7: Docker & Deployment
- [ ] Dockerfile optimizado
- [ ] Docker Compose (API + Streamlit)
- [ ] Deployment en GCP Cloud Run
- [ ] CI/CD con GitHub Actions

### Iteración 8: Dataset Chile
- [ ] Recolección de imágenes locales
- [ ] Anotación (Roboflow/LabelImg)
- [ ] Fine-tuning del modelo
- [ ] Validación en condiciones chilenas

---

## Estructura del Proyecto

```
epp-detector/
├── api/                    # FastAPI REST API
├── data/                   # Datasets
│   ├── chile/             # Imágenes minería chilena
│   └── scripts/           # Download/preprocessing
├── models/                # Modelos entrenados (.pt, .onnx)
├── notebooks/             # Jupyter notebooks (EDA, experimentos)
├── scripts/               # Scripts de entrenamiento/inference
├── streamlit_app/         # Dashboard Streamlit
├── tests/                 # Tests unitarios y de integración
├── docs/                  # Documentación adicional
├── .github/workflows/     # CI/CD
├── requirements.txt       # Dependencias producción
├── requirements-dev.txt   # Dependencias desarrollo
├── Dockerfile             # TODO: Iteración 7
└── README.md
```

---

## Contribución

Este proyecto está en desarrollo activo. Para contribuir:

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar detección de guantes'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

**Código de estilo:**
- Black para formateo
- isort para imports
- flake8 para linting
- Type hints en funciones públicas

---

## License

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## Contacto

**Autor**: Bastián Berríos
**Email**: [tu-email@ejemplo.com]
**LinkedIn**: [Tu perfil]
**GitHub**: [Tu usuario]

---

## Referencias

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Decreto Supremo 132 - Reglamento de Seguridad Minera](https://www.sernageomin.cl/)
- [SUSESO - Estadísticas de Accidentabilidad](https://www.suseso.cl/)
- [Roboflow Universe](https://universe.roboflow.com/)

---

**Made with ❤️ for Chilean mining safety**
