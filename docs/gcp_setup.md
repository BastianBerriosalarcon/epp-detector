# Guía de Configuración GCP para Training

Guía completa para entrenar modelos YOLOv8 en Google Cloud Platform con aceleración GPU.

## Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Configuración Inicial de GCP](#configuración-inicial-de-gcp)
3. [Crear Instancia de VM con GPU](#crear-instancia-de-vm-con-gpu)
4. [Configuración de SSH y Entorno](#configuración-de-ssh-y-entorno)
5. [Flujo de Entrenamiento](#flujo-de-entrenamiento)
6. [Descargar Resultados](#descargar-resultados)
7. [Limpieza (CRÍTICO)](#limpieza-crítico)
8. [Estimación de Costos](#estimación-de-costos)
9. [Solución de Problemas](#solución-de-problemas)

---

## Requisitos Previos

### 1. Cuenta de Google Cloud

- Cuenta GCP activa con facturación habilitada
- Free tier: $300 créditos por 90 días (usuarios nuevos)
- Tarjeta de crédito requerida para verificación de identidad

Regístrate en: https://cloud.google.com/free

### 2. Instalar gcloud CLI

**macOS:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

**Windows:**
Descargar instalador desde: https://cloud.google.com/sdk/docs/install

**Verificar instalación:**
```bash
gcloud --version
```

### 3. Cuota de GPU

Por defecto, las cuentas GCP tienen cuota 0 de GPU. Necesitas solicitar aumento de cuota:

1. Ir a: https://console.cloud.google.com/iam-admin/quotas
2. Filtrar por "GPUs (all regions)"
3. Seleccionar cuota y hacer clic en "EDIT QUOTAS"
4. Solicitar al menos 1 GPU (T4 recomendado)
5. Esperar aprobación (usualmente 1-2 días hábiles)

**Alternativa para estudiantes:**
Solicitar créditos de Google Cloud for Education: https://edu.google.com/programs/credits/

---

## Configuración Inicial de GCP

### 1. Autenticarse

```bash
gcloud auth login
```

Esto abre un navegador para autenticación.

### 2. Crear o Seleccionar Proyecto

```bash
# Crear nuevo proyecto
gcloud projects create hardhat-detection-PROJECT_ID --name="Hard Hat Detection"

# Configurar como proyecto activo
gcloud config set project hardhat-detection-PROJECT_ID

# Verificar
gcloud config get-value project
```

Reemplaza `PROJECT_ID` con un identificador único (ej: tu nombre + número aleatorio).

### 3. Habilitar APIs Requeridas

```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

### 4. Configurar Región/Zona por Defecto

Elige una región con disponibilidad de GPU y bajo costo:

```bash
# Buenas opciones:
# - us-central1-a (Iowa, económico)
# - us-west1-b (Oregon)
# - europe-west4-a (Países Bajos)

gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

Verificar disponibilidad de GPU por región: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones

---

## Crear Instancia de VM con GPU

### Opción 1: Comando Único (Recomendado)

```bash
gcloud compute instances create yolov8-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

**Configuración explicada:**
- `n1-standard-4`: 4 vCPUs, 15 GB RAM
- `nvidia-tesla-t4`: GPU de entrada (16GB VRAM, ~$0.35/hora)
- `pytorch-latest-gpu`: Imagen pre-configurada con CUDA, PyTorch
- `100GB boot disk`: Suficiente para dataset + modelos
- `TERMINATE on maintenance`: Requerido para instancias con GPU

### Opción 2: Usando la Consola

1. Ir a: https://console.cloud.google.com/compute/instances
2. Hacer clic en "CREAR INSTANCIA"
3. Configurar:
   - **Nombre:** yolov8-training-vm
   - **Región:** us-central1
   - **Zona:** us-central1-a
   - **Tipo de máquina:** n1-standard-4
   - **GPUs:** Agregar 1x NVIDIA T4
   - **Disco de arranque:**
     - SO: Deep Learning on Linux
     - Versión: PyTorch 1.x + CUDA
     - Tamaño: 100 GB
4. Hacer clic en "CREAR"

### Verificar Creación de Instancia

```bash
gcloud compute instances list

# Salida esperada:
# NAME                 ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
# yolov8-training-vm   us-central1-a  n1-standard-4               10.x.x.x     x.x.x.x        RUNNING
```

---

## Configuración de SSH y Entorno

### 1. Conectar por SSH a la VM

```bash
gcloud compute ssh yolov8-training-vm --zone=us-central1-a
```

La primera conexión configurará las claves SSH automáticamente.

### 2. Verificar GPU

Una vez conectado, verificar que la GPU sea accesible:

```bash
nvidia-smi
```

Salida esperada:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P0    26W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 3. Verificar PyTorch y CUDA

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Cantidad GPUs: {torch.cuda.device_count()}')"
```

Salida esperada:
```
PyTorch: 2.x.x
CUDA disponible: True
Cantidad GPUs: 1
```

### 4. Clonar Repositorio

```bash
cd ~
git clone https://github.com/TU_USUARIO/hardhat-detection.git
cd hardhat-detection
```

Reemplaza `TU_USUARIO` con tu usuario de GitHub.

### 5. Instalar Dependencias

```bash
# Crear entorno virtual (opcional pero recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar requirements
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalación de ultralytics
yolo version
```

### 6. Configurar Variables de Entorno

```bash
# Crear archivo .env
cat > .env << 'EOF'
ROBOFLOW_API_KEY=tu_api_key_aqui
MLFLOW_TRACKING_URI=./mlruns
EOF

# Reemplazar con tu API key real de Roboflow
nano .env
```

---

## Flujo de Entrenamiento

### Paso 1: Descargar Dataset

```bash
# Descargar dataset de Roboflow
python data/scripts/download_roboflow.py --output-dir ./data/roboflow

# Verificar descarga
ls -lh data/roboflow/
cat data/roboflow/data.yaml
```

### Paso 2: Iniciar Entrenamiento

**Inicio rápido (50 épocas):**
```bash
python scripts/train_gcp.py \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --dataset-path ./data/roboflow
```

**Entrenamiento productivo (100+ épocas con export):**
```bash
nohup python scripts/train_gcp.py \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --model yolov8n.pt \
    --dataset-path ./data/roboflow \
    --export-onnx \
    > training.log 2>&1 &
```

Usar `nohup` permite que el entrenamiento continúe si SSH se desconecta.

**Monitorear entrenamiento:**
```bash
# Ver logs en tiempo real
tail -f training.log

# Verificar uso de GPU
watch -n 1 nvidia-smi

# Verificar progreso del entrenamiento
ls -lh runs/train/
```

### Paso 3: Estimaciones de Tiempo de Entrenamiento

Con GPU NVIDIA T4:

| Tamaño Dataset | Épocas | Batch Size | Tiempo    |
|----------------|--------|------------|-----------|
| 5K imágenes    | 50     | 16         | ~2 horas  |
| 5K imágenes    | 100    | 16         | ~4 horas  |
| 10K imágenes   | 50     | 16         | ~4 horas  |
| 10K imágenes   | 100    | 16         | ~8 horas  |

**Tip:** Comienza con 10-20 épocas para verificar que todo funcione, luego ejecuta el entrenamiento completo.

### Paso 4: Monitorear Entrenamiento

Las salidas del entrenamiento se guardan en `runs/train/train_TIMESTAMP/`:

```bash
# Ver métricas de entrenamiento
cat runs/train/train_*/results.csv

# Ver mejor mAP
grep "best" runs/train/train_*/results.csv

# Verificar tamaño del modelo
ls -lh runs/train/train_*/weights/
```

---

## Descargar Resultados

### Opción 1: gcloud scp (Recomendado)

Desde tu **máquina local** (no la VM):

```bash
# Descargar directorio completo de entrenamiento
gcloud compute scp --recurse \
    yolov8-training-vm:~/hardhat-detection/runs/train \
    ./local-results/ \
    --zone=us-central1-a

# Descargar archivos específicos
gcloud compute scp \
    yolov8-training-vm:~/hardhat-detection/runs/train/train_*/weights/best.pt \
    ./models/ \
    --zone=us-central1-a
```

### Opción 2: Google Cloud Storage

En la VM:
```bash
# Crear bucket GCS (una sola vez)
gsutil mb -l us-central1 gs://hardhat-models-SUFIJO_UNICO

# Subir resultados
gsutil -m cp -r runs/train/train_* gs://hardhat-models-SUFIJO_UNICO/
```

Desde máquina local:
```bash
# Descargar desde bucket
gsutil -m cp -r gs://hardhat-models-SUFIJO_UNICO/train_* ./local-results/
```

### Opción 3: Descarga Directa vía Navegador

1. Instalar servidor de archivos en VM:
```bash
python3 -m http.server 8080 --directory runs/train
```

2. Crear túnel SSH desde máquina local:
```bash
gcloud compute ssh yolov8-training-vm \
    --zone=us-central1-a \
    -- -L 8080:localhost:8080
```

3. Abrir navegador en: http://localhost:8080

---

## Limpieza (CRÍTICO)

**DETENER LA VM INMEDIATAMENTE DESPUÉS DEL ENTRENAMIENTO PARA EVITAR CARGOS**

### Detener VM (Preserva disco, costo mínimo)

```bash
# Desde máquina local
gcloud compute instances stop yolov8-training-vm --zone=us-central1-a
```

Costo mientras está detenida: ~$0.04/día solo por almacenamiento de disco.

### Eliminar VM (Recomendado después de descargar resultados)

```bash
gcloud compute instances delete yolov8-training-vm --zone=us-central1-a
```

Esto elimina todo. Asegúrate de haber descargado los resultados primero.

### Verificar Limpieza

```bash
# Verificar instancias en ejecución
gcloud compute instances list

# Verificar discos (en caso de que la VM se eliminó pero el disco permaneció)
gcloud compute disks list
```

### Eliminar Discos Huérfanos

```bash
gcloud compute disks delete yolov8-training-vm --zone=us-central1-a
```

---

## Estimación de Costos

### Precios (us-central1, a partir de 2025)

| Recurso              | Costo Unitario | Notas                    |
|----------------------|----------------|--------------------------|
| n1-standard-4        | $0.19/hora     | 4 vCPU, 15 GB RAM        |
| NVIDIA Tesla T4      | $0.35/hora     | 16 GB memoria GPU        |
| Disco (100 GB SSD)   | $0.17/mes      | Mientras VM está detenida|
| Egreso de red        | $0.12/GB       | Descarga de resultados   |

**Total: ~$0.54/hora mientras está en ejecución**

### Costos de Ejemplo

| Escenario                             | Duración  | Costo    |
|---------------------------------------|-----------|----------|
| Entrenamiento corto (50 épocas)       | 2 horas   | $1.08    |
| Entrenamiento medio (100 épocas)      | 4 horas   | $2.16    |
| Entrenamiento largo (200 épocas)      | 8 horas   | $4.32    |
| Olvidaste detener por 1 día           | 24 horas  | $12.96   |
| Olvidaste detener por 1 semana        | 168 horas | $90.72   |

**Con $300 créditos gratuitos:**
- ~555 horas de entrenamiento
- ~138 entrenamientos completos (4 horas cada uno)

### Tips de Optimización de Costos

1. **Usar VMs Preemptibles** (70% más baratas pero pueden ser terminadas):
```bash
gcloud compute instances create ... --preemptible
```

2. **Comenzar con pocas épocas**: Prueba con 10 épocas primero

3. **Usar GPU más pequeña**: K80 es más barata ($0.45/hora total) pero más lenta

4. **Configurar alertas de presupuesto**:
   - Ir a: https://console.cloud.google.com/billing/budgets
   - Crear alerta en $10, $50, $100

5. **Apagado automático**: Programar VM para que se detenga después de N horas:
```bash
# Detener VM después de 6 horas
echo "gcloud compute instances stop yolov8-training-vm --zone=us-central1-a" | at now + 6 hours
```

---

## Solución de Problemas

### Problema: "Quota exceeded" al crear VM

**Solución:**
Solicitar aumento de cuota de GPU (ver [Requisitos Previos](#requisitos-previos)).

### Problema: nvidia-smi no encontrado

**Solución:**
```bash
# Instalar drivers NVIDIA
sudo /opt/deeplearning/install-driver.sh
sudo reboot
```

### Problema: CUDA out of memory durante entrenamiento

**Solución:**
Reducir batch size:
```bash
python scripts/train_gcp.py --batch-size 8  # En lugar de 16
```

### Problema: Timeout de conexión SSH

**Solución:**
```bash
# Agregar regla de firewall para SSH
gcloud compute firewall-rules create allow-ssh \
    --allow tcp:22 \
    --source-ranges 0.0.0.0/0
```

### Problema: El entrenamiento se detiene cuando SSH se desconecta

**Solución:**
Usar `nohup` o `screen`:
```bash
# Opción 1: nohup
nohup python scripts/train_gcp.py ... > training.log 2>&1 &

# Opción 2: screen (reanudar después de reconectar)
screen -S training
python scripts/train_gcp.py ...
# Ctrl+A, D para desconectar
# screen -r training para reanudar
```

### Problema: Velocidades de descarga lentas

**Solución:**
Usar Cloud Storage en lugar de scp directo:
```bash
gsutil -m cp -r runs/train gs://nombre-bucket/
```

### Problema: No encuentro los resultados del entrenamiento

**Solución:**
```bash
# Encontrar todas las salidas de entrenamiento
find ~ -name "best.pt" -type f 2>/dev/null
```

---

## Próximos Pasos

Después de un entrenamiento exitoso:

1. **Evaluar modelo**: Ver `docs/evaluation.md`
2. **Exportar a ONNX**: `python scripts/export_onnx.py --model runs/train/.../best.pt`
3. **Desplegar API**: Ver `docs/deployment.md`
4. **Monitorear en producción**: Configurar logging y alertas

---

## Recursos Adicionales

- Calculadora de Precios GCP: https://cloud.google.com/products/calculator
- Regiones con GPU: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
- Imágenes Deep Learning VM: https://cloud.google.com/deep-learning-vm
- Créditos Gratuitos: https://cloud.google.com/free

---

**¿Preguntas o problemas?**
Abre un issue en GitHub o contacta al mantenedor.
