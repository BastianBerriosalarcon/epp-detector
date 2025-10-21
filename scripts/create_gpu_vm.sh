#!/bin/bash
# Script para crear VM con GPU en GCP para entrenamiento de YOLOv8
# Proyecto: EPP Detector - Chilean Mining Safety

set -e

PROJECT_ID="lucky-kayak-475119-c7"
VM_NAME="yolov8-trainer"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
BOOT_DISK_SIZE="100GB"

echo "=================================================="
echo "Creando VM con GPU para entrenamiento YOLOv8"
echo "=================================================="
echo ""
echo "Configuración:"
echo "  Proyecto: $PROJECT_ID"
echo "  Nombre VM: $VM_NAME"
echo "  Zona: $ZONE"
echo "  Máquina: $MACHINE_TYPE (4 vCPUs, 15GB RAM)"
echo "  GPU: $GPU_TYPE x $GPU_COUNT"
echo "  Disco: $BOOT_DISK_SIZE"
echo ""
echo "Costo estimado: ~$0.56/hora (VM + GPU + disco)"
echo ""
read -p "¿Continuar? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelado."
    exit 1
fi

echo ""
echo "Creando VM..."
echo ""

gcloud compute instances create $VM_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --boot-disk-type=pd-standard \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=install-nvidia-driver=True

echo ""
echo "=================================================="
echo "✓ VM creada exitosamente!"
echo "=================================================="
echo ""
echo "Para conectarte a la VM:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "Para copiar tu proyecto a la VM:"
echo "  gcloud compute scp --recurse ~/epp-detector $VM_NAME:~/ --zone=$ZONE"
echo ""
echo "Para detener la VM (IMPORTANTE para no gastar créditos):"
echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "Para eliminar la VM:"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
