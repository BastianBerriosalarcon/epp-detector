# Dataset de EPP para Minería Chilena

## Resumen

Este proyecto utiliza el dataset **Construction Site Safety v27** de Roboflow como base para entrenar un detector de EPP adaptado a la minería chilena.

**Ubicación:** `/home/bastian_berrios/epp-detector/data/roboflow/`

## Estadísticas del Dataset

- **Total de imágenes:** 2,799
  - Training: 2,603 imágenes (93%)
  - Validation: 114 imágenes (4%)
  - Test: 82 imágenes (3%)
- **Total de anotaciones:** 34,780 objetos etiquetados
- **Formato:** YOLOv8 (txt files con formato: `class_id x_center y_center width height`)
- **Licencia:** CC BY 4.0

## Clases del Dataset (10 clases)

### Clases Relevantes para DS 132 (Reglamento Minero Chileno)

| ID | Nombre (EN) | Nombre (ES) | Tipo | Relevancia |
|----|-------------|-------------|------|------------|
| 0 | Hardhat | Casco de seguridad | EPP Obligatorio | Alta - Requisito DS 132 |
| 7 | Safety Vest | Chaleco reflectante | EPP Obligatorio | Alta - Requisito DS 132 |
| 2 | NO-Hardhat | Sin casco | Violación | Alta - Infracción crítica |
| 4 | NO-Safety Vest | Sin chaleco | Violación | Alta - Infracción crítica |
| 5 | Person | Persona | Contexto | Media - Para conteo de trabajadores |

### Clases de Contexto (útiles pero no son EPP)

| ID | Nombre (EN) | Nombre (ES) | Tipo | Relevancia |
|----|-------------|-------------|------|------------|
| 6 | Safety Cone | Cono de seguridad | Señalización | Baja - Útil para contexto de zona |
| 8 | machinery | Maquinaria | Equipo | Baja - Contexto de operaciones |
| 9 | vehicle | Vehículo | Transporte | Baja - Contexto de faena |

### Clases NO Relevantes (específicas de construcción/pandemia)

| ID | Nombre (EN) | Nombre (ES) | Tipo | Relevancia |
|----|-------------|-------------|------|------------|
| 1 | Mask | Mascarilla | EPP COVID | Ninguna - No aplica DS 132 |
| 3 | NO-Mask | Sin mascarilla | Violación COVID | Ninguna - No aplica DS 132 |

## Lógica de Cumplimiento de EPP (DS 132)

Según el Decreto Supremo 132 del Ministerio de Minería de Chile, el EPP mínimo obligatorio es:

**EPP Obligatorio:**
- Casco de seguridad (Hardhat)
- Chaleco reflectante (Safety Vest)

**Evaluación de Cumplimiento:**

```python
# Persona CUMPLE si:
- Detectado: Hardhat (clase 0) Y Safety Vest (clase 7)
- NO detectado: NO-Hardhat (clase 2) NI NO-Safety Vest (clase 4)

# Persona NO CUMPLE si:
- Detectado: NO-Hardhat (clase 2) O NO-Safety Vest (clase 4)

# Estado DESCONOCIDO si:
- Solo detectado: Person (clase 5) sin detección de EPP/violaciones
```

## Estrategia de Entrenamiento

### Opción 1: Usar todas las clases (Recomendado para comenzar)

Entrenar con las 10 clases originales del dataset para aprovechar toda la información disponible.

**Ventajas:**
- Más datos de entrenamiento
- Contexto completo de escena (maquinaria, vehículos)
- Detección de conos útil para zonas de riesgo

**Desventajas:**
- Clases de mascarilla desperdician capacidad del modelo
- Modelo más grande (10 clases vs 6 clases)

**Comando:**
```bash
python scripts/train_gcp.py \
    --data /home/bastian_berrios/epp-detector/data/roboflow/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640
```

### Opción 2: Filtrar clases irrelevantes

Crear un subset del dataset eliminando clases 1 (Mask) y 3 (NO-Mask).

**Ventajas:**
- Modelo más enfocado en EPP minero
- Menos clases = modelo más pequeño y rápido

**Desventajas:**
- Requiere pre-procesamiento del dataset
- Pierde algunas imágenes si solo contienen mascarillas

**Implementación:** Requiere script de filtrado (pendiente)

### Opción 3: Transfer Learning + Fine-tuning con datos chilenos (Largo plazo)

1. Pre-entrenar con dataset Roboflow completo (10 clases)
2. Recolectar imágenes de faenas mineras chilenas (objetivo: 1,000+ imágenes)
3. Fine-tune el modelo con datos localizados

**Ventajas:**
- Mejor adaptación a condiciones chilenas (iluminación desértica, tipos de casco/chaleco locales)
- Mayor precisión en producción

**Desventajas:**
- Requiere inversión en recolección de datos
- Proceso más largo

## Distribución de Clases en el Dataset

Para verificar la distribución de clases en el dataset:

```bash
cd /home/bastian_berrios/epp-detector/data/roboflow
python3 << 'EOF'
import os
from collections import Counter

classes = {
    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
    4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
    7: "Safety Vest", 8: "machinery", 9: "vehicle"
}

class_counts = Counter()

for split in ['train', 'valid', 'test']:
    label_dir = f'{split}/labels'
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(f'{label_dir}/{label_file}', 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

print("Distribución de clases en el dataset:")
print("-" * 50)
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    pct = (count / sum(class_counts.values())) * 100
    print(f"{class_id:2d} | {classes[class_id]:20s} | {count:6d} ({pct:5.2f}%)")
print("-" * 50)
print(f"Total de anotaciones: {sum(class_counts.values())}")
EOF
```

## Configuración del Modelo

El archivo `data.yaml` está configurado con:

```yaml
path: /home/bastian_berrios/epp-detector/data/roboflow
train: train/images
val: valid/images
test: test/images
nc: 10
names:
  - Hardhat
  - Mask
  - NO-Hardhat
  - NO-Mask
  - NO-Safety Vest
  - Person
  - Safety Cone
  - Safety Vest
  - machinery
  - vehicle
```

## Próximos Pasos

1. **Verificar distribución de clases** (ejecutar script arriba)
2. **Entrenar modelo base** con Opción 1 (todas las clases)
3. **Evaluar resultados** en conjunto de validación
4. **Analizar matriz de confusión** para clases críticas (Hardhat, Safety Vest, violaciones)
5. **Decidir** si filtrar clases irrelevantes o continuar con todas
6. **Planificar** recolección de datos chilenos para fine-tuning

## Referencias

- **Dataset Source:** [Roboflow Construction Site Safety v27](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27)
- **Regulación:** Decreto Supremo 132 - Reglamento de Seguridad Minera (Chile)
- **YOLOv8 Docs:** [Ultralytics YOLOv8 Training](https://docs.ultralytics.com/modes/train/)

## Notas Importantes

- **Desbalance de clases:** Es probable que algunas clases (Person, Hardhat) tengan muchas más anotaciones que violaciones (NO-Hardhat, NO-Safety Vest). Considerar class weights durante entrenamiento.
- **Calidad de anotaciones:** Dataset de Roboflow es de construcción, no minería. Puede haber diferencias en tipos de casco/chaleco entre construcción y minería chilena.
- **Contexto internacional:** Dataset incluye imágenes de varios países. Fine-tuning con datos chilenos mejorará significativamente la precisión en producción.
