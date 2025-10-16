# Sistema de Traducción Español-Inglés para EPP Detector

## Problema Resuelto

El proyecto necesitaba balancear dos requisitos:
1. **Código en inglés**: Dataset Roboflow usa clases en inglés (`hardhat`, `head`, `person`)
2. **UX en español**: Usuarios finales chilenos necesitan ver resultados en español

## Solución Implementada

### 1. Clases del Modelo (Inglés)

**Ubicación**: `api/__init__.py`

```python
EPP_CLASSES = {
    0: "hardhat",
    1: "head",   # head without hardhat (non-compliant)
    2: "person",
}
```

**Razón**: El modelo YOLOv8 entrenado con dataset Roboflow espera estos nombres. No se pueden cambiar sin re-entrenar.

### 2. Diccionario de Traducción (Español)

**Ubicación**: `api/__init__.py`

```python
EPP_CLASSES_ES = {
    "hardhat": "Casco de seguridad",
    "head": "Cabeza sin casco",
    "person": "Persona",
}
```

**Uso**: Mapeo para traducir respuestas al usuario final.

### 3. Modelo de Respuesta Actualizado

**Ubicación**: `api/main.py`

```python
class Detection(BaseModel):
    class_id: int
    class_name: str           # "hardhat" (para código)
    class_name_es: str        # "Casco de seguridad" (para usuario)
    confidence: float
    bbox: List[float]
```

### 4. Funciones de Utilidad Actualizadas

**Ubicación**: `api/utils.py`

- `format_detections()`: Agrega automáticamente `class_name_es` a cada detección
- `check_epp_compliance()`: Retorna mensajes en español
  - `summary`: "✓ Cumplimiento de EPP verificado: Casco de seguridad detectado"
  - `violations`: ["Persona sin casco detectada (violación crítica)"]
- `get_class_color()`: Usa nombres en inglés pero retorna colores semánticos
  - Verde: EPP presente
  - Rojo: Violación crítica
  - Azul: Persona neutral

## Ejemplo de Respuesta de API

**Antes** (solo inglés):
```json
{
  "class_id": 0,
  "class_name": "hardhat",
  "confidence": 0.92,
  "bbox": [120.5, 50.2, 200.8, 150.3]
}
```

**Después** (bilingüe):
```json
{
  "class_id": 0,
  "class_name": "hardhat",
  "class_name_es": "Casco de seguridad",
  "confidence": 0.92,
  "bbox": [120.5, 50.2, 200.8, 150.3]
}
```

## Ventajas del Enfoque

1. **Consistencia técnica**: Código usa nombres estándar del dataset
2. **UX chilena**: Usuario final ve todo en español
3. **Mantenibilidad**: Un solo lugar para actualizar traducciones
4. **Escalabilidad**: Fácil agregar más idiomas (EPP_CLASSES_PT, EPP_CLASSES_FR)
5. **Debugging**: Logs y código usan nombres estándar inglés

## Archivos Modificados

- ✅ `api/__init__.py`: Agregado `EPP_CLASSES_ES`
- ✅ `api/main.py`: Actualizado `Detection` model con `class_name_es`
- ✅ `api/utils.py`: Actualizadas funciones para usar traducciones
- ✅ `tests/conftest.py`: Se mantiene en inglés (consistente con modelo)

## Futuras Expansiones

Cuando se agreguen más clases de EPP al modelo:

```python
# Dataset futuro con chalecos y botas
EPP_CLASSES = {
    0: "hardhat",
    1: "head",
    2: "person",
    3: "safety_vest",      # Nuevo
    4: "no_vest",          # Nuevo
    5: "safety_boots",     # Nuevo
}

EPP_CLASSES_ES = {
    "hardhat": "Casco de seguridad",
    "head": "Cabeza sin casco",
    "person": "Persona",
    "safety_vest": "Chaleco reflectante",     # Agregar
    "no_vest": "Sin chaleco",                 # Agregar
    "safety_boots": "Zapatos de seguridad",   # Agregar
}
```

## Referencia de Clases Actuales

| ID | class_name | class_name_es | Color | Significado |
|----|------------|---------------|-------|-------------|
| 0  | hardhat    | Casco de seguridad | Verde | EPP correcto |
| 1  | head       | Cabeza sin casco | Rojo | Violación crítica |
| 2  | person     | Persona | Azul | Neutral |

---

**Autor**: Bastián Berríos
**Fecha**: 2025-01-15
**Proyecto**: EPP Detector - Minería Chile
