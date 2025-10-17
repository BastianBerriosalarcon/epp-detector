# Guía de Construcción de Dataset Personalizado para Minería Chilena

## Introducción

Esta guía proporciona instrucciones completas para construir un dataset personalizado de detección de EPP específico para operaciones mineras chilenas. Un dataset localizado mejorará significativamente la precisión del modelo al incluir:

- **Equipamiento específico**: Cascos y chalecos reflectantes usados en minería chilena
- **Condiciones reales**: Iluminación subterránea, polvo, diferentes ángulos de cámara
- **Cumplimiento DS 132**: Escenarios específicos de cumplimiento e incumplimiento de normativa chilena
- **Contexto local**: Trabajadores, maquinaria y entornos mineros chilenos

## Estimación de Tiempo y Recursos

### Escala Mínima Viable (MVP)
- **Imágenes requeridas**: 200-300 imágenes de alta calidad
- **Tiempo de recolección**: 1-2 días
- **Tiempo de anotación**: 3-5 días (2-3 minutos por imagen)
- **Costo**: $0 (anotación manual interna)
- **Resultado esperado**: Modelo funcional con precisión básica

### Escala Recomendada (Producción)
- **Imágenes requeridas**: 1000-2000 imágenes
- **Tiempo de recolección**: 1 semana
- **Tiempo de anotación**: 2-3 semanas (con 2 anotadores)
- **Costo**: $100-$1000 USD (si se externaliza anotación a $0.10-$0.50/imagen)
- **Resultado esperado**: Modelo robusto con alta precisión

### Escala Avanzada (Enterprise)
- **Imágenes requeridas**: 5000+ imágenes
- **Tiempo de recolección**: 2-3 semanas
- **Tiempo de anotación**: 1-2 meses (equipo de anotadores)
- **Costo**: $2500-$10000 USD (anotación profesional + validación)
- **Resultado esperado**: Modelo de grado industrial con precisión superior

## Requisitos de Diversidad del Dataset

### 1. Diversidad de Escenarios de Cumplimiento

**Prioridad ALTA**: Estas son las categorías críticas para detección de EPP

#### Cumplimiento Total (40% del dataset)
- Trabajadores usando casco + chaleco reflectante
- EPP visible claramente desde diferentes ángulos
- Varios colores de cascos (amarillo, blanco, rojo, naranja)
- Varios estilos de chalecos (naranja, amarillo, con bandas reflectantes)

**Target mínimo**: 80-120 imágenes (MVP) / 400-800 imágenes (Producción)

#### Violación: Sin Casco (30% del dataset)
- Personas claramente visibles sin casco
- Cabeza descubierta desde múltiples ángulos
- Diferentes tipos de cabello/gorras (pero sin casco)
- Contexto minero evidente

**Target mínimo**: 60-90 imágenes (MVP) / 300-600 imágenes (Producción)

**IMPORTANTE**: En operaciones reales, estas violaciones son raras. Deberá capturar intencionalmente o simular estos escenarios.

#### Violación: Sin Chaleco (20% del dataset)
- Personas con casco pero sin chaleco reflectante
- Ropa de trabajo sin elementos reflectantes
- Chaleco puesto pero no visible desde el ángulo de cámara

**Target mínimo**: 40-60 imágenes (MVP) / 200-400 imágenes (Producción)

#### Múltiples Personas (10% del dataset)
- 2-5+ trabajadores en la misma escena
- Mezcla de cumplimiento e incumplimiento
- Oclusiones parciales entre personas
- Diferentes distancias a la cámara

**Target mínimo**: 20-30 imágenes (MVP) / 100-200 imágenes (Producción)

### 2. Diversidad de Condiciones Ambientales

#### Iluminación (distribuir equitativamente)
- **Subterránea**: Iluminación artificial, sombras duras, alto contraste
- **Superficie (día)**: Luz solar directa, sombras naturales
- **Superficie (atardecer/amanecer)**: Luz baja, tonos cálidos
- **Nublado**: Luz difusa, bajo contraste
- **Nocturna**: Iluminación de reflectores, alta exposición

**Recomendación**: 20% por categoría de iluminación

#### Calidad del Aire
- **Limpia**: Visibilidad excelente
- **Polvo ligero**: Ligera bruma, reducción menor de contraste
- **Polvo moderado**: Partículas visibles, reducción notable de visibilidad
- **Polvo denso/niebla**: Visibilidad reducida, desafío para detección

**Recomendación**: 50% limpia, 30% polvo ligero, 15% polvo moderado, 5% polvo denso

#### Condiciones Meteorológicas (minas a cielo abierto)
- Soleado y claro
- Parcialmente nublado
- Llovizna/lluvia
- Viento (polvo suspendido)

### 3. Diversidad de Configuración de Cámara

#### Ángulos de Cámara
- **Frontal**: Vista directa del trabajador (0-30°)
- **Lateral**: Perfil o semi-perfil (30-60°)
- **Diagonal**: Vista en ángulo (60-90°)
- **Posterior**: Vista desde atrás (90-180°)
- **Elevada**: Cámara montada alta mirando hacia abajo (común en cámaras de seguridad)
- **Baja**: Cámara a nivel del suelo mirando hacia arriba

**Recomendación**: Priorizar ángulos frontales y laterales (70%), otros (30%)

#### Alturas de Cámara
- **Altura de ojos** (1.6-1.8m): Perspectiva natural
- **Montada alta** (2.5-4m): Típica de cámaras de seguridad fijas
- **Montada muy alta** (>4m): Cámaras de torre o dron
- **Baja** (<1m): Cámaras montadas en vehículos o equipos

#### Distancias al Sujeto
- **Primer plano**: 1-3 metros (detalle de EPP)
- **Media distancia**: 3-10 metros (escenario típico de operación)
- **Larga distancia**: 10-30 metros (supervisión general)
- **Muy larga distancia**: >30 metros (panorámica)

**Recomendación**: 40% media distancia, 30% primer plano, 20% larga, 10% muy larga

### 4. Diversidad de Tipos de Ubicación

#### Minas Subterráneas
- Túneles y galerías
- Puntos de extracción
- Áreas de transporte
- Estaciones de ventilación
- Refugios y áreas de descanso

#### Minas a Cielo Abierto (Rajo Abierto)
- Bancos de extracción
- Rampas de acceso
- Áreas de carga
- Zonas de tránsito de vehículos
- Puntos de control

#### Áreas de Superficie
- Plantas de procesamiento
- Talleres de mantención
- Oficinas de campo
- Zonas de almacenamiento
- Puntos de entrada/salida

### 5. Diversidad de Actividades

Capturar trabajadores realizando diferentes actividades:
- Caminando
- De pie (postura estática)
- Agachados/arrodillados
- Operando maquinaria
- Realizando mantención
- En reuniones de seguridad (grupos)
- Ascendiendo/descendiendo escaleras o rampas
- Gesticulando o señalizando

### 6. Diversidad Demográfica

**IMPORTANTE**: Respetar privacidad y obtener consentimiento

- Diferentes complexiones físicas
- Diferentes alturas
- Diferentes géneros (si aplica en su operación)
- Diferentes tipos de uniforme/ropa de trabajo

**Privacidad**: Considerar difuminar rostros si es requerido por política de empresa

## Estrategia de Recolección de Datos

### Opción 1: Extracción desde Videos (RECOMENDADA)

**Ventajas**:
- Más eficiente que fotos individuales
- Captura secuencias naturales de movimiento
- Fácil de obtener desde cámaras de seguridad existentes

**Proceso**:
1. Obtener grabaciones de cámaras de seguridad (CCTV)
2. Identificar segmentos con actividad relevante
3. Extraer frames usando `scripts/collect_images.py`
4. Filtrar frames borrosos o duplicados automáticamente

**Configuración recomendada**:
- Extraer 1 frame cada 2-3 segundos (0.33-0.5 fps)
- Usar detección de movimiento para evitar frames estáticos
- Filtrar duplicados con hashing perceptual

**Comando ejemplo**:
```bash
python scripts/collect_images.py \
    --source /path/to/videos/ \
    --output data/raw/images/ \
    --fps 0.33 \
    --min-blur-threshold 100 \
    --remove-duplicates
```

### Opción 2: Fotografía Directa

**Ventajas**:
- Control total sobre composición y calidad
- Puede capturar específicamente escenarios de violación

**Proceso**:
1. Coordinar con supervisores de seguridad
2. Visitar ubicaciones mineras durante diferentes turnos
3. Capturar fotos siguiendo checklist de diversidad
4. Organizar por ubicación/fecha

**Equipo recomendado**:
- Cámara/smartphone de buena calidad (>12MP)
- Resolución mínima: 1920x1080 (Full HD)
- Evitar zoom digital excesivo
- Usar estabilización de imagen

### Opción 3: Imágenes de Drones (Complementaria)

**Ventajas**:
- Perspectivas únicas (aérea)
- Buena para minas a cielo abierto
- Captura múltiples trabajadores simultáneamente

**Consideraciones**:
- Requiere permisos especiales
- Limitada para detalle de EPP (distancia)
- Útil para entrenamiento de contexto amplio

### Opción 4: Captura Híbrida (ÓPTIMA)

Combinar todas las estrategias:
- 60% desde videos de seguridad
- 30% fotografías directas dirigidas
- 10% otras fuentes (drones, cámaras corporales)

## Especificaciones Técnicas de Imágenes

### Resolución y Formato

**Resolución mínima**: 1280x720 (HD)
**Resolución recomendada**: 1920x1080 (Full HD)
**Resolución máxima útil**: 3840x2160 (4K) - evitar mayor por tiempo de procesamiento

**Formatos aceptados**:
- JPEG (.jpg, .jpeg) - recomendado, buen balance calidad/tamaño
- PNG (.png) - para máxima calidad, archivos grandes
- BMP (.bmp) - no recomendado, archivos muy grandes

**Configuración de compresión**:
- Calidad JPEG: 85-95% (evitar compresión excesiva)
- Evitar re-comprimir imágenes múltiples veces

### Criterios de Calidad

**Imágenes ACEPTABLES**:
- Enfoque nítido (sin blur de movimiento significativo)
- Exposición correcta (no sobre/subexpuesta)
- EPP visible y reconocible
- Al menos 1 persona con EPP presente

**Imágenes a RECHAZAR**:
- Extremadamente borrosas (blur score <100)
- Completamente sobre/subexpuestas
- Sin personas visibles
- EPP completamente oculto
- Duplicados exactos

### Herramientas de Filtrado Automático

El script `collect_images.py` incluye filtros automáticos:

```bash
# Filtrar imágenes de baja calidad
python scripts/collect_images.py \
    --source data/raw/ \
    --output data/filtered/ \
    --min-blur-threshold 100 \
    --min-brightness 30 \
    --max-brightness 225 \
    --remove-duplicates \
    --similarity-threshold 0.95
```

## Organización de Directorios

Estructura recomendada para el proyecto:

```
epp-detector/
├── data/
│   ├── raw/                          # Datos sin procesar
│   │   ├── videos/                   # Videos originales
│   │   │   ├── subterraneo/
│   │   │   ├── rajo_abierto/
│   │   │   └── superficie/
│   │   └── images/                   # Imágenes extraídas
│   │       ├── 2024-01-15/           # Organizadas por fecha
│   │       ├── 2024-01-16/
│   │       └── camera_01/            # O por cámara
│   │
│   ├── filtered/                     # Post-filtrado de calidad
│   │   └── images/
│   │
│   ├── annotations/                  # Imágenes + anotaciones
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/                   # Anotaciones en formato YOLO
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   ├── augmented/                    # Dataset aumentado
│   │   ├── images/
│   │   └── labels/
│   │
│   └── final/                        # Dataset final listo para entrenamiento
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── epp_dataset.yaml          # Configuración de dataset
```

## Consideraciones de Privacidad y Cumplimiento

### GDPR y Ley de Protección de Datos (Chile - Ley 19.628)

**Requisitos legales**:
- Obtener consentimiento informado de trabajadores fotografiados
- Informar sobre el propósito de recolección de datos
- Proporcionar opción de opt-out
- Establecer política de retención de datos

### Formulario de Consentimiento (Recomendado)

Incluir en el proceso:

```
CONSENTIMIENTO PARA USO DE IMAGEN EN SISTEMA DE DETECCIÓN DE EPP

Yo, [NOMBRE], RUT [RUT], autorizo a [EMPRESA] a utilizar mi imagen
capturada en las instalaciones mineras para entrenar un sistema de
inteligencia artificial de detección de equipos de protección personal (EPP).

Entiendo que:
- Las imágenes serán usadas exclusivamente para mejorar la seguridad laboral
- Mi rostro puede ser difuminado para proteger mi identidad
- Puedo revocar este consentimiento en cualquier momento
- Los datos no serán compartidos con terceros sin autorización adicional

Firma: ________________  Fecha: __________
```

### Anonimización (Opcional pero Recomendado)

Si la política de empresa lo requiere:

```bash
# Difuminar rostros automáticamente
python scripts/anonymize_faces.py \
    --input data/filtered/images/ \
    --output data/anonymized/images/ \
    --blur-strength 50
```

**Nota**: Esto requiere implementación adicional con detección facial

### Almacenamiento Seguro

**Mejores prácticas**:
- Almacenar datos en servidores con acceso restringido
- Encriptar datos en reposo y en tránsito
- Mantener logs de acceso a datos
- Establecer política de eliminación después de entrenamiento
- No subir imágenes originales a repositorios públicos

## Checklist de Recolección de Datos

### Antes de Comenzar
- [ ] Obtener aprobaciones de gerencia y departamento de seguridad
- [ ] Preparar formularios de consentimiento
- [ ] Identificar ubicaciones y horarios óptimos de captura
- [ ] Verificar equipamiento (cámara, almacenamiento, batería)
- [ ] Coordinar con supervisores de turno
- [ ] Revisar regulaciones de seguridad para fotógrafos en mina

### Durante la Recolección
- [ ] Capturar variedad de ángulos por cada escena
- [ ] Verificar calidad de imagen in-situ (enfoque, exposición)
- [ ] Etiquetar/organizar por ubicación y condición
- [ ] Mantener registro de metadatos (fecha, hora, ubicación, condiciones)
- [ ] Capturar tanto cumplimiento como violaciones
- [ ] Documentar cualquier particularidad o caso especial

### Después de la Recolección
- [ ] Respaldar datos inmediatamente (copia de seguridad)
- [ ] Ejecutar script de filtrado de calidad
- [ ] Revisar distribución de categorías (cumplimiento vs violaciones)
- [ ] Verificar suficiente diversidad de condiciones
- [ ] Organizar en estructura de directorios recomendada
- [ ] Generar reporte de estadísticas de recolección
- [ ] Archivar material original en almacenamiento seguro

## Consejos para Maximizar Calidad del Dataset

### 1. Balance de Clases
- Evitar desbalance extremo (mínimo 20% de cada categoría crítica)
- Si hay desbalance, usar técnicas de sobremuestreo o aumento de datos
- Priorizar calidad sobre cantidad en clases minoritarias

### 2. Captura de Casos Difíciles
Buscar intencionalmente escenarios desafiantes:
- Oclusiones parciales (persona detrás de objeto)
- Iluminación extrema (contraluces, sombras duras)
- EPP no estándar (cascos de colores inusuales)
- Múltiples personas superpuestas
- Distancias largas (EPP pequeño en imagen)

### 3. Consistencia de Captura
- Usar configuración similar de cámara cuando sea posible
- Mantener resolución consistente
- Evitar cambios drásticos de estilo visual entre batches

### 4. Documentación de Metadata
Registrar para cada imagen/video:
- Fecha y hora de captura
- Ubicación específica (coordenadas GPS si es posible)
- Tipo de mina (subterránea/superficie)
- Condiciones de iluminación
- Condiciones meteorológicas (si aplica)
- ID de cámara/dispositivo
- Nombre del operador

### 5. Validación con Expertos
- Revisar muestra del dataset con supervisores de seguridad
- Confirmar que escenarios de violación son realistas
- Validar que EPP capturado es representativo de operaciones reales

## Próximos Pasos

Una vez completada la recolección de imágenes:

1. **Ejecutar filtrado de calidad**:
```bash
python scripts/collect_images.py --source data/raw/ --output data/filtered/
```

2. **Generar estadísticas de recolección**:
```bash
python scripts/dataset_stats.py --data data/filtered/ --report results/collection_report.html
```

3. **Proceder a anotación**: Ver `docs/GUIA_ANOTACION.md`

4. **Validar anotaciones**:
```bash
python scripts/validate_dataset.py --data data/annotations/ --report results/quality_report.html
```

## Recursos Adicionales

- **GUIA_ANOTACION.md**: Instrucciones detalladas para anotar el dataset
- **dataset_config.yaml**: Configuración de parámetros de recolección
- **Scripts de recolección**: Automatización en `scripts/collect_images.py`

## Contacto y Soporte

Para preguntas o problemas durante la recolección de datos, contactar al equipo de ML/Seguridad de la empresa.

---

**Última actualización**: 2025-10-16
**Versión del documento**: 1.0
