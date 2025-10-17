# Guía de Anotación de Dataset para Detección de EPP

## Introducción

Esta guía establece los estándares y procedimientos para anotar correctamente el dataset de detección de EPP en minería chilena. La calidad de las anotaciones impacta directamente en la precisión del modelo, por lo que es fundamental seguir estas directrices con rigor.

## Definiciones de Clases

### Clase 1: `hardhat` (Casco de Seguridad)

**Definición**: Equipo de protección personal que cubre la parte superior de la cabeza.

**Cuándo anotar**:
- Casco completamente visible (>80% visible)
- Casco parcialmente visible pero claramente identificable (>50% visible)
- Cualquier color: amarillo, blanco, rojo, naranja, azul
- Con o sin accesorios (lámpara frontal, protección auditiva adjunta)

**Cuándo NO anotar**:
- Casco en el suelo o colgado (no siendo usado)
- Menos del 50% visible
- Gorra, sombrero o gorro que NO es un casco de seguridad

**Ejemplos de bounding box**:
```
[OK] CORRECTO:
- Ajustado alrededor del casco, incluyendo visera
- Incluye accesorios adjuntos (lámpara)
- Margen mínimo (2-5 píxeles) alrededor del objeto

[NO] INCORRECTO:
- Incluye parte del cuerpo o fondo
- Demasiado ajustado (corta parte del casco)
- Incluye sombra del casco
```

### Clase 2: `safety_vest` (Chaleco Reflectante)

**Definición**: Chaleco o prenda reflectante diseñada para alta visibilidad.

**Cuándo anotar**:
- Chaleco reflectante (naranja, amarillo, verde)
- Con bandas reflectantes horizontales o verticales
- Parcialmente visible pero identificable (>60% visible)
- Puesto correctamente sobre ropa

**Cuándo NO anotar**:
- Ropa de trabajo normal sin elementos reflectantes
- Chaleco en manos o colgado (no siendo usado)
- Menos del 60% visible
- Solo bandas reflectantes sin chaleco completo

**Casos especiales**:
- Si la persona está de espaldas y solo se ven las bandas reflectantes traseras: **SÍ anotar**
- Si el chaleco está parcialmente cubierto por otra prenda: **Anotar solo la parte visible**

### Clase 3: `no_hardhat` (Cabeza sin Protección)

**Definición**: Cabeza de una persona claramente visible sin casco de seguridad.

**Cuándo anotar**:
- Cabeza completamente descubierta (sin ningún tipo de protección)
- Usando gorra, sombrero o gorro (NO es casco de seguridad)
- Cabello visible sin protección
- Rostro visible sin casco

**Cuándo NO anotar**:
- Persona usando casco (usar clase `hardhat` en su lugar)
- Cabeza no visible o fuera de cuadro
- Solo parte del cabello visible sin contexto facial
- Cabeza completamente en sombra sin detalles visibles

**CRÍTICO**: Esta clase es fundamental para detectar violaciones de seguridad.

### Clase 4: `no_safety_vest` (Torso sin Chaleco)

**Definición**: Torso superior de una persona sin chaleco reflectante.

**Cuándo anotar**:
- Persona con ropa de trabajo normal (sin elementos reflectantes)
- Camiseta, camisa o chaqueta sin chaleco
- Torso visible sin ningún tipo de prenda reflectante
- Con casco pero sin chaleco (violación parcial)

**Cuándo NO anotar**:
- Persona usando chaleco reflectante (usar clase `safety_vest`)
- Torso no visible o fuera de cuadro
- Solo brazos visibles sin torso
- Persona completamente de espaldas sin torso discernible

**Nota**: Esta clase se usa en combinación con `hardhat` para casos donde hay cumplimiento parcial.

### Clase 5: `person` (Persona Completa)

**Definición**: Cuerpo completo o sustancial de una persona visible en la imagen.

**Cuándo anotar**:
- Persona completa visible (cabeza a pies o hasta cintura)
- Más del 70% del cuerpo visible
- Silueta humana claramente identificable
- Independiente del uso o no de EPP

**Cuándo NO anotar**:
- Solo cabeza o solo torso (usar clases específicas de EPP)
- Menos del 70% del cuerpo visible
- Extremidades solas (brazo, pierna) sin cuerpo principal
- Reflejo o sombra de persona

**Relación con otras clases**:
- Esta clase se usa en **CONJUNTO** con clases de EPP
- Una misma persona puede tener anotaciones de `person` + `hardhat` + `safety_vest`
- Permite al modelo entender el contexto completo del trabajador

## Reglas de Anotación de Bounding Boxes

### Tamaño y Ajuste

**Regla de oro: "Ajustado pero no cortado"**

```
Margen recomendado: 2-5 píxeles alrededor del objeto
```

**[OK] Bounding box CORRECTO**:
- Incluye todo el objeto visible
- Margen pequeño pero presente
- Bordes paralelos a los ejes de la imagen
- No incluye objetos adyacentes

**[NO] Bounding box INCORRECTO**:
- Corta parte del objeto (muy ajustado)
- Margen excesivo (incluye fondo o otros objetos)
- Incluye sombra o reflejo del objeto
- Diagonal o rotado (YOLO usa boxes rectangulares alineados)

### Oclusiones Parciales

**Oclusión Ligera (10-30% oculto)**: [OK] **Anotar normalmente**
- Box incluye parte visible y estima parte oculta
- Ejemplo: Persona parcialmente detrás de equipo

**Oclusión Moderada (30-50% oculto)**: [OK] **Anotar con precaución**
- Solo si el objeto es claramente identificable
- Box incluye solo la parte visible
- Ejemplo: Casco parcialmente detrás de columna

**Oclusión Severa (>50% oculto)**: [NO] **NO anotar**
- No hay suficiente información visual
- Riesgo de confusión para el modelo
- Ejemplo: Solo se ve 30% de un casco detrás de maquinaria

### Casos de Múltiples Personas

**Personas Separadas**: Anotar cada persona individualmente
- Cada persona tiene su propio conjunto de anotaciones
- No importa la distancia entre ellas

**Personas Superpuestas**: Anotar todas las visibles
- Persona en primer plano: Anotación completa
- Persona en segundo plano: Solo si >50% visible
- Boxes pueden superponerse (el modelo maneja esto con NMS)

**Grupo Denso (>5 personas)**: Priorizar personas en primer plano
- Anotar personas con EPP claramente visible
- Omitir personas al fondo con EPP no discernible
- Mantener consistencia de criterio en toda la imagen

## Procedimiento Paso a Paso

### Antes de Comenzar
1. Leer esta guía completa
2. Revisar imágenes de ejemplo anotadas
3. Configurar herramienta de anotación (LabelImg, CVAT)
4. Verificar que las clases estén correctamente configuradas

### Para Cada Imagen

**Paso 1: Inspección inicial** (10 segundos)
- Identificar cuántas personas hay en la imagen
- Evaluar calidad de la imagen (nitidez, iluminación)
- Identificar potenciales desafíos (oclusiones, distancia)

**Paso 2: Anotar personas completas** (30 segundos)
- Dibujar bounding box de clase `person` para cada individuo visible
- Priorizar personas en primer plano y media distancia
- Omitir personas en fondo lejano (EPP no discernible)

**Paso 3: Anotar EPP presente** (60 segundos)
- Para cada persona, identificar EPP visible:
  - ¿Usa casco? → Anotar `hardhat`
  - ¿Usa chaleco? → Anotar `safety_vest`
- Ajustar boxes con precisión

**Paso 4: Anotar violaciones** (30 segundos)
- Para cada persona sin EPP:
  - Cabeza visible sin casco? → Anotar `no_hardhat`
  - Torso visible sin chaleco? → Anotar `no_safety_vest`

**Paso 5: Revisión final** (10 segundos)
- Verificar que todas las personas tienen anotaciones
- Confirmar que boxes no se cortan ni son excesivamente grandes
- Verificar que no hay anotaciones duplicadas

**Tiempo total por imagen**: 2-3 minutos (varía según complejidad)

### Control de Calidad

**Auto-revisión** (cada 20 imágenes):
- Revisar últimas 5 imágenes anotadas
- Verificar consistencia de criterio
- Identificar patrones de error personal

**Revisión por pares** (cada 100 imágenes):
- Intercambiar muestra de 10 imágenes con otro anotador
- Comparar criterios y anotaciones
- Discutir discrepancias y ajustar

## Casos Especiales y Situaciones Difíciles

### Caso 1: Persona de Espaldas

**Situación**: Solo se ve la parte trasera del trabajador

**Reglas**:
- [OK] Anotar `person` si el cuerpo es claramente visible
- [OK] Anotar `safety_vest` si se ven bandas reflectantes traseras
- [?] `hardhat`: Solo si el casco es visible desde atrás (parte trasera del casco)
- [NO] NO anotar `no_hardhat` (no se puede confirmar ausencia)

**Razón**: El modelo debe aprender a detectar EPP desde todos los ángulos

### Caso 2: Iluminación Extrema

**Situación**: Contraluz, sombras duras, sobre-exposición

**Reglas**:
- Si el EPP es visible pero con dificultad: [OK] **Anotar normalmente**
- Si el EPP es completamente irreconocible: [NO] **Omitir la imagen** (marcar como baja calidad)
- Si solo parte de la persona es visible (resto en sombra): [OK] **Anotar solo la parte visible**

### Caso 3: Equipo en el Suelo o Colgado

**Situación**: Casco en el suelo, chaleco colgado en perchero

**Reglas**:
- [NO] **NO anotar** EPP que no está siendo usado
- Solo anotar EPP que está siendo portado por una persona
- Excepción: Si es ambiguo (persona agachada y casco cerca), anotar con precaución

**Razón**: El objetivo es detectar uso de EPP, no presencia de objetos

### Caso 4: Casco Parcialmente Visible

**Situación**: Solo se ve el borde del casco

**Reglas**:
- **>50% visible**: [OK] Anotar como `hardhat`
- **30-50% visible**: [OK] Anotar si es inequívocamente un casco
- **<30% visible**: [NO] NO anotar

### Caso 5: EPP No Estándar

**Situación**: Casco de color inusual (rosa, morado), chaleco de diseño diferente

**Reglas**:
- [OK] **Anotar normalmente** si cumple la función de protección
- No importa el color o diseño específico
- Si no está seguro: Consultar con supervisor o experto en seguridad

### Caso 6: Maniquíes, Posters o Imágenes de Personas

**Situación**: Foto de un poster de seguridad, maniquí de entrenamiento

**Reglas**:
- [NO] **NO anotar** representaciones de personas (solo personas reales)
- Excepción: Entrenar modelo para señalética requiere dataset separado

### Caso 7: Niños o Visitantes

**Situación**: Personal no operativo en sitio (visitas guiadas, inspectores)

**Reglas**:
- [OK] **Anotar normalmente** - las reglas de EPP aplican a todos
- Incluye: Visitantes, contratistas, gerencia
- El modelo debe detectar violaciones independiente del rol

### Caso 8: Reflejo en Vidrio o Espejo

**Situación**: Persona reflejada en ventana de vehículo o espejo

**Reglas**:
- [NO] **NO anotar** reflejos
- [OK] Anotar solo personas reales físicamente presentes
- Evita duplicación de detecciones

## Errores Comunes y Cómo Evitarlos

### Error 1: Box Demasiado Ajustado
**Síntoma**: Casco cortado en la parte superior, chaleco cortado a los lados

**Solución**: Dejar margen de 2-5 píxeles alrededor del objeto

### Error 2: Box Demasiado Grande
**Síntoma**: Box de casco incluye hombros, box de chaleco incluye piernas

**Solución**: Ajustar a los límites visuales del objeto específico

### Error 3: Inconsistencia en Oclusiones
**Síntoma**: A veces anotar objetos 40% ocultos, a veces no

**Solución**: Seguir regla estricta (>50% visible = anotar, <50% = omitir)

### Error 4: Confundir `no_hardhat` con Ausencia
**Síntoma**: Anotar `no_hardhat` cuando la cabeza no es visible

**Solución**: Solo anotar `no_hardhat` cuando la cabeza ES visible Y NO tiene casco

### Error 5: Omitir Personas en Segundo Plano
**Síntoma**: Solo anotar persona en primer plano, ignorar otras

**Solución**: Anotar TODAS las personas visibles con EPP discernible

### Error 6: Anotaciones Duplicadas
**Síntoma**: Misma persona anotada dos veces por accidente

**Solución**: Revisión final antes de guardar, usar función de preview de herramienta

### Error 7: Clases Incorrectas
**Síntoma**: Etiquetar `hardhat` cuando debería ser `no_hardhat`

**Solución**: Doble verificación antes de guardar, revisar lista de clases

## Herramientas de Anotación Recomendadas

### Opción 1: LabelImg (Desktop, Local)

**Ventajas**:
- Gratuito y open-source
- Interfaz simple y rápida
- Exporta directamente a formato YOLO
- No requiere conexión a internet

**Instalación**:
```bash
pip install labelImg
labelImg
```

**Uso**:
1. Abrir directorio con imágenes
2. Seleccionar clase actual
3. Dibujar bounding box (w + arrastrar mouse)
4. Guardar (Ctrl+S)
5. Siguiente imagen (D)

### Opción 2: CVAT (Web, Colaborativo)

**Ventajas**:
- Colaboración en equipo
- Control de progreso centralizado
- Interpolación para videos
- Revisión y validación integrada

**Instalación** (servidor local):
```bash
python scripts/setup_cvat_project.py --name "EPP Minería Chile"
```

**Uso**:
1. Subir imágenes al proyecto
2. Configurar clases (hardhat, safety_vest, etc.)
3. Asignar tareas a anotadores
4. Exportar en formato YOLO

**Recomendación**: CVAT para equipos >2 personas, LabelImg para anotación individual

### Opción 3: Label Studio (Web, Empresarial)

**Ventajas**:
- Interfaz moderna
- Machine learning assisted labeling
- Gestión avanzada de proyectos
- Analíticas de progreso

**Desventajas**:
- Configuración más compleja
- Puede requerir servidor

## Formato de Salida (YOLO)

Las anotaciones se guardan en formato YOLO (.txt):

```
<class_id> <x_center> <y_center> <width> <height>
```

**Ejemplo** (`image_001.txt`):
```
0 0.512 0.324 0.145 0.198    # hardhat
1 0.508 0.612 0.234 0.456    # safety_vest
4 0.510 0.520 0.298 0.890    # person
```

**Coordenadas normalizadas** (0.0 - 1.0):
- `x_center`: Coordenada X del centro del box / ancho de imagen
- `y_center`: Coordenada Y del centro del box / alto de imagen
- `width`: Ancho del box / ancho de imagen
- `height`: Alto del box / alto de imagen

**IDs de clase** (configurados en `epp_dataset.yaml`):
```yaml
names:
  0: hardhat
  1: safety_vest
  2: no_hardhat
  3: no_safety_vest
  4: person
```

**IMPORTANTE**: Las herramientas manejan esto automáticamente, no editar archivos .txt manualmente

## Métricas de Productividad

### Anotador Principiante
- Velocidad: 15-20 imágenes/hora
- Precisión inicial: 70-80%
- Requiere supervisión y revisión frecuente

### Anotador Experimentado
- Velocidad: 25-35 imágenes/hora
- Precisión: 90-95%
- Revisión periódica

### Objetivo de Calidad
- **Inter-annotator agreement (IAA)**: >85% (medido con IoU >0.7)
- **Tasa de error**: <5% en revisión por pares
- **Cobertura**: >98% de personas anotadas en cada imagen

## Validación y Revisión

### Auto-Validación con Script

Después de anotar un batch, ejecutar:

```bash
python scripts/validate_dataset.py \
    --data data/annotations/ \
    --report results/validation_report.html
```

**El script detecta**:
- Bounding boxes fuera de rango (coordenadas >1.0 o <0.0)
- Boxes demasiado pequeños (<10 píxeles)
- Boxes demasiado grandes (>90% de imagen)
- Imágenes sin anotaciones
- Clases desbalanceadas

### Revisión Manual

**Checklist de revisión** (10% de imágenes anotadas):
- [ ] Todas las personas tienen anotación de `person`
- [ ] Todo EPP visible está anotado
- [ ] Todas las violaciones (no_hardhat, no_safety_vest) están marcadas
- [ ] Boxes están bien ajustados (ni cortados ni excesivos)
- [ ] No hay anotaciones duplicadas
- [ ] Consistencia de criterio entre imágenes similares

## Preguntas Frecuentes (FAQ)

**P: ¿Anotar persona si solo se ve de cuello para arriba?**
R: Sí, anotar `person` + EPP correspondiente (o violación). El contexto del cuerpo ayuda al modelo.

**P: ¿Qué hacer si no estoy seguro si es un casco o una gorra?**
R: Principio de precaución: Si parece protección industrial (duro, con forma de casco), anotar como `hardhat`. Si es tela/blando, considerar como `no_hardhat`.

**P: ¿Anotar casco si está mal puesto (hacia atrás, ladeado)?**
R: Sí, anotar como `hardhat`. El modelo debe detectar uso incorrecto también (puede ser útil para alertas futuras).

**P: ¿Qué hacer con imágenes extremadamente difíciles?**
R: Marcar como "difícil" o "revisar" y consultar con supervisor. Si >50% de la imagen es inutilizable, considerar descartarla.

**P: ¿Cuántas anotaciones debe tener una imagen típica?**
R: Depende del contenido. Ejemplo:
- 1 persona con EPP completo: 3 anotaciones (person + hardhat + safety_vest)
- 2 personas, una sin casco: 5 anotaciones total
- No hay mínimo o máximo estricto

**P: ¿Anotar partes del cuerpo (brazos, piernas) por separado?**
R: No. Solo anotar el cuerpo principal con `person` y EPP en torso/cabeza.

## Recursos de Soporte

### Ejemplos Visuales

Consultar directorio `docs/annotation_examples/` para:
- `ejemplos_correctos/`: Anotaciones bien hechas
- `ejemplos_incorrectos/`: Errores comunes marcados
- `casos_especiales/`: Situaciones difíciles resueltas

### Canal de Consultas

Para dudas durante anotación:
- Crear issue en GitHub con etiqueta `annotation-question`
- Incluir captura de pantalla de la imagen problemática
- Describir la duda específica

### Sesiones de Calibración

Recomendado antes de iniciar anotación masiva:
- Sesión 1 (2 horas): Anotar 20 imágenes, revisar en grupo, calibrar criterios
- Sesión 2 (1 hora): Anotar 20 imágenes más, validar consistencia
- Inicio de producción: Proceder con anotación completa

## Actualización de Esta Guía

Esta guía es un documento vivo. Si durante la anotación se identifican:
- Casos no cubiertos
- Ambigüedades en las reglas
- Errores sistemáticos

Por favor reportar para actualizar el documento y mejorar calidad del dataset.

---

**Última actualización**: 2025-10-16
**Versión del documento**: 1.0
**Autores**: Equipo EPP Detector
