# Tests - EPP Detector

Suite de tests para el proyecto EPP Detector.

## Estructura

```
tests/
├── __init__.py         # Documentación del paquete de tests
├── conftest.py         # Fixtures compartidos y configuración pytest
├── test_api.py         # Tests de endpoints FastAPI
├── test_model.py       # Tests de EPPDetector (inferencia)
├── test_utils.py       # Tests de funciones helper
├── pytest.ini          # Configuración de pytest
└── README.md           # Este archivo
```

## Ejecutar tests

### Todos los tests

```bash
pytest tests/
```

### Tests específicos

```bash
# Un archivo
pytest tests/test_api.py

# Una función específica
pytest tests/test_api.py::test_health_endpoint

# Tests por marker
pytest -m unit          # Solo tests unitarios
pytest -m "not slow"    # Excluir tests lentos
pytest -m integration   # Solo tests de integración
```

### Con coverage

```bash
# Coverage en terminal
pytest --cov=api --cov-report=term-missing

# Coverage en HTML
pytest --cov=api --cov-report=html
# Abrir htmlcov/index.html
```

### Verbose output

```bash
pytest -v              # Verbose
pytest -vv             # Extra verbose
pytest -s              # Mostrar prints
pytest --tb=short      # Traceback corto
```

### Tests en paralelo

```bash
# Instalar pytest-xdist primero
pip install pytest-xdist

# Ejecutar en paralelo
pytest -n auto         # Auto-detectar CPUs
pytest -n 4            # 4 workers
```

## Tipos de tests

### Unit Tests

Tests rápidos (<0.1s) que prueban funciones individuales aisladas.
Usan mocks para dependencias externas.

```bash
pytest -m unit
```

### Integration Tests

Tests que prueban integración entre componentes.
Pueden requerir modelo real cargado.

```bash
pytest -m integration
```

### Slow Tests

Tests que toman >1s (inferencia real, benchmarks).
Útil excluirlos durante desarrollo.

```bash
# Excluir tests lentos
pytest -m "not slow"
```

### GPU Tests

Tests que requieren GPU disponible.

```bash
pytest -m gpu
```

## Fixtures disponibles

### FastAPI

- `client`: TestClient básico
- `client_with_mock_model`: Cliente con modelo mockeado

### Datos de prueba

- `sample_image_bytes`: Imagen dummy (bytes)
- `sample_image_file`: Imagen como file object
- `large_image_bytes`: Imagen >10MB
- `invalid_image_bytes`: Datos inválidos
- `sample_detection`: Detección individual
- `sample_detections`: Lista de detecciones

### Modelo

- `mock_model`: Mock de EPPDetector
- `mock_model_no_detections`: Mock sin detecciones
- `mock_model_with_violations`: Mock con violaciones
- `temp_model_path`: Path temporal a modelo

### Configuración

- `test_config`: Dict con configuración de test

## Cobertura objetivo

| Módulo | Target | Actual | Status |
|--------|--------|--------|--------|
| `api/main.py` | 80% | TBD | 🔴 |
| `api/model.py` | 80% | TBD | 🔴 |
| `api/utils.py` | 85% | TBD | 🔴 |
| `api/config.py` | 70% | TBD | 🔴 |
| **Total** | **80%** | **TBD** | 🔴 |

## Estado de implementación

### ✅ Completado

- Estructura base de tests
- Fixtures compartidos (conftest.py)
- Tests básicos de endpoints
- Configuración de pytest
- Markers personalizados

### 🟡 En progreso

- Tests de `test_api.py` (5/20 implementados)
- Tests de `test_model.py` (0/25 skipped)
- Tests de `test_utils.py` (1/30 implementados)

### 🔴 Pendiente

- Implementar tests skipped
- Generar imágenes reales para tests
- Agregar tests de performance
- Agregar tests de memory leaks
- Tests end-to-end con modelo real

## TODOs por prioridad

### Alta

- [ ] Implementar generación de imágenes dummy (PIL)
- [ ] Descomentar tests cuando se implemente código en `api/`
- [ ] Agregar tests para todos los endpoints
- [ ] Implementar reset de métricas globales

### Media

- [ ] Tests de concurrent requests
- [ ] Tests de video inference
- [ ] Tests de batch inference
- [ ] Benchmarks de latencia

### Baja

- [ ] Tests de memory profiling
- [ ] Tests con diferentes modelos (YOLOv8s, YOLOv8m)
- [ ] Tests de cuantización (INT8, FP16)

## Comandos útiles

```bash
# Limpiar cache de pytest
pytest --cache-clear

# Mostrar fixtures disponibles
pytest --fixtures

# Mostrar markers
pytest --markers

# Ejecutar último test que falló
pytest --lf

# Ejecutar tests que fallaron + algunos más
pytest --ff

# Modo verbose con traceback completo
pytest -vv --tb=long

# Stop en el primer error
pytest -x

# Stop después de N errores
pytest --maxfail=3

# Modo watch (requiere pytest-watch)
ptw -- -v
```

## Integración con CI/CD

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=api --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Debugging tests

### Con pdb

```bash
# Abrir debugger en errores
pytest --pdb

# Abrir debugger en inicio de test
# Agregar breakpoint() en código
pytest -s
```

### Con VS Code

Agregar a `.vscode/launch.json`:

```json
{
  "name": "Pytest: Current File",
  "type": "python",
  "request": "launch",
  "module": "pytest",
  "args": ["${file}", "-v"],
  "console": "integratedTerminal"
}
```

## Buenas prácticas

1. **Nombrar tests descriptivamente**: `test_predict_rejects_invalid_format`
2. **Un assert por concepto**: Evitar múltiples asserts no relacionados
3. **Usar fixtures**: Reutilizar setup común
4. **Marcar tests lentos**: `@pytest.mark.slow`
5. **Mockear dependencias externas**: No cargar modelo real en tests unitarios
6. **Documentar con docstrings**: Explicar qué y por qué se testea
7. **Tests independientes**: No depender de orden de ejecución
8. **Cleanup**: Usar fixtures con yield para cleanup

## Referencias

- [Pytest docs](https://docs.pytest.org/)
- [Pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Última actualización**: Iteración 4
**Tests implementados**: 6 / ~75 total
**Coverage actual**: TBD (target: 80%)
