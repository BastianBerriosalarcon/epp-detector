"""
Suite de tests para EPP Detector.

Este paquete contiene tests unitarios, de integración y end-to-end
para todos los módulos del proyecto.

Estructura:
- test_api.py: Tests de endpoints FastAPI
- test_model.py: Tests de EPPDetector (inferencia)
- test_utils.py: Tests de funciones helper
- conftest.py: Fixtures compartidos

Para ejecutar:
    pytest tests/
    pytest tests/ -v
    pytest tests/ --cov=api
    pytest tests/test_api.py::test_health_endpoint
"""
