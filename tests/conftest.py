"""
Configuración de pytest y fixtures compartidos.

Este módulo define fixtures reutilizables para todos los tests,
incluyendo clientes de API, datos de prueba y mocks.
"""

import io
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model import EPPDetector

# TODO: Descomentar cuando se implementen
# import numpy as np
# from PIL import Image


# ============================================================================
# Fixtures de FastAPI
# ============================================================================


@pytest.fixture
def client() -> TestClient:
    """
    Cliente de prueba para FastAPI.

    Permite hacer requests HTTP a la API sin levantar servidor.
    Usa TestClient de Starlette que simula requests reales.

    Returns:
        TestClient configurado con la app FastAPI

    Example:
        >>> def test_health(client):
        ...     response = client.get("/health")
        ...     assert response.status_code == 200
    """
    return TestClient(app)


@pytest.fixture
def client_with_mock_model(client: TestClient) -> Generator[TestClient, None, None]:
    """
    Cliente con modelo mockeado para evitar carga real.

    Útil para tests que no requieren inferencia real pero
    necesitan que el modelo esté "cargado".

    Yields:
        TestClient con detector mockeado

    TODO: Implementar mock más sofisticado con comportamiento configurable
    TODO: Mockear diferentes respuestas según input
    """
    mock_detector = Mock(spec=EPPDetector)
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = [
        {
            "class_id": 0,
            "class_name": "hardhat",
            "confidence": 0.92,
            "bbox": [120.5, 50.2, 200.8, 150.3],
            "bbox_norm": [0.251, 0.156, 0.125, 0.156],
        }
    ]

    # Mock validator that returns image dimensions
    mock_validator = Mock()
    mock_validator.validate_all.return_value = (640, 640)  # width, height

    with patch("api.main.app_state") as mock_state:
        mock_state.detector = mock_detector
        mock_state.validator = mock_validator
        mock_state.request_count = 0
        mock_state.total_inference_time = 0.0
        yield client


# ============================================================================
# Fixtures de datos de prueba
# ============================================================================


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Imagen dummy en formato bytes para tests.

    Genera una imagen simple de 640x640 píxeles para testing.
    No requiere archivos externos.

    Returns:
        Bytes de una imagen PNG válida

    TODO: Agregar fixtures para diferentes tamaños (small, medium, large)
    TODO: Agregar fixtures para diferentes formatos (jpg, png, bmp)
    """
    from PIL import Image

    img = Image.new("RGB", (640, 640), color=(73, 109, 137))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes: bytes) -> io.BytesIO:
    """
    Imagen dummy como file object para upload.

    Returns:
        BytesIO que simula un archivo subido

    Example:
        >>> def test_upload(client, sample_image_file):
        ...     response = client.post("/predict", files={"file": sample_image_file})
    """
    return io.BytesIO(sample_image_bytes)


@pytest.fixture
def large_image_bytes() -> bytes:
    """
    Imagen grande (>10MB) para tests de validación.

    Returns:
        Bytes de una imagen que excede el límite

    TODO: Generar imagen real que supere MAX_IMAGE_SIZE_MB
    """
    # TODO: Generar imagen grande
    # img = Image.new("RGB", (4000, 4000), color=(255, 0, 0))
    # img_bytes = io.BytesIO()
    # img.save(img_bytes, format="PNG", quality=100)
    # return img_bytes.getvalue()

    # MOCK: Simular imagen grande (11MB)
    return b"x" * (11 * 1024 * 1024)


@pytest.fixture
def invalid_image_bytes() -> bytes:
    """
    Datos inválidos que no son una imagen.

    Returns:
        Bytes que no corresponden a una imagen válida
    """
    return b"not_an_image_just_random_bytes"


@pytest.fixture
def sample_detection() -> dict:
    """
    Detección de EPP de ejemplo.

    Returns:
        Diccionario con estructura de detección estándar
    """
    return {
        "class_id": 0,
        "class_name": "hardhat",
        "confidence": 0.92,
        "bbox": [120.5, 50.2, 200.8, 150.3],
        "bbox_norm": [0.251, 0.156, 0.125, 0.156],
    }


@pytest.fixture
def sample_detections() -> list:
    """
    Lista de múltiples detecciones para tests.

    Returns:
        Lista con detecciones de diferentes clases EPP
    """
    return [
        {
            "class_id": 0,
            "class_name": "hardhat",
            "confidence": 0.92,
            "bbox": [120.5, 50.2, 200.8, 150.3],
            "bbox_norm": [0.251, 0.156, 0.125, 0.156],
        },
        {
            "class_id": 2,
            "class_name": "person",
            "confidence": 0.87,
            "bbox": [100.0, 180.5, 250.3, 400.7],
            "bbox_norm": [0.273, 0.453, 0.234, 0.344],
        },
    ]


# ============================================================================
# Fixtures de modelo
# ============================================================================


@pytest.fixture
def mock_model() -> Mock:
    """
    Mock de EPPDetector para tests sin modelo real.

    Simula comportamiento del detector sin cargar modelo pesado.
    Útil para tests rápidos de lógica de negocio.

    Returns:
        Mock configurado con métodos de EPPDetector

    Example:
        >>> def test_prediction(mock_model):
        ...     result = mock_model.predict(image)
        ...     assert len(result) > 0

    TODO: Configurar mock más realista con side_effects
    TODO: Mockear diferentes escenarios (sin EPP, EPP completo, errores)
    """
    mock = Mock(spec=EPPDetector)

    # Simular modelo cargado
    mock.is_loaded = True
    mock.model_type = "onnx"
    mock.input_size = 640
    mock.confidence_threshold = 0.5

    # Simular predicción exitosa
    mock.predict.return_value = [
        {
            "class_id": 0,
            "class_name": "hardhat",
            "confidence": 0.92,
            "bbox": [120.5, 50.2, 200.8, 150.3],
            "bbox_norm": [0.251, 0.156, 0.125, 0.156],
        }
    ]

    # Simular get_model_info (usar clases reales del dataset Roboflow)
    mock.get_model_info.return_value = {
        "model_path": "models/yolov8n_epp.onnx",
        "model_type": "onnx",
        "input_size": 640,
        "classes": {
            0: "hardhat",
            1: "mask",
            2: "no_hardhat",
            3: "no_mask",
            4: "no_safety_vest",
            5: "person",
            6: "safety_cone",
            7: "safety_vest",
            8: "machinery",
            9: "vehicle",
        },
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "is_loaded": True,
    }

    return mock


@pytest.fixture
def mock_model_no_detections(mock_model: Mock) -> Mock:
    """
    Mock de detector que no encuentra EPP.

    Returns:
        Mock que retorna lista vacía

    TODO: Implementar
    """
    mock_model.predict.return_value = []
    return mock_model


@pytest.fixture
def mock_model_with_violations(mock_model: Mock) -> Mock:
    """
    Mock de detector que encuentra violaciones de EPP.

    Returns:
        Mock que detecta personas sin casco (no_hardhat - violación)

    TODO: Implementar detecciones de violaciones
    """
    mock_model.predict.return_value = [
        {
            "class_id": 2,
            "class_name": "no_hardhat",
            "confidence": 0.88,
            "bbox": [120.5, 50.2, 200.8, 150.3],
            "bbox_norm": [0.251, 0.156, 0.125, 0.156],
        }
    ]
    return mock_model


# ============================================================================
# Fixtures de configuración
# ============================================================================


@pytest.fixture
def temp_model_path(tmp_path) -> str:
    """
    Path temporal para modelo de prueba.

    Args:
        tmp_path: Fixture de pytest que provee directorio temporal

    Returns:
        String con path a archivo de modelo temporal

    TODO: Crear archivo dummy de modelo para tests
    """
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"fake_model_data")
    return str(model_file)


@pytest.fixture
def test_config() -> dict:
    """
    Configuración de prueba para la aplicación.

    Returns:
        Diccionario con configuración de test

    TODO: Implementar override de settings para tests
    """
    return {
        "model_path": "tests/fixtures/test_model.onnx",
        "confidence_threshold": 0.5,
        "input_size": 640,
        "enable_gpu": False,  # Desactivar GPU en tests
        "debug": True,
    }


# ============================================================================
# Fixtures de cleanup
# ============================================================================


@pytest.fixture(autouse=True)
def reset_metrics():
    """
    Resetea métricas globales antes de cada test.

    Evita que tests interfieran entre sí al modificar
    variables globales de métricas.

    TODO: Implementar reset real de métricas en api.main
    """
    # TODO: Resetear request_count, total_inference_time, etc.
    yield
    # Cleanup después del test si es necesario


# ============================================================================
# Markers personalizados
# ============================================================================

# Registrar markers personalizados en pytest.ini o pyproject.toml
# @pytest.mark.slow: Tests lentos (inferencia real)
# @pytest.mark.integration: Tests de integración
# @pytest.mark.unit: Tests unitarios rápidos
# @pytest.mark.gpu: Tests que requieren GPU


# ============================================================================
# Hooks de pytest
# ============================================================================


def pytest_configure(config):
    """
    Configuración inicial de pytest.

    TODO: Configurar plugins adicionales
    TODO: Configurar coverage settings
    TODO: Configurar parallel execution
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
