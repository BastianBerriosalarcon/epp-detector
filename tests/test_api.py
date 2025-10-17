"""
Tests para endpoints de la API REST.

Suite de tests que verifica el comportamiento de todos los endpoints
de FastAPI, incluyendo validaciones, responses y manejo de errores.
"""

import io
from unittest.mock import patch, Mock

import pytest
from fastapi.testclient import TestClient
from fastapi import status


# ============================================================================
# Tests de endpoint raíz
# ============================================================================


def test_root_endpoint(client: TestClient):
    """
    Test del endpoint raíz (/).

    Verifica que retorne información básica de la API y enlaces
    a documentación.

    Args:
        client: Cliente de prueba FastAPI
    """
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert data["docs"] == "/docs"


# ============================================================================
# Tests de /health
# ============================================================================


def test_health_endpoint_success(client: TestClient):
    """
    Test de health check con servicio saludable.

    Verifica que /health retorne status 200 y estructura correcta
    cuando el servicio está operativo.
    """
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert "timestamp" in data

    # Version debe ser string no vacío
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


def test_health_endpoint_model_not_loaded(client: TestClient):
    """
    Test de health check cuando modelo no está cargado.

    Debería retornar status "degraded" pero HTTP 200.

    TODO: Mockear detector como None para simular modelo no cargado
    TODO: Verificar que status sea "degraded"
    """
    # TODO: Implementar mock de detector = None
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Si modelo no está cargado, model_loaded debe ser False
    # assert data["model_loaded"] is False
    # assert data["status"] == "degraded"


# ============================================================================
# Tests de /predict
# ============================================================================


def test_predict_endpoint_success(client_with_mock_model: TestClient, sample_image_bytes: bytes):
    """
    Test de predicción exitosa con imagen válida.

    Verifica que el endpoint /predict procese correctamente una imagen
    y retorne detecciones en el formato esperado.

    Args:
        client_with_mock_model: Cliente con detector mockeado
        sample_image_bytes: Imagen de prueba

    TODO: Generar imagen real para test más realista
    TODO: Verificar que detections tengan estructura correcta
    """
    # Simular upload de archivo
    files = {"file": ("test_image.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    response = client_with_mock_model.post("/predict", files=files)

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    # Verificar estructura de respuesta
    assert "success" in data
    assert data["success"] is True
    assert "detections" in data
    assert "inference_time_ms" in data
    assert "image_size" in data
    assert "total_detections" in data
    assert "epp_compliant" in data

    # Verificar que detections sea lista
    assert isinstance(data["detections"], list)

    # Si hay detecciones, verificar estructura
    if len(data["detections"]) > 0:
        detection = data["detections"][0]
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection


def test_predict_endpoint_no_file(client: TestClient):
    """
    Test de predicción sin archivo.

    Verifica que retorne error 422 cuando no se envía imagen.
    """
    response = client.post("/predict")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_endpoint_invalid_file_format(client: TestClient):
    """
    Test de predicción con formato inválido.

    Verifica que rechace archivos que no son imágenes.

    TODO: Implementar validación real en API
    TODO: Verificar mensaje de error específico
    """
    # Archivo de texto en lugar de imagen
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}

    response = client.post("/predict", files=files)

    # TODO: Cuando se implemente validación, debe retornar 400
    # assert response.status_code == status.HTTP_400_BAD_REQUEST
    # data = response.json()
    # assert "error" in data


def test_predict_endpoint_large_image(client: TestClient, large_image_bytes: bytes):
    """
    Test de predicción con imagen muy grande.

    Verifica que rechace imágenes que excedan MAX_IMAGE_SIZE_MB.

    TODO: Implementar validación de tamaño en API
    """
    files = {"file": ("large.jpg", io.BytesIO(large_image_bytes), "image/jpeg")}

    response = client.post("/predict", files=files)

    # TODO: Cuando se implemente validación, debe retornar 413
    # assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


def test_predict_endpoint_model_not_available(client: TestClient, sample_image_bytes: bytes):
    """
    Test de predicción cuando modelo no está disponible.

    Verifica que retorne error 503 si el detector es None.

    TODO: Mockear detector como None
    """
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # TODO: Mock detector = None
    # with patch("api.main.detector", None):
    #     response = client.post("/predict", files=files)
    #     assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_predict_endpoint_with_violations(client: TestClient, sample_image_bytes: bytes):
    """
    Test de predicción que detecta violaciones de EPP.

    Verifica que epp_compliant sea False cuando se detecta
    ausencia de EPP obligatorio.

    TODO: Mockear detector para retornar violaciones
    TODO: Verificar estructura de violaciones en response
    """
    # TODO: Mock detector con violaciones (sin_casco, sin_chaleco)
    pass


def test_predict_endpoint_multiple_detections(client: TestClient, sample_image_bytes: bytes):
    """
    Test de predicción con múltiples personas/EPP.

    Verifica manejo correcto de múltiples detecciones.

    TODO: Mockear múltiples detecciones
    TODO: Verificar ordenamiento por confianza
    """
    pass


# ============================================================================
# Tests de /metrics
# ============================================================================


def test_metrics_endpoint(client: TestClient):
    """
    Test del endpoint de métricas.

    Verifica que retorne estadísticas de uso del servicio.
    """
    response = client.get("/metrics")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "requests_total" in data
    assert "avg_latency_ms" in data
    assert "uptime_seconds" in data
    assert "model_loaded" in data

    # Verificar tipos de datos
    assert isinstance(data["requests_total"], int)
    assert isinstance(data["avg_latency_ms"], (int, float))
    assert isinstance(data["uptime_seconds"], (int, float))
    assert isinstance(data["model_loaded"], bool)

    # Métricas deben ser no negativas
    assert data["requests_total"] >= 0
    assert data["avg_latency_ms"] >= 0
    assert data["uptime_seconds"] >= 0


def test_metrics_increment_after_prediction(
    client_with_mock_model: TestClient, sample_image_bytes: bytes
):
    """
    Test de incremento de métricas después de predicción.

    Verifica que request_count aumente después de cada /predict.

    TODO: Resetear métricas antes del test
    TODO: Verificar incremento exacto de requests_total
    """
    # Obtener métricas iniciales
    initial_response = client_with_mock_model.get("/metrics")
    initial_data = initial_response.json()
    initial_count = initial_data["requests_total"]

    # Hacer predicción
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    client_with_mock_model.post("/predict", files=files)

    # Obtener métricas después
    final_response = client_with_mock_model.get("/metrics")
    final_data = final_response.json()
    final_count = final_data["requests_total"]

    # TODO: Verificar incremento (requiere reset de métricas)
    # assert final_count == initial_count + 1


# ============================================================================
# Tests de /info
# ============================================================================


def test_info_endpoint(client: TestClient):
    """
    Test del endpoint de información del modelo.

    Verifica que retorne metadata del modelo y configuración.
    """
    response = client.get("/info")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "model_version" in data
    assert "model_type" in data
    assert "classes" in data
    assert "confidence_threshold" in data
    assert "dataset_info" in data

    # Verificar que classes sea diccionario
    assert isinstance(data["classes"], dict)

    # Verificar que confidence_threshold esté en rango válido
    assert 0.0 <= data["confidence_threshold"] <= 1.0


def test_info_endpoint_classes_structure(client: TestClient):
    """
    Test de estructura de clases en /info.

    Verifica que las clases EPP estén correctamente definidas.
    """
    response = client.get("/info")
    assert response.status_code == 200

    data = response.json()

    classes = data["classes"]

    # Verificar que sea un diccionario con IDs numéricos
    assert isinstance(classes, dict)

    # Verificar clases esperadas del dataset (nombres en INGLÉS)
    expected_classes = ["hardhat", "head", "person"]

    classes_values = list(classes.values())
    for expected_class in expected_classes:
        assert (
            expected_class in classes_values
        ), f"Clase esperada '{expected_class}' no encontrada en modelo"


# ============================================================================
# Tests de documentación
# ============================================================================


def test_openapi_docs_available(client: TestClient):
    """
    Test de disponibilidad de OpenAPI docs.

    Verifica que /docs esté accesible.
    """
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK


def test_redoc_available(client: TestClient):
    """
    Test de disponibilidad de ReDoc.

    Verifica que /redoc esté accesible.
    """
    response = client.get("/redoc")
    assert response.status_code == status.HTTP_200_OK


def test_openapi_json_available(client: TestClient):
    """
    Test de disponibilidad de OpenAPI JSON schema.

    Verifica que /openapi.json esté accesible y sea JSON válido.
    """
    response = client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data


# ============================================================================
# Tests de manejo de errores
# ============================================================================


def test_404_not_found(client: TestClient):
    """
    Test de endpoint inexistente.

    Verifica que retorne 404 para rutas no definidas.
    """
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_method_not_allowed(client: TestClient):
    """
    Test de método HTTP no permitido.

    Verifica que retorne 405 cuando se usa método incorrecto.
    """
    # /health solo acepta GET
    response = client.post("/health")
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


# ============================================================================
# Tests de CORS (si está habilitado)
# ============================================================================


def test_cors_headers_present(client: TestClient):
    """
    Test de headers CORS.

    Verifica que headers CORS estén presentes si está habilitado.

    TODO: Configurar CORS en FastAPI
    TODO: Verificar headers Access-Control-Allow-Origin
    """
    pass


# ============================================================================
# Tests de performance
# ============================================================================


@pytest.mark.slow
def test_predict_endpoint_latency(client_with_mock_model: TestClient, sample_image_bytes: bytes):
    """
    Test de latencia del endpoint /predict.

    Verifica que inference_time_ms esté dentro de límites aceptables.

    TODO: Definir SLA de latencia (target <100ms CPU)
    TODO: Agregar benchmark con diferentes tamaños de imagen
    """
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client_with_mock_model.post("/predict", files=files)

    data = response.json()
    latency = data["inference_time_ms"]

    # TODO: Definir threshold realista
    # assert latency < 100.0, f"Latencia muy alta: {latency}ms"


@pytest.mark.slow
def test_concurrent_requests(client_with_mock_model: TestClient, sample_image_bytes: bytes):
    """
    Test de requests concurrentes.

    Verifica que la API maneje múltiples requests simultáneos.

    TODO: Implementar test con ThreadPoolExecutor
    TODO: Verificar que no haya race conditions en métricas
    """
    pass


# ============================================================================
# Tests de integración
# ============================================================================


@pytest.mark.integration
def test_full_prediction_workflow(client: TestClient):
    """
    Test de flujo completo end-to-end.

    Simula workflow real: health check → predict → metrics.

    TODO: Implementar con modelo real (no mock)
    TODO: Usar imagen real de minería chilena
    """
    # 1. Verificar health
    health_response = client.get("/health")
    assert health_response.status_code == status.HTTP_200_OK

    # 2. Hacer predicción
    # TODO: Implementar con imagen real

    # 3. Verificar métricas
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == status.HTTP_200_OK


# ============================================================================
# Parametrized tests
# ============================================================================


@pytest.mark.parametrize(
    "endpoint",
    ["/", "/health", "/metrics", "/info", "/docs", "/redoc"],
)
def test_all_get_endpoints_respond(client: TestClient, endpoint: str):
    """
    Test parametrizado de todos los endpoints GET.

    Verifica que todos los endpoints GET respondan 200.

    Args:
        client: Cliente de prueba
        endpoint: Ruta del endpoint a testear
    """
    response = client.get(endpoint)
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.parametrize(
    "invalid_extension",
    ["test.txt", "test.pdf", "test.docx", "test.exe"],
)
def test_predict_rejects_invalid_extensions(client: TestClient, invalid_extension: str):
    """
    Test parametrizado de rechazo de extensiones inválidas.

    Args:
        client: Cliente de prueba
        invalid_extension: Nombre de archivo con extensión inválida

    TODO: Implementar validación de extensión en API
    """
    files = {"file": (invalid_extension, io.BytesIO(b"fake data"), "application/octet-stream")}

    response = client.post("/predict", files=files)

    # TODO: Cuando se implemente validación
    # assert response.status_code == status.HTTP_400_BAD_REQUEST
