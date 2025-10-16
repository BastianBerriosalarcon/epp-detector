#!/usr/bin/env bash
# =============================================================================
# Docker Build Script para EPP Detector
# =============================================================================
# Script para automatizar la construcción de imágenes Docker con diferentes
# configuraciones (desarrollo, producción, limpieza).
#
# Uso:
#   ./scripts/docker-build.sh dev      # Build desarrollo
#   ./scripts/docker-build.sh prod     # Build producción
#   ./scripts/docker-build.sh clean    # Limpiar imágenes y containers
#
# Autor: Bastian Berrios
# Fecha: 2025-01-15
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME="epp-detector"
REGISTRY="ghcr.io"  # GitHub Container Registry
IMAGE_NAME="${REGISTRY}/${USER}/${PROJECT_NAME}"

# Get version from git or use default
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
Uso: $0 <comando> [opciones]

Comandos:
  dev       Build imagen de desarrollo
  prod      Build imagen de producción
  clean     Limpiar imágenes y containers
  help      Mostrar esta ayuda

Opciones:
  --no-cache     Build sin usar cache
  --push         Push imagen a registry después de build
  --tag TAG      Tag personalizado (default: latest)

Ejemplos:
  $0 dev
  $0 prod --push
  $0 prod --tag v1.0.0 --push
  $0 clean

EOF
}

# =============================================================================
# Build Functions
# =============================================================================

build_dev() {
    local tag="${1:-latest}"
    local cache_flag="${2:-}"

    log_info "Building development image..."
    log_info "Tag: ${IMAGE_NAME}:${tag}-dev"

    docker build \
        ${cache_flag} \
        -f Dockerfile \
        --target development \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" \
        -t "${IMAGE_NAME}:${tag}-dev" \
        -t "${IMAGE_NAME}:dev" \
        .

    log_success "Development image built successfully"
    log_info "Image: ${IMAGE_NAME}:${tag}-dev"
}

build_prod() {
    local tag="${1:-latest}"
    local cache_flag="${2:-}"

    log_info "Building production image..."
    log_info "Tag: ${IMAGE_NAME}:${tag}"

    docker build \
        ${cache_flag} \
        -f Dockerfile \
        --target production \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" \
        -t "${IMAGE_NAME}:${tag}" \
        -t "${IMAGE_NAME}:latest" \
        .

    log_success "Production image built successfully"
    log_info "Image: ${IMAGE_NAME}:${tag}"
}

push_image() {
    local tag="${1:-latest}"
    local env="${2:-prod}"

    if [ "${env}" == "dev" ]; then
        local full_tag="${IMAGE_NAME}:${tag}-dev"
    else
        local full_tag="${IMAGE_NAME}:${tag}"
    fi

    log_info "Pushing image to registry..."
    log_info "Image: ${full_tag}"

    # Check if logged in
    if ! docker info 2>/dev/null | grep -q "Username"; then
        log_warning "Not logged in to Docker registry"
        log_info "Run: echo \$GITHUB_TOKEN | docker login ${REGISTRY} -u \$GITHUB_USER --password-stdin"
        return 1
    fi

    docker push "${full_tag}"

    # Also push latest tag
    if [ "${env}" == "dev" ]; then
        docker push "${IMAGE_NAME}:dev"
    else
        docker push "${IMAGE_NAME}:latest"
    fi

    log_success "Image pushed successfully"
}

clean_docker() {
    log_warning "Cleaning Docker resources..."

    # Stop and remove containers
    if [ "$(docker ps -aq -f name=${PROJECT_NAME})" ]; then
        log_info "Stopping and removing containers..."
        docker ps -aq -f name="${PROJECT_NAME}" | xargs docker rm -f
    fi

    # Remove images
    if [ "$(docker images -q ${IMAGE_NAME})" ]; then
        log_info "Removing images..."
        docker images -q "${IMAGE_NAME}" | xargs docker rmi -f
    fi

    # Remove dangling images
    if [ "$(docker images -f "dangling=true" -q)" ]; then
        log_info "Removing dangling images..."
        docker images -f "dangling=true" -q | xargs docker rmi -f
    fi

    # Prune build cache
    log_info "Pruning build cache..."
    docker builder prune -f

    log_success "Cleanup completed"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi

    # Parse command
    local command="${1:-}"
    shift || true

    # Parse options
    local no_cache=""
    local push=false
    local tag="latest"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache)
                no_cache="--no-cache"
                shift
                ;;
            --push)
                push=true
                shift
                ;;
            --tag)
                tag="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Execute command
    case "${command}" in
        dev)
            build_dev "${tag}" "${no_cache}"
            if [ "${push}" = true ]; then
                push_image "${tag}" "dev"
            fi
            ;;
        prod)
            build_prod "${tag}" "${no_cache}"
            if [ "${push}" = true ]; then
                push_image "${tag}" "prod"
            fi
            ;;
        clean)
            clean_docker
            ;;
        help|--help|-h)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown command: ${command}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

# Run main if script is executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
