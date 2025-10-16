"""
Setup configuration for epp-detector package.

This allows installation of the package and its dependencies.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

# Read dev requirements
dev_requirements_file = Path(__file__).parent / "requirements-dev.txt"
dev_requirements = []
if dev_requirements_file.exists():
    dev_requirements = dev_requirements_file.read_text(encoding="utf-8").strip().split("\n")
    dev_requirements = [
        req.strip()
        for req in dev_requirements
        if req.strip() and not req.startswith("#") and not req.startswith("-r")
    ]

setup(
    name="epp-detector",
    version="0.1.0",
    author="Bastian Berrios",
    author_email="",  # TODO: Add email if desired
    description="API REST para detección de EPP en minería chilena",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/epp-detector",  # TODO: Update with actual repo
    project_urls={
        "Bug Tracker": "https://github.com/tu-usuario/epp-detector/issues",
        "Documentation": "https://github.com/tu-usuario/epp-detector/tree/main/docs",
        "Source Code": "https://github.com/tu-usuario/epp-detector",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "httpx>=0.25.1",
        ],
        "lint": [
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "epp-detector=api.main:main",  # TODO: Create main() function in api/main.py if needed
        ],
    },
    include_package_data=True,
    package_data={
        "api": ["*.yaml", "*.json"],
        "": ["README.md", "LICENSE"],
    },
    zip_safe=False,
    keywords=[
        "computer-vision",
        "object-detection",
        "yolov8",
        "ppe-detection",
        "mining-safety",
        "fastapi",
        "deep-learning",
    ],
)
