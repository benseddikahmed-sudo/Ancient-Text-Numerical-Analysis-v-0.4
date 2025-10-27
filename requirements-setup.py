# =============================================================================
# requirements.txt
# Python dependencies for Ancient Text Numerical Analysis Framework
# =============================================================================

# Core scientific computing
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0
pandas>=1.3.0,<3.0.0

# Visualization
matplotlib>=3.4.0,<4.0.0
seaborn>=0.11.0,<1.0.0

# Optional: Bayesian analysis (recommended for full functionality)
pymc>=5.0.0,<6.0.0
arviz>=0.11.0,<1.0.0

# Optional: Progress bars
tqdm>=4.62.0

# Development dependencies (optional)
pytest>=7.0.0              # Testing
pytest-cov>=3.0.0          # Coverage reporting
black>=22.0.0              # Code formatting
flake8>=4.0.0              # Linting
mypy>=0.950                # Type checking

# Documentation (optional)
sphinx>=4.0.0              # Documentation generation
sphinx-rtd-theme>=1.0.0    # ReadTheDocs theme

# =============================================================================
# setup.py
# Installation script for Ancient Text Numerical Analysis Framework
# =============================================================================

#!/usr/bin/env python3
"""
Setup script for Ancient Text Numerical Analysis Framework

Installation:
    pip install -e .                    # Development mode
    pip install .                       # Standard installation
    pip install .[bayesian]             # With Bayesian support
    pip install .[dev]                  # With development tools
    pip install .[all]                  # Everything
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
    ]

# Optional dependencies
extras_require = {
    'bayesian': [
        'pymc>=5.0.0',
        'arviz>=0.11.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.950',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'progress': [
        'tqdm>=4.62.0',
    ],
}

# Meta dependency groups
extras_require['all'] = list(set(sum(extras_require.values(), [])))
extras_require['full'] = extras_require['bayesian'] + extras_require['progress']

setup(
    name='ancient-text-analysis',
    version='4.0.0',
    
    # Metadata
    author='Ahmed Benseddik',
    author_email='benseddik.ahmed@gmail.com',
    description='Multi-cultural statistical framework for ancient text numerical analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4,
    project_urls={
        'Documentation': https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.readthedocs.io',
        'Source': 'https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4,
        'Bug Reports': https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues',
        'Paper': 'https://doi.org/pending',
    },
    
    # License
    license='MIT',
    
    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    
    # Python version
    python_requires='>=3.9',
    
    # Package discovery
    packages=find_packages(exclude=['tests', 'docs']),
    
    # Dependencies
    install_requires=requirements,
    extras_require=extras_require,
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'ancient-text-analysis=ancient_text_numerical_analysis:main',
            'generate-dsh-figures=generate_dsh_figures:main',
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', 'LICENSE'],
        'data': ['*.txt'],
    },
    
    # Keywords
    keywords=[
        'digital humanities',
        'ancient texts',
        'gematria',
        'isopsephy',
        'abjad',
        'bayesian statistics',
        'computational text analysis',
        'ethical computing',
    ],
    
    # Zip safe
    zip_safe=False,
)

# =============================================================================
# MANIFEST.in
# Specifies additional files to include in distribution
# =============================================================================

include README.md
include LICENSE
include requirements.txt
include setup.py

# Data files
recursive-include data *.txt
recursive-include figures *.png *.pdf *.svg

# Documentation
recursive-include docs *.md *.rst *.txt

# Tests
recursive-include tests *.py

# Exclude development files
global-exclude __pycache__
global-exclude *.py[co]
global-exclude .DS_Store
global-exclude .git*

# =============================================================================
# .gitignore
# Files to exclude from version control
# =============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
docs/_static/
docs/_templates/

# Results and figures (too large for git)
data/results/*.json
data/results/*.txt
figures/supplementary/pdf/
figures/supplementary/png/
figures/supplementary/tiff/
figures/supplementary/svg/

# Keep example files
!data/results/example_results.json
!figures/supplementary/example_figure.png

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# Temporary files
*.log
*.tmp
temp/

# =============================================================================
# Makefile
# Common development tasks
# =============================================================================

.PHONY: help install install-dev test lint format clean docs figures

help:
	@echo "Ancient Text Analysis Framework - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make install-full   Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make lint           Check code style"
	@echo "  make format         Format code with black"
	@echo "  make typecheck      Run mypy type checking"
	@echo ""
	@echo "Analysis:"
	@echo "  make analyze        Run full analysis pipeline"
	@echo "  make figures        Generate all DSH figures"
	@echo "  make demo           Run demonstration"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make clean-all      Remove all generated files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[all]"

# Testing
test:
	python ancient_text_numerical_analysis.py --test
	pytest tests/ -v --cov=. --cov-report=html

# Code quality
lint:
	flake8 *.py --max-line-length=100 --ignore=E501,W503
	pylint *.py --max-line-length=100 || true

format:
	black *.py --line-length=100
	isort *.py

typecheck:
	mypy *.py --ignore-missing-imports

# Analysis
analyze:
	python ancient_text_numerical_analysis.py --data-dir ./data --enable-bayesian --verbose

figures:
	python generate_dsh_figures.py --profile dsh --figures all --verbose

demo:
	python ancient_text_numerical_analysis.py --cross-cultural-demo

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean
	rm -rf data/results/*.json
	rm -rf data/results/*.txt
	rm -rf figures/supplementary/pdf/*
	rm -rf figures/supplementary/png/*
	rm -rf figures/supplementary/tiff/*
	rm -rf venv/
	rm -rf env/

# =============================================================================
# .github/workflows/ci.yml
# GitHub Actions CI/CD pipeline
# =============================================================================

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        python ancient_text_numerical_analysis.py --test
        pytest tests/ -v --cov=.
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
  
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install flake8 black mypy
    
    - name: Lint with flake8
      run: flake8 *.py --max-line-length=100
    
    - name: Check formatting with black
      run: black --check *.py --line-length=100
    
    - name: Type check with mypy
      run: mypy *.py --ignore-missing-imports

# =============================================================================
# docker/Dockerfile
# Docker container for reproducible execution
# =============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="Ahmed Benseddik <benseddik.ahmed@gmail.com>"
LABEL description="Ancient Text Numerical Analysis Framework"
LABEL version="4.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ancient_text_numerical_analysis.py .
COPY generate_dsh_figures.py .
COPY data/ ./data/

# Create output directories
RUN mkdir -p data/results figures/supplementary

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Default command
CMD ["python", "ancient_text_numerical_analysis.py", "--help"]

# Usage:
# docker build -t ancient-text-analysis -f docker/Dockerfile .
# docker run -v $(pwd)/data:/app/data -v $(pwd)/figures:/app/figures ancient-text-analysis \
#     python ancient_text_numerical_analysis.py --data-dir /app/data

# =============================================================================
# docker-compose.yml
# Multi-container orchestration
# =============================================================================

version: '3.8'

services:
  analysis:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./data:/app/data
      - ./figures:/app/figures
    command: python ancient_text_numerical_analysis.py --data-dir /app/data --verbose
    environment:
      - PYTHONUNBUFFERED=1
  
  figures:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./data:/app/data
      - ./figures:/app/figures
    command: python generate_dsh_figures.py --profile dsh --figures all
    depends_on:
      - analysis

# Usage:
# docker-compose up

# =============================================================================
# pytest.ini
# PyTest configuration
# =============================================================================

[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --cov=.
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    bayesian: marks tests requiring PyMC

# =============================================================================
# .editorconfig
# Editor configuration for consistent code style
# =============================================================================

root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.py]
indent_style = space
indent_size = 4
max_line_length = 100

[*.{yml,yaml,json}]
indent_style = space
indent_size = 2

[Makefile]
indent_style = tab

# =============================================================================
# pyproject.toml
# Modern Python project configuration
# =============================================================================

[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "ancient-text-analysis"
version = "4.0.0"
description = "Multi-cultural statistical framework for ancient text numerical analysis"
authors = [{name = "Ahmed Benseddik", email = "benseddik.ahmed@gmail.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
keywords = ["digital-humanities", "gematria", "bayesian-statistics"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
]

[project.optional-dependencies]
bayesian = ["pymc>=5.0.0", "arviz>=0.11.0"]
dev = ["pytest>=7.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"]
all = ["pymc>=5.0.0", "arviz>=0.11.0", "pytest>=7.0.0", "tqdm>=4.62.0"]

[project.urls]
Homepage = https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
Documentation = https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.readthedocs.io"
Repository = https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
"Bug Tracker" =https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues"

[project.scripts]
ancient-text-analysis = "ancient_text_numerical_analysis:main"
generate-dsh-figures = "generate_dsh_figures:main"

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true

[tool.pylint.messages_control]
max-line-length = 100
disable = ["C0103", "C0114", "C0115", "C0116"]

[tool.coverage.run]
source = ["."]
omit = ["*/tests/*", "*/venv/*", "setup.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]