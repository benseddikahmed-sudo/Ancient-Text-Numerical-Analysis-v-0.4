# Contributing to Ancient Text Analysis Framework

Thank you for your interest in contributing to this project! This document provides guidelines for contributing code, documentation, and other improvements.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Background, identity, or experience level
- Discipline (computer science, humanities, linguistics, etc.)
- Technical expertise
- Geographic location

### Expected Behavior

- **Be respectful**: Value diverse perspectives and experiences
- **Be collaborative**: Work together constructively
- **Be professional**: Focus on what's best for the project
- **Be patient**: Especially with newcomers and across disciplines
- **Be culturally sensitive**: Respect the cultural contexts of texts analyzed

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Dismissing concerns about cultural sensitivity
- Any conduct inappropriate for an academic environment

### Reporting

Report violations to: benseddik.ahmed@gmail.com

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of statistical methods
- Familiarity with ancient text traditions (helpful but not required)

### Development Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/ancient-text-analysis.git
cd ancient-text-analysis

# 3. Add upstream remote
git remote add upstream https://github.com/ahmedbenseddik/ancient-text-analysis.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in development mode
pip install -e ".[all]"

# 6. Install pre-commit hooks
pre-commit install

# 7. Run tests to verify setup
pytest tests/ -v
```

---

## Development Workflow

### 1. Choose or Create an Issue

- Browse existing issues: https://github.com/ahmedbenseddik/ancient-text-analysis/issues
- For new features, create an issue first to discuss
- Comment on the issue to indicate you're working on it

### 2. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write code following our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Commit regularly with clear messages

### 4. Keep Your Branch Updated

```bash
# Periodically sync with upstream
git fetch upstream
git rebase upstream/main
```

### 5. Run Tests Locally

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ancient_text_analysis --cov-report=html

# Run specific test file
pytest tests/test_gematria.py -v

# Run only fast tests
pytest tests/ -m "not slow"
```

### 6. Submit Pull Request

See [Submitting Changes](#submitting-changes) below.

---

## Coding Standards

### Python Style

We follow **PEP 8** with some modifications:

```python
# Good: Clear, documented function
def compute_gematria(word: str, system: CulturalSystem = CulturalSystem.HEBREW_STANDARD) -> int:
    """
    Compute numerical value using specified cultural system.
    
    Args:
        word: Input text string
        system: Cultural numerical system to use
    
    Returns:
        Integer numerical value
    
    Examples:
        >>> compute_gematria('בראשית')
        913
    """
    # Implementation
```

### Type Hints

- Use type hints for all function signatures
- Use `typing` module for complex types

```python
from typing import Dict, List, Optional, Tuple

def analyze_text(text: str, divisors: List[int]) -> Dict[str, Any]:
    ...
```

### Docstrings

- Use Google style docstrings
- Include examples for public functions
- Document all parameters and return values

### Naming Conventions

```python
# Classes: PascalCase
class StatisticalAnalyzer:
    pass

# Functions/methods: snake_case
def compute_effect_size():
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_SEED = 42

# Private functions: _leading_underscore
def _internal_helper():
    pass
```

### Code Organization

```python
# 1. Standard library imports
import json
import logging
from pathlib import Path

# 2. Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# 3. Local imports
from ancient_text_analysis import compute_gematria
```

### Linting and Formatting

We use multiple tools (enforced by pre-commit):

```bash
# Auto-format code
black ancient_text_analysis/

# Sort imports
isort ancient_text_analysis/

# Check style
flake8 ancient_text_analysis/

# Type checking
mypy ancient_text_analysis/

# Comprehensive linting
pylint ancient_text_analysis/
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_gematria.py
import pytest
from ancient_text_analysis import compute_gematria, CulturalSystem

class TestGematriaStandard:
    """Test standard Hebrew gematria."""
    
    def test_single_letter(self):
        """Test single letter values."""
        assert compute_gematria('א') == 1
        assert compute_gematria('ת') == 400
    
    def test_known_words(self):
        """Test known word values."""
        assert compute_gematria('בראשית') == 913
    
    @pytest.mark.parametrize("word,expected", [
        ('א', 1),
        ('אב', 3),
        ('אלהים', 86),
    ])
    def test_parametrized(self, word, expected):
        """Parametrized test for multiple cases."""
        assert compute_gematria(word) == expected
```

### Test Types

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test general properties (using Hypothesis)
4. **Regression Tests**: Prevent known bugs from recurring

### Test Coverage

- Aim for >80% code coverage
- 100% coverage for critical functions
- Test edge cases and error conditions

```bash
# Generate coverage report
pytest tests/ --cov=ancient_text_analysis --cov-report=html

# View report
open htmlcov/index.html
```

### Fixtures

Use fixtures for common test data:

```python
@pytest.fixture
def sample_hebrew_text():
    """Sample Hebrew text for testing."""
    return 'בראשיתבראאלהים'

@pytest.fixture
def sample_analysis_config():
    """Sample configuration for testing."""
    return AnalysisConfig(
        data_dir=Path('test_data'),
        random_seed=42
    )

def test_with_fixture(sample_hebrew_text):
    """Test using fixture."""
    result = compute_gematria(sample_hebrew_text)
    assert result > 0
```

---

## Documentation

### Docstring Format

```python
def complex_function(param1: int, param2: str, param3: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Brief one-line description.
    
    More detailed explanation that can span multiple lines.
    Explain the purpose, behavior, and any important notes.
    
    Args:
        param1: Description of param1. Can be multiple lines
            if needed, indented.
        param2: Description of param2.
        param3: Description of param3. Default behavior explained.
    
    Returns:
        Dictionary containing:
            - 'key1': description
            - 'key2': description
    
    Raises:
        ValueError: If param1 is negative.
        TypeError: If param2 is not a string.
    
    Examples:
        >>> result = complex_function(42, "test")
        >>> result['key1']
        'value1'
    
    Note:
        Additional notes about edge cases, performance, etc.
    
    See Also:
        related_function: For related functionality.
    """
```

### Documentation Files

Update relevant documentation files:

- **README.md**: Overview and quick start
- **METHODOLOGY.md**: Detailed methodology
- **API documentation**: Auto-generated from docstrings

### Sphinx Documentation

```bash
# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

---

## Submitting Changes

### Pull Request Process

1. **Prepare Your Changes**
   ```bash
   # Ensure all tests pass
   pytest tests/ -v
   
   # Run linters
   black ancient_text_analysis/
   flake8 ancient_text_analysis/
   mypy ancient_text_analysis/
   
   # Update documentation
   # Update CHANGELOG.md
   ```

2. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Go to GitHub and create PR from your fork
   - Use clear, descriptive title
   - Fill out PR template completely

### PR Template

```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed code
- [ ] Commented complex sections
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Added tests
- [ ] All tests pass

## Related Issues
Closes #123
```

---

## Review Process

### What Reviewers Look For

1. **Correctness**: Does it work as intended?
2. **Testing**: Adequate test coverage?
3. **Style**: Follows coding standards?
4. **Documentation**: Clear and complete?
5. **Cultural Sensitivity**: Respects cultural contexts?
6. **Performance**: No obvious inefficiencies?

### Timeline

- Initial review: Within 3-5 days
- Revisions: As needed
- Merge: After approval from 1+ maintainers

### Addressing Feedback

```bash
# Make requested changes
# Commit changes
git add .
git commit -m "Address review feedback"

# Update PR
git push origin feature/your-feature-name
```

---

## Types of Contributions

### Code Contributions

- **Bug fixes**: Always welcome
- **New features**: Discuss in issue first
- **Performance improvements**: With benchmarks
- **Refactoring**: Maintain backward compatibility

### Non-Code Contributions

- **Documentation**: Improve clarity, add examples
- **Examples**: Jupyter notebooks, use cases
- **Bug reports**: Detailed, reproducible
- **Feature requests**: With use cases
- **Cultural expertise**: Guidance on traditions
- **Translation**: Internationalization

---

## Cultural Sensitivity Guidelines

When working with cultural systems:

1. **Research thoroughly**: Understand the tradition
2. **Consult experts**: Especially for new systems
3. **Document context**: Explain cultural background
4. **Respect tradition**: Don't reduce to mere numbers
5. **Acknowledge limitations**: Be humble about interpretations
6. **Use appropriate language**: Respectful terminology

### Example: Adding New Cultural System

```python
class CulturalSystem(Enum):
    """Enumeration of supported cultural numerical systems."""
    HEBREW_STANDARD = ("hebrew_standard", "Hebrew Gematria", "Jewish tradition")
    # Add with full context
    SYRIAC_ABJAD = ("syriac_abjad", "Syriac Abjad", "Syriac Christian tradition")
    
    def __init__(self, code: str, name: str, tradition: str):
        self.code = code
        self.display_name = name
        self.tradition = tradition  # Cultural context

# Include documentation about:
# - Historical origins
# - Cultural significance
# - Appropriate use cases
# - Community perspectives
```

---

## Questions?

- **GitHub Discussions**: For general questions
- **Issues**: For specific bugs/features
- **Email**: benseddik.ahmed@gmail.com

---

## Recognition

Contributors are recognized in:
- **AUTHORS.md**: All contributors listed
- **Release notes**: Contributions credited
- **Academic citations**: Where appropriate

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing ethical, rigorous digital scholarship! 🙏