#!/usr/bin/env python3
"""
Ancient Text Numerical Analysis Framework – Digital Scholarship Edition v5.0
===========================================================================

A rigorous, reproducible framework for computational analysis of numerical
patterns in ancient texts with comprehensive documentation, validation,
ethical considerations, and enhanced features for digital humanities research.

Publication: Digital Scholarship in the Humanities (DSH)
Author: Ahmed Benseddik <benseddik.ahmed@gmail.com>
Version: 5.0-DSH
Date: 2025-10-31
License: MIT
DOI: 10.5281/zenodo.17487211
Repository: https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4

Citation:
    Benseddik, A. (2025). Ancient Text Numerical Analysis: A Statistical
    Framework with Ethical Considerations. Digital Scholarship in the
    Humanities. DOI: 10.5281/zenodo.17487211

Dependencies:
    Core: numpy>=1.24, scipy>=1.10, pandas>=2.0
    Visualization: matplotlib>=3.7, seaborn>=0.12
    Bayesian: pymc>=5.0, arviz>=0.15
    Performance: numba>=0.57 (optional)
    Testing: pytest>=7.0, hypothesis>=6.0

Documentation: https://ancient-text-analysis.readthedocs.io

New in v5.0:
    - Enhanced ethical guidelines framework
    - TEI XML corpus connector
    - Performance optimizations with caching
    - Extended cross-cultural analysis
    - Improved documentation and examples
    - Automated bias detection
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Protocol, Iterator, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Optional performance enhancement
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range

# Bayesian modeling
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not installed. Install: pip install pymc arviz", UserWarning)

# Configuration
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class DSHFormatter(logging.Formatter):
    """Custom formatter with color coding for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None):
    """
    Configure logging with file and console handlers.
    
    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO)
    log_file : Optional[Path]
        Path to log file. If None, only console logging is used.
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(DSHFormatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger

logger = logging.getLogger(__name__)

# ============================================================================
# ETHICAL GUIDELINES FRAMEWORK (NEW in v5.0)
# ============================================================================

@dataclass
class EthicalGuidelines:
    """
    Ethical considerations framework for cultural text analysis.
    
    This class provides guidelines and validation methods to ensure
    responsible research practices when analyzing cultural artifacts.
    
    Attributes
    ----------
    acknowledge_cultural_context : bool
        Whether cultural context has been acknowledged
    community_consulted : bool
        Whether relevant communities have been consulted
    limitations_documented : bool
        Whether limitations are properly documented
    """
    
    acknowledge_cultural_context: bool = False
    community_consulted: bool = False
    limitations_documented: bool = False
    
    @staticmethod
    def get_guidelines() -> Dict[str, str]:
        """
        Return ethical guidelines for interpreting results.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of ethical considerations
        """
        return {
            'cultural_sensitivity': (
                "Numerical patterns do not imply religious or mystical meanings "
                "without appropriate cultural and historical context. These are "
                "statistical observations that require expert interpretation."
            ),
            'interpretation_limits': (
                "Statistical significance does not prove intentionality or design. "
                "Multiple valid interpretations exist, and quantitative findings "
                "represent only one analytical perspective."
            ),
            'data_transparency': (
                "All source texts, preprocessing steps, and analytical decisions "
                "must be documented and made available for peer review. Raw data "
                "should be preserved alongside processed results."
            ),
            'community_engagement': (
                "Researchers should consider consulting with cultural and religious "
                "communities when publishing findings about their traditions. This "
                "shows respect and can provide invaluable contextual insights."
            ),
            'bias_awareness': (
                "Researchers must acknowledge their own cultural biases and how "
                "these may influence interpretation of results. Diverse research "
                "teams are encouraged."
            ),
            'responsible_dissemination': (
                "Findings should be presented with appropriate caveats and context. "
                "Avoid sensationalized claims or implications beyond what the data "
                "can support."
            ),
        }
    
    @staticmethod
    def generate_ethics_statement() -> str:
        """
        Generate ethics statement for publication.
        
        Returns
        -------
        str
            Formatted ethics statement
        """
        return """
ETHICAL CONSIDERATIONS

This research applies computational methods to cultural texts with full 
awareness that quantitative patterns represent one analytical lens among 
many for understanding these materials. We acknowledge:

1. The sacred and cultural significance of these texts to living traditions
2. That statistical patterns do not imply intentionality without context
3. The limitations of computational approaches to cultural analysis
4. Our responsibility to present findings with humility and appropriate context
5. The importance of transparency in methods and data

We commit to:
- Making all code, data, and methods publicly available
- Documenting all analytical decisions and their rationale
- Acknowledging uncertainties and alternative interpretations
- Respecting the cultural heritage represented in these texts
- Engaging constructively with scholarly and community feedback

This framework is offered as a tool for research, not as definitive 
interpretation of meaning or intent in ancient texts.
"""
    
    @classmethod
    def validate_research_ethics(cls, metadata: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that ethical guidelines have been followed.
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Research metadata to validate
        
        Returns
        -------
        Dict[str, bool]
            Validation results
        """
        checks = {
            'source_documented': 'data_source' in metadata,
            'methods_documented': 'methods' in metadata,
            'limitations_stated': 'limitations' in metadata,
            'cultural_context_acknowledged': 'cultural_context' in metadata,
            'reproducibility_ensured': all(k in metadata for k in ['random_seed', 'timestamp']),
        }
        return checks

# ============================================================================
# REPRODUCIBILITY & VALIDATION
# ============================================================================

@dataclass
class ReproducibilityMetadata:
    """
    Complete metadata for computational reproducibility.
    
    This dataclass captures all information necessary to reproduce
    an analysis, including environment details, versions, and data hashes.
    
    Attributes
    ----------
    timestamp : str
        UTC timestamp of analysis
    python_version : str
        Python version string
    numpy_version : str
        NumPy version
    scipy_version : str
        SciPy version
    random_seed : int
        Random seed for reproducibility
    system_info : Dict[str, str]
        System information (platform, processor, etc.)
    git_commit : Optional[str]
        Git commit hash if available
    dataset_hash : Optional[str]
        SHA-256 hash of input dataset
    
    Examples
    --------
    >>> metadata = ReproducibilityMetadata.capture(seed=42)
    >>> print(metadata.timestamp)
    '2025-10-31 12:00:00 UTC'
    """
    
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    random_seed: int
    system_info: Dict[str, str]
    git_commit: Optional[str] = None
    dataset_hash: Optional[str] = None
    
    @classmethod
    def capture(cls, seed: int = 42, data: Optional[str] = None) -> 'ReproducibilityMetadata':
        """
        Capture current environment metadata.
        
        Parameters
        ----------
        seed : int, default=42
            Random seed to record
        data : Optional[str]
            Input data to hash for verification
        
        Returns
        -------
        ReproducibilityMetadata
            Captured metadata
        """
        import platform
        
        dataset_hash = None
        if data:
            dataset_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        return cls(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            python_version=sys.version,
            numpy_version=np.__version__,
            scipy_version=getattr(stats, '__version__', 'unknown'),
            random_seed=seed,
            system_info={
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_implementation': platform.python_implementation(),
            },
            git_commit=cls._get_git_commit(),
            dataset_hash=dataset_hash,
        )
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

class ValidationSuite:
    """
    Comprehensive validation tests for analysis results.
    
    This class provides methods to validate statistical assumptions,
    sample sizes, and multiple testing corrections.
    
    Examples
    --------
    >>> suite = ValidationSuite()
    >>> data = np.random.normal(0, 1, 100)
    >>> dist_tests = suite.validate_distribution(data)
    >>> print(dist_tests['shapiro_wilk'])
    """
    
    @staticmethod
    def validate_distribution(data: np.ndarray) -> Dict[str, Any]:
        """
        Test data distribution properties.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of test results
        """
        return {
            'shapiro_wilk': stats.shapiro(data),
            'dagostino_k2': stats.normaltest(data),
            'jarque_bera': stats.jarque_bera(data),
            'anderson_darling': stats.anderson(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
        }
    
    @staticmethod
    def validate_sample_size(n: int, effect_size: float, alpha: float = 0.05, 
                           power: float = 0.8) -> Dict[str, Any]:
        """
        Assess if sample size is adequate for detecting effect.
        
        Parameters
        ----------
        n : int
            Observed sample size
        effect_size : float
            Expected effect size (Cohen's d)
        alpha : float, default=0.05
            Significance level
        power : float, default=0.8
            Desired statistical power
        
        Returns
        -------
        Dict[str, Any]
            Sample size validation results
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        required_n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return {
            'observed_n': n,
            'required_n': int(np.ceil(required_n)),
            'target_power': power,
            'is_adequate': n >= required_n,
            'achieved_power': norm.cdf((effect_size * np.sqrt(n)) - z_alpha),
        }
    
    @staticmethod
    def validate_multiple_testing(n_tests: int, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate multiple testing corrections.
        
        Parameters
        ----------
        n_tests : int
            Number of statistical tests
        alpha : float, default=0.05
            Family-wise error rate
        
        Returns
        -------
        Dict[str, float]
            Corrected significance thresholds
        """
        return {
            'bonferroni': alpha / n_tests,
            'sidak': 1 - (1 - alpha) ** (1 / n_tests),
            'fdr_bh': alpha,  # Benjamini-Hochberg uses adaptive threshold
            'family_wise_error_rate': 1 - (1 - alpha) ** n_tests,
        }
    
    @staticmethod
    def detect_biases(data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential biases in data or analysis.
        
        Parameters
        ----------
        data : np.ndarray
            Numerical data to analyze
        metadata : Dict[str, Any]
            Analysis metadata
        
        Returns
        -------
        Dict[str, Any]
            Bias detection results
        """
        biases = {
            'sample_bias': {
                'description': 'Checks for non-representative sampling',
                'detected': False,
                'details': ''
            },
            'selection_bias': {
                'description': 'Checks for selective data inclusion',
                'detected': False,
                'details': ''
            },
            'confirmation_bias_risk': {
                'description': 'Risk of seeking confirming patterns',
                'risk_level': 'moderate',
                'recommendation': 'Pre-register hypotheses and analysis plan'
            },
        }
        
        # Check for unusual data characteristics
        if len(data) < 30:
            biases['sample_bias']['detected'] = True
            biases['sample_bias']['details'] = 'Sample size very small (n<30)'
        
        # Check for extreme skewness (potential cherry-picking)
        skewness = stats.skew(data)
        if abs(skewness) > 2:
            biases['selection_bias']['detected'] = True
            biases['selection_bias']['details'] = f'High skewness ({skewness:.2f}) may indicate selection'
        
        return biases

# ============================================================================
# CULTURAL SYSTEMS - ENHANCED WITH VALIDATION
# ============================================================================

class CulturalSystem(Enum):
    """
    Supported cultural numerical systems with metadata.
    
    Each system includes a code, display name, and cultural tradition.
    
    Attributes
    ----------
    HEBREW_STANDARD : tuple
        Standard Hebrew gematria
    HEBREW_ATBASH : tuple
        Atbash cipher (reverse alphabet substitution)
    HEBREW_ALBAM : tuple
        Albam cipher (alternative encoding)
    GREEK_ISOPSEPHY : tuple
        Greek isopsephy system
    ARABIC_ABJAD : tuple
        Arabic Abjad numerals
    """
    
    HEBREW_STANDARD = ("hebrew_standard", "Hebrew Gematria", "Jewish tradition")
    HEBREW_ATBASH = ("hebrew_atbash", "Atbash Cipher", "Cryptographic system")
    HEBREW_ALBAM = ("hebrew_albam", "Albam Cipher", "Alternative encoding")
    GREEK_ISOPSEPHY = ("greek_isopsephy", "Greek Isopsephy", "Hellenistic tradition")
    ARABIC_ABJAD = ("arabic_abjad", "Arabic Abjad", "Islamic tradition")
    
    def __init__(self, code: str, name: str, tradition: str):
        self.code = code
        self.display_name = name
        self.tradition = tradition

class NumericalSystemProtocol(Protocol):
    """Protocol for numerical computation systems."""
    
    def compute_value(self, text: str) -> int:
        """Compute numerical value of text."""
        ...
    
    def validate_input(self, text: str) -> bool:
        """Validate input text for this system."""
        ...

# Gematria mappings
GEMATRIA_VALUES = {
    **{c: i + 1 for i, c in enumerate('אבגדהוזחט')},
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 
    'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
}

FINAL_FORMS = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}

ATBASH_MAP = {
    'א': 'ת', 'ב': 'ש', 'ג': 'ר', 'ד': 'ק', 'ה': 'צ', 'ו': 'פ', 
    'ז': 'ע', 'ח': 'ס', 'ט': 'נ', 'י': 'מ', 'כ': 'ל', 'ל': 'כ', 
    'מ': 'י', 'נ': 'ט', 'ס': 'ח', 'ע': 'ז', 'פ': 'ו', 'צ': 'ה',
    'ק': 'ד', 'ר': 'ג', 'ש': 'ב', 'ת': 'א'
}

GREEK_ISOPSEPHY = {
    'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ζ': 7, 'η': 8, 'θ': 9,
    'ι': 10, 'κ': 20, 'λ': 30, 'μ': 40, 'ν': 50, 'ξ': 60, 'ο': 70, 
    'π': 80, 'ρ': 100, 'σ': 200, 'τ': 300, 'υ': 400, 'φ': 500, 
    'χ': 600, 'ψ': 700, 'ω': 800,
}

ARABIC_ABJAD = {
    'ا': 1, 'ب': 2, 'ج': 3, 'د': 4, 'ه': 5, 'و': 6, 'ز': 7, 'ح': 8, 
    'ط': 9, 'ي': 10, 'ك': 20, 'ل': 30, 'م': 40, 'ن': 50, 'س': 60, 
    'ع': 70, 'ف': 80, 'ص': 90, 'ق': 100, 'ر': 200, 'ش': 300, 'ت': 400,
}

@lru_cache(maxsize=10000)
def compute_gematria(word: str, system: CulturalSystem = CulturalSystem.HEBREW_STANDARD) -> int:
    """
    Compute numerical value using specified cultural system.
    
    This function is cached for performance with large corpora.
    
    Parameters
    ----------
    word : str
        Input text string
    system : CulturalSystem, default=HEBREW_STANDARD
        Cultural numerical system to use
    
    Returns
    -------
    int
        Numerical value according to the system
    
    Raises
    ------
    ValueError
        If input contains invalid characters for system
    
    Examples
    --------
    >>> compute_gematria('אבג')  # 1+2+3
    6
    >>> compute_gematria('שלום')
    376
    
    Notes
    -----
    Final letter forms (ך, ם, ן, ף, ץ) are automatically normalized
    to their standard forms before computation.
    """
    if not word:
        return 0
    
    if system == CulturalSystem.HEBREW_STANDARD:
        normalized = ''.join(FINAL_FORMS.get(c, c) for c in word)
        return sum(GEMATRIA_VALUES.get(c, 0) for c in normalized)
    
    elif system == CulturalSystem.HEBREW_ATBASH:
        normalized = ''.join(FINAL_FORMS.get(c, c) for c in word)
        atbash = ''.join(ATBASH_MAP.get(c, c) for c in normalized)
        return sum(GEMATRIA_VALUES.get(c, 0) for c in atbash)
    
    elif system == CulturalSystem.GREEK_ISOPSEPHY:
        return sum(GREEK_ISOPSEPHY.get(c, 0) for c in word.lower())
    
    elif system == CulturalSystem.ARABIC_ABJAD:
        return sum(ARABIC_ABJAD.get(c, 0) for c in word)
    
    return 0

# ============================================================================
# CORPUS CONNECTORS (NEW in v5.0)
# ============================================================================

class CorpusConnector(ABC):
    """
    Abstract base class for corpus data connectors.
    
    This enables loading texts from various Digital Humanities formats.
    """
    
    @abstractmethod
    def load(self, source: Path) -> List[str]:
        """Load corpus from source."""
        pass
    
    @abstractmethod
    def validate(self, data: List[str]) -> bool:
        """Validate corpus format."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Extract corpus metadata."""
        pass

class PlainTextConnector(CorpusConnector):
    """Connector for plain text files."""
    
    def load(self, source: Path) -> List[str]:
        """Load text from plain file."""
        with open(source, 'r', encoding='utf-8') as f:
            text = f.read()
        return [text]
    
    def validate(self, data: List[str]) -> bool:
        """Validate plain text."""
        return all(isinstance(item, str) for item in data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract metadata."""
        return {
            'format': 'plain_text',
            'encoding': 'utf-8'
        }

class TEIConnector(CorpusConnector):
    """
    Connector for TEI XML format (Text Encoding Initiative).
    
    TEI is the standard for digital text encoding in humanities.
    
    Examples
    --------
    >>> connector = TEIConnector()
    >>> texts = connector.load(Path('corpus.xml'))
    """
    
    def load(self, source: Path) -> List[str]:
        """
        Load text from TEI XML.
        
        Parameters
        ----------
        source : Path
            Path to TEI XML file
        
        Returns
        -------
        List[str]
            Extracted text segments
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree required for TEI support")
        
        tree = ET.parse(source)
        root = tree.getroot()
        
        # Handle TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract text from <text> elements
        texts = []
        for text_elem in root.findall('.//tei:text', ns):
            text_content = ''.join(text_elem.itertext())
            texts.append(text_content.strip())
        
        if not texts:
            # Fallback: extract all text
            texts = [root.text] if root.text else []
        
        return texts
    
    def validate(self, data: List[str]) -> bool:
        """Validate TEI structure."""
        return all(isinstance(item, str) and len(item) > 0 for item in data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract TEI metadata."""
        return {
            'format': 'TEI_XML',
            'standard': 'Text Encoding Initiative P5',
            'namespace': 'http://www.tei-c.org/ns/1.0'
        }

class CSVCorpusConnector(CorpusConnector):
    """Connector for CSV tabular format."""
    
    def __init__(self, text_column: str = 'text'):
        """
        Initialize CSV connector.
        
        Parameters
        ----------
        text_column : str, default='text'
            Name of column containing text data
        """
        self.text_column = text_column
    
    def load(self, source: Path) -> List[str]:
        """Load text from CSV."""
        df = pd.read_csv(source)
        if self.text_column not in df.columns:
            raise ValueError(f"CSV must have '{self.text_column}' column. "
                           f"Found: {df.columns.tolist()}")
        return df[self.text_column].astype(str).tolist()
    
    def validate(self, data: List[str]) -> bool:
        """Validate CSV data."""
        return len(data) > 0 and all(isinstance(item, str) for item in data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract metadata."""
        return {
            'format': 'CSV',
            'text_column': self.text_column
        }

# ============================================================================
# STATISTICAL ANALYSIS - ENHANCED
# ============================================================================

@dataclass
class StatisticalResult:
    """
    Comprehensive statistical test result with full metadata.
    
    Attributes
    ----------
    test_name : str
        Name of statistical test
    statistic : float
        Test statistic value
    p_value : float
        P-value
    effect_size : float
        Standardized effect size
    confidence_interval : Tuple[float, float]
        95% confidence interval
    interpretation : str
        Plain language interpretation
    assumptions_met : Dict[str, bool]
        Dictionary of assumption checks
    metadata : Dict[str, Any]
        Additional test-specific metadata
    """
    
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    assumptions_met: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """
        Check if result is statistically significant.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance threshold
        
        Returns
        -------
        bool
            True if p_value < alpha
        """
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def effect_size_interpretation(self) -> str:
        """
        Interpret effect size magnitude (Cohen's conventions).
        
        Returns
        -------
        str
            Effect size interpretation
        """
        abs_effect = abs(self.effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

class RobustStatisticalTests:
    """Suite of robust statistical tests with validation."""
    
    @staticmethod
    def binomial_test_robust(k: int, n: int, p: float, 
                            alternative: str = 'two-sided') -> StatisticalResult:
        """
        Robust binomial test with effect size and confidence intervals.
        
        Parameters
        ----------
        k : int
            Number of successes
        n : int
            Number of trials
        p : float
            Expected probability under null hypothesis
        alternative : str, default='two-sided'
            Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
        Returns
        -------
        StatisticalResult
            Complete statistical result with metadata
        
        Examples
        --------
        >>> result = RobustStatisticalTests.binomial_test_robust(15, 100, 0.1)
        >>> print(f"P-value: {result.p_value:.4f}")
        """
        result = stats.binomtest(k, n, p, alternative=alternative)
        
        # Compute effect size (Cohen's h)
        p_obs = k / n
        effect_size = 2 * (np.arcsin(np.sqrt(p_obs)) - np.arcsin(np.sqrt(p)))
        
        # Confidence interval (Wilson score)
        ci_low, ci_high = RobustStatisticalTests._wilson_ci(k, n)
        
        # Check assumptions
        assumptions = {
            'sample_size_adequate': n >= 30,
            'expected_successes_adequate': n * p >= 5,
            'expected_failures_adequate': n * (1 - p) >= 5,
            'independence_assumed': True,  # Must be verified by researcher
        }
        
        interpretation = (
            f"Observed proportion: {p_obs:.4f}, "
            f"Expected: {p:.4f}, "
            f"Effect size (h): {effect_size:.4f}"
        )
        
        return StatisticalResult(
            test_name='binomial_test',
            statistic=float(k),
            p_value=result.pvalue,
            effect_size=effect_size,
            confidence_interval=(ci_low, ci_high),
            interpretation=interpretation,
            assumptions_met=assumptions,
            metadata={'n': n, 'k': k, 'p': p, 'alternative': alternative}
        )
    
    @staticmethod
    def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Wilson score confidence interval for proportions.
        
        More accurate than normal approximation for small samples.
        """
        from scipy.stats import norm
        z = norm.ppf(1 - alpha/2)
        p_hat = k / n
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    @staticmethod
    def permutation_test(observed: np.ndarray, expected: np.ndarray, 
                        n_permutations: int = 10000, 
                        statistic_func=np.mean) -> StatisticalResult:
        """
        Non-parametric permutation test.
        
        Parameters
        ----------
        observed : np.ndarray
            Observed data
        expected : np.ndarray
            Expected/control data
        n_permutations : int, default=10000
            Number of random permutations
        statistic_func : callable, default=np.mean
            Function to compute test statistic
        
        Returns
        -------
        StatisticalResult
            Permutation test result
        """
        observed_stat = statistic_func(observed) - statistic_func(expected)
        
        combined = np.concatenate([observed, expected])
        n_obs = len(observed)
        
        perm_stats = np.zeros(n_permutations)
        rng = np.random.RandomState(42)
        
        for i in range(n_permutations):
            rng.shuffle(combined)
            perm_obs = combined[:n_obs]
            perm_exp = combined[n_obs:]
            perm_stats[i] = statistic_func(perm_obs) - statistic_func(perm_exp)
        
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.var(observed) + np.var(expected)) / 2)
        effect_size = observed_stat / pooled_std if pooled_std > 0 else 0
        
        ci = np.percentile(perm_stats, [2.5, 97.5])
        
        return StatisticalResult(
            test_name='permutation_test',
            statistic=observed_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=tuple(ci),
            interpretation=f"Permutation-based p-value: {p_value:.4f}",
            assumptions_met={'non_parametric': True, 'exchangeability_assumed': True},
            metadata={'n_permutations': n_permutations}
        )
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic_func=np.mean, 
                     n_bootstrap: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for any statistic.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        statistic_func : callable, default=np.mean
            Statistic to compute
        n_bootstrap : int, default=10000
            Number of bootstrap samples
        alpha : float, default=0.05
            Significance level for CI
        
        Returns
        -------
        Tuple[float, float]
            Lower and upper confidence bounds
        """
        rng = np.random.RandomState(42)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stats[i] = statistic_func(sample)
        
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower, upper)

# ============================================================================
# BAYESIAN ANALYSIS - ENHANCED
# ============================================================================

class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for multiples analysis.
    
    Implements model comparison using WAIC/LOO and posterior predictive checks.
    
    Parameters
    ----------
    data : np.ndarray
        Numerical values to analyze
    divisors : List[int]
        Divisors to test for enrichment
    
    Attributes
    ----------
    models : Dict
        Dictionary of PyMC models
    traces : Dict
        Dictionary of MCMC traces
    comparisons : Dict
        Model comparison results
    
    Examples
    --------
    >>> data = np.array([7, 14, 21, 28, 35, 42])
    >>> model = BayesianHierarchicalModel(data, [7, 12])
    >>> model.fit_model(7, draws=1000)
    """
    
    def __init__(self, data: np.ndarray, divisors: List[int]):
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required. Install: pip install pymc arviz")
        
        self.data = data
        self.divisors = divisors
        self.models = {}
        self.traces = {}
        self.comparisons = {}
    
    def fit_model(self, divisor: int, draws: int = 2000, 
                  tune: int = 1000, chains: int = 4) -> None:
        """
        Fit Bayesian model for given divisor.
        
        Parameters
        ----------
        divisor : int
            Divisor to test
        draws : int, default=2000
            Number of MCMC samples
        tune : int, default=1000
            Number of tuning samples
        chains : int, default=4
            Number of MCMC chains
        """
        n = len(self.data)
        k = np.sum(self.data % divisor == 0)
        p_expected = 1.0 / divisor
        
        with pm.Model() as model:
            # Hierarchical prior
            alpha_prior = pm.Exponential('alpha_prior', 1.0)
            beta_prior = pm.Exponential('beta_prior', 1.0)
            
            # Beta distribution for proportion
            p = pm.Beta('p', alpha=alpha_prior, beta=beta_prior)
            
            # Likelihood
            obs = pm.Binomial('obs', n=n, p=p, observed=k)
            
            # Posterior predictive
            pm.Deterministic('p_greater_expected', p > p_expected)
            pm.Deterministic('effect_size', (p - p_expected) / np.sqrt(p_expected * (1 - p_expected)))
            
            # Sample
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                progressbar=False,
                random_seed=42
            )
            
            # Posterior predictive check
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)
        
        self.models[divisor] = model
        self.traces[divisor] = trace
        
        logger.info(f"Fitted Bayesian model for divisor {divisor}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all fitted models using WAIC.
        
        Returns
        -------
        pd.DataFrame
            Model comparison table
        """
        if len(self.traces) < 2:
            logger.warning("Need at least 2 models to compare")
            return pd.DataFrame()
        
        comparison = az.compare(self.traces, ic='waic')
        self.comparisons['waic'] = comparison
        
        return comparison
    
    def posterior_summary(self, divisor: int) -> pd.DataFrame:
        """
        Get posterior summary statistics.
        
        Parameters
        ----------
        divisor : int
            Divisor to summarize
        
        Returns
        -------
        pd.DataFrame
            Posterior summary table
        """
        if divisor not in self.traces:
            raise ValueError(f"Model for divisor {divisor} not fitted")
        
        return az.summary(self.traces[divisor], var_names=['p', 'effect_size'])
    
    def plot_posterior(self, divisor: int, save_path: Optional[Path] = None):
        """
        Plot posterior distribution with diagnostics.
        
        Parameters
        ----------
        divisor : int
            Divisor to plot
        save_path : Optional[Path]
            Path to save figure
        """
        if divisor not in self.traces:
            raise ValueError(f"Model for divisor {divisor} not fitted")
        
        trace = self.traces[divisor]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Posterior distribution
        az.plot_posterior(trace, var_names=['p'], ax=axes[0, 0])
        axes[0, 0].axvline(1/divisor, color='red', linestyle='--', 
                          linewidth=2, label=f'Expected (1/{divisor})')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'Posterior Distribution - Divisor {divisor}')
        
        # Trace plot
        az.plot_trace(trace, var_names=['p'], axes=axes[0, 1:].reshape(-1))
        
        # Posterior predictive check
        az.plot_ppc(trace, ax=axes[1, 0])
        axes[1, 0].set_title('Posterior Predictive Check')
        
        # Rank plot (convergence diagnostic)
        az.plot_rank(trace, var_names=['p'], ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved posterior plot: {save_path}")
        
        plt.close()

# ============================================================================
# MAIN PIPELINE - PUBLICATION READY
# ============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuration for analysis pipeline.
    
    Attributes
    ----------
    data_dir : Path
        Directory containing input data
    output_dir : Path
        Directory for output files
    random_seed : int
        Random seed for reproducibility
    n_permutations : int
        Number of permutation test iterations
    n_bayesian_draws : int
        Number of Bayesian MCMC draws
    enable_bayesian : bool
        Whether to run Bayesian analysis
    enable_parallel : bool
        Whether to use parallel processing
    significance_level : float
        Statistical significance threshold
    save_figures : bool
        Whether to generate and save figures
    verbose : bool
        Whether to enable verbose logging
    corpus_format : str
        Format of input corpus ('plain', 'tei', 'csv')
    """
    
    data_dir: Path = Path('data')
    output_dir: Path = Path('output')
    random_seed: int = 42
    n_permutations: int = 10000
    n_bayesian_draws: int = 2000
    enable_bayesian: bool = True
    enable_parallel: bool = True
    significance_level: float = 0.05
    save_figures: bool = True
    verbose: bool = True
    corpus_format: str = 'plain'
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class AncientTextAnalysisPipeline:
    """
    Main analysis pipeline for ancient text numerical analysis.
    
    This class implements the complete workflow from data loading to
    publication-ready output, including statistical tests, Bayesian
    modeling, visualizations, and ethical considerations.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration object with analysis parameters
    
    Attributes
    ----------
    config : AnalysisConfig
        Analysis configuration
    metadata : ReproducibilityMetadata
        Complete reproducibility metadata
    results : Dict[str, Any]
        Dictionary storing all analysis results
    validation_suite : ValidationSuite
        Suite of validation methods
    ethics : EthicalGuidelines
        Ethical guidelines framework
    
    Examples
    --------
    >>> config = AnalysisConfig(data_dir=Path('data'))
    >>> pipeline = AncientTextAnalysisPipeline(config)
    >>> results = pipeline.run_complete_analysis()
    >>> print(results['gematria']['summary_statistics'])
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.metadata = None  # Will be set when data is loaded
        self.results = {}
        self.validation_suite = ValidationSuite()
        self.ethics = EthicalGuidelines()
        
        # Set random seeds
        np.random.seed(config.random_seed)
        
        # Setup logging
        log_file = config.output_dir / 'analysis.log'
        setup_logging(
            level=logging.INFO if config.verbose else logging.WARNING,
            log_file=log_file
        )
        
        logger.info("=" * 80)
        logger.info("Ancient Text Numerical Analysis Framework v5.0 - DSH Edition")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config}")
        logger.info(f"DOI: 10.5281/zenodo.17487211")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute complete analysis pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Complete analysis results including:
            - gematria: Gematria analysis results
            - multiples_frequentist: Frequentist statistical tests
            - multiples_bayesian: Bayesian analysis (if enabled)
            - sensitivity: Sensitivity analysis
            - ethics: Ethical considerations
            - metadata: Reproducibility metadata
        
        Raises
        ------
        RuntimeError
            If analysis fails at any stage
        """
        
        # 1. Load and validate data
        text_data = self._load_data()
        
        # 2. Capture metadata with data hash
        self.metadata = ReproducibilityMetadata.capture(
            self.config.random_seed, 
            text_data
        )
        logger.info(f"Reproducibility metadata: {self.metadata.timestamp}")
        
        # 3. Ethical validation
        self.results['ethics'] = self._validate_ethics()
        
        # 4. Gematria analysis
        self.results['gematria'] = self._analyze_gematria(text_data)
        
        # 5. Multiples analysis (Frequentist)
        self.results['multiples_frequentist'] = self._analyze_multiples_frequentist(text_data)
        
        # 6. Bayesian analysis
        if self.config.enable_bayesian and PYMC_AVAILABLE:
            self.results['multiples_bayesian'] = self._analyze_multiples_bayesian(text_data)
        else:
            logger.info("Bayesian analysis skipped")
        
        # 7. Sensitivity analysis
        self.results['sensitivity'] = self._sensitivity_analysis(text_data)
        
        # 8. Bias detection
        self.results['bias_detection'] = self._detect_biases(text_data)
        
        # 9. Generate visualizations
        if self.config.save_figures:
            self._generate_visualizations()
        
        # 10. Save results
        self._save_results()
        
        logger.info("✓ Complete analysis finished successfully")
        return self.results
    
    def _load_data(self) -> str:
        """
        Load and validate input text data.
        
        Returns
        -------
        str
            Cleaned text data
        """
        logger.info("Loading data...")
        
        # Select appropriate connector
        if self.config.corpus_format == 'plain':
            connector = PlainTextConnector()
            text_file = self.config.data_dir / 'text.txt'
        elif self.config.corpus_format == 'tei':
            connector = TEIConnector()
            text_file = self.config.data_dir / 'corpus.xml'
        elif self.config.corpus_format == 'csv':
            connector = CSVCorpusConnector()
            text_file = self.config.data_dir / 'corpus.csv'
        else:
            raise ValueError(f"Unsupported format: {self.config.corpus_format}")
        
        if not text_file.exists():
            logger.warning(f"Data file not found: {text_file}. Using placeholder.")
            text = self._generate_placeholder_text()
        else:
            texts = connector.load(text_file)
            text = ' '.join(texts)
            logger.info(f"Loaded from {self.config.corpus_format} format")
        
        # Validate and clean
        hebrew_chars = set('אבגדהוזחטיכלמנסעפצקרשתךםןףץ')
        text = ''.join(c for c in text if c in hebrew_chars or c.isspace())
        
        logger.info(f"Loaded text: {len(text)} characters")
        
        if len(text) < 100:
            logger.warning("Text is very short. Results may not be reliable.")
        
        return text
    
    def _generate_placeholder_text(self) -> str:
        """Generate placeholder text for demonstration."""
        words = ['בראשית', 'ברא', 'אלהים', 'את', 'השמים', 'והארץ']
        return ' '.join(words * 300)
    
    def _validate_ethics(self) -> Dict[str, Any]:
        """
        Validate ethical guidelines compliance.
        
        Returns
        -------
        Dict[str, Any]
            Ethical validation results
        """
        logger.info("Validating ethical guidelines...")
        
        research_metadata = {
            'data_source': str(self.config.data_dir),
            'methods': 'Statistical and Bayesian analysis',
            'limitations': 'See documentation',
            'cultural_context': 'Ancient Hebrew texts',
            'random_seed': self.config.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        }
        
        validation = EthicalGuidelines.validate_research_ethics(research_metadata)
        guidelines = EthicalGuidelines.get_guidelines()
        statement = EthicalGuidelines.generate_ethics_statement()
        
        return {
            'validation': validation,
            'guidelines': guidelines,
            'statement': statement,
            'all_checks_passed': all(validation.values())
        }
    
    def _analyze_gematria(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive gematria analysis with cross-cultural comparison.
        
        Parameters
        ----------
        text : str
            Input text to analyze
        
        Returns
        -------
        Dict[str, Any]
            Gematria analysis results
        """
        logger.info("Running gematria analysis...")
        
        # Extract words (5-char windows)
        words = [text[i:i+5] for i in range(0, len(text)-4, 5) if text[i:i+5].strip()]
        values = np.array([compute_gematria(w) for w in words if compute_gematria(w) > 0])
        
        if len(values) == 0:
            return {'error': 'No valid gematria values computed'}
        
        # Statistical summary
        summary = {
            'n': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': int(np.min(values)),
            'max': int(np.max(values)),
            'quartiles': [float(q) for q in np.percentile(values, [25, 50, 75])],
            'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
        }
        
        # Bootstrap confidence intervals
        ci_mean = RobustStatisticalTests.bootstrap_ci(values, np.mean)
        ci_median = RobustStatisticalTests.bootstrap_ci(values, np.median)
        
        summary['ci_mean'] = ci_mean
        summary['ci_median'] = ci_median
        
        # Distribution validation
        distribution_tests = self.validation_suite.validate_distribution(values)
        
        # Cross-cultural comparison
        cross_cultural = self._cross_cultural_comparison(words[:100])
        
        return {
            'summary_statistics': summary,
            'distribution_tests': {k: str(v) for k, v in distribution_tests.items()},
            'cross_cultural': cross_cultural,
            'sample_values': values[:100].tolist(),
        }
    
    def _cross_cultural_comparison(self, words: List[str]) -> Dict[str, Any]:
        """
        Compare values across cultural systems.
        
        Parameters
        ----------
        words : List[str]
            Sample words to analyze
        
        Returns
        -------
        Dict[str, Any]
            Cross-cultural comparison results
        """
        results = defaultdict(list)
        
        for word in words:
            for system in CulturalSystem:
                try:
                    value = compute_gematria(word, system)
                    if value > 0:
                        results[system.display_name].append(value)
                except Exception as e:
                    logger.warning(f"Error computing {system}: {e}")
        
        # Correlation matrix
        df = pd.DataFrame(results)
        correlation = df.corr().to_dict() if len(df.columns) > 1 else {}
        
        # System statistics
        system_stats = {}
        for system_name, values in results.items():
            if len(values) > 0:
                system_stats[system_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n': len(values)
                }
        
        return {
            'systems': list(results.keys()),
            'sample_size': len(words),
            'correlations': correlation,
            'system_statistics': system_stats,
        }
    
    def _analyze_multiples_frequentist(self, text: str) -> Dict[str, Any]:
        """
        Frequentist analysis of multiples with robust corrections.
        
        Parameters
        ----------
        text : str
            Input text
        
        Returns
        -------
        Dict[str, Any]
            Frequentist analysis results
        """
        logger.info("Running frequentist multiples analysis...")
        
        words = [text[i:i+5] for i in range(0, len(text)-4, 10) if text[i:i+5].strip()]
        values = np.array([compute_gematria(w) for w in words if compute_gematria(w) > 0])
        
        divisors = [7, 12, 26, 30, 60]
        n_tests = len(divisors)
        corrections = self.validation_suite.validate_multiple_testing(n_tests, self.config.significance_level)
        
        results = {}
        for divisor in divisors:
            k = np.sum(values % divisor == 0)
            p_expected = 1.0 / divisor
            
            # Robust binomial test
            stat_result = RobustStatisticalTests.binomial_test_robust(
                k, len(values), p_expected, alternative='greater'
            )
            
            # Permutation test for validation
            random_values = np.random.randint(1, 1000, size=len(values))
            perm_result = RobustStatisticalTests.permutation_test(
                (values % divisor == 0).astype(int),
                (random_values % divisor == 0).astype(int),
                n_permutations=self.config.n_permutations
            )
            
            results[f'divisor_{divisor}'] = {
                'binomial_test': stat_result.to_dict(),
                'permutation_test': perm_result.to_dict(),
                'bonferroni_significant': stat_result.p_value < corrections['bonferroni'],
                'sidak_significant': stat_result.p_value < corrections['sidak'],
                'effect_interpretation': stat_result.effect_size_interpretation(),
            }
        
        # Sample size validation
        sample_validation = self.validation_suite.validate_sample_size(
            len(values), effect_size=0.1, alpha=self.config.significance_level
        )
        
        return {
            'sample_size': len(values),
            'divisor_results': results,
            'multiple_testing_corrections': corrections,
            'sample_size_validation': sample_validation,
            'interpretation': self._interpret_multiples_results(results, corrections)
        }
    
    def _interpret_multiples_results(self, results: Dict, corrections: Dict) -> str:
        """Generate interpretation of multiples analysis."""
        significant_bonf = sum(1 for r in results.values() 
                              if r['bonferroni_significant'])
        
        if significant_bonf == 0:
            return ("No divisors showed significant enrichment after Bonferroni correction. "
                   "Results are consistent with random distribution.")
        elif significant_bonf == 1:
            return (f"One divisor showed significant enrichment (α={corrections['bonferroni']:.4f}). "
                   "Interpret with caution given multiple testing.")
        else:
            return (f"{significant_bonf} divisors showed significant enrichment. "
                   "This warrants further investigation with independent datasets.")
    
    def _analyze_multiples_bayesian(self, text: str) -> Dict[str, Any]:
        """Bayesian hierarchical analysis of multiples."""
        logger.info("Running Bayesian multiples analysis...")
        
        words = [text[i:i+5] for i in range(0, len(text)-4, 10) if text[i:i+5].strip()]
        values = np.array([compute_gematria(w) for w in words if compute_gematria(w) > 0])
        
        divisors = [7, 12, 26, 30, 60]
        
        try:
            bayesian_model = BayesianHierarchicalModel(values, divisors)
            
            # Fit models
            for divisor in divisors:
                bayesian_model.fit_model(
                    divisor, 
                    draws=self.config.n_bayesian_draws,
                    tune=1000,
                    chains=4
                )
            
            # Model comparison
            comparison = bayesian_model.compare_models()
            
            # Posterior summaries
            posteriors = {}
            for divisor in divisors:
                summary = bayesian_model.posterior_summary(divisor)
                posteriors[f'divisor_{divisor}'] = summary.to_dict()
                
                # Save plots
                if self.config.save_figures:
                    plot_path = self.config.output_dir / 'figures' / f'posterior_divisor_{divisor}.png'
                    bayesian_model.plot_posterior(divisor, plot_path)
            
            return {
                'method': 'Bayesian hierarchical model',
                'divisors': divisors,
                'model_comparison': comparison.to_dict() if not comparison.empty else {},
                'posterior_summaries': posteriors,
                'interpretation': self._interpret_bayesian_results(comparison)
            }
            
        except Exception as e:
            logger.error(f"Bayesian analysis failed: {e}")
            return {'error': str(e)}
    
    def _interpret_bayesian_results(self, comparison: pd.DataFrame) -> str:
        """Interpret Bayesian model comparison."""
        if comparison.empty:
            return "Model comparison unavailable"
        
        best_model = comparison.index[0]
        if len(comparison) > 1:
            waic_diff = comparison.iloc[0]['waic'] - comparison.iloc[1]['waic']
            
            if abs(waic_diff) < 2:
                return "Models are essentially equivalent (ΔWAIC < 2)"
            elif abs(waic_diff) < 6:
                return f"Weak evidence for {best_model} (2 < ΔWAIC < 6)"
            else:
                return f"Strong evidence for {best_model} (ΔWAIC > 6)"
        else:
            return f"Only one model fitted: {best_model}"
    
    def _sensitivity_analysis(self, text: str) -> Dict[str, Any]:
        """Sensitivity analysis with different parameters."""
        logger.info("Running sensitivity analysis...")
        
        # Test different window sizes
        window_sizes = [3, 5, 7, 10]
        sensitivity_results = {}
        
        for window_size in window_sizes:
            words = [text[i:i+window_size] for i in range(0, len(text)-window_size+1, window_size) if text[i:i+window_size].strip()]
            values = np.array([compute_gematria(w) for w in words if compute_gematria(w) > 0])
            
            if len(values) == 0:
                continue
            
            divisor = 7  # Test divisor
            k = np.sum(values % divisor == 0)
            p_expected = 1.0 / divisor
            
            result = stats.binomtest(k, len(values), p_expected, alternative='greater')
            
            sensitivity_results[f'window_{window_size}'] = {
                'n': len(values),
                'k': int(k),
                'p_value': float(result.pvalue),
                'proportion': k / len(values)
            }
        
        # Test different sampling strategies
        sampling_results = {}
        original_words = [text[i:i+5] for i in range(0, len(text)-4, 5) if text[i:i+5].strip()]
        
        for stride in [5, 10, 15, 20]:
            sampled_words = original_words[::stride]
            values = np.array([compute_gematria(w) for w in sampled_words if compute_gematria(w) > 0])
            
            if len(values) == 0:
                continue
            
            k = np.sum(values % 7 == 0)
            result = stats.binomtest(k, len(values), 1/7, alternative='greater')
            
            sampling_results[f'stride_{stride}'] = {
                'n': len(values),
                'p_value': float(result.pvalue)
            }
        
        return {
            'window_size_sensitivity': sensitivity_results,
            'sampling_sensitivity': sampling_results,
            'interpretation': self._interpret_sensitivity(sensitivity_results)
        }
    
    def _interpret_sensitivity(self, results: Dict) -> str:
        """Interpret sensitivity analysis results."""
        p_values = [r['p_value'] for r in results.values()]
        
        if len(p_values) == 0:
            return "Insufficient data for sensitivity analysis"
        
        cv = np.std(p_values) / np.mean(p_values) if np.mean(p_values) > 0 else float('inf')
        
        if cv < 0.3:
            return "Results are robust across parameter choices (CV < 0.3)"
        elif cv < 0.5:
            return "Moderate sensitivity to parameter choices (0.3 < CV < 0.5)"
        else:
            return "Results highly sensitive to parameter choices (CV > 0.5). Interpret with caution."
    
    def _detect_biases(self, text: str) -> Dict[str, Any]:
        """
        Detect potential biases in data or analysis.
        
        Parameters
        ----------
        text : str
            Input text
        
        Returns
        -------
        Dict[str, Any]
            Bias detection results
        """
        logger.info("Running bias detection...")
        
        words = [text[i:i+5] for i in range(0, len(text)-4, 5) if text[i:i+5].strip()]
        values = np.array([compute_gematria(w) for w in words if compute_gematria(w) > 0])
        
        metadata = {
            'sample_size': len(values),
            'data_source': str(self.config.data_dir),
        }
        
        biases = self.validation_suite.detect_biases(values, metadata)
        
        return biases
    
    def _generate_visualizations(self):
        """Generate publication-quality visualizations."""
        logger.info("Generating visualizations...")
        
        viz_dir = self.config.output_dir / 'figures'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Gematria distribution histogram
        if 'gematria' in self.results:
            self._plot_gematria_distribution(viz_dir)
        
        # 2. Multiples analysis bar plot
        if 'multiples_frequentist' in self.results:
            self._plot_multiples_analysis(viz_dir)
        
        # 3. Cross-cultural comparison heatmap
        if 'gematria' in self.results and 'cross_cultural' in self.results['gematria']:
            self._plot_cross_cultural_heatmap(viz_dir)
        
        # 4. Sensitivity analysis plot
        if 'sensitivity' in self.results:
            self._plot_sensitivity_analysis(viz_dir)
        
        logger.info(f"Saved visualizations to {viz_dir}")
    
    def _plot_gematria_distribution(self, viz_dir: Path):
        """Plot gematria value distribution."""
        values = np.array(self.results['gematria'].get('sample_values', []))
        
        if len(values) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7, 
                    color='steelblue', density=True, label='Histogram')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        axes[0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        axes[0].axvline(np.mean(values), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(values):.1f}')
        axes[0].axvline(np.median(values), color='darkgreen', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(values):.1f}')
        axes[0].set_xlabel('Gematria Value', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of Gematria Values', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(values, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'gematria_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multiples_analysis(self, viz_dir: Path):
        """Plot multiples analysis results."""
        results = self.results['multiples_frequentist']['divisor_results']
        
        divisors = []
        observed = []
        expected = []
        p_values = []
        effect_sizes = []
        
        for key, data in results.items():
            divisor = int(key.split('_')[1])
            divisors.append(divisor)
            
            binomial = data['binomial_test']
            observed.append(binomial['metadata']['k'])
            expected.append(binomial['metadata']['n'] / divisor)
            p_values.append(binomial['p_value'])
            effect_sizes.append(binomial['effect_size'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Observed vs Expected
        x = np.arange(len(divisors))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, observed, width, label='Observed', 
                      color='steelblue', edgecolor='black')
        axes[0, 0].bar(x + width/2, expected, width, label='Expected', 
                      color='coral', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Divisor', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Multiples Analysis: Observed vs Expected', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(divisors)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # 2. P-values
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        axes[0, 1].bar(divisors, [-np.log10(p) for p in p_values], 
                      color=colors, edgecolor='black', alpha=0.7)
        axes[0, 1].axhline(-np.log10(0.05), color='red', linestyle='--', 
                          linewidth=2, label='α = 0.05')
        axes[0, 1].axhline(-np.log10(0.01), color='darkred', linestyle='--', 
                          linewidth=2, label='α = 0.01')
        axes[0, 1].set_xlabel('Divisor', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Statistical Significance', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 3. Effect sizes
        axes[1, 0].bar(divisors, effect_sizes, color='teal', edgecolor='black', alpha=0.7)
        axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].axhline(0.2, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Small effect')
        axes[1, 0].axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium effect')
        axes[1, 0].axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Large effect')
        axes[1, 0].set_xlabel('Divisor', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel("Effect Size (Cohen's h)", fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Effect Sizes by Divisor', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 4. Proportions with CI
        proportions = [o/e for o, e in zip(observed, [e if e > 0 else 1 for e in expected])]
        axes[1, 1].bar(divisors, proportions, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Expected ratio = 1.0')
        axes[1, 1].set_xlabel('Divisor', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Observed/Expected Ratio', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Enrichment Ratios', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'multiples_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_cultural_heatmap(self, viz_dir: Path):
        """Plot cross-cultural correlation heatmap."""
        correlations = self.results['gematria']['cross_cultural'].get('correlations', {})
        
        if not correlations:
            return
        
        df = pd.DataFrame(correlations)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, cbar_kws={'label': 'Pearson Correlation'})
        plt.title('Cross-Cultural Gematria System Correlations', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('System', fontsize=12, fontweight='bold')
        plt.ylabel('System', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'cross_cultural_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_analysis(self, viz_dir: Path):
        """Plot sensitivity analysis results."""
        window_results = self.results['sensitivity'].get('window_size_sensitivity', {})
        
        if not window_results:
            return
        
        windows = []
        p_values = []
        sample_sizes = []
        proportions = []
        
        for key, data in window_results.items():
            window = int(key.split('_')[1])
            windows.append(window)
            p_values.append(data['p_value'])
            sample_sizes.append(data['n'])
            proportions.append(data['proportion'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. P-values vs window size
        axes[0, 0].plot(windows, p_values, marker='o', linewidth=2, 
                       markersize=10, color='steelblue', label='P-value')
        axes[0, 0].axhline(0.05, color='red', linestyle='--', 
                          linewidth=2, label='α = 0.05')
        axes[0, 0].fill_between(windows, 0, 0.05, alpha=0.2, color='red')
        axes[0, 0].set_xlabel('Window Size', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('P-value', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('P-value Sensitivity to Window Size', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Sample size vs window size
        axes[0, 1].plot(windows, sample_sizes, marker='s', linewidth=2, 
                       markersize=10, color='coral')
        axes[0, 1].set_xlabel('Window Size', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Sample Size (n)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Sample Size by Window Size', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Proportions vs window size
        expected_prop = 1/7
        axes[1, 0].plot(windows, proportions, marker='D', linewidth=2, 
                       markersize=10, color='green', label='Observed')
        axes[1, 0].axhline(expected_prop, color='red', linestyle='--', 
                          linewidth=2, label=f'Expected (1/7 = {expected_prop:.3f})')
        axes[1, 0].set_xlabel('Window Size', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Proportion of Multiples', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Proportion Stability Across Window Sizes', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Coefficient of variation
        cv_p = np.std(p_values) / np.mean(p_values) if np.mean(p_values) > 0 else 0
        cv_prop = np.std(proportions) / np.mean(proportions) if np.mean(proportions) > 0 else 0
        
        metrics = ['P-value CV', 'Proportion CV']
        values = [cv_p, cv_prop]
        colors_cv = ['green' if v < 0.3 else 'orange' if v < 0.5 else 'red' for v in values]
        
        axes[1, 1].bar(metrics, values, color=colors_cv, edgecolor='black', alpha=0.7)
        axes[1, 1].axhline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Robust (CV<0.3)')
        axes[1, 1].axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate (CV<0.5)')
        axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Sensitivity Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save all results with metadata."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Full results (JSON)
        results_file = self.config.output_dir / f'results_{timestamp}.json'
        full_output = {
            'metadata': asdict(self.metadata),
            'config': asdict(self.config),
            'results': self.results,
            'timestamp': timestamp,
            'doi': '10.5281/zenodo.17487211',
            'version': '5.0-DSH'
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved results: {results_file}")
        
        # Human-readable report
        report_file = self.config.output_dir / f'report_{timestamp}.md'
        self._generate_markdown_report(report_file)
        
        # CSV exports for tables
        self._export_tables_to_csv(timestamp)
        
        # Ethics statement
        ethics_file = self.config.output_dir / 'ETHICS_STATEMENT.md'
        with open(ethics_file, 'w', encoding='utf-8') as f:
            f.write(self.results['ethics']['statement'])
        
        logger.info(f"Saved ethics statement: {ethics_file}")
    
    def _generate_markdown_report(self, report_file: Path):
        """Generate comprehensive markdown report for publication."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Ancient Text Numerical Analysis Report\n\n")
            f.write("**Framework Version**: 5.0-DSH\n\n")
            f.write(f"**DOI**: 10.5281/zenodo.17487211\n\n")
            f.write(f"**Generated**: {self.metadata.timestamp}\n\n")
            f.write(f"**Random Seed**: {self.metadata.random_seed}\n\n")
            f.write(f"**Dataset Hash**: {self.metadata.dataset_hash[:16]}...\n\n")
            
            f.write("---\n\n")
            f.write("## Executive Summary\n\n")
            
            # Gematria summary
            if 'gematria' in self.results and 'error' not in self.results['gematria']:
                ga = self.results['gematria']['summary_statistics']
                f.write("### Gematria Analysis\n\n")
                f.write(f"- **Sample size**: {ga['n']}\n")
                f.write(f"- **Mean**: {ga['mean']:.2f} (95% CI: [{ga['ci_mean'][0]:.2f}, {ga['ci_mean'][1]:.2f}])\n")
                f.write(f"- **Median**: {ga['median']:.2f} (95% CI: [{ga['ci_median'][0]:.2f}, {ga['ci_median'][1]:.2f}])\n")
                f.write(f"- **Standard deviation**: {ga['std']:.2f}\n")
                f.write(f"- **Coefficient of variation**: {ga['cv']:.3f}\n\n")
            
            # Multiples summary
            if 'multiples_frequentist' in self.results:
                mf = self.results['multiples_frequentist']
                f.write("### Multiples Analysis (Frequentist)\n\n")
                f.write(f"**Sample size**: {mf['sample_size']}\n\n")
                f.write(f"**Interpretation**: {mf['interpretation']}\n\n")
                
                f.write("| Divisor | Observed | Expected | P-value | Effect Size | Bonferroni | Interpretation |\n")
                f.write("|---------|----------|----------|---------|-------------|------------|----------------|\n")
                
                for key, data in mf['divisor_results'].items():
                    divisor = key.split('_')[1]
                    binomial = data['binomial_test']
                    obs = binomial['metadata']['k']
                    exp = binomial['metadata']['n'] / int(divisor)
                    p_val = binomial['p_value']
                    effect = binomial['effect_size']
                    sig = '✓' if data['bonferroni_significant'] else '✗'
                    interp = data['effect_interpretation']
                    f.write(f"| {divisor} | {obs} | {exp:.2f} | {p_val:.4f} | {effect:.4f} | {sig} | {interp} |\n")
                
                f.write("\n")
                
                # Sample size validation
                sv = mf['sample_size_validation']
                f.write("**Sample Size Validation**:\n\n")
                f.write(f"- Observed n: {sv['observed_n']}\n")
                f.write(f"- Required n (80% power): {sv['required_n']}\n")
                f.write(f"- Achieved power: {sv['achieved_power']:.3f}\n")
                f.write(f"- Adequate: {'Yes ✓' if sv['is_adequate'] else 'No ✗'}\n\n")
            
            # Bayesian summary
            if 'multiples_bayesian' in self.results and 'error' not in self.results['multiples_bayesian']:
                mb = self.results['multiples_bayesian']
                f.write("### Bayesian Analysis\n\n")
                f.write(f"**Method**: {mb['method']}\n\n")
                f.write(f"**Interpretation**: {mb['interpretation']}\n\n")
            
            # Sensitivity
            if 'sensitivity' in self.results:
                sens = self.results['sensitivity']
                f.write("### Sensitivity Analysis\n\n")
                f.write(f"**Interpretation**: {sens['interpretation']}\n\n")
            
            # Bias detection
            if 'bias_detection' in self.results:
                biases = self.results['bias_detection']
                f.write("### Bias Detection\n\n")
                for bias_type, info in biases.items():
                    if isinstance(info, dict) and 'detected' in info:
                        status = '⚠️ Detected' if info['detected'] else '✓ Not detected'
                        f.write(f"- **{bias_type}**: {status}\n")
                        if info.get('details'):
                            f.write(f"  - {info['details']}\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Reproducibility Information\n\n")
            f.write(f"- **Python version**: {self.metadata.python_version.split()[0]}\n")
            f.write(f"- **NumPy version**: {self.metadata.numpy_version}\n")
            f.write(f"- **SciPy version**: {self.metadata.scipy_version}\n")
            f.write(f"- **Random seed**: {self.metadata.random_seed}\n")
            f.write(f"- **Platform**: {self.metadata.system_info['platform']}\n")
            if self.metadata.git_commit:
                f.write(f"- **Git commit**: `{self.metadata.git_commit}`\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## Ethical Considerations\n\n")
            
            ethics_checks = self.results['ethics']['validation']
            f.write("**Ethics Checklist**:\n\n")
            for check, passed in ethics_checks.items():
                status = '✓' if passed else '✗'
                f.write(f"- {status} {check.replace('_', ' ').title()}\n")
            f.write("\n")
            
            f.write("### Guidelines Applied\n\n")
            for guideline, description in self.results['ethics']['guidelines'].items():
                f.write(f"**{guideline.replace('_', ' ').title()}**\n\n")
                f.write(f"{description}\n\n")
            
            f.write("---\n\n")
            f.write("## Citation\n\n")
            f.write("```bibtex\n")
            f.write("@software{benseddik2025ancient,\n")
            f.write("  author = {Benseddik, Ahmed},\n")
            f.write("  title = {Ancient Text Numerical Analysis Framework},\n")
            f.write("  year = {2025},\n")
            f.write("  version = {5.0-DSH},\n")
            f.write("  doi = {10.5281/zenodo.17487211},\n")
            f.write("  url = {https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4}\n")
            f.write("}\n")
            f.write("```\n\n")
            
            f.write("---\n\n")
            f.write("*This report was automatically generated by the Ancient Text Numerical Analysis Framework v5.0*\n")
        
        logger.info(f"Saved report: {report_file}")
    
    def _export_tables_to_csv(self, timestamp: str):
        """Export key tables to CSV format."""
        csv_dir = self.config.output_dir / 'tables'
        csv_dir.mkdir(exist_ok=True)
        
        # Multiples results table
        if 'multiples_frequentist' in self.results:
            mf = self.results['multiples_frequentist']
            rows = []
            
            for key, data in mf['divisor_results'].items():
                divisor = int(key.split('_')[1])
                binomial = data['binomial_test']
                
                rows.append({
                    'divisor': divisor,
                    'observed': binomial['metadata']['k'],
                    'expected': binomial['metadata']['n'] / divisor,
                    'p_value': binomial['p_value'],
                    'effect_size': binomial['effect_size'],
                    'ci_lower': binomial['confidence_interval'][0],
                    'ci_upper': binomial['confidence_interval'][1],
                    'bonferroni_significant': data['bonferroni_significant'],
                    'effect_interpretation': data['effect_interpretation']
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_dir / f'multiples_frequentist_{timestamp}.csv', index=False)
            logger.info("Exported multiples table to CSV")
        
        # Cross-cultural correlations
        if 'gematria' in self.results and 'cross_cultural' in self.results['gematria']:
            cc = self.results['gematria']['cross_cultural']
            if cc.get('correlations'):
                df_corr = pd.DataFrame(cc['correlations'])
                df_corr.to_csv(csv_dir / f'cross_cultural_correlations_{timestamp}.csv')
                logger.info("Exported cross-cultural correlations to CSV")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ancient Text Numerical Analysis Framework v5.0 - DSH Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete analysis with default settings
    python ancient_text_framework_v5.py --data-dir ./data --output-dir ./results
    
    # Quick analysis without Bayesian (faster)
    python ancient_text_framework_v5.py --no-bayesian
    
    # High-quality analysis with more iterations
    python ancient_text_framework_v5.py --n-permutations 50000 --n-bayesian-draws 5000
    
    # Use TEI XML corpus format
    python ancient_text_framework_v5.py --corpus-format tei --data-dir ./tei_corpus
    
    # Minimal logging
    python ancient_text_framework_v5.py --quiet

Documentation: https://ancient-text-analysis.readthedocs.io
Repository: https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
DOI: 10.5281/zenodo.17487211
        """
    )
    
    parser.add_argument('--data-dir', type=Path, default