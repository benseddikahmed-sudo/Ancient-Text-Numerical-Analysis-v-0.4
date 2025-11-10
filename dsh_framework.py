#!/usr/bin/env python3
"""
Ancient Text Numerical Analysis Framework — Digital Scholarship Edition
========================================================================

A rigorous, reproducible framework for computational analysis of numerical
patterns in ancient texts with comprehensive documentation, validation,
and ethical considerations for digital humanities research.

Publication: Digital Scholarship in the Humanities (DSH)
Author: Ahmed Benseddik <benseddik.ahmed@gmail.com>
Version: 4.0-DSH
Date: 2025-10-26
License: MIT
DOI: [To be assigned]

Citation:
    Benseddik, A. (2025). Ancient Text Numerical Analysis: A Statistical
    Framework with Ethical Considerations. Digital Scholarship in the
    Humanities. [DOI]

Dependencies:
    Core: numpy>=1.24, scipy>=1.10, pandas>=2.0
    Visualization: matplotlib>=3.7, seaborn>=0.12
    Bayesian: pymc>=5.0, arviz>=0.15
    Performance: numba>=0.57 (optional)
    Testing: pytest>=7.0, hypothesis>=6.0

Repository: https://github.com/[username]/ancient-text-analysis
Documentation: https://ancient-text-analysis.readthedocs.io
"""

import argparse
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Protocol, Iterator
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
    # Fallback decorator that does nothing
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
    """Configure logging with file and console handlers."""
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
# REPRODUCIBILITY & VALIDATION
# ============================================================================

@dataclass
class ReproducibilityMetadata:
    """Complete metadata for computational reproducibility."""
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    random_seed: int
    system_info: Dict[str, str]
    git_commit: Optional[str] = None
    dataset_hash: Optional[str] = None
    
    @classmethod
    def capture(cls, seed: int = 42) -> 'ReproducibilityMetadata':
        """Capture current environment metadata."""
        import platform
        import hashlib
        
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
    """Comprehensive validation tests for analysis results."""
    
    @staticmethod
    def validate_distribution(data: np.ndarray) -> Dict[str, Any]:
        """Test data distribution properties."""
        return {
            'shapiro_wilk': stats.shapiro(data),
            'dagostino_k2': stats.normaltest(data),
            'jarque_bera': stats.jarque_bera(data),
            'anderson_darling': stats.anderson(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
        }
    
    @staticmethod
    def validate_sample_size(n: int, effect_size: float, alpha: float = 0.05) -> Dict[str, Any]:
        """Assess if sample size is adequate."""
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(0.8)  # 80% power
        required_n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return {
            'observed_n': n,
            'required_n_80pct': int(np.ceil(required_n)),
            'is_adequate': n >= required_n,
            'achieved_power': norm.cdf((effect_size * np.sqrt(n)) - z_alpha),
        }
    
    @staticmethod
    def validate_multiple_testing(n_tests: int, alpha: float = 0.05) -> Dict[str, float]:
        """Calculate multiple testing corrections."""
        return {
            'bonferroni': alpha / n_tests,
            'sidak': 1 - (1 - alpha) ** (1 / n_tests),
            'fdr_bh': alpha,  # Benjamini-Hochberg uses adaptive threshold
            'family_wise_error_rate': 1 - (1 - alpha) ** n_tests,
        }

# ============================================================================
# CULTURAL SYSTEMS - ENHANCED WITH VALIDATION
# ============================================================================

class CulturalSystem(Enum):
    """Supported cultural numerical systems with metadata."""
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

# Gematria mappings (unchanged but with validation)
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

@jit(nopython=NUMBA_AVAILABLE)
def _gematria_compute_fast(chars: np.ndarray, values: np.ndarray) -> int:
    """Numba-optimized gematria computation."""
    total = 0
    for char in chars:
        for i in range(len(values)):
            if char == values[i, 0]:
                total += int(values[i, 1])
                break
    return total

def compute_gematria(word: str, system: CulturalSystem = CulturalSystem.HEBREW_STANDARD) -> int:
    """
    Compute numerical value using specified cultural system.
    
    Args:
        word: Input text string
        system: Cultural numerical system
    
    Returns:
        Integer numerical value
    
    Raises:
        ValueError: If input contains invalid characters for system
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
# STATISTICAL ANALYSIS - ENHANCED
# ============================================================================

@dataclass
class StatisticalResult:
    """Comprehensive statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    assumptions_met: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class RobustStatisticalTests:
    """Suite of robust statistical tests with validation."""
    
    @staticmethod
    def binomial_test_robust(k: int, n: int, p: float, 
                            alternative: str = 'two-sided') -> StatisticalResult:
        """
        Robust binomial test with effect size and confidence intervals.
        
        Args:
            k: Number of successes
            n: Number of trials
            p: Expected probability
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            StatisticalResult with comprehensive statistics
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
        """Wilson score confidence interval for proportions."""
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
        
        Args:
            observed: Observed data
            expected: Expected/control data
            n_permutations: Number of random permutations
            statistic_func: Function to compute test statistic
        
        Returns:
            StatisticalResult
        """
        observed_stat = statistic_func(observed) - statistic_func(expected)
        
        combined = np.concatenate([observed, expected])
        n_obs = len(observed)
        
        perm_stats = np.zeros(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
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
            assumptions_met={'non_parametric': True},
            metadata={'n_permutations': n_permutations}
        )

# ============================================================================
# BAYESIAN ANALYSIS - ENHANCED
# ============================================================================

class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for multiples analysis.
    
    Implements model comparison using WAIC/LOO and posterior predictive checks.
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
        """Fit Bayesian model for given divisor."""
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
        """Compare all fitted models using WAIC."""
        if len(self.traces) < 2:
            logger.warning("Need at least 2 models to compare")
            return pd.DataFrame()
        
        comparison = az.compare(self.traces, ic='waic')
        self.comparisons['waic'] = comparison
        
        return comparison
    
    def posterior_summary(self, divisor: int) -> pd.DataFrame:
        """Get posterior summary statistics."""
        if divisor not in self.traces:
            raise ValueError(f"Model for divisor {divisor} not fitted")
        
        return az.summary(self.traces[divisor], var_names=['p'])
    
    def plot_posterior(self, divisor: int, save_path: Optional[Path] = None):
        """Plot posterior distribution with diagnostics."""
        if divisor not in self.traces:
            raise ValueError(f"Model for divisor {divisor} not fitted")
        
        trace = self.traces[divisor]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Posterior distribution
        az.plot_posterior(trace, var_names=['p'], ax=axes[0, 0])
        axes[0, 0].axvline(1/divisor, color='red', linestyle='--', 
                          label=f'Expected (1/{divisor})')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'Posterior Distribution - Divisor {divisor}')
        
        # Trace plot
        az.plot_trace(trace, var_names=['p'], axes=axes[0, 1:])
        
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
    """Configuration for analysis pipeline."""
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
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class AncientTextAnalysisPipeline:
    """
    Main analysis pipeline for ancient text numerical analysis.
    
    Implements complete workflow from data loading to publication-ready output.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.metadata = ReproducibilityMetadata.capture(config.random_seed)
        self.results = {}
        self.validation_suite = ValidationSuite()
        
        # Set random seeds
        np.random.seed(config.random_seed)
        
        # Setup logging
        log_file = config.output_dir / 'analysis.log'
        setup_logging(
            level=logging.INFO if config.verbose else logging.WARNING,
            log_file=log_file
        )
        
        logger.info("=" * 80)
        logger.info("Ancient Text Numerical Analysis Framework - DSH Edition")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config}")
        logger.info(f"Reproducibility metadata: {self.metadata}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis pipeline."""
        
        # 1. Load and validate data
        text_data = self._load_data()
        
        # 2. Gematria analysis
        self.results['gematria'] = self._analyze_gematria(text_data)
        
        # 3. Multiples analysis (Frequentist)
        self.results['multiples_frequentist'] = self._analyze_multiples_frequentist(text_data)
        
        # 4. Bayesian analysis
        if self.config.enable_bayesian and PYMC_AVAILABLE:
            self.results['multiples_bayesian'] = self._analyze_multiples_bayesian(text_data)
        
        # 5. Sensitivity analysis
        self.results['sensitivity'] = self._sensitivity_analysis(text_data)
        
        # 6. Generate visualizations
        if self.config.save_figures:
            self._generate_visualizations()
        
        # 7. Save results
        self._save_results()
        
        logger.info("✓ Complete analysis finished successfully")
        return self.results
    
    def _load_data(self) -> str:
        """Load and validate input text data."""
        logger.info("Loading data...")
        
        text_file = self.config.data_dir / 'text.txt'
        if not text_file.exists():
            logger.warning(f"Data file not found: {text_file}. Using placeholder.")
            text = self._generate_placeholder_text()
        else:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Validate and clean
        hebrew_chars = set('אבגדהוזחטיכלמנסעפצקרשתךםןףץ')
        text = ''.join(c for c in text if c in hebrew_chars)
        
        logger.info(f"Loaded text: {len(text)} characters")
        
        if len(text) < 100:
            logger.warning("Text is very short. Results may not be reliable.")
        
        return text
    
    def _generate_placeholder_text(self) -> str:
        """Generate placeholder text for demonstration."""
        words = ['בראשית', 'ברא', 'אלהים', 'את', 'השמים', 'והארץ']
        return ''.join(words * 300)
    
    def _analyze_gematria(self, text: str) -> Dict[str, Any]:
        """Comprehensive gematria analysis."""
        logger.info("Running gematria analysis...")
        
        # Extract words (5-char windows)
        words = [text[i:i+5] for i in range(0, len(text)-4, 5)]
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
        }
        
        # Distribution validation
        distribution_tests = self.validation_suite.validate_distribution(values)
        
        # Cross-cultural comparison
        cross_cultural = self._cross_cultural_comparison(words[:50])
        
        return {
            'summary_statistics': summary,
            'distribution_tests': {k: str(v) for k, v in distribution_tests.items()},
            'cross_cultural': cross_cultural,
            'sample_values': values[:100].tolist(),
        }
    
    def _cross_cultural_comparison(self, words: List[str]) -> Dict[str, Any]:
        """Compare values across cultural systems."""
        results = defaultdict(list)
        
        for word in words:
            for system in CulturalSystem:
                try:
                    value = compute_gematria(word, system)
                    results[system.display_name].append(value)
                except Exception as e:
                    logger.warning(f"Error computing {system}: {e}")
        
        # Correlation matrix
        df = pd.DataFrame(results)
        correlation = df.corr().to_dict()
        
        return {
            'systems': list(results.keys()),
            'sample_size': len(words),
            'correlations': correlation,
        }
    
    def _analyze_multiples_frequentist(self, text: str) -> Dict[str, Any]:
        """Frequentist analysis of multiples."""
        logger.info("Running frequentist multiples analysis...")
        
        words = [text[i:i+5] for i in range(0, len(text)-4, 10)]
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
                values % divisor == 0,
                random_values % divisor == 0,
                n_permutations=self.config.n_permutations
            )
            
            results[f'divisor_{divisor}'] = {
                'binomial_test': stat_result.to_dict(),
                'permutation_test': perm_result.to_dict(),
                'bonferroni_significant': stat_result.p_value < corrections['bonferroni'],
                'sidak_significant': stat_result.p_value < corrections['sidak'],
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
        
        words = [text[i:i+5] for i in range(0, len(text)-4, 10)]
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
                    plot_path = self.config.output_dir / f'posterior_divisor_{divisor}.png'
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
        waic_diff = comparison.loc[best_model, 'waic'] - comparison.iloc[1]['waic']
        
        if abs(waic_diff) < 2:
            return "Models are essentially equivalent (ΔWAIC < 2)"
        elif abs(waic_diff) < 6:
            return f"Weak evidence for {best_model} (2 < ΔWAIC < 6)"
        else:
            return f"Strong evidence for {best_model} (ΔWAIC > 6)"
    
    def _sensitivity_analysis(self, text: str) -> Dict[str, Any]:
        """Sensitivity analysis with different parameters."""
        logger.info("Running sensitivity analysis...")
        
        # Test different window sizes
        window_sizes = [3, 5, 7, 10]
        sensitivity_results = {}
        
        for window_size in window_sizes:
            words = [text[i:i+window_size] for i in range(0, len(text)-window_size+1, window_size)]
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
        original_words = [text[i:i+5] for i in range(0, len(text)-4, 5)]
        
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
        
        # Histogram
        axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(np.mean(values), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(values):.1f}')
        axes[0].axvline(np.median(values), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(values):.1f}')
        axes[0].set_xlabel('Gematria Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Gematria Values', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot
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
        
        for key, data in results.items():
            divisor = int(key.split('_')[1])
            divisors.append(divisor)
            
            binomial = data['binomial_test']
            observed.append(binomial['metadata']['k'])
            expected.append(binomial['metadata']['n'] / divisor)
            p_values.append(binomial['p_value'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot: Observed vs Expected
        x = np.arange(len(divisors))
        width = 0.35
        
        axes[0].bar(x - width/2, observed, width, label='Observed', 
                   color='steelblue', edgecolor='black')
        axes[0].bar(x + width/2, expected, width, label='Expected', 
                   color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Divisor', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Multiples Analysis: Observed vs Expected', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(divisors)
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        # P-values plot
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        axes[1].bar(divisors, [-np.log10(p) for p in p_values], 
                   color=colors, edgecolor='black', alpha=0.7)
        axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', 
                       linewidth=2, label='α = 0.05')
        axes[1].set_xlabel('Divisor', fontsize=12)
        axes[1].set_ylabel('-log₁₀(p-value)', fontsize=12)
        axes[1].set_title('Statistical Significance', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'multiples_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_cultural_heatmap(self, viz_dir: Path):
        """Plot cross-cultural correlation heatmap."""
        correlations = self.results['gematria']['cross_cultural'].get('correlations', {})
        
        if not correlations:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(correlations)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
        plt.title('Cross-Cultural Gematria System Correlations', 
                 fontsize=14, fontweight='bold', pad=20)
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
        
        for key, data in window_results.items():
            window = int(key.split('_')[1])
            windows.append(window)
            p_values.append(data['p_value'])
            sample_sizes.append(data['n'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # P-values vs window size
        axes[0].plot(windows, p_values, marker='o', linewidth=2, 
                    markersize=8, color='steelblue')
        axes[0].axhline(0.05, color='red', linestyle='--', 
                       linewidth=2, label='α = 0.05')
        axes[0].set_xlabel('Window Size', fontsize=12)
        axes[0].set_ylabel('P-value', fontsize=12)
        axes[0].set_title('Sensitivity to Window Size', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Sample size vs window size
        axes[1].plot(windows, sample_sizes, marker='s', linewidth=2, 
                    markersize=8, color='coral')
        axes[1].set_xlabel('Window Size', fontsize=12)
        axes[1].set_ylabel('Sample Size (n)', fontsize=12)
        axes[1].set_title('Sample Size by Window Size', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
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
            'timestamp': timestamp
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved results: {results_file}")
        
        # Human-readable report
        report_file = self.config.output_dir / f'report_{timestamp}.md'
        self._generate_markdown_report(report_file)
        
        # CSV exports for tables
        self._export_tables_to_csv(timestamp)
    
    def _generate_markdown_report(self, report_file: Path):
        """Generate markdown report for publication."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Ancient Text Numerical Analysis Report\n\n")
            f.write(f"**Generated**: {self.metadata.timestamp}\n\n")
            f.write(f"**Random Seed**: {self.metadata.random_seed}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Gematria summary
            if 'gematria' in self.results:
                ga = self.results['gematria']['summary_statistics']
                f.write(f"### Gematria Analysis\n\n")
                f.write(f"- **Sample size**: {ga['n']}\n")
                f.write(f"- **Mean**: {ga['mean']:.2f}\n")
                f.write(f"- **Median**: {ga['median']:.2f}\n")
                f.write(f"- **Standard deviation**: {ga['std']:.2f}\n\n")
            
            # Multiples summary
            if 'multiples_frequentist' in self.results:
                mf = self.results['multiples_frequentist']
                f.write(f"### Multiples Analysis (Frequentist)\n\n")
                f.write(f"**Sample size**: {mf['sample_size']}\n\n")
                f.write(f"**Interpretation**: {mf['interpretation']}\n\n")
                
                f.write("| Divisor | Observed | Expected | P-value | Bonferroni Sig. |\n")
                f.write("|---------|----------|----------|---------|------------------|\n")
                
                for key, data in mf['divisor_results'].items():
                    divisor = key.split('_')[1]
                    binomial = data['binomial_test']
                    obs = binomial['metadata']['k']
                    exp = binomial['metadata']['n'] / int(divisor)
                    p_val = binomial['p_value']
                    sig = '✓' if data['bonferroni_significant'] else '✗'
                    f.write(f"| {divisor} | {obs} | {exp:.2f} | {p_val:.4f} | {sig} |\n")
                
                f.write("\n")
            
            # Bayesian summary
            if 'multiples_bayesian' in self.results and 'error' not in self.results['multiples_bayesian']:
                mb = self.results['multiples_bayesian']
                f.write(f"### Bayesian Analysis\n\n")
                f.write(f"**Method**: {mb['method']}\n\n")
                f.write(f"**Interpretation**: {mb['interpretation']}\n\n")
            
            # Sensitivity
            if 'sensitivity' in self.results:
                sens = self.results['sensitivity']
                f.write(f"### Sensitivity Analysis\n\n")
                f.write(f"**Interpretation**: {sens['interpretation']}\n\n")
            
            f.write("## Reproducibility\n\n")
            f.write(f"- **Python version**: {self.metadata.python_version.split()[0]}\n")
            f.write(f"- **NumPy version**: {self.metadata.numpy_version}\n")
            f.write(f"- **Random seed**: {self.metadata.random_seed}\n")
            if self.metadata.git_commit:
                f.write(f"- **Git commit**: `{self.metadata.git_commit}`\n")
            f.write("\n")
            
            f.write("## Ethical Considerations\n\n")
            f.write("- This analysis applies statistical methods to cultural artifacts\n")
            f.write("- Numerical patterns do not imply intentionality or specific meanings\n")
            f.write("- Results should be interpreted within appropriate cultural contexts\n")
            f.write("- Multiple valid interpretations may exist beyond quantitative findings\n")
        
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
                    'bonferroni_significant': data['bonferroni_significant']
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_dir / f'multiples_frequentist_{timestamp}.csv', index=False)
            logger.info(f"Exported multiples table to CSV")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ancient Text Numerical Analysis Framework - DSH Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete analysis
    python ancient_text_dsh.py --data-dir ./data --output-dir ./results
    
    # Quick analysis without Bayesian (faster)
    python ancient_text_dsh.py --no-bayesian
    
    # High-quality analysis with more permutations
    python ancient_text_dsh.py --n-permutations 50000 --n-bayesian-draws 5000
    
    # Generate only visualizations from existing results
    python ancient_text_dsh.py --visualizations-only

For documentation: https://ancient-text-analysis.readthedocs.io
For issues: https://github.com/[username]/ancient-text-analysis/issues
        """
    )
    
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Input data directory (default: data)')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                       help='Output directory for results (default: output)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n-permutations', type=int, default=10000,
                       help='Number of permutation test iterations (default: 10000)')
    parser.add_argument('--n-bayesian-draws', type=int, default=2000,
                       help='Number of Bayesian MCMC draws (default: 2000)')
    parser.add_argument('--no-bayesian', action='store_false', dest='enable_bayesian',
                       help='Disable Bayesian analysis')
    parser.add_argument('--no-parallel', action='store_false', dest='enable_parallel',
                       help='Disable parallel processing')
    parser.add_argument('--no-figures', action='store_false', dest='save_figures',
                       help='Do not generate figures')
    parser.add_argument('--significance-level', type=float, default=0.05,
                       help='Statistical significance level (default: 0.05)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Minimal logging output')
    parser.add_argument('--version', action='version', version='%(prog)s 4.0-DSH')
    
    return parser

def main():
    """Main execution function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
        n_permutations=args.n_permutations,
        n_bayesian_draws=args.n_bayesian_draws,
        enable_bayesian=args.enable_bayesian and PYMC_AVAILABLE,
        enable_parallel=args.enable_parallel,
        significance_level=args.significance_level,
        save_figures=args.save_figures,
        verbose=args.verbose
    )
    
    # Check dependencies
    if config.enable_bayesian and not PYMC_AVAILABLE:
        logger.warning("PyMC not available. Bayesian analysis disabled.")
        logger.warning("Install with: pip install pymc arviz")
        config.enable_bayesian = False
    
    if NUMBA_AVAILABLE:
        logger.info("Numba acceleration enabled")
    
    # Run pipeline
    try:
        pipeline = AncientTextAnalysisPipeline(config)
        results = pipeline.run_complete_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {config.output_dir}")
        print(f"Figures saved to: {config.output_dir / 'figures'}")
        print(f"Tables saved to: {config.output_dir / 'tables'}")
        print("\nNext steps:")
        print("  1. Review the markdown report for summary")
        print("  2. Check figures for visual analysis")
        print("  3. Examine JSON file for complete data")
        print("  4. Cite using provided DOI and metadata")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())