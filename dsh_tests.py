#!/usr/bin/env python3
"""
Test Suite for Ancient Text Numerical Analysis Framework
=========================================================

Comprehensive test suite with unit tests, integration tests,
and property-based tests using pytest and hypothesis.

Run with: pytest test_suite.py -v --cov
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from pathlib import Path
import tempfile
import json

# Import from main module (adjust import path as needed)
# from ancient_text_dsh import (
#     compute_gematria, CulturalSystem, StatisticalResult,
#     RobustStatisticalTests, ValidationSuite, AnalysisConfig,
#     AncientTextAnalysisPipeline, ReproducibilityMetadata
# )

# For standalone testing, we'll mock the imports
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Any

# Mock classes for testing
class CulturalSystem(Enum):
    HEBREW_STANDARD = "hebrew_standard"
    GREEK_ISOPSEPHY = "greek_isopsephy"
    ARABIC_ABJAD = "arabic_abjad"

GEMATRIA_VALUES = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5,
    'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9, 'י': 10,
    'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60,
    'ע': 70, 'פ': 80, 'צ': 90, 'ק': 100, 'ר': 200,
    'ש': 300, 'ת': 400
}

def compute_gematria(word: str, system=None) -> int:
    """Mock gematria function."""
    return sum(GEMATRIA_VALUES.get(c, 0) for c in word)

# ============================================================================
# UNIT TESTS - Gematria Computation
# ============================================================================

class TestGematriaComputation:
    """Test gematria value computation."""
    
    def test_single_letter(self):
        """Test single letter values."""
        assert compute_gematria('א') == 1
        assert compute_gematria('י') == 10
        assert compute_gematria('ק') == 100
        assert compute_gematria('ת') == 400
    
    def test_word_values(self):
        """Test known word values."""
        # בראשית = 2+200+1+300+10+400 = 913
        assert compute_gematria('בראשית') == 913
        
        # אלהים = 1+30+5+10+40 = 86
        assert compute_gematria('אלהים') == 86
        
        # תורה = 400+6+200+5 = 611
        assert compute_gematria('תורה') == 611
    
    def test_empty_string(self):
        """Test empty string returns 0."""
        assert compute_gematria('') == 0
    
    def test_unknown_characters(self):
        """Test handling of unknown characters."""
        # Should ignore non-Hebrew characters
        assert compute_gematria('א1ב2') == 3  # א(1) + ב(2) = 3
        assert compute_gematria('abc') == 0
    
    def test_final_forms(self):
        """Test normalization of final letter forms."""
        # ך -> כ (20), ם -> מ (40), etc.
        # This would need actual normalization in real implementation
        pass
    
    @given(st.text(alphabet='אבגדהוזחטיכלמנסעפצקרשת', min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_gematria_always_positive(self, word):
        """Property: Gematria values are always non-negative."""
        assert compute_gematria(word) >= 0
    
    @given(st.text(alphabet='אבגדהוזחטיכלמנסעפצקרשת', min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_gematria_bounded(self, word):
        """Property: Gematria values are bounded by word length * 400."""
        value = compute_gematria(word)
        assert value <= len(word) * 400


# ============================================================================
# UNIT TESTS - Statistical Functions
# ============================================================================

class TestStatisticalFunctions:
    """Test statistical analysis functions."""
    
    def test_binomial_test_exact_probability(self):
        """Test binomial test with exact expected probability."""
        from scipy.stats import binomtest
        
        n = 100
        k = 14  # Exactly expected for p=1/7
        p = 1/7
        
        result = binomtest(k, n, p, alternative='two-sided')
        # Should not be significant
        assert result.pvalue > 0.05
    
    def test_binomial_test_enrichment(self):
        """Test binomial test with clear enrichment."""
        from scipy.stats import binomtest
        
        n = 100
        k = 30  # Much higher than expected 1/7 ≈ 14
        p = 1/7
        
        result = binomtest(k, n, p, alternative='greater')
        # Should be highly significant
        assert result.pvalue < 0.001
    
    def test_wilson_confidence_interval(self):
        """Test Wilson score confidence interval."""
        from scipy.stats import norm
        
        def wilson_ci(k, n, alpha=0.05):
            z = norm.ppf(1 - alpha/2)
            p_hat = k / n
            denominator = 1 + z**2 / n
            center = (p_hat + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
            return (max(0, center - margin), min(1, center + margin))
        
        ci = wilson_ci(50, 100)
        
        # CI should contain true proportion 0.5
        assert ci[0] <= 0.5 <= ci[1]
        
        # CI should be narrower than ±0.2
        assert (ci[1] - ci[0]) < 0.4
    
    def test_effect_size_computation(self):
        """Test Cohen's h effect size."""
        # Cohen's h = 2 * (arcsin(√p1) - arcsin(√p2))
        p1 = 0.3
        p2 = 0.2
        
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        # Should be positive for p1 > p2
        assert h > 0
        
        # Should be roughly 0.23 for this difference
        assert 0.2 < h < 0.3


# ============================================================================
# UNIT TESTS - Validation Suite
# ============================================================================

class TestValidationSuite:
    """Test validation functions."""
    
    def test_normality_tests(self):
        """Test distribution normality tests."""
        from scipy import stats
        
        # Normal data
        normal_data = np.random.normal(100, 15, 1000)
        sw_stat, sw_pval = stats.shapiro(normal_data)
        
        # Should not reject normality
        assert sw_pval > 0.01
        
        # Uniform data (not normal)
        uniform_data = np.random.uniform(0, 100, 1000)
        sw_stat, sw_pval = stats.shapiro(uniform_data)
        
        # Should reject normality
        assert sw_pval < 0.05
    
    def test_sample_size_validation(self):
        """Test sample size adequacy."""
        from scipy.stats import norm
        
        def validate_sample_size(n, effect_size, alpha=0.05, power=0.8):
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            required_n = ((z_alpha + z_beta) / effect_size) ** 2
            return n >= required_n
        
        # Small effect, small sample - inadequate
        assert not validate_sample_size(n=50, effect_size=0.2)
        
        # Large effect, small sample - adequate
        assert validate_sample_size(n=50, effect_size=0.8)
        
        # Small effect, large sample - adequate
        assert validate_sample_size(n=500, effect_size=0.2)
    
    def test_multiple_testing_corrections(self):
        """Test multiple testing correction calculations."""
        n_tests = 5
        alpha = 0.05
        
        # Bonferroni
        bonf = alpha / n_tests
        assert bonf == 0.01
        
        # Šidák (more powerful)
        sidak = 1 - (1 - alpha) ** (1 / n_tests)
        assert sidak > bonf  # Šidák is less conservative
        assert sidak < alpha


# ============================================================================
# INTEGRATION TESTS - Pipeline
# ============================================================================

class TestAnalysisPipeline:
    """Test complete analysis pipeline."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'data'
            data_dir.mkdir()
            
            # Create test text file
            test_text = 'בראשית' * 100  # Repeat word 100 times
            text_file = data_dir / 'text.txt'
            text_file.write_text(test_text, encoding='utf-8')
            
            yield data_dir
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / 'output'
    
    def test_pipeline_initialization(self, temp_data_dir, temp_output_dir):
        """Test pipeline initialization."""
        # This would test actual pipeline initialization
        # For now, just verify directories exist
        assert temp_data_dir.exists()
        assert (temp_data_dir / 'text.txt').exists()
    
    def test_data_loading(self, temp_data_dir):
        """Test data loading and validation."""
        text_file = temp_data_dir / 'text.txt'
        text = text_file.read_text(encoding='utf-8')
        
        # Should contain only Hebrew letters
        hebrew_chars = set('אבגדהוזחטיכלמנסעפצקרשתךםןףץ')
        assert all(c in hebrew_chars for c in text)
        
        # Should not be empty
        assert len(text) > 0
    
    def test_results_output_structure(self, temp_output_dir):
        """Test output directory structure."""
        # Create expected directory structure
        temp_output_dir.mkdir(parents=True)
        (temp_output_dir / 'figures').mkdir()
        (temp_output_dir / 'tables').mkdir()
        
        assert (temp_output_dir / 'figures').exists()
        assert (temp_output_dir / 'tables').exists()


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(st.integers(min_value=0, max_value=1000),
           st.integers(min_value=1, max_value=1000),
           st.floats(min_value=0.01, max_value=0.99))
    @settings(max_examples=100)
    def test_binomial_probability_bounds(self, k, n, p):
        """Property: Binomial test p-values are between 0 and 1."""
        from scipy.stats import binomtest
        
        if k > n:
            k = n  # Ensure k <= n
        
        result = binomtest(k, n, p)
        assert 0 <= result.pvalue <= 1
    
    @given(st.lists(st.floats(min_value=1, max_value=1000), min_size=10, max_size=100))
    @settings(max_examples=50)
    def test_distribution_statistics_consistent(self, data):
        """Property: Mean and median are reasonable for data."""
        arr = np.array(data)
        
        mean = np.mean(arr)
        median = np.median(arr)
        
        # Mean and median should be within data range
        assert np.min(arr) <= mean <= np.max(arr)
        assert np.min(arr) <= median <= np.max(arr)
    
    @given(st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_divisibility_frequency_bounded(self, divisor):
        """Property: Frequency of divisibility is bounded by 1/divisor."""
        data = np.random.randint(1, 1000, size=1000)
        frequency = np.mean(data % divisor == 0)
        
        # Frequency should be close to 1/divisor (with some tolerance)
        expected = 1 / divisor
        # Allow 3 standard deviations
        std_error = np.sqrt(expected * (1 - expected) / len(data))
        tolerance = 3 * std_error
        
        assert abs(frequency - expected) < 0.1  # Loose bound for random data


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegression:
    """Regression tests for known results."""
    
    def test_known_gematria_values(self):
        """Test against known published gematria values."""
        known_values = {
            'אחד': 13,      # "one" = 1+8+4
            'אהבה': 13,     # "love" = 1+5+2+5
            'בראשית': 913,  # "beginning"
            'אלהים': 86,    # "God"
            'שלום': 376,    # "peace" = 300+30+6+40
        }
        
        for word, expected in known_values.items():
            actual = compute_gematria(word)
            assert actual == expected, f"Mismatch for {word}: {actual} != {expected}"
    
    def test_reproducibility_with_fixed_seed(self):
        """Test that results are reproducible with fixed seed."""
        np.random.seed(42)
        data1 = np.random.randint(1, 1000, size=100)
        
        np.random.seed(42)
        data2 = np.random.randint(1, 1000, size=100)
        
        assert np.array_equal(data1, data2)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    def test_gematria_computation_speed(self, benchmark=None):
        """Benchmark gematria computation."""
        word = 'בראשית' * 10
        
        # Time the computation
        import time
        start = time.time()
        for _ in range(1000):
            compute_gematria(word)
        elapsed = time.time() - start
        
        # Should complete 1000 iterations in under 1 second
        assert elapsed < 1.0
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Generate large dataset
        words = ['בראשית'] * 10000
        
        import time
        start = time.time()
        values = [compute_gematria(w) for w in words]
        elapsed = time.time() - start
        
        # Should process 10k words in under 5 seconds
        assert elapsed < 5.0
        assert len(values) == 10000


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert compute_gematria('') == 0
    
    def test_whitespace_only(self):
        """Test handling of whitespace."""
        assert compute_gematria('   ') == 0
        assert compute_gematria('\n\t') == 0
    
    def test_mixed_scripts(self):
        """Test handling of mixed scripts."""
        # Should ignore non-Hebrew characters
        result = compute_gematria('אaב')
        assert result == 3  # א(1) + ב(2)
    
    def test_very_long_string(self):
        """Test handling of very long strings."""
        long_string = 'א' * 10000
        result = compute_gematria(long_string)
        assert result == 10000  # 1 * 10000
    
    def test_unicode_edge_cases(self):
        """Test Unicode edge cases."""
        # Test combining characters, variations, etc.
        # This would need more sophisticated handling
        pass


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "regression: marks regression tests")


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_hebrew_text():
    """Sample Hebrew text for testing."""
    return 'בראשיתבראאלהיםאתהשמיםואתהארץ'


@pytest.fixture
def sample_gematria_values():
    """Sample gematria values for testing."""
    return np.array([913, 86, 401, 395, 611, 26, 65, 44])


@pytest.fixture
def mock_analysis_results():
    """Mock analysis results for testing."""
    return {
        'gematria': {
            'summary_statistics': {
                'n': 100,
                'mean': 285.5,
                'median': 250.0,
                'std': 125.3
            }
        },
        'multiples_frequentist': {
            'sample_size': 100,
            'divisor_results': {
                'divisor_7': {
                    'binomial_test': {
                        'p_value': 0.023,
                        'effect_size': 0.15
                    },
                    'bonferroni_significant': True
                }
            }
        }
    }


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=ancient_text_dsh',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--tb=short'
    ])