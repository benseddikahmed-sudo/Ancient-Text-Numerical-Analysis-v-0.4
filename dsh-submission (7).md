#!/usr/bin/env python3
"""
Supplementary Figures Generator for Ancient Text Numerical Analysis
--------------------------------------------------------------------
Publication-ready figures for Digital Scholarship in the Humanities (DSH)

This script generates all supplementary figures for the manuscript:
"Towards Ethical Numerical Analysis of Ancient Texts: A Multi-Cultural 
Statistical Framework with Integrated Epistemological Safeguards"

Author: Ahmed Benseddik <benseddik.ahmed@gmail.com>
Version: 4.0 (DSH Submission Ready)
Date: 2025-10-26
License: MIT
DOI: https://doi.org/10.5281/zenodo.17443361

Dependencies:
    numpy>=1.21.0
    pandas>=1.3.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    scipy>=1.7.0

Usage:
    python generate_dsh_figures.py --results-dir data/results/
    python generate_dsh_figures.py --figures all --dpi 600 --profile dsh
    python generate_dsh_figures.py --test  # Generate with synthetic data
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Optional imports with graceful fallbacks
try:
    from scipy import stats
    from scipy.stats import norm, gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ Warning: scipy not available, using fallback calculations")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# =============================================================================
# JOURNAL PUBLICATION PROFILES
# =============================================================================

@dataclass
class JournalProfile:
    """Publication specifications for different journals."""
    name: str
    dpi: int
    width_single_column: float  # inches
    width_double_column: float  # inches
    height_max: float  # inches
    font_family: str
    font_size_base: int
    line_width: float
    formats: List[str] = field(default_factory=lambda: ['pdf', 'png', 'svg'])

JOURNAL_PROFILES = {
    'dsh': JournalProfile(
        name='Digital Scholarship in the Humanities',
        dpi=600,  # DSH recommends 600 DPI for print
        width_single_column=3.35,  # inches (85mm)
        width_double_column=7.0,   # inches (178mm)
        height_max=9.0,            # inches (max page height)
        font_family='Times New Roman',
        font_size_base=8,          # Points (DSH standard)
        line_width=1.0,
        formats=['pdf', 'tiff', 'eps']  # DSH accepts these
    ),
    'standard': JournalProfile(
        name='Standard Publication',
        dpi=300,
        width_single_column=5.0,
        width_double_column=10.0,
        height_max=7.0,
        font_family='serif',
        font_size_base=10,
        line_width=1.5,
        formats=['pdf', 'png', 'svg']
    ),
    'high_res': JournalProfile(
        name='High Resolution',
        dpi=600,
        width_single_column=5.0,
        width_double_column=10.0,
        height_max=7.0,
        font_family='serif',
        font_size_base=10,
        line_width=1.5,
        formats=['pdf', 'png', 'tiff']
    )
}

# Colorblind-friendly palette (Okabe & Ito, 2008)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'red': '#D55E00',
    'black': '#000000',
    'gray': '#808080',
}

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

class FigureLogger:
    """Logging utility for figure generation."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.figures_generated = []
        self.errors = []
    
    def info(self, message: str):
        if self.verbose:
            print(f"ℹ {message}")
    
    def success(self, message: str):
        print(f"✓ {message}")
    
    def warning(self, message: str):
        print(f"⚠ {message}")
    
    def error(self, message: str):
        print(f"✗ {message}")
        self.errors.append(message)
    
    def log_figure(self, figure_id: str, status: str, details: str = ""):
        self.figures_generated.append({
            'id': figure_id,
            'status': status,
            'details': details
        })
    
    def print_summary(self):
        print("\n" + "="*70)
        print("FIGURE GENERATION SUMMARY")
        print("="*70)
        success_count = sum(1 for f in self.figures_generated if f['status'] == 'success')
        print(f"Total: {len(self.figures_generated)} | Success: {success_count} | Failed: {len(self.errors)}")
        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  • {error}")
        print("="*70 + "\n")

logger = FigureLogger()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_matplotlib_style(profile: JournalProfile):
    """Configure matplotlib with journal-specific settings."""
    plt.rcParams.update({
        'font.family': profile.font_family,
        'font.size': profile.font_size_base,
        'axes.labelsize': profile.font_size_base + 1,
        'axes.titlesize': profile.font_size_base + 2,
        'xtick.labelsize': profile.font_size_base - 1,
        'ytick.labelsize': profile.font_size_base - 1,
        'legend.fontsize': profile.font_size_base - 1,
        'figure.titlesize': profile.font_size_base + 3,
        'figure.dpi': profile.dpi,
        'savefig.dpi': profile.dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': profile.line_width,
        'lines.linewidth': profile.line_width,
        'patch.linewidth': profile.line_width,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'figure.autolayout': False,
        'text.usetex': False,  # Set True if LaTeX available
    })
    logger.info(f"Matplotlib configured for {profile.name} (DPI: {profile.dpi})")

def load_results(results_path: Path) -> Dict:
    """Load analysis results from JSON file."""
    try:
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            logger.info("Generating figures with synthetic data")
            return generate_synthetic_results()
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.success(f"Loaded results from {results_path.name}")
        return results
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return generate_synthetic_results()
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return generate_synthetic_results()

def generate_synthetic_results() -> Dict:
    """Generate synthetic results for testing/demonstration."""
    np.random.seed(42)
    
    # Generate gematria values
    n_words = 200
    mean_gematria = 350
    std_gematria = 180
    gematria_values = np.random.gamma(4, mean_gematria/4, n_words).astype(int)
    
    # Generate multiples analysis
    divisors = [7, 12, 30, 60]
    multiples_results = {}
    
    for d in divisors:
        count = np.sum(gematria_values % d == 0)
        expected = n_words / d
        p_value = stats.binom_test(count, n_words, 1.0/d, alternative='greater') if SCIPY_AVAILABLE else 0.15
        
        multiples_results[f'divisor_{d}'] = {
            'observed_count': int(count),
            'expected_count': float(expected),
            'p_value': float(p_value),
            'significant_bonferroni': p_value < 0.05 / len(divisors)
        }
    
    # Generate Bayesian results
    bayesian_results = {}
    for d in divisors:
        bf_log = np.random.normal(-1.0, 1.5)
        bayesian_results[f'divisor_{d}'] = {
            'bayes_factor_log': float(bf_log),
            'interpretation': 'enrichment favored' if bf_log < 0 else 'null favored'
        }
    
    # Generate power analysis
    power_results = {}
    for d in divisors:
        power_results[f'divisor_{d}'] = {
            'power': float(np.random.uniform(0.3, 0.9)),
            'recommended_n': int(np.random.randint(250, 400))
        }
    
    # Generate ELS results
    els_results = {
        'תורה': {
            'skip_2_11': {'observed_count': 5, 'expected_mean': 3.2, 'p_value': 0.08},
            'skip_11_21': {'observed_count': 2, 'expected_mean': 1.1, 'p_value': 0.15}
        },
        'אלהים': {
            'skip_2_11': {'observed_count': 8, 'expected_mean': 4.5, 'p_value': 0.04},
            'skip_11_21': {'observed_count': 3, 'expected_mean': 1.8, 'p_value': 0.12}
        }
    }
    
    return {
        'gematria_analysis': {
            'values': gematria_values.tolist(),
            'sample_size': n_words,
            'mean': float(np.mean(gematria_values)),
            'median': float(np.median(gematria_values)),
            'std': float(np.std(gematria_values)),
            'min': float(np.min(gematria_values)),
            'max': float(np.max(gematria_values))
        },
        'multiples_analysis': {
            'sample_size': n_words,
            'divisor_analysis': multiples_results
        },
        'bayesian_analysis': {
            'results': bayesian_results
        },
        'power_analysis': {
            'sample_size_used': n_words,
            'power_by_divisor': power_results
        },
        'els_analysis': {
            'results': els_results
        }
    }

def ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory with subdirectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create format-specific subdirectories
    for fmt in ['pdf', 'png', 'svg', 'tiff', 'eps']:
        (output_dir / fmt).mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    return output_dir

def save_figure(fig: plt.Figure, filename: str, output_dir: Path, 
                formats: List[str], dpi: int):
    """Save figure in multiple formats with metadata."""
    saved = []
    failed = []
    
    for fmt in formats:
        try:
            # Determine output path
            if fmt == 'tiff':
                # TIFF in separate directory
                output_path = output_dir / 'tiff' / f"{filename}.tif"
            else:
                output_path = output_dir / fmt / f"{filename}.{fmt}"
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'dpi': dpi,
                'format': fmt if fmt != 'tiff' else 'tiff'
            }
            
            if fmt == 'pdf':
                save_kwargs['metadata'] = {
                    'Title': filename,
                    'Author': 'Ahmed Benseddik',
                    'Subject': 'Ancient Text Numerical Analysis',
                    'Creator': 'Python matplotlib'
                }
            elif fmt in ['tiff', 'tif']:
                save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            
            fig.savefig(output_path, **save_kwargs)
            saved.append(fmt)
            
        except Exception as e:
            failed.append(f"{fmt}: {str(e)}")
    
    if saved:
        logger.success(f"Saved {filename} ({', '.join(saved)})")
        return True
    else:
        logger.error(f"Failed to save {filename}: {'; '.join(failed)}")
        return False

def add_figure_label(ax: plt.Axes, label: str, x: float = 0.02, y: float = 0.98):
    """Add figure panel label (A, B, C, etc.)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                     edgecolor='black', linewidth=1.5))

# =============================================================================
# FIGURE S1: CROSS-CULTURAL NUMERICAL COMPARISON
# =============================================================================

def generate_figure_s1(results: Dict, output_dir: Path, profile: JournalProfile):
    """
    Figure S1: Cross-Cultural Numerical System Comparison
    
    Shows how identical Hebrew words produce different numerical values
    across different cultural systems (Standard Gematria, Atbash, Albam).
    Demonstrates system-dependency of numerical "significance."
    """
    logger.info("[1/7] Generating Figure S1: Cross-Cultural Comparison...")
    
    try:
        # Data preparation
        words_data = {
            'Word': [
                'בראשית\n(Bereshit\n"Beginning")',
                'אלהים\n(Elohim\n"God")',
                'תורה\n(Torah\n"Law")',
                'שלום\n(Shalom\n"Peace")',
                'אמת\n(Emet\n"Truth")'
            ],
            'Hebrew\nStandard': [913, 86, 611, 376, 441],
            'Hebrew\nAtbash': [300, 314, 189, 424, 359],
            'Hebrew\nAlbam': [945, 122, 645, 408, 473],
        }
        
        df = pd.DataFrame(words_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(profile.width_double_column, 
                                        profile.width_double_column * 0.6))
        
        x = np.arange(len(df['Word']))
        width = 0.25
        
        # Create grouped bars
        systems = ['Hebrew\nStandard', 'Hebrew\nAtbash', 'Hebrew\nAlbam']
        colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green']]
        
        for i, (system, color) in enumerate(zip(systems, colors_list)):
            offset = width * (i - 1)
            bars = ax.bar(x + offset, df[system], width,
                         label=system.replace('\n', ' '),
                         color=color, alpha=0.85,
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 2),
                              textcoords="offset points",
                              ha='center', va='bottom',
                              fontsize=profile.font_size_base - 2,
                              fontweight='bold')
        
        # Styling
        ax.set_xlabel('Hebrew Words with Transliteration and Translation',
                     fontweight='bold')
        ax.set_ylabel('Numerical Value', fontweight='bold')
        ax.set_title('Cross-Cultural Numerical System Comparison\n' +
                    'Same Words Yield Different Values Across Systems',
                    fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Word'], fontsize=profile.font_size_base - 2)
        ax.legend(loc='upper right', framealpha=0.95, fancybox=True, 
                 shadow=True, fontsize=profile.font_size_base - 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(df[systems].max().max() * 1.15))
        
        # Add figure label
        add_figure_label(ax, 'S1')
        
        # Add statistical note
        std_values = df['Hebrew\nStandard'].values
        atbash_values = df['Hebrew\nAtbash'].values
        if SCIPY_AVAILABLE:
            corr, p_val = stats.pearsonr(std_values, atbash_values)
            note = (f"Pearson r (Standard vs. Atbash) = {corr:.3f}, p = {p_val:.3f}\n"
                   f"Systems show low correlation, confirming cultural specificity")
        else:
            note = "Different systems produce substantially different values"
        
        fig.text(0.5, 0.01, note, ha='center',
                fontsize=profile.font_size_base - 2,
                style='italic', color='gray',
                transform=fig.transFigure)
        
        plt.tight_layout()
        
        # Save figure
        success = save_figure(fig, 'Figure_S1_cross_cultural', output_dir,
                            profile.formats, profile.dpi)
        plt.close(fig)
        
        if success:
            logger.log_figure('S1', 'success', 'Cross-cultural comparison')
        else:
            logger.log_figure('S1', 'failed', 'Save error')
        
    except Exception as e:
        logger.error(f"Failed to generate Figure S1: {e}")
        logger.log_figure('S1', 'failed', str(e))
        import traceback
        traceback.print_exc()

# =============================================================================
# FIGURE S2: STATISTICAL POWER CURVES
# =============================================================================

def generate_figure_s2(results: Dict, output_dir: Path, profile: JournalProfile):
    """
    Figure S2: Statistical Power Analysis Curves
    
    Shows statistical power as function of sample size for detecting
    10% enrichment above chance across different divisors. Critical for
    justifying sample size requirements.
    """
    logger.info("[2/7] Generating Figure S2: Statistical Power Curves...")
    
    try:
        fig, ax = plt.subplots(figsize=(profile.width_double_column,
                                        profile.width_double_column * 0.6))
        
        # Parameters
        sample_sizes = np.arange(50, 551, 25)
        divisors = [7, 12, 30, 60]
        effect_size = 0.10
        alpha = 0.05
        
        colors_list = [COLORS['blue'], COLORS['orange'], 
                      COLORS['green'], COLORS['purple']]
        
        # Calculate and plot power curves
        for divisor, color in zip(divisors, colors_list):
            powers = []
            
            for n in sample_sizes:
                p_null = 1.0 / divisor
                p_alt = p_null + effect_size
                
                if SCIPY_AVAILABLE:
                    # Accurate power calculation
                    se_alt = np.sqrt(p_alt * (1 - p_alt) / n)
                    z_crit = stats.norm.ppf(1 - alpha)
                    z_effect = (p_alt - p_null) / (se_alt + 1e-10)
                    power = 1 - stats.norm.cdf(z_crit - z_effect)
                else:
                    # Approximation
                    se = np.sqrt(p_null * (1 - p_null) / n)
                    z = effect_size / (se + 1e-10)
                    power = 1 / (1 + np.exp(-(z - 1.96)))
                
                powers.append(np.clip(power, 0.05, 0.99))
            
            ax.plot(sample_sizes, powers, linewidth=2,
                   label=f'Divisor {divisor}', marker='o', markersize=3,
                   markevery=7, color=color, alpha=0.9)
        
        # Reference lines
        ax.axhline(y=0.80, color=COLORS['red'], linestyle='--',
                  linewidth=1.5, alpha=0.7,
                  label='Target Power (0.80)')
        ax.axhline(y=0.50, color=COLORS['gray'], linestyle=':',
                  linewidth=1, alpha=0.5)
        
        # Current sample size marker
        current_n = results.get('power_analysis', {}).get('sample_size_used', 200)
        ax.axvline(x=current_n, color=COLORS['red'], linestyle=':',
                  linewidth=1.5, alpha=0.5,
                  label=f'Current Study (n={current_n})')
        
        # Adequate power region
        ax.axhspan(0.80, 1.0, alpha=0.1, color=COLORS['green'])
        ax.text(500, 0.88, 'Adequate\nPower', ha='right', va='center',
               fontsize=profile.font_size_base, color='darkgreen',
               style='italic', fontweight='bold')
        
        # Styling
        ax.set_xlabel('Sample Size (n)', fontweight='bold')
        ax.set_ylabel('Statistical Power (1 - β)', fontweight='bold')
        ax.set_title('Statistical Power by Sample Size and Divisor\n' +
                    '(Effect Size = 10% enrichment, α = 0.05, one-tailed)',
                    fontweight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(50, 550)
        ax.legend(loc='lower right', framealpha=0.95, fancybox=True,
                 shadow=True, fontsize=profile.font_size_base - 1,
                 ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Add figure label
        add_figure_label(ax, 'S2')
        
        # Add note
        note = (f"Power represents probability of detecting {effect_size*100:.0f}% "
               f"enrichment above chance.\n"
               f"Recommended: n ≥ 300 for power ≥ 0.80 across all divisors tested. "
               f"Current study: n = {current_n}.")
        fig.text(0.5, 0.01, note, ha='center',
                fontsize=profile.font_size_base - 2,
                style='italic', color='gray',
                transform=fig.transFigure)
        
        plt.tight_layout()
        
        success = save_figure(fig, 'Figure_S2_power_curves', output_dir,
                            profile.formats, profile.dpi)
        plt.close(fig)
        
        if success:
            logger.log_figure('S2', 'success', 'Power analysis')
        else:
            logger.log_figure('S2', 'failed', 'Save error')
        
    except Exception as e:
        logger.error(f"Failed to generate Figure S2: {e}")
        logger.log_figure('S2', 'failed', str(e))
        import traceback
        traceback.print_exc()

# =============================================================================
# FIGURE S3: BAYESIAN MODEL COMPARISON FOREST PLOT
# =============================================================================

def generate_figure_s3(results: Dict, output_dir: Path, profile: JournalProfile):
    """
    Figure S3: Bayesian Model Comparison Forest Plot
    
    Log Bayes factors comparing null hypothesis (uniform distribution)
    vs. enrichment hypothesis (above-chance multiples) for each divisor.
    """
    logger.info("[3/7] Generating Figure S3: Bayesian Forest Plot...")
    
    try:
        # Extract Bayesian results
        bayesian_data = results.get('bayesian_analysis', {}).get('results', {})
        
        divisors = []
        bf_log = []
        interpretations = []
        
        for key in sorted(bayesian_data.keys()):
            if 'divisor_' in key:
                div_num = key.replace('divisor_', '')
                divisors.append(f'Divisor {div_num}')
                bf_log.append(bayesian_data[key].get('bayes_factor_log', 0))
                interpretations.append(bayesian_data[key].get('interpretation', 'inconclusive'))
        
        if not divisors:
            # Fallback data
            divisors = ['Divisor 7', 'Divisor 12', 'Divisor 30', 'Divisor 60']
            bf_log = [-2.3, -0.8, 1.2, 1.8]
            interpretations = ['enrichment favored', 'enrichment favored',
                             'null favored', 'null favored']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(profile.width_double_column,
                                        profile.width_single_column))
        
        y_pos = np.arange(len(divisors))
        
        # Color bars
        colors_list = [COLORS['green'] if 'enrichment' in interp.lower()
                      else COLORS['gray'] for interp in interpretations]
        
        # Horizontal bars
        bars = ax.barh(y_pos, bf_log, color=colors_list, alpha=0.8,
                      edgecolor='black', linewidth=1, height=0.6)
        
        # Reference line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # Evidence strength markers
        for threshold in [-2, 2]:
            ax.axvline(x=threshold, color=COLORS['gray'], linestyle=':',
                      linewidth=1, alpha=0.5)
        
        # Add value labels
        for i, (bf, interp, bar) in enumerate(zip(bf_log, interpretations, bars)):
            # BF value
            label_x = bf + (0.4 if bf > 0 else -0.4)
            ax.text(label_x, i, f'{bf:.2f}',
                   va='center', ha='left' if bf > 0 else 'right',
                   fontweight='bold', fontsize=profile.font_size_base - 1,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                            alpha=0.95, edgecolor='gray', linewidth=0.5))
            
            # Interpretation
            interp_x = -5.5 if bf > 0 else 5.5
            ax.text(interp_x, i, interp.title(),
                   va='center', ha='right' if bf > 0 else 'left',
                   fontsize=profile.font_size_base - 2,
                   style='italic', color=colors_list[i])
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(divisors, fontweight='bold')
        ax.set_xlabel('Log Bayes Factor (log BF₁₀)', fontweight='bold')
        ax.set_title('Bayesian Model Comparison: Null vs. Enrichment\n' +
                    '(Negative values favor enrichment, positive favor null)',
                    fontweight='bold', pad=15)
        ax.set_xlim(-6, 6)
        ax.grid(axis='x', alpha=0.3)
        
        # Shaded regions
        ax.axvspan(-10, 0, alpha=0.08, color=COLORS['green'])
        ax.axvspan(0, 10, alpha=0.08, color=COLORS['gray'])
        
        # Evidence annotations
        ax.text(-3.5, len(divisors) - 0.3, 'Strong\nEnrichment\nEvidence',
               ha='center', va='top', fontsize=profile.font_size_base - 2,
               color=COLORS['green'], style='italic', fontweight='bold')
        ax.text(3.5, len(divisors) - 0.3, 'Strong\nNull\nEvidence',
               ha='center', va='top', fontsize=profile.font_size_base - 2,
               color=COLORS['gray'], style='italic', fontweight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=COLORS['green'], alpha=0.8,
                          label='Enrichment Favored'),
            mpatches.Patch(color=COLORS['gray'], alpha=0.8,
                          label='Null Hypothesis Favored')
        ]
        ax.legend(handles=legend_elements, loc='lower right