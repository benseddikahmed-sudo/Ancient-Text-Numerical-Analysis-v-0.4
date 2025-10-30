# Supplementary Figures Generator for Ancient Text Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17443361.svg)](https://doi.org/10.5281/zenodo.17443361)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4)

## Description

Ce script Python génère des figures supplémentaires publication-ready pour l'analyse statistique de textes anciens utilisant la gématrie hébraïque. Il produit cinq figures principales couvrant les comparaisons inter-culturelles, l'analyse de puissance statistique, la modélisation bayésienne, le flux méthodologique et les cartes thermiques de signification.

Ce générateur fait partie intégrante du projet **Ancient Text Numerical Analysis** et permet de visualiser les résultats de l'analyse statistique des systèmes de numération hébraïque (Standard, Atbash, Albam, etc.).

**Auteur:** Ahmed Benseddik <benseddik.ahmed@gmail.com>  
**Version:** 3.1 (Optimisé & Vérifié)  
**Date:** 2025-10-25  
**Licence:** MIT  
**Repository:** [GitHub - Ancient-Text-Numerical-Analysis-v-0.4](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4)  
**DOI:** [10.5281/zenodo.17443361](https://doi.org/10.5281/zenodo.17443361)

## Contexte du projet

Ce générateur de figures est conçu pour accompagner l'analyse principale des textes anciens disponible sur GitHub. Il transforme les résultats JSON produits par le pipeline d'analyse en visualisations scientifiques de haute qualité, prêtes pour la publication académique.

### Projet parent
- **Nom:** Ancient Text Numerical Analysis v0.4
- **GitHub:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.17443361

## Figures générées

### Figure S1 - Comparaison inter-culturelle
Graphique à barres comparant les valeurs numériques de mots hébreux identiques dans différents systèmes de calcul (Standard et Atbash). Démontre comment un même mot peut avoir des valeurs différentes selon le système utilisé.

**Exemples de mots analysés:**
- בראשית (Bereshit - "Au commencement")
- אלהים (Elohim - "Dieu")
- תורה (Torah - "Loi")
- שלום (Shalom - "Paix")
- אמת (Emet - "Vérité")

### Figure S2 - Courbes de puissance statistique
Analyse de puissance montrant la relation entre la taille d'échantillon (n=50 à n=500), les diviseurs testés (7, 12, 30, 60) et la capacité à détecter un enrichissement de 10% au-dessus du hasard. Permet de déterminer la taille d'échantillon optimale pour l'étude.

**Paramètres:**
- Taille d'effet: 10% d'enrichissement
- Niveau alpha: 0.05
- Puissance cible: 0.80

### Figure S3 - Comparaison de modèles bayésiens
Diagramme en forêt (forest plot) présentant les facteurs de Bayes (log BF₁₀) pour différents diviseurs, indiquant la force de l'évidence pour ou contre l'hypothèse d'enrichissement par rapport à l'hypothèse nulle.

**Interprétation:**
- log BF₁₀ < -2: Évidence forte pour l'enrichissement
- log BF₁₀ > 2: Évidence forte pour l'hypothèse nulle
- -2 < log BF₁₀ < 2: Évidence non concluante

### Figure S4 - Flux méthodologique
Diagramme de workflow illustrant le pipeline d'analyse complet, de l'entrée des textes anciens à l'interprétation finale, incluant les étapes de traitement, calcul numérique, analyses fréquentiste et bayésienne, correction pour tests multiples et intégration du cadre éthique.

**Étapes principales:**
1. Input Text (Ancient Corpus)
2. Text Processing
3. Numerical Calculation
4. Parallel Statistical Analyses
5. Validation & Correction
6. Ethical Framework Integration
7. Final Interpretation

### Figure S5 - Carte thermique des p-values
Heatmap visualisant la signification statistique (échelle -log₁₀ des p-values) à travers plusieurs systèmes numériques et diviseurs. Les régions vertes indiquent une signification statistique (p < 0.05), les régions rouges indiquent l'absence de signification.

**Systèmes testés:**
- Standard (Gématrie classique)
- Atbash (Inversion alphabétique)
- Albam (Substitution par paires)
- Mispar Gadol (Valeurs finales étendues)

## Prérequis

### Dépendances Python obligatoires
```
python >= 3.8
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

### Dépendances optionnelles
```
scipy >= 1.5.0  (pour calculs de puissance précis)
tqdm >= 4.50.0  (pour barres de progression)
```

## Installation

### Option 1: Depuis GitHub
```bash
# Cloner le dépôt complet
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
cd Ancient-Text-Numerical-Analysis-v-0.4

# Installer les dépendances
pip install -r requirements.txt
```

### Option 2: Depuis Zenodo
```bash
# Télécharger depuis Zenodo
wget https://zenodo.org/record/17443361/files/generate_supplementary_figures.py

# Installer les dépendances
pip install numpy pandas matplotlib seaborn scipy tqdm
```

### Option 3: Installation manuelle
```bash
# Télécharger le script directement
curl -O https://raw.githubusercontent.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/main/generate_supplementary_figures.py

# Installer les dépendances
pip install numpy pandas matplotlib seaborn scipy tqdm
```

## Utilisation

### Workflow complet avec le projet parent

```bash
# 1. Exécuter l'analyse principale (depuis le dépôt GitHub)
python ancient_text_analysis.py --input data/hebrew_texts.txt

# 2. Générer les figures supplémentaires
python generate_supplementary_figures.py --results-dir data/results/

# 3. Les figures sont créées dans figures/supplementary/
```

### Utilisation basique
```bash
# Génère toutes les figures avec les paramètres par défaut
python generate_supplementary_figures.py

# Génère des figures spécifiques
python generate_supplementary_figures.py --figures S1 S3 S5

# Utilise un fichier de résultats spécifique
python generate_supplementary_figures.py --results-file data/results/analysis_results.json
```

### Options avancées
```bash
# Haute résolution pour publication
python generate_supplementary_figures.py --dpi 600 --font-scale 1.2

# Personnaliser les répertoires
python generate_supplementary_figures.py \
  --results-dir data/my_results/ \
  --output-dir figures/publication/

# Mode verbose pour débogage
python generate_supplementary_figures.py --verbose

# Génération rapide sans PDF (pour tests)
python generate_supplementary_figures.py --figures S1 S4 --no-pdf
```

### Arguments de ligne de commande

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--results-dir` | str | `data/results/` | Répertoire contenant les résultats d'analyse |
| `--results-file` | str | None | Fichier de résultats spécifique (prioritaire) |
| `--output-dir` | str | `figures/supplementary/` | Répertoire de sortie pour les figures |
| `--figures` | list | `all` | Figures à générer: S1, S2, S3, S4, S5, all |
| `--font-scale` | float | 1.0 | Facteur d'échelle pour la taille des polices |
| `--dpi` | int | 300 | Résolution des figures (DPI) |
| `--no-pdf` | flag | False | Ne pas générer les PDF (plus rapide) |
| `--verbose` | flag | False | Affichage détaillé pour débogage |

## Format des données d'entrée

Le script attend un fichier JSON contenant les résultats d'analyse produits par le pipeline principal. Structure attendue :

```json
{
  "metadata": {
    "analysis_date": "2025-10-25",
    "version": "0.4",
    "corpus_size": 1000
  },
  "power_analysis": {
    "sample_size_used": 200,
    "effect_size": 0.1,
    "divisors": [7, 12, 30, 60]
  },
  "bayesian_analysis": {
    "results": {
      "divisor_7": {
        "bayes_factor_log": -2.3,
        "interpretation": "enrichment favored",
        "posterior_probability": 0.91
      },
      "divisor_12": {
        "bayes_factor_log": -1.1,
        "interpretation": "enrichment favored",
        "posterior_probability": 0.75
      },
      "divisor_30": {
        "bayes_factor_log": 0.8,
        "interpretation": "null favored",
        "posterior_probability": 0.31
      },
      "divisor_60": {
        "bayes_factor_log": 1.5,
        "interpretation": "null favored",
        "posterior_probability": 0.18
      }
    }
  },
  "multiples_analysis": {
    "divisors_tested": [7, 12, 18, 26, 30, 60, 70, 120],
    "systems": ["Standard", "Atbash", "Albam", "Mispar Gadol"],
    "pvalue_matrix": [
      [0.001, 0.05, 0.3, 0.8, 0.15, 0.6, 0.002, 0.9],
      [0.08, 0.12, 0.4, 0.005, 0.25, 0.7, 0.35, 0.85],
      [0.2, 0.3, 0.6, 0.1, 0.45, 0.03, 0.5, 0.75],
      [0.5, 0.4, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4]
    ]
  }
}
```

**Note:** Si les données ne sont pas disponibles, le script génère automatiquement des données simulées réalistes pour démonstration.

## Formats de sortie

Chaque figure est générée en trois formats pour maximiser la compatibilité :
- **PDF** : Vectoriel, idéal pour publications académiques (LaTeX, Word)
- **PNG** : Raster haute résolution (300-600 DPI), pour présentations
- **SVG** : Vectoriel éditable, compatible Inkscape/Illustrator/Figma

Structure de sortie :
```
figures/supplementary/
├── Figure_S1_cross_cultural.pdf
├── Figure_S1_cross_cultural.png
├── Figure_S1_cross_cultural.svg
├── Figure_S2_power_curves.pdf
├── Figure_S2_power_curves.png
├── Figure_S2_power_curves.svg
├── Figure_S3_bayesian_forest.pdf
├── Figure_S3_bayesian_forest.png
├── Figure_S3_bayesian_forest.svg
├── Figure_S4_workflow.pdf
├── Figure_S4_workflow.png
├── Figure_S4_workflow.svg
├── Figure_S5_pvalue_heatmap.pdf
├── Figure_S5_pvalue_heatmap.png
└── Figure_S5_pvalue_heatmap.svg
```

## Caractéristiques techniques

### Palette de couleurs
Le script utilise une palette colorblind-friendly (Tol bright scheme) :
- **Primaire:** #0173B2 (Bleu) - Données principales
- **Secondaire:** #DE8F05 (Orange) - Données alternatives
- **Accent:** #029E73 (Vert) - Succès/Enrichissement
- **Highlight:** #CC78BC (Violet) - Mise en évidence
- **Neutre:** #949494 (Gris) - Hypothèse nulle
- **Danger:** #CA3433 (Rouge) - Seuils critiques

### Style typographique
- **Police par défaut:** Times New Roman / DejaVu Serif / Liberation Serif
- **Taille de base:** 10pt (configurable avec `--font-scale`)
- **Épaisseur de ligne:** 2.0-2.5pt pour les graphiques principaux
- **Résolution:** 300 DPI par défaut (configurable jusqu'à 600+ DPI)
- **Format de sortie:** Tight bounding box pour éliminer les marges

### Standards de publication
- Conforme aux exigences de Nature, Science, PLOS
- Figures vectorielles pour reproduction sans perte
- Annotations en haute résolution
- Légendes complètes et informatives

### Gestion des erreurs
- ✅ Validation automatique des données d'entrée
- ✅ Génération de données simulées si manquantes
- ✅ Messages d'erreur détaillés en mode verbose
- ✅ Gestion gracieuse des interruptions (Ctrl+C)
- ✅ Fallbacks pour bibliothèques optionnelles

## Exemples d'utilisation

### Exemple 1 : Génération standard après analyse complète
```bash
# Analyse complète du corpus
python ancient_text_analysis.py \
  --input data/genesis.txt \
  --systems standard atbash albam \
  --divisors 7 12 30 60

# Génération des figures
python generate_supplementary_figures.py \
  --results-dir data/results/ \
  --output-dir figures/supplementary/
```

### Exemple 2 : Publication haute qualité
```bash
python generate_supplementary_figures.py \
  --dpi 600 \
  --font-scale 1.2 \
  --figures all \
  --output-dir figures/publication_ready/
```

### Exemple 3 : Test rapide d'une sous-sélection
```bash
python generate_supplementary_figures.py \
  --figures S1 S4 \
  --no-pdf \
  --dpi 150
```

### Exemple 4 : Débogage avec données spécifiques
```bash
python generate_supplementary_figures.py \
  --verbose \
  --results-file data/test_run_2025_10_25.json \
  --output-dir figures/debug/
```

### Exemple 5 : Batch processing
```bash
# Boucle sur plusieurs analyses
for results_file in data/results/*.json; do
  python generate_supplementary_figures.py \
    --results-file "$results_file" \
    --output-dir "figures/$(basename $results_file .json)/"
done
```

## Intégration avec le projet parent

Ce générateur de figures s'intègre parfaitement avec le pipeline d'analyse principal :

```bash
# Pipeline complet automatisé
#!/bin/bash

# 1. Analyse du corpus
python ancient_text_analysis.py \
  --input data/hebrew_corpus.txt \
  --output data/results/ \
  --all-systems

# 2. Génération des figures
python generate_supplementary_figures.py \
  --results-dir data/results/ \
  --output-dir figures/supplementary/ \
  --dpi 600

# 3. Génération du rapport
python generate_report.py \
  --figures figures/supplementary/ \
  --output reports/analysis_report.pdf

echo "✓ Pipeline complet terminé!"
```

## Dépannage

### Erreur : "No results files found"
**Cause:** Le répertoire spécifié ne contient pas de fichiers JSON.

**Solutions:**
```bash
# Vérifier le contenu du répertoire
ls -la data/results/

# Utiliser un fichier spécifique
python generate_supplementary_figures.py \
  --results-file data/results/analysis_results_2025_10_25.json

# Vérifier que l'analyse principale a bien produit des résultats
python ancient_text_analysis.py --input data/test.txt
```

### Erreur : "Invalid JSON in results file"
**Cause:** Le fichier JSON est mal formé ou corrompu.

**Solutions:**
```bash
# Valider le JSON en ligne de commande
python -m json.tool data/results/file.json

# Ou utiliser jq
jq . data/results/file.json

# Vérifier l'encodage
file data/results/file.json  # Doit être UTF-8
```

### Problème : Polices manquantes
**Cause:** Times New Roman non disponible sur le système.

**Solutions:**
```bash
# Linux/Ubuntu
sudo apt-get install msttcorefonts -qq
fc-cache -f

# macOS (incluses par défaut)
# Rien à faire

# Vérifier les polices disponibles
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'Times' in f.name])"
```

### Figures vides ou incorrectes
**Cause:** Données manquantes ou structure JSON incorrecte.

**Solutions:**
```bash
# Mode verbose pour diagnostiquer
python generate_supplementary_figures.py --verbose

# Tester avec données simulées
python generate_supplementary_figures.py --figures S1

# Vérifier la structure JSON
python -c "import json; print(json.load(open('data/results/file.json')).keys())"
```

### Erreur de mémoire avec haute résolution
**Cause:** DPI trop élevé pour la mémoire disponible.

**Solutions:**
```bash
# Générer les figures une par une
for fig in S1 S2 S3 S4 S5; do
  python generate_supplementary_figures.py --figures $fig --dpi 600
done

# Ou réduire le DPI
python generate_supplementary_figures.py --dpi 300  # Au lieu de 600
```

## Citation

Si vous utilisez ce script dans vos travaux de recherche, veuillez citer à la fois le générateur de figures et le projet principal :

### Citation BibTeX pour le générateur de figures
```bibtex
@software{benseddik2025figgen,
  author       = {Benseddik, Ahmed},
  title        = {{Supplementary Figures Generator for Ancient Text 
                   Numerical Analysis}},
  year         = 2025,
  month        = oct,
  version      = {3.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17443361},
  url          = {https://doi.org/10.5281/zenodo.17443361},
  note         = {Part of Ancient Text Numerical Analysis v0.4}
}
```

### Citation BibTeX pour le projet principal
```bibtex
@software{benseddik2025ancient,
  author       = {Benseddik, Ahmed},
  title        = {{Ancient Text Numerical Analysis}},
  year         = 2025,
  month        = oct,
  version      = {0.4},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17443361},
  url          = {https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4},
  note         = {Statistical analysis of Hebrew gematria systems}
}
```

### Citation textuelle
> Benseddik, A. (2025). Supplementary Figures Generator for Ancient Text Numerical Analysis (Version 3.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17443361

## Licence

Ce logiciel est distribué sous licence MIT. Voir le fichier LICENSE pour plus de détails.

```
MIT License

Copyright (c) 2025 Ahmed Benseddik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ressources supplémentaires

### Documentation
- **Guide complet:** [GitHub Wiki](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/wiki)
- **Tutoriels:** [GitHub Discussions](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/discussions)
- **API Reference:** Voir docstrings dans le code source

### Liens utiles
- 🏠 **Homepage:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
- 📚 **Documentation:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/wiki
- 🐛 **Issues:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues
- 💬 **Discussions:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/discussions
- 📦 **Zenodo Archive:** https://doi.org/10.5281/zenodo.17443361

### Projets connexes
- **Ancient Text Analysis (Main):** Analyse statistique complète
- **Gematria Calculator:** Calculateur de valeurs numériques
- **Hebrew Text Processor:** Prétraitement de corpus hébraïques

## Contact et support

**Auteur :** Ahmed Benseddik  
**Email :** benseddik.ahmed@gmail.com  
**GitHub :** [@benseddikahmed-sudo](https://github.com/benseddikahmed-sudo)  
**Issues :** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues

Pour toute question, suggestion ou rapport de bug, n'hésitez pas à :
1. Ouvrir une issue sur GitHub
2. Démarrer une discussion dans l'onglet Discussions
3. Contacter directement par email

## Historique des versions

### Version 3.1 (2025-10-25) - **Current**
- ✨ Optimisation complète du code
- 🐛 Correction de la gestion des erreurs et fermeture des figures
- 📊 Amélioration de la qualité visuelle (palette colorblind-friendly)
- 🚀 Ajout d'options CLI avancées (--verbose, --no-pdf)
- 📝 Documentation enrichie pour Zenodo
- 🔗 Intégration avec GitHub repository
- 🎯 Support complet des données simulées
- 🌐 Publication sur Zenodo (DOI: 10.5281/zenodo.17443361)

### Version 3.0 (2025-10-24)
- 🎨 Refonte complète de l'interface graphique
- 📈 Ajout de la Figure S5 (heatmap des p-values)
- 🔧 Amélioration de la configuration (FIGURE_CONFIG dict)
- 🎯 Support des données simulées pour démonstration
- 📐 Standardisation des tailles de figures

### Version 2.0 (2025-08)
- 📊 Ajout de l'analyse bayésienne (Figure S3)
- 📁 Support multi-format (PDF/PNG/SVG)
- 🔄 Amélioration du workflow (Figure S4)
- 🎨 Palette de couleurs professionnelle

### Version 1.0 (2025-06)
- 🎉 Release initiale
- 📊 Figures S1-S2 (comparaison culturelle et puissance)
- 📈 Support basique des graphiques matplotlib
- 💾 Export PNG uniquement

## Remerciements

Ce projet utilise les bibliothèques open-source suivantes :
- **NumPy** - Calculs numériques et algèbre linéaire
- **Pandas** - Manipulation et analyse de données
- **Matplotlib** - Visualisation scientifique
- **Seaborn** - Graphiques statistiques élégants
- **SciPy** - Calculs scientifiques et tests statistiques

Merci à la communauté scientifique Python pour ces outils exceptionnels.

## Conformité et standards

Ce logiciel respecte les standards suivants :
- ✅ **PEP 8** - Style guide Python
- ✅ **Semantic Versioning 2.0.0** - Versioning
- ✅ **FAIR Principles** - Findable, Accessible, Interoperable, Reusable
- ✅ **Open Source Initiative** - Licence MIT approuvée
- ✅ **Nature Figure Guidelines** - Standards de publication scientifique

---

**Note:** Ce README accompagne la version 3.1 du générateur de figures supplémentaires, publié sur Zenodo sous le DOI [10.5281/zenodo.17443361](https://doi.org/10.5281/zenodo.17443361) dans le cadre du projet [Ancient Text Numerical Analysis v0.4](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4).

**Dernière mise à jour:** 30 octobre 2025