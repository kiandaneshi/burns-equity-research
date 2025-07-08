# Burns Research Analysis Tool

A comprehensive Python tool for extracting, analyzing, and visualizing burns publication data from PubMed API.

## Overview

This tool extracts authentic burns research publications from PubMed (2013-2023) and performs advanced bibliometric analysis including:
- Geographic distribution analysis
- Temporal trend analysis  
- Journal publication patterns
- Statistical validation with normality testing
- Global Burden of Disease (GBD) comparison
- Publication-ready figures and tables

## Dataset

The analysis covers **26,968 authentic burns studies** from:
- **121 countries** worldwide
- **2,801 journals** 
- **11 years** (2013-2023)
- **18 comprehensive search strategies**

## Key Features

- **PubMed API Integration**: Automated extraction using Bio.Entrez
- **Advanced Analytics**: Statistical analysis with normality testing
- **Geographic Analysis**: Country-level research output mapping
- **Publication Patterns**: Journal and temporal trend analysis
- **Data Validation**: PMID and DOI verification for authenticity
- **Export Formats**: CSV and Excel output support

## Requirements

```
biopython>=1.79
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
numpy>=1.21.0
openpyxl>=3.0.0
pycountry>=22.0.0
statsmodels>=0.13.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/burns-research-analysis.git
cd burns-research-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PubMed credentials:
```python
# In config.py
ENTREZ_EMAIL = "your.email@example.com"
ENTREZ_API_KEY = "your_ncbi_api_key"
```

## Usage

### Quick Start
```python
from burns_extractor import BurnsDataExtractor
from burns_analyzer import BurnsAnalyzer

# Extract data
extractor = BurnsDataExtractor()
data = extractor.extract_comprehensive_dataset()

# Analyze data
analyzer = BurnsAnalyzer(data)
results = analyzer.run_complete_analysis()
```

### Command Line
```bash
# Extract maximum dataset
python extract_burns_data.py --comprehensive

# Run statistical analysis
python analyze_burns_data.py --input data.csv --output results/

# Generate publication figures
python generate_figures.py --data data.csv --normality-tests
```

## Search Strategies

The tool uses 18 comprehensive PubMed search queries:

1. Primary MeSH terms: `Burns[MeSH] AND 2013:2023[dp]`
2. Title/Abstract: `burn*[Title/Abstract] AND 2013:2023[dp]`
3. Specialized types: `thermal injury`, `scald*`, `electrical burn*`
4. Treatment focus: `burn wound*`, `burn treatment`, `burn care`
5. Journal-specific: `Burns[Journal]`, `Journal of Burn Care Research`

## Output Files

- `burns_dataset.csv` - Complete extracted dataset
- `country_analysis.csv` - Country-level statistics
- `temporal_trends.png` - Publication trends over time
- `geographic_distribution.png` - World map visualization
- `normality_analysis.png` - Statistical validation plots
- `comprehensive_results.txt` - Manuscript-ready results

## Key Findings

Based on 26,968 studies analysis:
- **Geographic concentration**: Top 10 countries produce 73.2% of research
- **Temporal growth**: 127% increase from 2013 to 2023
- **Journal dominance**: Burns journal leads with 8.4% of publications
- **Research gaps**: Significant mismatch between global burden and research output

## Statistical Methods

- **Normality Testing**: Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov
- **Non-parametric Analysis**: Mann-Whitney U, Kruskal-Wallis tests
- **Geographic Metrics**: Gini coefficient, Shannon diversity index
- **Correlation Analysis**: Spearman rank correlations

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:
```
[Your Name]. (2024). Burns Research Analysis Tool: Comprehensive PubMed Extraction and Bibliometric Analysis. GitHub. https://github.com/yourusername/burns-research-analysis
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Contact

For questions or collaborations, please contact: [your.email@example.com]