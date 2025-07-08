"""
Configuration settings for the Burns Research Analysis Tool
"""

# PubMed API Configuration
ENTREZ_EMAIL = "your.email@example.com"  # Replace with your email
ENTREZ_API_KEY = "your_ncbi_api_key"     # Replace with your NCBI API key

# Analysis Parameters
DEFAULT_DATE_RANGE = "2013:2023[dp]"
MAX_BATCH_SIZE = 1000
EXTRACTION_DELAY = 0.5  # seconds between API calls

# Output Settings
OUTPUT_FORMATS = ['csv', 'xlsx']
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Search Configuration
SEARCH_STRATEGIES = [
    # Primary strategies
    ("Burns[MeSH] AND 2013:2023[dp]", 20000),
    ("burn*[Title/Abstract] AND 2013:2023[dp] NOT Burns[MeSH]", 50000),
    
    # Specialized burn types
    ("thermal injury[Title/Abstract] AND 2013:2023[dp]", 5000),
    ("scald*[Title/Abstract] AND 2013:2023[dp]", 5000),
    ("electrical burn*[Title/Abstract] AND 2013:2023[dp]", 2000),
    ("chemical burn*[Title/Abstract] AND 2013:2023[dp]", 2000),
    ("flame burn*[Title/Abstract] AND 2013:2023[dp]", 2000),
    
    # Treatment and care
    ("burn wound*[Title/Abstract] AND 2013:2023[dp]", 10000),
    ("burn injury[Title/Abstract] AND 2013:2023[dp]", 10000),
    ("burn treatment[Title/Abstract] AND 2013:2023[dp]", 8000),
    ("burn care[Title/Abstract] AND 2013:2023[dp]", 8000),
    ("burn surgery[Title/Abstract] AND 2013:2023[dp]", 6000),
    ("burn reconstruction[Title/Abstract] AND 2013:2023[dp]", 4000),
    
    # Journal-specific
    ("Burns[Journal] AND 2013:2023[dp]", 5000),
    ("Journal of Burn Care Research[Journal] AND 2013:2023[dp]", 3000),
    
    # Additional clinical terms
    ("inhalation injury[Title/Abstract] AND 2013:2023[dp]", 3000),
    ("burn shock[Title/Abstract] AND 2013:2023[dp]", 2000),
    ("burn sepsis[Title/Abstract] AND 2013:2023[dp]", 2000),
]

# Statistical Analysis Settings
NORMALITY_TESTS = ['shapiro', 'dagostino', 'kstest']
SIGNIFICANCE_LEVEL = 0.05
CORRELATION_METHOD = 'spearman'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'