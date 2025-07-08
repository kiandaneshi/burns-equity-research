"""
Country parsing utilities for extracting and standardizing country names from affiliations
"""

import re
import pycountry
from typing import Optional, Dict

class CountryParser:
    """Parses and standardizes country names from affiliation strings"""
    
    def __init__(self):
        self.country_patterns = self._build_country_patterns()
        self.manual_mappings = self._build_manual_mappings()
        self.stats = {'total_parsed': 0, 'successful_matches': 0}
    
    def _build_country_patterns(self) -> Dict[str, str]:
        """Build regex patterns for country detection"""
        patterns = {}
        
        for country in pycountry.countries:
            # Official name
            patterns[country.name.lower()] = country.name
            
            # Common name (if different)
            if hasattr(country, 'common_name'):
                patterns[country.common_name.lower()] = country.name
            
            # Alpha-2 and Alpha-3 codes
            patterns[country.alpha_2.lower()] = country.name
            patterns[country.alpha_3.lower()] = country.name
        
        return patterns
    
    def _build_manual_mappings(self) -> Dict[str, str]:
        """Manual mappings for common variations and abbreviations"""
        return {
            'usa': 'United States',
            'united states of america': 'United States',
            'uk': 'United Kingdom',
            'england': 'United Kingdom',
            'scotland': 'United Kingdom',
            'wales': 'United Kingdom',
            'northern ireland': 'United Kingdom',
            'south korea': 'Korea, Republic of',
            'north korea': "Korea, Democratic People's Republic of",
            'russia': 'Russian Federation',
            'iran': 'Iran, Islamic Republic of',
            'venezuela': 'Venezuela, Bolivarian Republic of',
            'syria': 'Syrian Arab Republic',
            'taiwan': 'Taiwan, Province of China',
            'hong kong': 'Hong Kong',
            'macau': 'Macao',
            'palestine': 'Palestine, State of',
            'czech republic': 'Czechia',
            'ivory coast': "Côte d'Ivoire",
            'cape verde': 'Cabo Verde',
            'swaziland': 'Eswatini',
            'macedonia': 'North Macedonia',
            'bosnia': 'Bosnia and Herzegovina',
        }
    
    def parse_country(self, affiliation: str) -> Optional[str]:
        """
        Extract country name from affiliation string
        
        Args:
            affiliation: Raw affiliation string
            
        Returns:
            Standardized country name or None if not found
        """
        if not affiliation:
            return None
        
        self.stats['total_parsed'] += 1
        
        # Clean affiliation
        cleaned = self._clean_affiliation(affiliation)
        
        # Check manual mappings first
        country = self._check_manual_mappings(cleaned)
        if country:
            self.stats['successful_matches'] += 1
            return country
        
        # Check country patterns
        country = self._check_country_patterns(cleaned)
        if country:
            self.stats['successful_matches'] += 1
            return country
        
        # Extract from common formats
        country = self._extract_from_common_formats(cleaned)
        if country:
            self.stats['successful_matches'] += 1
            return country
        
        return None
    
    def _clean_affiliation(self, affiliation: str) -> str:
        """Clean and normalize affiliation string"""
        # Convert to lowercase
        cleaned = affiliation.lower()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove common punctuation
        cleaned = re.sub(r'[,;.\(\)\[\]{}]', ' ', cleaned)
        
        # Remove email patterns
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        
        # Remove zip codes
        cleaned = re.sub(r'\b\d{5}(-\d{4})?\b', '', cleaned)
        
        return cleaned.strip()
    
    def _check_manual_mappings(self, affiliation: str) -> Optional[str]:
        """Check against manual country mappings"""
        for pattern, country in self.manual_mappings.items():
            if pattern in affiliation:
                return country
        return None
    
    def _check_country_patterns(self, affiliation: str) -> Optional[str]:
        """Check against country name patterns"""
        # Sort by length (longest first) to avoid partial matches
        sorted_patterns = sorted(self.country_patterns.items(), 
                               key=lambda x: len(x[0]), reverse=True)
        
        for pattern, country in sorted_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', affiliation):
                return country
        
        return None
    
    def _extract_from_common_formats(self, affiliation: str) -> Optional[str]:
        """Extract country from common affiliation formats"""
        # Pattern: "..., Country"
        parts = affiliation.split(',')
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            country = self._match_country_name(last_part)
            if country:
                return country
        
        # Pattern: "Country ..."
        words = affiliation.split()
        if words:
            first_word = words[0]
            country = self._match_country_name(first_word)
            if country:
                return country
        
        return None
    
    def _match_country_name(self, text: str) -> Optional[str]:
        """Try to match a text string to a country name"""
        text = text.strip()
        
        # Check manual mappings
        if text in self.manual_mappings:
            return self.manual_mappings[text]
        
        # Check country patterns
        if text in self.country_patterns:
            return self.country_patterns[text]
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        success_rate = 0
        if self.stats['total_parsed'] > 0:
            success_rate = (self.stats['successful_matches'] / 
                          self.stats['total_parsed']) * 100
        
        return {
            'total_parsed': self.stats['total_parsed'],
            'successful_matches': self.stats['successful_matches'],
            'success_rate': round(success_rate, 2)
        }