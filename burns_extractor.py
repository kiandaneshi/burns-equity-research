"""
Burns Research Data Extractor
Comprehensive extraction of burns studies from PubMed API
"""

import os
import time
import json
import pandas as pd
import xml.etree.ElementTree as ET
from Bio import Entrez
from country_parser import CountryParser
from config import *
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class BurnsDataExtractor:
    """Extract comprehensive burns research data from PubMed"""
    
    def __init__(self, email: str = None, api_key: str = None):
        """
        Initialize extractor with PubMed credentials
        
        Args:
            email: NCBI email address
            api_key: NCBI API key
        """
        Entrez.email = email or ENTREZ_EMAIL
        Entrez.api_key = api_key or ENTREZ_API_KEY
        
        self.country_parser = CountryParser()
        self.all_studies = []
        self.processed_pmids = set()
        
        logger.info("Burns Data Extractor initialized")
    
    def extract_comprehensive_dataset(self, output_file: str = "burns_dataset.csv") -> pd.DataFrame:
        """
        Extract comprehensive burns dataset using all search strategies
        
        Args:
            output_file: Output CSV filename
            
        Returns:
            DataFrame containing all extracted studies
        """
        logger.info("Starting comprehensive burns data extraction")
        
        total_extracted = 0
        
        for strategy_idx, (query, max_results) in enumerate(SEARCH_STRATEGIES):
            logger.info(f"Strategy {strategy_idx+1}/{len(SEARCH_STRATEGIES)}: {query[:60]}...")
            
            try:
                # Get all PMIDs for this strategy
                all_pmids = self._get_all_pmids(query, max_results)
                
                # Filter new PMIDs
                new_pmids = [pmid for pmid in all_pmids if pmid not in self.processed_pmids]
                logger.info(f"New PMIDs: {len(new_pmids):,}")
                
                if not new_pmids:
                    continue
                
                # Extract studies in batches
                strategy_extracted = self._extract_pmid_batch(new_pmids, strategy_idx+1)
                total_extracted += strategy_extracted
                
                logger.info(f"Strategy {strategy_idx+1}: +{strategy_extracted:,} | Total: {total_extracted:,}")
                
                # Save progress at major milestones
                if total_extracted > 0 and total_extracted % 5000 == 0:
                    self._save_progress_checkpoint(total_extracted)
                
            except Exception as e:
                logger.error(f"Error in strategy {strategy_idx+1}: {e}")
                continue
        
        # Finalize dataset
        df = self._finalize_dataset()
        
        # Save to file
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset saved to {output_file}")
        
        logger.info(f"Extraction complete: {len(df):,} studies from {df['country'].nunique()} countries")
        return df
    
    def _get_all_pmids(self, query: str, max_results: int) -> List[str]:
        """Get all PMIDs for a search query"""
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="pub_date",
                retmode="xml"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results.get("IdList", [])
            logger.info(f"Found {len(pmids):,} PMIDs for query")
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PMIDs: {e}")
            return []
    
    def _extract_pmid_batch(self, pmids: List[str], strategy_idx: int) -> int:
        """Extract data for a batch of PMIDs"""
        extracted_count = 0
        
        # Process in batches
        for i in range(0, len(pmids), MAX_BATCH_SIZE):
            batch = pmids[i:i + MAX_BATCH_SIZE]
            batch_data = self._fetch_batch_details(batch)
            
            if batch_data:
                self.all_studies.extend(batch_data)
                self.processed_pmids.update([study['pmid'] for study in batch_data])
                extracted_count += len(batch_data)
                
                logger.info(f"Batch {i//MAX_BATCH_SIZE + 1}: +{len(batch_data)} studies")
            
            # Respect API rate limits
            time.sleep(EXTRACTION_DELAY)
        
        return extracted_count
    
    def _fetch_batch_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for a batch of PMIDs"""
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="xml",
                retmode="xml"
            )
            xml_data = handle.read()
            handle.close()
            
            return self._parse_xml_batch(xml_data)
            
        except Exception as e:
            logger.error(f"Error fetching batch details: {e}")
            return []
    
    def _parse_xml_batch(self, xml_data: str) -> List[Dict]:
        """Parse XML data and extract study information"""
        studies = []
        
        try:
            root = ET.fromstring(xml_data)
            articles = root.findall(".//PubmedArticle")
            
            for article in articles:
                study_data = self._extract_study_data(article)
                if study_data:
                    studies.append(study_data)
        
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
        return studies
    
    def _extract_study_data(self, article_xml) -> Optional[Dict]:
        """Extract comprehensive data from a single article"""
        try:
            # Basic article information
            pmid_elem = article_xml.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Title
            title_elem = article_xml.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_elem = article_xml.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Journal
            journal_elem = article_xml.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Publication date
            year = self._extract_publication_year(article_xml)
            
            # Authors and affiliations
            authors_data = self._extract_authors_affiliations(article_xml)
            
            # Extract country from first author affiliation
            country = None
            if authors_data and authors_data[0].get('affiliation'):
                country = self.country_parser.parse_country(authors_data[0]['affiliation'])
            
            # DOI
            doi = self._extract_doi(article_xml)
            
            # Publication types
            pub_types = self._extract_publication_types(article_xml)
            
            # Keywords and MeSH terms
            keywords = self._extract_keywords_mesh(article_xml)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'country': country,
                'doi': doi,
                'authors': json.dumps(authors_data) if authors_data else "",
                'publication_types': "|".join(pub_types) if pub_types else "",
                'keywords': "|".join(keywords) if keywords else "",
                'author_count': len(authors_data) if authors_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting study data: {e}")
            return None
    
    def _extract_publication_year(self, article_xml) -> str:
        """Extract publication year from article XML"""
        # Try different date elements
        date_elements = [
            ".//PubDate/Year",
            ".//PubDate/MedlineDate",
            ".//ArticleDate/Year"
        ]
        
        for element_path in date_elements:
            elem = article_xml.find(element_path)
            if elem is not None and elem.text:
                # Extract year from various formats
                text = elem.text
                if text.isdigit() and len(text) == 4:
                    return text
                # Extract year from "2023 Jan" format
                import re
                year_match = re.search(r'\b(20\d{2})\b', text)
                if year_match:
                    return year_match.group(1)
        
        return ""
    
    def _extract_authors_affiliations(self, article_xml) -> List[Dict]:
        """Extract authors and their affiliations"""
        authors = []
        
        author_elements = article_xml.findall(".//Author")
        for author in author_elements:
            # Name
            lastname_elem = author.find("LastName")
            firstname_elem = author.find("ForeName")
            
            lastname = lastname_elem.text if lastname_elem is not None else ""
            firstname = firstname_elem.text if firstname_elem is not None else ""
            
            # Affiliation
            affiliation_elem = author.find("AffiliationInfo/Affiliation")
            affiliation = affiliation_elem.text if affiliation_elem is not None else ""
            
            if lastname or firstname:
                authors.append({
                    'lastname': lastname,
                    'firstname': firstname,
                    'affiliation': affiliation
                })
        
        return authors
    
    def _extract_doi(self, article_xml) -> str:
        """Extract DOI from article"""
        doi_elem = article_xml.find(".//ELocationID[@EIdType='doi']")
        return doi_elem.text if doi_elem is not None else ""
    
    def _extract_publication_types(self, article_xml) -> List[str]:
        """Extract publication types"""
        pub_types = []
        type_elements = article_xml.findall(".//PublicationType")
        
        for elem in type_elements:
            if elem.text:
                pub_types.append(elem.text)
        
        return pub_types
    
    def _extract_keywords_mesh(self, article_xml) -> List[str]:
        """Extract keywords and MeSH terms"""
        keywords = []
        
        # MeSH terms
        mesh_elements = article_xml.findall(".//MeshHeading/DescriptorName")
        for elem in mesh_elements:
            if elem.text:
                keywords.append(elem.text)
        
        # Keywords
        keyword_elements = article_xml.findall(".//Keyword")
        for elem in keyword_elements:
            if elem.text:
                keywords.append(elem.text)
        
        return list(set(keywords))  # Remove duplicates
    
    def _save_progress_checkpoint(self, total_count: int):
        """Save extraction progress checkpoint"""
        checkpoint_file = f"extraction_checkpoint_{total_count}.json"
        
        checkpoint_data = {
            'total_extracted': total_count,
            'processed_pmids': list(self.processed_pmids),
            'timestamp': time.time()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _finalize_dataset(self) -> pd.DataFrame:
        """Finalize and clean the complete dataset"""
        if not self.all_studies:
            logger.warning("No studies extracted")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_studies)
        
        # Clean and standardize
        df = df.drop_duplicates(subset=['pmid'])
        df = df[df['pmid'].notna() & (df['pmid'] != '')]
        
        # Filter by year range
        df = df[df['year'].str.match(r'^20(1[3-9]|2[0-3])$', na=False)]
        
        # Clean text fields
        text_columns = ['title', 'abstract', 'journal']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Sort by year and PMID
        df = df.sort_values(['year', 'pmid'])
        
        logger.info(f"Dataset finalized: {len(df):,} studies")
        return df

def main():
    """Run comprehensive extraction"""
    extractor = BurnsDataExtractor()
    dataset = extractor.extract_comprehensive_dataset("comprehensive_burns_dataset.csv")
    
    print(f"\nExtraction Summary:")
    print(f"Total studies: {len(dataset):,}")
    print(f"Countries: {dataset['country'].nunique()}")
    print(f"Journals: {dataset['journal'].nunique()}")
    print(f"Year range: {dataset['year'].min()} - {dataset['year'].max()}")

if __name__ == "__main__":
    main()