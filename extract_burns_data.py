#!/usr/bin/env python3
"""
Command-line interface for Burns Research Data Extraction
"""

import argparse
import sys
import logging
from burns_extractor import BurnsDataExtractor
from config import *

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="Extract burns research data from PubMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_burns_data.py --comprehensive
  python extract_burns_data.py --output my_dataset.csv --email me@example.com
  python extract_burns_data.py --quick --max-studies 1000
        """
    )
    
    parser.add_argument(
        '--comprehensive', 
        action='store_true',
        help='Run comprehensive extraction using all 18 search strategies'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true', 
        help='Run quick extraction with limited strategies'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='burns_dataset.csv',
        help='Output CSV filename (default: burns_dataset.csv)'
    )
    
    parser.add_argument(
        '--email',
        help='NCBI email address (overrides config)'
    )
    
    parser.add_argument(
        '--api-key',
        help='NCBI API key (overrides config)'
    )
    
    parser.add_argument(
        '--max-studies',
        type=int,
        help='Maximum number of studies to extract'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    
    # Validate arguments
    if not args.comprehensive and not args.quick:
        print("Error: Must specify either --comprehensive or --quick")
        sys.exit(1)
    
    # Initialize extractor
    try:
        extractor = BurnsDataExtractor(
            email=args.email,
            api_key=args.api_key
        )
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        sys.exit(1)
    
    # Run extraction
    try:
        if args.comprehensive:
            print("Starting comprehensive extraction (all 18 search strategies)...")
            dataset = extractor.extract_comprehensive_dataset(args.output)
        else:
            print("Starting quick extraction...")
            # Quick extraction with limited strategies
            limited_strategies = SEARCH_STRATEGIES[:5]  # First 5 strategies
            extractor.search_strategies = limited_strategies
            dataset = extractor.extract_comprehensive_dataset(args.output)
        
        # Display results
        print(f"\n{'='*50}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"Total studies extracted: {len(dataset):,}")
        print(f"Countries represented: {dataset['country'].nunique()}")
        print(f"Journals: {dataset['journal'].nunique()}")
        print(f"Year range: {dataset['year'].min()} - {dataset['year'].max()}")
        print(f"Output file: {args.output}")
        
        # Top countries
        print(f"\nTop 10 countries:")
        top_countries = dataset['country'].value_counts().head(10)
        for i, (country, count) in enumerate(top_countries.items(), 1):
            print(f"  {i:2d}. {country}: {count:,} studies")
        
        # Top journals
        print(f"\nTop 5 journals:")
        top_journals = dataset['journal'].value_counts().head(5)
        for i, (journal, count) in enumerate(top_journals.items(), 1):
            print(f"  {i}. {journal}: {count:,} studies")
        
        print(f"\nDataset ready for analysis!")
        
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()