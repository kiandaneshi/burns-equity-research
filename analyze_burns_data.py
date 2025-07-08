#!/usr/bin/env python3
"""
Command-line interface for Burns Research Data Analysis
"""

import argparse
import sys
import os
import logging
from burns_analyzer import BurnsAnalyzer
from config import *

def main():
    """Main command-line interface for analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze burns research data with statistical validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_burns_data.py --input data.csv --output results/
  python analyze_burns_data.py --input data.csv --normality-tests
  python analyze_burns_data.py --input data.csv --quick-analysis
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file containing burns data'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='analysis_results',
        help='Output directory for results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--normality-tests',
        action='store_true',
        help='Run comprehensive normality testing'
    )
    
    parser.add_argument(
        '--quick-analysis',
        action='store_true',
        help='Run quick analysis (descriptive stats only)'
    )
    
    parser.add_argument(
        '--generate-figures',
        action='store_true',
        help='Generate publication-ready figures'
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
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize analyzer
        print(f"Loading data from {args.input}...")
        analyzer = BurnsAnalyzer(data_file=args.input)
        analyzer.output_dir = args.output
        
        print(f"Loaded {len(analyzer.df):,} studies for analysis")
        
        # Run analysis based on options
        if args.quick_analysis:
            print("Running quick analysis...")
            results = {}
            results.update(analyzer._generate_descriptive_analysis())
            results.update(analyzer._generate_temporal_analysis())
            
        elif args.normality_tests:
            print("Running normality testing...")
            results = analyzer._run_normality_testing()
            
        elif args.generate_figures:
            print("Generating publication figures...")
            results = analyzer._generate_manuscript_figures()
            
        else:
            print("Running comprehensive analysis...")
            results = analyzer.run_complete_analysis()
        
        # Display results
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print(f"Output directory: {args.output}")
        print(f"Generated files:")
        
        for analysis_type, file_path in results.items():
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {analysis_type}: {file_path} ({file_size:,} bytes)")
        
        # Key statistics
        if hasattr(analyzer, 'df'):
            df = analyzer.df
            print(f"\nKey Statistics:")
            print(f"  Total studies: {len(df):,}")
            print(f"  Countries: {df['country'].nunique()}")
            print(f"  Journals: {df['journal'].nunique()}")
            print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
            
            # Top countries
            top_countries = df['country'].value_counts().head(5)
            print(f"  Top countries: {', '.join(top_countries.index)}")
            
            # Geographic concentration
            total_studies = len(df)
            top_10_percentage = (top_countries.head(10).sum() / total_studies) * 100
            print(f"  Geographic concentration: Top 10 countries = {top_10_percentage:.1f}%")
        
        print(f"\nAnalysis ready for manuscript preparation!")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()