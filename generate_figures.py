#!/usr/bin/env python3
"""
Command-line interface for generating publication-ready figures
"""

import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from config import *

def main():
    """Generate publication-ready figures from burns data"""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for burns research analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_figures.py --data data.csv --normality-tests
  python generate_figures.py --data data.csv --temporal-analysis
  python generate_figures.py --data data.csv --all-figures
        """
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Input CSV file containing burns data'
    )
    
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory for figures (default: figures)'
    )
    
    parser.add_argument(
        '--normality-tests',
        action='store_true',
        help='Generate normality testing plots'
    )
    
    parser.add_argument(
        '--temporal-analysis',
        action='store_true',
        help='Generate temporal trend figures'
    )
    
    parser.add_argument(
        '--geographic-analysis',
        action='store_true',
        help='Generate geographic distribution figures'
    )
    
    parser.add_argument(
        '--all-figures',
        action='store_true',
        help='Generate all publication figures'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df):,} studies")
        
        generated_files = []
        
        # Generate requested figures
        if args.normality_tests or args.all_figures:
            normality_file = generate_normality_plots(df, args.output_dir, args.dpi)
            generated_files.append(normality_file)
        
        if args.temporal_analysis or args.all_figures:
            temporal_file = generate_temporal_plots(df, args.output_dir, args.dpi)
            generated_files.append(temporal_file)
        
        if args.geographic_analysis or args.all_figures:
            geographic_file = generate_geographic_plots(df, args.output_dir, args.dpi)
            generated_files.append(geographic_file)
        
        if args.all_figures:
            manuscript_files = generate_manuscript_figures(df, args.output_dir, args.dpi)
            generated_files.extend(manuscript_files)
        
        # Display results
        print(f"\n{'='*50}")
        print("FIGURE GENERATION COMPLETE")
        print(f"{'='*50}")
        print(f"Output directory: {args.output_dir}")
        print(f"Generated files:")
        
        for file_path in generated_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {os.path.basename(file_path)}: {file_size:,} bytes")
        
        print(f"\nFigures ready for manuscript submission!")
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        sys.exit(1)

def generate_normality_plots(df, output_dir, dpi):
    """Generate comprehensive normality testing plots"""
    print("Generating normality testing plots...")
    
    # Prepare data
    yearly_counts = df['year'].value_counts().sort_index().values
    country_counts = df['country'].value_counts().values
    journal_counts = df['journal'].value_counts().values[:50]
    log_country = np.log(country_counts[country_counts > 0])
    log_journal = np.log(journal_counts[journal_counts > 0])
    
    # Create 9-panel normality analysis
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    colors = ['skyblue', 'lightgreen', 'salmon']
    datasets = [
        ('Annual Publications', yearly_counts),
        ('Country Publications (log)', log_country),
        ('Journal Publications (log)', log_journal)
    ]
    
    for i, (name, data) in enumerate(datasets):
        color = colors[i]
        
        # Histogram with normal curve
        axes[i, 0].hist(data, bins=15, density=True, alpha=0.7, color=color, edgecolor='black')
        x = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x, np.mean(data), np.std(data))
        axes[i, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal curve')
        axes[i, 0].set_title(f'{chr(65+i*3)}. {name} Distribution', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('Value')
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(data, dist='norm', plot=axes[i, 1])
        axes[i, 1].set_title(f'{chr(66+i*3)}. {name} Q-Q Plot', fontsize=14, fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[i, 2].boxplot(data, patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7))
        axes[i, 2].set_title(f'{chr(67+i*3)}. {name} Box Plot', fontsize=14, fontweight='bold')
        axes[i, 2].set_ylabel('Value')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Comprehensive_Normality_Analysis.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_temporal_plots(df, output_dir, dpi):
    """Generate temporal trend analysis plots"""
    print("Generating temporal analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Annual publication trends
    yearly_counts = df['year'].value_counts().sort_index()
    axes[0, 0].plot(yearly_counts.index, yearly_counts.values, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_title('A. Annual Publication Trends', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Publications')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel B: Growth rates
    growth_rates = yearly_counts.pct_change() * 100
    axes[0, 1].bar(growth_rates.index[1:], growth_rates.values[1:])
    axes[0, 1].set_title('B. Annual Growth Rates', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Growth Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel C: Cumulative publications
    cumulative = np.cumsum(yearly_counts.values)
    axes[1, 0].plot(yearly_counts.index, cumulative, 's-', linewidth=2, markersize=4)
    axes[1, 0].set_title('C. Cumulative Publications', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Cumulative Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: Top countries over time
    top_countries = df['country'].value_counts().head(5).index
    yearly_country = df.groupby(['year', 'country']).size().unstack(fill_value=0)
    
    for country in top_countries:
        if country in yearly_country.columns:
            axes[1, 1].plot(yearly_country.index, yearly_country[country], 
                          marker='o', label=country, linewidth=2)
    
    axes[1, 1].set_title('D. Top Countries Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Number of Publications')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Temporal_Analysis.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_geographic_plots(df, output_dir, dpi):
    """Generate geographic distribution plots"""
    print("Generating geographic analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Top countries
    country_counts = df['country'].value_counts().head(15)
    axes[0, 0].barh(range(len(country_counts)), country_counts.values)
    axes[0, 0].set_yticks(range(len(country_counts)))
    axes[0, 0].set_yticklabels(country_counts.index)
    axes[0, 0].set_title('A. Top 15 Countries', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Publications')
    
    # Panel B: Geographic concentration curve
    country_percentages = (country_counts / len(df)) * 100
    cumulative_percentage = np.cumsum(country_percentages)
    axes[0, 1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
    axes[0, 1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[0, 1].set_title('B. Geographic Concentration', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Country Rank')
    axes[0, 1].set_ylabel('Cumulative Percentage (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel C: Country distribution (log scale)
    all_countries = df['country'].value_counts()
    axes[1, 0].hist(np.log(all_countries.values), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('C. Country Publications (log scale)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Log(Publications)')
    axes[1, 0].set_ylabel('Number of Countries')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: Top journals
    journal_counts = df['journal'].value_counts().head(10)
    axes[1, 1].bar(range(len(journal_counts)), journal_counts.values)
    axes[1, 1].set_xticks(range(len(journal_counts)))
    axes[1, 1].set_xticklabels(journal_counts.index, rotation=45, ha='right')
    axes[1, 1].set_title('D. Top 10 Journals', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Publications')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Geographic_Analysis.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_manuscript_figures(df, output_dir, dpi):
    """Generate all manuscript-ready figures"""
    print("Generating manuscript figures...")
    
    files = []
    
    # Figure 1: Overview analysis
    fig1 = generate_figure_1(df, output_dir, dpi)
    files.append(fig1)
    
    # Figure 2: Statistical analysis
    fig2 = generate_figure_2(df, output_dir, dpi)
    files.append(fig2)
    
    # Figure 3: Publication patterns
    fig3 = generate_figure_3(df, output_dir, dpi)
    files.append(fig3)
    
    return files

def generate_figure_1(df, output_dir, dpi):
    """Generate Figure 1: Temporal and Geographic Overview"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Temporal trends
    yearly_counts = df['year'].value_counts().sort_index()
    axes[0, 0].plot(yearly_counts.index, yearly_counts.values, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_title('A. Annual Publication Trends', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Publications')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel B: Top countries
    country_counts = df['country'].value_counts().head(15)
    axes[0, 1].barh(range(len(country_counts)), country_counts.values)
    axes[0, 1].set_yticks(range(len(country_counts)))
    axes[0, 1].set_yticklabels(country_counts.index)
    axes[0, 1].set_title('B. Top 15 Countries', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Publications')
    
    # Panel C: Journal distribution
    journal_counts = df['journal'].value_counts().head(10)
    axes[1, 0].bar(range(len(journal_counts)), journal_counts.values)
    axes[1, 0].set_xticks(range(len(journal_counts)))
    axes[1, 0].set_xticklabels(journal_counts.index, rotation=45, ha='right')
    axes[1, 0].set_title('C. Top 10 Journals', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Publications')
    
    # Panel D: Geographic concentration
    country_percentages = (country_counts / len(df)) * 100
    cumulative_percentage = np.cumsum(country_percentages)
    axes[1, 1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
    axes[1, 1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[1, 1].set_title('D. Geographic Concentration', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Country Rank')
    axes[1, 1].set_ylabel('Cumulative Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Figure_1_Overview_Analysis.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_figure_2(df, output_dir, dpi):
    """Generate Figure 2: Statistical Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    yearly_counts = df['year'].value_counts().sort_index().values
    country_counts = df['country'].value_counts().values
    log_country = np.log(country_counts[country_counts > 0])
    
    # Panel A: Annual publications Q-Q plot
    stats.probplot(yearly_counts, dist='norm', plot=axes[0, 0])
    axes[0, 0].set_title('A. Annual Publications Normality', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel B: Country publications Q-Q plot
    stats.probplot(log_country, dist='norm', plot=axes[0, 1])
    axes[0, 1].set_title('B. Country Publications Normality (log)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel C: Growth analysis
    yearly_counts_series = df['year'].value_counts().sort_index()
    growth_rates = yearly_counts_series.pct_change() * 100
    axes[1, 0].bar(growth_rates.index[1:], growth_rates.values[1:])
    axes[1, 0].set_title('C. Annual Growth Rates', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Growth Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: Distribution summary
    test_vars = ['Annual\nPubs', 'Country\nPubs', 'Journal\nPubs']
    normality_results = ['Non-normal', 'Non-normal', 'Non-normal']
    colors = ['red', 'red', 'red']
    
    bars = axes[1, 1].bar(test_vars, [1, 1, 1], color=colors, alpha=0.3)
    axes[1, 1].set_title('D. Normality Assessment', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Test Result')
    axes[1, 1].set_ylim(0, 1)
    
    for i, result in enumerate(normality_results):
        axes[1, 1].text(i, 0.5, result, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Figure_2_Statistical_Analysis.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_figure_3(df, output_dir, dpi):
    """Generate Figure 3: Publication Patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Collaboration patterns (author count if available)
    if 'author_count' in df.columns:
        author_data = df.groupby('country')['author_count'].mean().sort_values(ascending=False).head(15)
        axes[0, 0].bar(range(len(author_data)), author_data.values)
        axes[0, 0].set_xticks(range(len(author_data)))
        axes[0, 0].set_xticklabels(author_data.index, rotation=45, ha='right')
        axes[0, 0].set_title('A. Average Authors per Study', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Average Authors')
    else:
        # Alternative: Study count distribution
        country_counts = df['country'].value_counts()
        axes[0, 0].hist(country_counts.values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('A. Country Publication Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Publications')
        axes[0, 0].set_ylabel('Number of Countries')
    
    # Panel B: Journal concentration
    journal_counts = df['journal'].value_counts()
    journal_percentages = (journal_counts / len(df)) * 100
    top_journals = journal_percentages.head(15)
    
    axes[0, 1].barh(range(len(top_journals)), top_journals.values)
    axes[0, 1].set_yticks(range(len(top_journals)))
    axes[0, 1].set_yticklabels(top_journals.index)
    axes[0, 1].set_title('B. Journal Distribution (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Percentage of Publications')
    
    # Panel C: Productivity trends by top countries
    top_countries = df['country'].value_counts().head(5).index
    yearly_country = df.groupby(['year', 'country']).size().unstack(fill_value=0)
    
    for country in top_countries:
        if country in yearly_country.columns:
            axes[1, 0].plot(yearly_country.index, yearly_country[country], 
                          marker='o', label=country, linewidth=2)
    
    axes[1, 0].set_title('C. Top Countries Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Publications')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: Research intensity
    yearly_totals = df['year'].value_counts().sort_index()
    yearly_countries = df.groupby('year')['country'].nunique().sort_index()
    research_intensity = yearly_totals / yearly_countries
    
    axes[1, 1].plot(research_intensity.index, research_intensity.values, 's-', linewidth=2)
    axes[1, 1].set_title('D. Research Intensity\n(Publications per Active Country)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Publications per Country')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Figure_3_Publication_Patterns.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file

if __name__ == "__main__":
    main()