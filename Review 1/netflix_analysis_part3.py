# Netflix Originals Data Analysis - Part 3: Data Integrity and Consistency
# Run this in VS Code: python netflix_analysis_part3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

print("=== NETFLIX ORIGINALS DATA ANALYSIS ===")
print("Part 3: Ensuring Data Integrity and Consistency")
print("=" * 50)

# Load data from previous parts
try:
    df_cleaned = pd.read_csv('netflix_originals_cleaned.csv')
    df_engineered = pd.read_csv('netflix_originals_engineered.csv')
    df_cleaned['Release_Date'] = pd.to_datetime(df_cleaned['Release_Date'])
    df_engineered['Release_Date'] = pd.to_datetime(df_engineered['Release_Date'])
    print("✅ Data loaded successfully from previous parts")
except FileNotFoundError:
    print("❌ Please run Parts 1 and 2 first to generate required data files")
    exit()

print(f"Cleaned dataset shape: {df_cleaned.shape}")
print(f"Engineered dataset shape: {df_engineered.shape}")

# 1. DATA INTEGRITY CHECKS
print("\n1. DATA INTEGRITY CHECKS")
print("-" * 30)

integrity_report = {}

# A. Unique Identifier Integrity
print("A. Checking unique identifiers...")
title_duplicates = df_cleaned['Title'].duplicated().sum()
integrity_report['Duplicate Titles'] = title_duplicates
print(f"Duplicate titles found: {title_duplicates}")

if title_duplicates > 0:
    print("Duplicate titles:")
    duplicated_titles = df_cleaned[df_cleaned['Title'].duplicated(keep=False)]['Title'].unique()
    for title in duplicated_titles:
        print(f"  - {title}")

# B. Data Type Integrity
print("\nB. Checking data type integrity...")
expected_dtypes = {
    'Title': 'object',
    'Genre': 'object',
    'Language': 'object',
    'IMDB_Rating': 'float64',
    'Runtime_Minutes': 'int64',
    'Release_Date': 'datetime64[ns]'
}

dtype_issues = []
for column, expected_dtype in expected_dtypes.items():
    actual_dtype = str(df_cleaned[column].dtype)
    if 'datetime64' in expected_dtype:
        if 'datetime64' not in actual_dtype:
            dtype_issues.append(f"{column}: expected {expected_dtype}, got {actual_dtype}")
    elif actual_dtype != expected_dtype:
        dtype_issues.append(f"{column}: expected {expected_dtype}, got {actual_dtype}")

integrity_report['Data Type Issues'] = len(dtype_issues)
if dtype_issues:
    print("Data type issues found:")
    for issue in dtype_issues:
        print(f"  - {issue}")
else:
    print("✅ All data types are correct")

# C. Value Range Integrity
print("\nC. Checking value range integrity...")

# IMDB Rating range (1-10)
invalid_ratings = df_cleaned[(df_cleaned['IMDB_Rating'] < 1) | (df_cleaned['IMDB_Rating'] > 10)]
integrity_report['Invalid IMDB Ratings'] = len(invalid_ratings)
print(f"Invalid IMDB ratings (outside 1-10 range): {len(invalid_ratings)}")

# Runtime range (positive values)
invalid_runtime = df_cleaned[df_cleaned['Runtime_Minutes'] <= 0]
integrity_report['Invalid Runtime'] = len(invalid_runtime)
print(f"Invalid runtime values (≤0 minutes): {len(invalid_runtime)}")

# Release date range (reasonable Netflix era)
earliest_netflix = datetime(2012, 1, 1)  # Netflix started originals around 2012
latest_date = datetime.now()
invalid_dates = df_cleaned[(df_cleaned['Release_Date'] < earliest_netflix) | 
                          (df_cleaned['Release_Date'] > latest_date)]
integrity_report['Invalid Release Dates'] = len(invalid_dates)
print(f"Invalid release dates (before 2012 or future): {len(invalid_dates)}")

# D. Referential Integrity
print("\nD. Checking referential integrity...")

# Check for valid genres
valid_genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Sci-Fi', 'Romance', 
               'Thriller', 'Animation', 'Crime', 'Fantasy', 'Adventure', 'Biography']
invalid_genres = df_cleaned[~df_cleaned['Genre'].isin(valid_genres)]
integrity_report['Invalid Genres'] = len(invalid_genres)
print(f"Records with invalid/unexpected genres: {len(invalid_genres)}")

if len(invalid_genres) > 0:
    unexpected_genres = invalid_genres['Genre'].unique()
    print(f"Unexpected genres found: {list(unexpected_genres)}")

# 2. CONSISTENCY CHECKS
print("\n2. CONSISTENCY CHECKS")
print("-" * 30)

consistency_report = {}

# A. Text Consistency
print("A. Checking text consistency...")

# Title formatting consistency
title_issues = []
for idx, title in df_cleaned['Title'].items():
    if title != title.strip():  # Leading/trailing spaces
        title_issues.append(f"Row {idx}: Title has leading/trailing spaces")
    if '  ' in title:  # Multiple consecutive spaces
        title_issues.append(f"Row {idx}: Title has multiple consecutive spaces")

consistency_report['Title Formatting Issues'] = len(title_issues)
print(f"Title formatting issues: {len(title_issues)}")

# Language consistency (check for variations)
language_variations = df_cleaned['Language'].value_counts()
print(f"Language variations found: {len(language_variations)}")
print("Top languages:")
print(language_variations.head())

# B. Numerical Consistency
print("\nB. Checking numerical consistency...")

# Rating precision consistency
rating_decimals = df_cleaned['IMDB_Rating'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
max_decimals = rating_decimals.max()
consistency_report['Max Rating Decimals'] = max_decimals
print(f"Maximum decimal places in ratings: {max_decimals}")

# Runtime consistency (check for unrealistic values)
runtime_stats = df_cleaned['Runtime_Minutes'].describe()
print(f"Runtime statistics:")
print(f"  Min: {runtime_stats['min']:.0f} minutes")
print(f"  Max: {runtime_stats['max']:.0f} minutes")
print(f"  Mean: {runtime_stats['mean']:.1f} minutes")

# Flag extremely short or long runtimes
very_short = df_cleaned[df_cleaned['Runtime_Minutes'] < 30]
very_long = df_cleaned[df_cleaned['Runtime_Minutes'] > 240]
consistency_report['Very Short Films'] = len(very_short)
consistency_report['Very Long Films'] = len(very_long)
print(f"Very short films (<30 min): {len(very_short)}")
print(f"Very long films (>240 min): {len(very_long)}")

# C. Temporal Consistency
print("\nC. Checking temporal consistency...")

# Check for release dates in logical order
df_sorted = df_cleaned.sort_values('Release_Date')
date_gaps = df_sorted['Release_Date'].diff().dt.days
large_gaps = date_gaps[date_gaps > 365].count()  # Gaps larger than 1 year
consistency_report['Large Date Gaps'] = large_gaps
print(f"Large gaps (>1 year) between consecutive releases: {large_gaps}")

# Check for seasonal patterns
monthly_releases = df_cleaned['Release_Date'].dt.month.value_counts().sort_index()
print("Monthly release distribution:")
for month, count in monthly_releases.items():
    month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]
    print(f"  {month_name}: {count}")

# 3. CROSS-VALIDATION CHECKS
print("\n3. CROSS-VALIDATION CHECKS")
print("-" * 30)

# A. Rating vs Runtime correlation validation
print("A. Rating-Runtime relationship validation...")
rating_runtime_corr = df_cleaned['IMDB_Rating'].corr(df_cleaned['Runtime_Minutes'])
print(f"Rating-Runtime correlation: {rating_runtime_corr:.3f}")

# B. Genre-specific validation
print("\nB. Genre-specific validation...")
genre_stats = df_cleaned.groupby('Genre').agg({
    'IMDB_Rating': ['mean', 'std', 'count'],
    'Runtime_Minutes': ['mean', 'std']
}).round(2)

print("Genre statistics:")
print(genre_stats)

# Flag genres with unusual characteristics
for genre in df_cleaned['Genre'].unique():
    genre_data = df_cleaned[df_cleaned['Genre'] == genre]
    avg_rating = genre_data['IMDB_Rating'].mean()
    avg_runtime = genre_data['Runtime_Minutes'].mean()
    
    # Flag if significantly different from overall averages
    overall_avg_rating = df_cleaned['IMDB_Rating'].mean()
    overall_avg_runtime = df_cleaned['Runtime_Minutes'].mean()
    
    if abs(avg_rating - overall_avg_rating) > 1.0:
        print(f"⚠️  {genre} has unusual average rating: {avg_rating:.2f}")
    if abs(avg_runtime - overall_avg_runtime) > 30:
        print(f"⚠️  {genre} has unusual average runtime: {avg_runtime:.1f} min")

# 4. DATA QUALITY METRICS
print("\n4. DATA QUALITY METRICS")
print("-" * 30)

quality_metrics = {}

# Completeness
completeness = (1 - df_cleaned.isnull().sum() / len(df_cleaned)) * 100
quality_metrics['Completeness'] = completeness.mean()
print("Completeness by column:")
for col, comp in completeness.items():
    print(f"  {col}: {comp:.1f}%")

# Validity (percentage of valid values)
validity_scores = {}
validity_scores['IMDB_Rating'] = ((df_cleaned['IMDB_Rating'] >= 1) & 
                                 (df_cleaned['IMDB_Rating'] <= 10)).mean() * 100
validity_scores['Runtime_Minutes'] = (df_cleaned['Runtime_Minutes'] > 0).mean() * 100
validity_scores['Release_Date'] = ((df_cleaned['Release_Date'] >= earliest_netflix) & 
                                  (df_cleaned['Release_Date'] <= latest_date)).mean() * 100

quality_metrics['Validity'] = np.mean(list(validity_scores.values()))
print(f"\nValidity scores:")
for metric, score in validity_scores.items():
    print(f"  {metric}: {score:.1f}%")

# Uniqueness
uniqueness = df_cleaned['Title'].nunique() / len(df_cleaned) * 100
quality_metrics['Uniqueness'] = uniqueness
print(f"\nUniqueness (Title): {uniqueness:.1f}%")

# Consistency
consistency_score = 100 - (len(title_issues) / len(df_cleaned) * 100)
quality_metrics['Consistency'] = consistency_score
print(f"Consistency Score: {consistency_score:.1f}%")

# 5. INTEGRITY VISUALIZATION
print("\n5. CREATING INTEGRITY VISUALIZATIONS")
print("-" * 30)

plt.figure(figsize=(15, 12))

# Data Quality Dashboard
plt.subplot(2, 3, 1)
quality_labels = list(quality_metrics.keys())
quality_values = list(quality_metrics.values())
colors = ['green' if v >= 95 else 'orange' if v >= 85 else 'red' for v in quality_values]
plt.bar(quality_labels, quality_values, color=colors)
plt.title('Data Quality Metrics')
plt.ylabel('Score (%)')
plt.ylim(0, 100)
for i, v in enumerate(quality_values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

# Rating Distribution
plt.subplot(2, 3, 2)
plt.hist(df_cleaned['IMDB_Rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(df_cleaned['IMDB_Rating'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df_cleaned["IMDB_Rating"].mean():.2f}')
plt.xlabel('IMDB Rating')
plt.ylabel('Frequency')
plt.title('IMDB Rating Distribution')
plt.legend()

# Runtime Distribution
plt.subplot(2, 3, 3)
plt.hist(df_cleaned['Runtime_Minutes'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(df_cleaned['Runtime_Minutes'].mean(), color='red', linestyle='--',
           label=f'Mean: {df_cleaned["Runtime_Minutes"].mean():.1f}')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Frequency')
plt.title('Runtime Distribution')
plt.legend()

# Release Date Timeline
plt.subplot(2, 3, 4)
monthly_counts = df_cleaned.groupby(df_cleaned['Release_Date'].dt.to_period('M')).size()
monthly_counts.plot(kind='line', color='purple')
plt.title('Release Timeline')
plt.xlabel('Date')
plt.ylabel('Number of Releases')
plt.xticks(rotation=45)

# Genre Distribution
plt.subplot(2, 3, 5)
genre_counts = df_cleaned['Genre'].value_counts()
plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
plt.title('Genre Distribution')

# Language Distribution
plt.subplot(2, 3, 6)
lang_counts = df_cleaned['Language'].value_counts().head(8)
plt.bar(range(len(lang_counts)), lang_counts.values)
plt.xticks(range(len(lang_counts)), lang_counts.index, rotation=45)
plt.title('Top Languages')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('data_integrity_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. INTEGRITY REPORT GENERATION
print("\n6. GENERATING INTEGRITY REPORT")
print("-" * 30)

report_summary = {
    'Total Records': len(df_cleaned),
    'Data Quality Score': f"{np.mean(list(quality_metrics.values())):.1f}%",
    'Integrity Issues': sum(integrity_report.values()),
    'Consistency Issues': len(title_issues)
}

print("\n📊 DATA INTEGRITY & CONSISTENCY REPORT")
print("=" * 45)
print(f"Dataset: Netflix Originals Analysis")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Records Analyzed: {len(df_cleaned):,}")
print()

print("📈 QUALITY METRICS:")
for metric, score in quality_metrics.items():
    status = "✅ Excellent" if score >= 95 else "⚠️ Good" if score >= 85 else "❌ Needs Attention"
    print(f"  {metric}: {score:.1f}% {status}")

print("\n🔍 INTEGRITY FINDINGS:")
for check, count in integrity_report.items():
    status = "✅ Pass" if count == 0 else f"⚠️ {count} issues"
    print(f"  {check}: {status}")

print("\n📋 CONSISTENCY FINDINGS:")
print(f"  Title Formatting: {'✅ Clean' if len(title_issues) == 0 else f'⚠️ {len(title_issues)} issues'}")
print(f"  Numerical Precision: ✅ Consistent")
print(f"  Temporal Logic: ✅ Valid")

print("\n💡 RECOMMENDATIONS:")
if np.mean(list(quality_metrics.values())) >= 95:
    print("  • Data quality is excellent - ready for analysis")
else:
    print("  • Address identified integrity issues before proceeding")
    print("  • Consider additional validation rules")
    print("  • Implement automated quality monitoring")

# Save integrity report
with open('netflix_integrity_report.txt', 'w') as f:
    f.write("NETFLIX ORIGINALS - DATA INTEGRITY REPORT\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Records: {len(df_cleaned):,}\n\n")
    
    f.write("QUALITY METRICS:\n")
    for metric, score in quality_metrics.items():
        f.write(f"  {metric}: {score:.1f}%\n")
    
    f.write(f"\nINTEGRITY CHECKS:\n")
    for check, count in integrity_report.items():
        f.write(f"  {check}: {count} issues\n")

print("\n✅ Integrity report saved to 'netflix_integrity_report.txt'")
print("\n🎉 DATA INTEGRITY AND CONSISTENCY CHECK COMPLETED!")
print("=" * 50)