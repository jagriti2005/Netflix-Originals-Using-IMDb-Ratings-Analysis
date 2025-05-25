# Netflix Originals Data Analysis - Part 1: Data Cleaning and Missing Values
# Run this in VS Code: python netflix_analysis_part1.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== NETFLIX ORIGINALS DATA ANALYSIS ===")
print("Part 1: Data Cleaning and Handling Missing Values")
print("=" * 50)

# You'll need to download Netflix dataset from Kaggle or similar source
# For this example, I'll create a sample dataset structure
# Replace this with your actual data loading
try:
    # Try to load actual Netflix data
    df = pd.read_csv('netflix_originals.csv')
    print("✅ Data loaded successfully from netflix_originals.csv")
except FileNotFoundError:
    # Create sample data if file not found
    print("⚠️ netflix_originals.csv not found. Creating sample data for demonstration.")
    
    # Sample Netflix Originals data structure
    np.random.seed(42)
    n_movies = 500
    
    titles = [f"Netflix Original {i}" for i in range(1, n_movies + 1)]
    genres = np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Sci-Fi', 'Romance'], n_movies)
    languages = np.random.choice(['English', 'Spanish', 'French', 'Korean', 'Japanese', 'German'], n_movies, p=[0.4, 0.2, 0.1, 0.15, 0.1, 0.05])
    imdb_ratings = np.random.normal(6.5, 1.2, n_movies)
    imdb_ratings = np.clip(imdb_ratings, 1, 10)  # Clip to valid IMDB range
    runtimes = np.random.normal(95, 25, n_movies)
    runtimes = np.clip(runtimes, 30, 300)  # Reasonable runtime range
    
    # Add some missing values intentionally
    imdb_ratings[np.random.choice(n_movies, 50, replace=False)] = np.nan
    runtimes[np.random.choice(n_movies, 30, replace=False)] = np.nan
    languages[np.random.choice(n_movies, 20, replace=False)] = None
    
    # Create release dates
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    release_dates = [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n_movies)]
    
    df = pd.DataFrame({
        'Title': titles,
        'Genre': genres,
        'Language': languages,
        'IMDB_Rating': imdb_ratings,
        'Runtime_Minutes': runtimes,
        'Release_Date': release_dates
    })
    
    print("📊 Sample dataset created with intentional missing values")

# 1. INITIAL DATA EXPLORATION
print("\n1. INITIAL DATA EXPLORATION")
print("-" * 30)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# 2. MISSING VALUES ANALYSIS
print("\n2. MISSING VALUES ANALYSIS")
print("-" * 30)

# Count missing values
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_percentages
})

print("Missing values summary:")
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))

# Missing values heatmap
plt.subplot(1, 2, 1)
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Columns')

# Missing values bar chart
plt.subplot(1, 2, 2)
missing_data = missing_summary[missing_summary['Missing_Count'] > 0]
if not missing_data.empty:
    plt.bar(missing_data.index, missing_data['Missing_Percentage'])
    plt.title('Missing Values Percentage by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. DATA CLEANING STRATEGIES
print("\n3. DATA CLEANING STRATEGIES")
print("-" * 30)

# Create a copy for cleaning
df_cleaned = df.copy()

# Handle missing values in IMDB_Rating
if df_cleaned['IMDB_Rating'].isnull().sum() > 0:
    print(f"Handling {df_cleaned['IMDB_Rating'].isnull().sum()} missing IMDB ratings...")
    
    # Strategy: Fill with median rating by genre
    median_ratings_by_genre = df_cleaned.groupby('Genre')['IMDB_Rating'].median()
    print("Median ratings by genre:")
    print(median_ratings_by_genre)
    
    # Fill missing values
    df_cleaned['IMDB_Rating'] = df_cleaned.apply(
        lambda row: median_ratings_by_genre[row['Genre']] 
        if pd.isna(row['IMDB_Rating']) else row['IMDB_Rating'], 
        axis=1
    )
    print("✅ IMDB ratings filled with genre-specific medians")

# Handle missing values in Runtime_Minutes
if df_cleaned['Runtime_Minutes'].isnull().sum() > 0:
    print(f"\nHandling {df_cleaned['Runtime_Minutes'].isnull().sum()} missing runtime values...")
    
    # Strategy: Fill with median runtime by genre
    median_runtime_by_genre = df_cleaned.groupby('Genre')['Runtime_Minutes'].median()
    print("Median runtime by genre:")
    print(median_runtime_by_genre)
    
    df_cleaned['Runtime_Minutes'] = df_cleaned.apply(
        lambda row: median_runtime_by_genre[row['Genre']] 
        if pd.isna(row['Runtime_Minutes']) else row['Runtime_Minutes'], 
        axis=1
    )
    print("✅ Runtime values filled with genre-specific medians")

# Handle missing values in Language
if df_cleaned['Language'].isnull().sum() > 0:
    print(f"\nHandling {df_cleaned['Language'].isnull().sum()} missing language values...")
    
    # Strategy: Fill with most common language
    most_common_language = df_cleaned['Language'].mode()[0]
    df_cleaned['Language'].fillna(most_common_language, inplace=True)
    print(f"✅ Missing languages filled with most common: {most_common_language}")

# 4. DATA TYPE CONVERSIONS
print("\n4. DATA TYPE CONVERSIONS")
print("-" * 30)

# Convert Release_Date to datetime if it's not already
if df_cleaned['Release_Date'].dtype == 'object':
    df_cleaned['Release_Date'] = pd.to_datetime(df_cleaned['Release_Date'])
    print("✅ Release_Date converted to datetime")

# Round numeric columns to appropriate precision
df_cleaned['IMDB_Rating'] = df_cleaned['IMDB_Rating'].round(1)
df_cleaned['Runtime_Minutes'] = df_cleaned['Runtime_Minutes'].round(0).astype(int)

print("✅ Numeric columns rounded to appropriate precision")

# 5. DUPLICATE REMOVAL
print("\n5. DUPLICATE REMOVAL")
print("-" * 30)

initial_count = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
final_count = len(df_cleaned)
duplicates_removed = initial_count - final_count

print(f"Initial records: {initial_count}")
print(f"Final records: {final_count}")
print(f"Duplicates removed: {duplicates_removed}")

# 6. FINAL CLEANING SUMMARY
print("\n6. FINAL CLEANING SUMMARY")
print("-" * 30)

print("Before cleaning:")
print(f"- Shape: {df.shape}")
print(f"- Missing values: {df.isnull().sum().sum()}")

print("\nAfter cleaning:")
print(f"- Shape: {df_cleaned.shape}")
print(f"- Missing values: {df_cleaned.isnull().sum().sum()}")

# Display final cleaned data info
print("\nCleaned dataset info:")
print(df_cleaned.info())

print("\nCleaned dataset sample:")
print(df_cleaned.head())

# Save cleaned data
df_cleaned.to_csv('netflix_originals_cleaned.csv', index=False)
print("\n✅ Cleaned data saved to 'netflix_originals_cleaned.csv'")

# 7. CLEANING VALIDATION
print("\n7. CLEANING VALIDATION")
print("-" * 30)

# Check for any remaining issues
print("Validation checks:")
print(f"✓ No missing values: {df_cleaned.isnull().sum().sum() == 0}")
print(f"✓ IMDB ratings in valid range (1-10): {df_cleaned['IMDB_Rating'].between(1, 10).all()}")
print(f"✓ Runtime values positive: {(df_cleaned['Runtime_Minutes'] > 0).all()}")
print(f"✓ All titles unique: {df_cleaned['Title'].nunique() == len(df_cleaned)}")

print("\n🎉 DATA CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 50)