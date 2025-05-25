# Netflix Originals IMDb Ratings Analysis - Summary Statistics and Insights
# Rubric 4: Summary statistics and insights (4 marks)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset (assuming previous rubrics have been completed)
# If you don't have a cleaned dataset, uncomment the next line and load your raw data
# df = pd.read_csv('netflix_titles.csv')

# For demonstration, I'll create a sample structure - replace this with your actual data loading
# df = pd.read_csv('your_cleaned_netflix_data.csv')

# Sample data structure (replace with your actual data)
# This is just to show the structure - use your actual cleaned dataset
print("=== NETFLIX ORIGINALS IMDB RATINGS ANALYSIS ===")
print("=== SUMMARY STATISTICS AND INSIGHTS ===\n")

# Assuming your cleaned dataset has these columns:
# - title, genre, imdb_rating, release_year, runtime, language, etc.

def generate_summary_statistics(df):
    """
    Generate comprehensive summary statistics for Netflix Originals dataset
    """
    
    print("1. DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total number of Netflix Originals: {len(df)}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names and data types:")
    print(df.dtypes)
    print("\n")
    
    # Basic descriptive statistics for numerical columns
    print("2. NUMERICAL FEATURES SUMMARY")
    print("=" * 50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("Numerical columns found:", list(numerical_cols))
    
    if 'imdb_rating' in df.columns:
        print("\n📊 IMDb Rating Statistics:")
        print(f"   Mean Rating: {df['imdb_rating'].mean():.2f}")
        print(f"   Median Rating: {df['imdb_rating'].median():.2f}")
        print(f"   Standard Deviation: {df['imdb_rating'].std():.2f}")
        print(f"   Minimum Rating: {df['imdb_rating'].min():.2f}")
        print(f"   Maximum Rating: {df['imdb_rating'].max():.2f}")
        print(f"   25th Percentile: {df['imdb_rating'].quantile(0.25):.2f}")
        print(f"   75th Percentile: {df['imdb_rating'].quantile(0.75):.2f}")
        print(f"   Range: {df['imdb_rating'].max() - df['imdb_rating'].min():.2f}")
        print(f"   IQR: {df['imdb_rating'].quantile(0.75) - df['imdb_rating'].quantile(0.25):.2f}")
    
    if 'runtime' in df.columns:
        print(f"\n⏱️ Runtime Statistics (minutes):")
        print(f"   Average Runtime: {df['runtime'].mean():.1f} minutes")
        print(f"   Median Runtime: {df['runtime'].median():.1f} minutes")
        print(f"   Shortest Content: {df['runtime'].min():.0f} minutes")
        print(f"   Longest Content: {df['runtime'].max():.0f} minutes")
    
    if 'release_year' in df.columns:
        print(f"\n📅 Release Year Statistics:")
        print(f"   Earliest Release: {int(df['release_year'].min())}")
        print(f"   Latest Release: {int(df['release_year'].max())}")
        print(f"   Most Common Year: {int(df['release_year'].mode()[0])}")
        print(f"   Average Release Year: {df['release_year'].mean():.1f}")
    
    print("\n")
    
    # Categorical features summary
    print("3. CATEGORICAL FEATURES SUMMARY")
    print("=" * 50)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if 'genre' in df.columns:
        print("🎭 Genre Distribution:")
        genre_counts = df['genre'].value_counts().head(10)
        for genre, count in genre_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {genre}: {count} titles ({percentage:.1f}%)")
        print(f"   Total unique genres: {df['genre'].nunique()}")
    
    if 'language' in df.columns:
        print(f"\n🌍 Language Distribution:")
        lang_counts = df['language'].value_counts().head(5)
        for lang, count in lang_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {lang}: {count} titles ({percentage:.1f}%)")
        print(f"   Total languages: {df['language'].nunique()}")
    
    if 'type' in df.columns:
        print(f"\n🎬 Content Type Distribution:")
        type_counts = df['type'].value_counts()
        for content_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {content_type}: {count} titles ({percentage:.1f}%)")
    
    print("\n")
    
    # Advanced insights
    print("4. KEY INSIGHTS AND FINDINGS")
    print("=" * 50)
    
    # Rating quality insights
    if 'imdb_rating' in df.columns:
        high_rated = df[df['imdb_rating'] >= 8.0]
        low_rated = df[df['imdb_rating'] <= 5.0]
        
        print(f"🌟 Content Quality Insights:")
        print(f"   High-rated content (≥8.0): {len(high_rated)} titles ({len(high_rated)/len(df)*100:.1f}%)")
        print(f"   Low-rated content (≤5.0): {len(low_rated)} titles ({len(low_rated)/len(df)*100:.1f}%)")
        print(f"   Average rating trend: {'Above average' if df['imdb_rating'].mean() > 6.5 else 'Below average'}")
    
    # Release trends
    if 'release_year' in df.columns:
        recent_content = df[df['release_year'] >= 2020]
        print(f"\n📈 Release Trends:")
        print(f"   Recent content (2020+): {len(recent_content)} titles ({len(recent_content)/len(df)*100:.1f}%)")
        
        yearly_counts = df['release_year'].value_counts().sort_index()
        peak_year = yearly_counts.idxmax()
        print(f"   Peak production year: {int(peak_year)} ({yearly_counts[peak_year]} titles)")
    
    # Genre performance
    if 'genre' in df.columns and 'imdb_rating' in df.columns:
        print(f"\n🎭 Genre Performance:")
        genre_ratings = df.groupby('genre')['imdb_rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        genre_ratings = genre_ratings[genre_ratings['count'] >= 5]  # Only genres with 5+ titles
        
        if not genre_ratings.empty:
            best_genre = genre_ratings.index[0]
            worst_genre = genre_ratings.index[-1]
            print(f"   Best performing genre: {best_genre} (avg: {genre_ratings.loc[best_genre, 'mean']:.2f})")
            print(f"   Lowest performing genre: {worst_genre} (avg: {genre_ratings.loc[worst_genre, 'mean']:.2f})")
    
    print("\n")
    
    # Data quality summary
    print("5. DATA QUALITY SUMMARY")
    print("=" * 50)
    print("Missing values per column:")
    missing_data = df.isnull().sum()
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count} ({percentage:.1f}%)")
        else:
            print(f"   {col}: 0 (0.0%)")
    
    total_missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {total_missing}")
    print(f"Data completeness: {((df.size - total_missing) / df.size * 100):.1f}%")
    
    return df

def create_summary_visualizations(df):
    """
    Create visualizations to support summary statistics
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Netflix Originals - Summary Statistics Visualizations', fontsize=16, fontweight='bold')
    
    # IMDb Rating Distribution
    if 'imdb_rating' in df.columns:
        axes[0, 0].hist(df['imdb_rating'], bins=20, color='red', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['imdb_rating'].mean(), color='blue', linestyle='--', 
                          label=f'Mean: {df["imdb_rating"].mean():.2f}')
        axes[0, 0].axvline(df['imdb_rating'].median(), color='orange', linestyle='--', 
                          label=f'Median: {df["imdb_rating"].median():.2f}')
        axes[0, 0].set_title('IMDb Rating Distribution')
        axes[0, 0].set_xlabel('IMDb Rating')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Release Year Trend
    if 'release_year' in df.columns:
        year_counts = df['release_year'].value_counts().sort_index()
        axes[0, 1].plot(year_counts.index, year_counts.values, marker='o', color='green', linewidth=2)
        axes[0, 1].set_title('Netflix Originals Release Trend')
        axes[0, 1].set_xlabel('Release Year')
        axes[0, 1].set_ylabel('Number of Titles')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Top Genres
    if 'genre' in df.columns:
        top_genres = df['genre'].value_counts().head(8)
        axes[1, 0].barh(range(len(top_genres)), top_genres.values, color='purple', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_genres)))
        axes[1, 0].set_yticklabels(top_genres.index)
        axes[1, 0].set_title('Top 8 Genres by Count')
        axes[1, 0].set_xlabel('Number of Titles')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Rating vs Release Year
    if 'imdb_rating' in df.columns and 'release_year' in df.columns:
        axes[1, 1].scatter(df['release_year'], df['imdb_rating'], alpha=0.6, color='orange')
        axes[1, 1].set_title('IMDb Rating vs Release Year')
        axes[1, 1].set_xlabel('Release Year')
        axes[1, 1].set_ylabel('IMDb Rating')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_statistical_report(df):
    """
    Generate a comprehensive statistical report
    """
    print("6. DETAILED STATISTICAL REPORT")
    print("=" * 50)
    
    # Correlation analysis for numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        print("Correlation Matrix (numerical variables):")
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix.round(3))
        print("\n")
    
    # Quartile analysis for IMDb ratings
    if 'imdb_rating' in df.columns:
        print("IMDb Rating Quartile Analysis:")
        quartiles = df['imdb_rating'].quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = quartiles[0.25], quartiles[0.5], quartiles[0.75]
        
        print(f"Q1 (25th percentile): {q1:.2f}")
        print(f"Q2 (50th percentile/Median): {q2:.2f}")
        print(f"Q3 (75th percentile): {q3:.2f}")
        
        # Classify content by rating quartiles
        excellent = df[df['imdb_rating'] >= q3]
        good = df[(df['imdb_rating'] >= q2) & (df['imdb_rating'] < q3)]
        average = df[(df['imdb_rating'] >= q1) & (df['imdb_rating'] < q2)]
        below_average = df[df['imdb_rating'] < q1]
        
        print(f"\nContent Classification by Rating:")
        print(f"Excellent (Q4): {len(excellent)} titles ({len(excellent)/len(df)*100:.1f}%)")
        print(f"Good (Q3): {len(good)} titles ({len(good)/len(df)*100:.1f}%)")
        print(f"Average (Q2): {len(average)} titles ({len(average)/len(df)*100:.1f}%)")
        print(f"Below Average (Q1): {len(below_average)} titles ({len(below_average)/len(df)*100:.1f}%)")

# Main execution function
def main():
    """
    Main function to run all summary statistics and insights
    """
    try:
        # Load your cleaned dataset here
        # df = pd.read_csv('your_cleaned_netflix_data.csv')
        
        # For demonstration purposes, create a sample dataset
        # Replace this section with your actual data loading
        print("⚠️  Please replace this sample data with your actual cleaned Netflix dataset")
        print("   Use: df = pd.read_csv('your_cleaned_netflix_data.csv')\n")
        
        # Sample data creation (replace with your actual data)
        np.random.seed(42)
        sample_data = {
            'title': [f'Netflix Original {i}' for i in range(100)],
            'genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary', 'Thriller'], 100),
            'imdb_rating': np.random.normal(6.5, 1.2, 100).clip(1, 10),
            'release_year': np.random.randint(2015, 2024, 100),
            'runtime': np.random.randint(80, 180, 100),
            'language': np.random.choice(['English', 'Spanish', 'French', 'Korean'], 100),
            'type': np.random.choice(['Movie', 'TV Series'], 100)
        }
        df = pd.DataFrame(sample_data)
        
        # Run summary statistics analysis
        df = generate_summary_statistics(df)
        
        # Create visualizations
        create_summary_visualizations(df)
        
        # Generate detailed report
        generate_statistical_report(df)
        
        print("\n" + "="*60)
        print("✅ SUMMARY STATISTICS AND INSIGHTS ANALYSIS COMPLETE")
        print("="*60)
        
    except FileNotFoundError:
        print("❌ Error: Dataset file not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()