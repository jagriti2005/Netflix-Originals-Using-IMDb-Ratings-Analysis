# Netflix Originals IMDb Ratings Analysis - Patterns, Trends, and Anomalies
# Rubric 5: Identifying patterns, trends, and anomalies (5 marks)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_temporal_trends(df):
    """
    Analyze temporal patterns and trends in Netflix Originals
    """
    print("🔍 TEMPORAL TRENDS ANALYSIS")
    print("=" * 60)
    
    if 'release_year' in df.columns and 'imdb_rating' in df.columns:
        # Yearly release trends
        yearly_stats = df.groupby('release_year').agg({
            'imdb_rating': ['count', 'mean', 'median', 'std'],
            'title': 'count'
        }).round(2)
        
        yearly_stats.columns = ['Title_Count', 'Avg_Rating', 'Median_Rating', 'Rating_StdDev', 'Total_Titles']
        yearly_stats = yearly_stats.drop('Total_Titles', axis=1)
        
        print("📊 Year-over-Year Statistics:")
        print(yearly_stats.tail(10))
        
        # Identify trending years
        yearly_growth = yearly_stats['Title_Count'].pct_change() * 100
        peak_growth_year = yearly_growth.idxmax()
        peak_decline_year = yearly_growth.idxmin()
        
        print(f"\n📈 Growth Patterns:")
        print(f"   Peak growth year: {peak_growth_year} ({yearly_growth[peak_growth_year]:.1f}% increase)")
        print(f"   Biggest decline year: {peak_decline_year} ({yearly_growth[peak_decline_year]:.1f}% decrease)")
        
        # Quality trends over time
        correlation_year_rating = df['release_year'].corr(df['imdb_rating'])
        print(f"\n⭐ Quality Trend Analysis:")
        print(f"   Correlation between release year and rating: {correlation_year_rating:.3f}")
        
        if correlation_year_rating > 0.1:
            print("   📈 Quality is generally improving over time")
        elif correlation_year_rating < -0.1:
            print("   📉 Quality is generally declining over time")
        else:
            print("   ➡️  Quality remains relatively stable over time")
        
        # Recent vs historical performance
        recent_content = df[df['release_year'] >= 2020]
        historical_content = df[df['release_year'] < 2020]
        
        if len(recent_content) > 0 and len(historical_content) > 0:
            recent_avg = recent_content['imdb_rating'].mean()
            historical_avg = historical_content['imdb_rating'].mean()
            
            print(f"\n🆚 Recent vs Historical Comparison:")
            print(f"   Recent content (2020+) avg rating: {recent_avg:.2f}")
            print(f"   Historical content (<2020) avg rating: {historical_avg:.2f}")
            print(f"   Difference: {recent_avg - historical_avg:.2f}")
    
    print("\n")

def analyze_genre_patterns(df):
    """
    Analyze genre-based patterns and performance
    """
    print("🎭 GENRE PATTERNS ANALYSIS")
    print("=" * 60)
    
    if 'genre' in df.columns and 'imdb_rating' in df.columns:
        # Genre performance analysis
        genre_stats = df.groupby('genre').agg({
            'imdb_rating': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'title': 'count'
        }).round(2)
        
        genre_stats.columns = ['Count', 'Mean_Rating', 'Median_Rating', 'Std_Dev', 'Min_Rating', 'Max_Rating', 'Total']
        genre_stats = genre_stats.drop('Total', axis=1)
        genre_stats = genre_stats[genre_stats['Count'] >= 3]  # Only genres with 3+ titles
        
        # Sort by mean rating
        genre_stats_sorted = genre_stats.sort_values('Mean_Rating', ascending=False)
        
        print("🏆 Genre Performance Ranking:")
        print(genre_stats_sorted)
        
        # Identify genre patterns
        best_genre = genre_stats_sorted.index[0]
        worst_genre = genre_stats_sorted.index[-1]
        most_consistent = genre_stats_sorted.loc[genre_stats_sorted['Std_Dev'].idxmin()]
        most_variable = genre_stats_sorted.loc[genre_stats_sorted['Std_Dev'].idxmax()]
        
        print(f"\n🎯 Genre Insights:")
        print(f"   Best performing genre: {best_genre} (avg: {genre_stats_sorted.loc[best_genre, 'Mean_Rating']:.2f})")
        print(f"   Worst performing genre: {worst_genre} (avg: {genre_stats_sorted.loc[worst_genre, 'Mean_Rating']:.2f})")
        print(f"   Most consistent quality: {most_consistent.name} (std: {most_consistent['Std_Dev']:.2f})")
        print(f"   Most variable quality: {most_variable.name} (std: {most_variable['Std_Dev']:.2f})")
        
        # Genre evolution over time
        if 'release_year' in df.columns:
            print(f"\n📅 Genre Evolution Analysis:")
            recent_years = df[df['release_year'] >= 2020]
            if len(recent_years) > 0:
                recent_genre_dist = recent_years['genre'].value_counts(normalize=True) * 100
                overall_genre_dist = df['genre'].value_counts(normalize=True) * 100
                
                print("   Recent genre focus vs overall distribution:")
                for genre in recent_genre_dist.head(5).index:
                    recent_pct = recent_genre_dist[genre]
                    overall_pct = overall_genre_dist.get(genre, 0)
                    change = recent_pct - overall_pct
                    print(f"   {genre}: {recent_pct:.1f}% recent vs {overall_pct:.1f}% overall ({change:+.1f}%)")
    
    print("\n")

def detect_rating_anomalies(df):
    """
    Detect anomalies in IMDb ratings using statistical methods
    """
    print("🚨 ANOMALY DETECTION ANALYSIS")
    print("=" * 60)
    
    if 'imdb_rating' in df.columns:
        ratings = df['imdb_rating']
        
        # Method 1: Z-Score based anomalies
        z_scores = np.abs(stats.zscore(ratings))
        z_threshold = 3
        z_anomalies = df[z_scores > z_threshold]
        
        # Method 2: IQR based anomalies
        Q1 = ratings.quantile(0.25)
        Q3 = ratings.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = df[(ratings < lower_bound) | (ratings > upper_bound)]
        
        # Method 3: Modified Z-Score (using median)
        median = ratings.median()
        mad = np.median(np.abs(ratings - median))
        modified_z_scores = 0.6745 * (ratings - median) / mad
        modified_z_anomalies = df[np.abs(modified_z_scores) > 3.5]
        
        print(f"📊 Anomaly Detection Results:")
        print(f"   Z-Score method (|z| > 3): {len(z_anomalies)} anomalies")
        print(f"   IQR method: {len(iqr_anomalies)} anomalies")
        print(f"   Modified Z-Score method: {len(modified_z_anomalies)} anomalies")
        
        # Display extreme cases
        if len(iqr_anomalies) > 0:
            print(f"\n🔍 Detected Anomalies (IQR method):")
            anomaly_details = iqr_anomalies[['title', 'imdb_rating', 'genre', 'release_year']].sort_values('imdb_rating')
            
            print("   Extremely Low Rated:")
            low_rated = anomaly_details[anomaly_details['imdb_rating'] < lower_bound]
            if len(low_rated) > 0:
                for _, row in low_rated.head(3).iterrows():
                    print(f"   • {row['title']}: {row['imdb_rating']:.1f} ({row['genre']}, {row['release_year']})")
            
            print("   Extremely High Rated:")
            high_rated = anomaly_details[anomaly_details['imdb_rating'] > upper_bound]
            if len(high_rated) > 0:
                for _, row in high_rated.tail(3).iterrows():
                    print(f"   • {row['title']}: {row['imdb_rating']:.1f} ({row['genre']}, {row['release_year']})")
        
        # Statistical summary of anomalies
        print(f"\n📈 Rating Distribution Analysis:")
        print(f"   Normal range (IQR): {lower_bound:.2f} - {upper_bound:.2f}")
        print(f"   Mean rating: {ratings.mean():.2f}")
        print(f"   Median rating: {ratings.median():.2f}")
        print(f"   Skewness: {ratings.skew():.3f}")
        print(f"   Kurtosis: {ratings.kurtosis():.3f}")
        
        return iqr_anomalies
    
    print("\n")
    return pd.DataFrame()

def analyze_clustering_patterns(df):
    """
    Use clustering to identify hidden patterns in the data
    """
    print("🔬 CLUSTERING PATTERN ANALYSIS")
    print("=" * 60)
    
    # Prepare numerical features for clustering
    numerical_features = []
    if 'imdb_rating' in df.columns:
        numerical_features.append('imdb_rating')
    if 'release_year' in df.columns:
        numerical_features.append('release_year')
    if 'runtime' in df.columns:
        numerical_features.append('runtime')
    
    if len(numerical_features) >= 2:
        # Prepare data for clustering
        cluster_data = df[numerical_features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 8)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Use 4 clusters for analysis
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to dataframe
        df_clustered = cluster_data.copy()
        df_clustered['Cluster'] = cluster_labels
        
        # Analyze clusters
        print(f"🎯 Identified {optimal_k} distinct content clusters:")
        
        cluster_analysis = df_clustered.groupby('Cluster').agg({
            col: ['mean', 'count'] for col in numerical_features
        }).round(2)
        
        for cluster_id in range(optimal_k):
            cluster_data_subset = df_clustered[df_clustered['Cluster'] == cluster_id]
            print(f"\n   Cluster {cluster_id + 1} ({len(cluster_data_subset)} titles):")
            
            for feature in numerical_features:
                mean_val = cluster_data_subset[feature].mean()
                print(f"     • Avg {feature}: {mean_val:.2f}")
            
            # Characterize cluster
            if 'imdb_rating' in numerical_features:
                avg_rating = cluster_data_subset['imdb_rating'].mean()
                if avg_rating >= 7.5:
                    print("     🌟 Cluster Type: HIGH QUALITY content")
                elif avg_rating <= 5.5:
                    print("     ⚠️  Cluster Type: LOW QUALITY content")
                else:
                    print("     📊 Cluster Type: AVERAGE QUALITY content")
        
        return df_clustered
    
    print("   ⚠️  Insufficient numerical features for clustering analysis")
    print("\n")
    return df

def analyze_seasonal_patterns(df):
    """
    Analyze seasonal and monthly release patterns
    """
    print("🗓️ SEASONAL PATTERNS ANALYSIS")
    print("=" * 60)
    
    if 'release_date' in df.columns:
        # Convert to datetime if not already
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_month'] = df['release_date'].dt.month
        df['release_quarter'] = df['release_date'].dt.quarter
        
        # Monthly release patterns
        monthly_releases = df['release_month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        print("📅 Monthly Release Patterns:")
        for month, count in monthly_releases.items():
            print(f"   {month_names[month-1]}: {count} releases")
        
        # Quarterly analysis
        quarterly_releases = df['release_quarter'].value_counts().sort_index()
        print(f"\n📊 Quarterly Release Distribution:")
        for quarter, count in quarterly_releases.items():
            print(f"   Q{quarter}: {count} releases ({count/len(df)*100:.1f}%)")
        
        # Identify peak seasons
        peak_month = monthly_releases.idxmax()
        peak_quarter = quarterly_releases.idxmax()
        print(f"\n🏆 Peak Release Periods:")
        print(f"   Peak month: {month_names[peak_month-1]} ({monthly_releases[peak_month]} releases)")
        print(f"   Peak quarter: Q{peak_quarter} ({quarterly_releases[peak_quarter]} releases)")
    
    elif 'release_year' in df.columns:
        print("   ⚠️  Detailed release dates not available")
        print("   📊 Using year-based analysis instead")
        
        # Analyze release frequency by year
        yearly_releases = df['release_year'].value_counts().sort_index()
        recent_trend = yearly_releases.tail(5)
        
        print(f"\n📈 Recent Year Trend (last 5 years):")
        for year, count in recent_trend.items():
            print(f"   {year}: {count} releases")
    
    print("\n")

def create_pattern_visualizations(df, anomalies_df=None):
    """
    Create comprehensive visualizations for patterns and trends
    """
    print("📊 GENERATING PATTERN VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Netflix Originals - Patterns, Trends & Anomalies Analysis', fontsize=16, fontweight='bold')
    
    # 1. Temporal trend analysis
    if 'release_year' in df.columns and 'imdb_rating' in df.columns:
        yearly_avg = df.groupby('release_year')['imdb_rating'].mean()
        yearly_count = df.groupby('release_year').size()
        
        # Plot 1: Rating trend over years
        ax1 = axes[0, 0]
        ax1.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6, color='red')
        ax1.set_title('Average IMDb Rating Trend Over Time', fontweight='bold')
        ax1.set_xlabel('Release Year')
        ax1.set_ylabel('Average IMDb Rating')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(yearly_avg.index, yearly_avg.values, 1)
        p = np.poly1d(z)
        ax1.plot(yearly_avg.index, p(yearly_avg.index), "--", color='blue', alpha=0.7, label=f'Trend: {z[0]:.3f}x')
        ax1.legend()
        
        # Plot 2: Volume trend over years
        ax2 = axes[0, 1]
        bars = ax2.bar(yearly_count.index, yearly_count.values, color='green', alpha=0.7)
        ax2.set_title('Number of Releases per Year', fontweight='bold')
        ax2.set_xlabel('Release Year')
        ax2.set_ylabel('Number of Titles')
        ax2.grid(True, alpha=0.3)
        
        # Highlight peak years
        max_year = yearly_count.idxmax()
        max_idx = list(yearly_count.index).index(max_year)
        bars[max_idx].set_color('orange')
    
    # 2. Genre performance analysis
    if 'genre' in df.columns and 'imdb_rating' in df.columns:
        genre_stats = df.groupby('genre')['imdb_rating'].agg(['mean', 'count']).reset_index()
        genre_stats = genre_stats[genre_stats['count'] >= 3].sort_values('mean', ascending=True)
        
        # Plot 3: Genre performance
        ax3 = axes[1, 0]
        bars = ax3.barh(genre_stats['genre'], genre_stats['mean'], color='purple', alpha=0.7)
        ax3.set_title('Average IMDb Rating by Genre', fontweight='bold')
        ax3.set_xlabel('Average IMDb Rating')
        ax3.grid(True, alpha=0.3)
        
        # Color code best and worst
        if len(bars) > 0:
            bars[-1].set_color('gold')  # Best
            bars[0].set_color('red')    # Worst
    
    # 3. Rating distribution with anomalies
    ax4 = axes[1, 1]
    ax4.hist(df['imdb_rating'], bins=25, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.set_title('IMDb Rating Distribution', fontweight='bold')
    ax4.set_xlabel('IMDb Rating')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Highlight anomalies if available
    if anomalies_df is not None and len(anomalies_df) > 0:
        ax4.axvline(anomalies_df['imdb_rating'].min(), color='red', linestyle='--', alpha=0.7, label='Anomalies')
        ax4.axvline(anomalies_df['imdb_rating'].max(), color='red', linestyle='--', alpha=0.7)
        ax4.legend()
    
    # 4. Correlation heatmap
    ax5 = axes[2, 0]
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax5.set_title('Feature Correlation Matrix', fontweight='bold')
    
    # 5. Box plot for genre ratings
    if 'genre' in df.columns and 'imdb_rating' in df.columns:
        ax6 = axes[2, 1]
        top_genres = df['genre'].value_counts().head(6).index
        genre_data = [df[df['genre'] == genre]['imdb_rating'].values for genre in top_genres]
        
        box_plot = ax6.boxplot(genre_data, labels=top_genres, patch_artist=True)
        ax6.set_title('Rating Distribution by Top Genres', fontweight='bold')
        ax6.set_xlabel('Genre')
        ax6.set_ylabel('IMDb Rating')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
    
    plt.tight_layout()
    plt.show()
    print("✅ Visualizations generated successfully!\n")

def generate_insights_summary(df):
    """
    Generate comprehensive insights summary
    """
    print("💡 KEY INSIGHTS SUMMARY")
    print("=" * 60)
    
    insights = []
    
    # Temporal insights
    if 'release_year' in df.columns and 'imdb_rating' in df.columns:
        year_rating_corr = df['release_year'].corr(df['imdb_rating'])
        if abs(year_rating_corr) > 0.1:
            trend_direction = "improving" if year_rating_corr > 0 else "declining"
            insights.append(f"📈 Netflix content quality is {trend_direction} over time (correlation: {year_rating_corr:.3f})")
        
        recent_years = df[df['release_year'] >= 2020]
        if len(recent_years) > 0:
            recent_avg = recent_years['imdb_rating'].mean()
            overall_avg = df['imdb_rating'].mean()
            if recent_avg > overall_avg + 0.2:
                insights.append(f"🌟 Recent content (2020+) shows higher quality: {recent_avg:.2f} vs {overall_avg:.2f}")
            elif recent_avg < overall_avg - 0.2:
                insights.append(f"⚠️  Recent content (2020+) shows lower quality: {recent_avg:.2f} vs {overall_avg:.2f}")
    
    # Genre insights
    if 'genre' in df.columns and 'imdb_rating' in df.columns:
        genre_performance = df.groupby('genre')['imdb_rating'].agg(['mean', 'count'])
        genre_performance = genre_performance[genre_performance['count'] >= 3]
        
        if len(genre_performance) > 0:
            best_genre = genre_performance['mean'].idxmax()
            worst_genre = genre_performance['mean'].idxmin()
            best_rating = genre_performance.loc[best_genre, 'mean']
            worst_rating = genre_performance.loc[worst_genre, 'mean']
            
            insights.append(f"🏆 Best performing genre: {best_genre} (avg: {best_rating:.2f})")
            insights.append(f"🔻 Lowest performing genre: {worst_genre} (avg: {worst_rating:.2f})")
            
            rating_range = best_rating - worst_rating
            if rating_range > 1.5:
                insights.append(f"📊 Significant quality variance across genres (range: {rating_range:.2f})")
    
    # Production volume insights
    if 'release_year' in df.columns:
        yearly_counts = df['release_year'].value_counts()
        peak_year = yearly_counts.idxmax()
        peak_count = yearly_counts[peak_year]
        
        insights.append(f"📅 Peak production year: {peak_year} with {peak_count} releases")
        
        recent_growth = yearly_counts[yearly_counts.index >= 2020].sum()
        total_content = len(df)
        recent_percentage = (recent_growth / total_content) * 100
        
        if recent_percentage > 40:
            insights.append(f"🚀 Heavy focus on recent content: {recent_percentage:.1f}% released since 2020")
    
    # Quality distribution insights
    if 'imdb_rating' in df.columns:
        high_quality = len(df[df['imdb_rating'] >= 8.0])
        low_quality = len(df[df['imdb_rating'] <= 5.0])
        total = len(df)
        
        if high_quality / total > 0.15:
            insights.append(f"⭐ Strong quality portfolio: {high_quality/total*100:.1f}% rated 8.0+")
        
        if low_quality / total > 0.15:
            insights.append(f"⚠️  Quality concerns: {low_quality/total*100:.1f}% rated 5.0 or below")
        
        rating_std = df['imdb_rating'].std()
        if rating_std > 1.5:
            insights.append(f"📈 High rating variability (std: {rating_std:.2f}) - inconsistent quality")
    
    # Display insights
    print("🔍 Discovered Patterns & Trends:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    if not insights:
        print("   📊 Basic statistical patterns identified - consider deeper analysis")
    
    print("\n")
    return insights

# Main execution function
def main():
    """
    Main function to run comprehensive pattern and trend analysis
    """
    try:
        print("🔍 NETFLIX ORIGINALS - PATTERNS, TRENDS & ANOMALIES ANALYSIS")
        print("=" * 80)
        print("This analysis will identify hidden patterns, trends, and anomalies in your Netflix dataset\n")
        
        # Load your cleaned dataset here
        # df = pd.read_csv('your_cleaned_netflix_data.csv')
        
        # For demonstration purposes, create a more realistic sample dataset
        print("⚠️  Please replace this sample data with your actual cleaned Netflix dataset")
        print("   Use: df = pd.read_csv('your_cleaned_netflix_data.csv')\n")
        
        # Enhanced sample data creation (replace with your actual data)
        np.random.seed(42)
        n_samples = 200
        
        # Create more realistic data with patterns
        years = np.random.choice(range(2015, 2024), n_samples)
        genres = np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary', 'Thriller', 'Horror', 'Romance'], n_samples)
        
        # Create rating patterns based on genre and year
        base_ratings = np.random.normal(6.5, 1.0, n_samples)
        
        # Genre-based adjustments
        genre_adjustments = {
            'Documentary': 0.8, 'Drama': 0.4, 'Thriller': 0.2,
            'Comedy': 0.0, 'Romance': -0.2, 'Action': -0.4, 'Horror': -0.6
        }
        
        # Year-based trend (slight improvement over time)
        year_factor = (years - 2015) * 0.05
        
        ratings = base_ratings + [genre_adjustments.get(g, 0) for g in genres] + year_factor
        ratings = np.clip(ratings, 1.0, 10.0)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, 10, replace=False)
        ratings[anomaly_indices[:5]] = np.random.uniform(1.0, 2.5, 5)  # Very low
        ratings[anomaly_indices[5:]] = np.random.uniform(9.0, 10.0, 5)  # Very high
        
        sample_data = {
            'title': [f'Netflix Original {i}' for i in range(n_samples)],
            'genre': genres,
            'imdb_rating': ratings,
            'release_year': years,
            'runtime': np.random.randint(80, 180, n_samples),
            'language': np.random.choice(['English', 'Spanish', 'French', 'Korean', 'German'], n_samples),
            'type': np.random.choice(['Movie', 'TV Series'], n_samples)
        }
        df = pd.DataFrame(sample_data)
        
        print(f"📊 Dataset loaded: {len(df)} Netflix Originals\n")
        
        # Run comprehensive analysis
        analyze_temporal_trends(df)
        analyze_genre_patterns(df)
        anomalies = detect_rating_anomalies(df)
        df_clustered = analyze_clustering_patterns(df)
        analyze_seasonal_patterns(df)
        
        # Generate visualizations
        create_pattern_visualizations(df, anomalies)
        
        # Generate insights summary
        insights = generate_insights_summary(df)
        
        print("=" * 80)
        print("✅ COMPREHENSIVE PATTERN ANALYSIS COMPLETE")
        print("=" * 80)
        print("📋 Analysis included:")
        print("   • Temporal trends and quality evolution")
        print("   • Genre performance patterns")
        print("   • Statistical anomaly detection")
        print("   • Hidden clustering patterns")
        print("   • Seasonal release patterns")
        print("   • Comprehensive visualizations")
        print("   • Actionable insights summary")
        
    except FileNotFoundError:
        print("❌ Error: Dataset file not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()