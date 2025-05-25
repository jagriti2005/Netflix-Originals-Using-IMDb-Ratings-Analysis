# Netflix Originals Data Analysis - Part 2: Feature Selection and Engineering
# Run this in VS Code: python netflix_analysis_part2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

print("=== NETFLIX ORIGINALS DATA ANALYSIS ===")
print("Part 2: Feature Selection and Engineering")
print("=" * 50)

# Load cleaned data from Part 1
try:
    df_cleaned = pd.read_csv('netflix_originals_cleaned.csv')
    df_cleaned['Release_Date'] = pd.to_datetime(df_cleaned['Release_Date'])
    print("✅ Cleaned data loaded successfully")
except FileNotFoundError:
    print("❌ Please run Part 1 first to generate cleaned data")
    exit()

print(f"Dataset shape: {df_cleaned.shape}")
print("\nInitial features:")
print(df_cleaned.columns.tolist())

# 1. FEATURE ENGINEERING
print("\n1. FEATURE ENGINEERING")
print("-" * 30)

# Create a copy for feature engineering
df_features = df_cleaned.copy()

# A. Temporal Features
print("Creating temporal features...")

# Extract date components
df_features['Release_Year'] = df_features['Release_Date'].dt.year
df_features['Release_Month'] = df_features['Release_Date'].dt.month
df_features['Release_Quarter'] = df_features['Release_Date'].dt.quarter
df_features['Release_Day_of_Week'] = df_features['Release_Date'].dt.dayofweek

# Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_features['Release_Season'] = df_features['Release_Month'].apply(get_season)

# Age of content (days since release)
current_date = datetime.now()
df_features['Content_Age_Days'] = (current_date - df_features['Release_Date']).dt.days

# Netflix era categorization
def netflix_era(year):
    if year <= 2016:
        return 'Early_Era'
    elif year <= 2019:
        return 'Growth_Era'
    else:
        return 'Mature_Era'

df_features['Netflix_Era'] = df_features['Release_Year'].apply(netflix_era)

print("✅ Temporal features created")

# B. Content-Based Features
print("\nCreating content-based features...")

# Runtime categories
def runtime_category(runtime):
    if runtime < 60:
        return 'Short'
    elif runtime < 120:
        return 'Medium'
    elif runtime < 180:
        return 'Long'
    else:
        return 'Very_Long'

df_features['Runtime_Category'] = df_features['Runtime_Minutes'].apply(runtime_category)

# Rating categories
def rating_category(rating):
    if rating < 5.0:
        return 'Poor'
    elif rating < 6.5:
        return 'Below_Average'
    elif rating < 7.5:
        return 'Good'
    elif rating < 8.5:
        return 'Excellent'
    else:
        return 'Outstanding'

df_features['Rating_Category'] = df_features['IMDB_Rating'].apply(rating_category)

# Language popularity (based on count)
language_counts = df_features['Language'].value_counts()
def language_popularity(language):
    # Handle NaN/missing values
    if pd.isna(language) or language is None:
        return 'Low'  # Default for missing values
    
    if language not in language_counts:
        return 'Low'  # Default for unknown languages
    
    count = language_counts[language]
    if count >= 50:
        return 'High'
    elif count >= 20:
        return 'Medium'
    else:
        return 'Low'

df_features['Language_Popularity'] = df_features['Language'].apply(language_popularity)

print("✅ Content-based features created")

# C. Derived Numerical Features
print("\nCreating derived numerical features...")

# Rating deviation from genre average
genre_avg_rating = df_features.groupby('Genre')['IMDB_Rating'].transform('mean')
df_features['Rating_vs_Genre_Avg'] = df_features['IMDB_Rating'] - genre_avg_rating

# Runtime deviation from genre average
genre_avg_runtime = df_features.groupby('Genre')['Runtime_Minutes'].transform('mean')
df_features['Runtime_vs_Genre_Avg'] = df_features['Runtime_Minutes'] - genre_avg_runtime

# Popularity score (combining multiple factors)
# Normalize ratings to 0-1 scale
normalized_rating = (df_features['IMDB_Rating'] - df_features['IMDB_Rating'].min()) / (df_features['IMDB_Rating'].max() - df_features['IMDB_Rating'].min())
# Weight by recency (more recent content gets slight boost)
recency_weight = 1 - (df_features['Content_Age_Days'] / df_features['Content_Age_Days'].max()) * 0.2
df_features['Popularity_Score'] = normalized_rating * recency_weight

print("✅ Derived numerical features created")

# D. Interaction Features
print("\nCreating interaction features...")

# Genre-Language interaction
df_features['Genre_Language'] = df_features['Genre'] + '_' + df_features['Language']

# Era-Genre interaction
df_features['Era_Genre'] = df_features['Netflix_Era'] + '_' + df_features['Genre']

print("✅ Interaction features created")

# 2. FEATURE ENCODING
print("\n2. FEATURE ENCODING")
print("-" * 30)

# Create a copy for encoding
df_encoded = df_features.copy()

# Label encoding for ordinal features
ordinal_features = ['Rating_Category', 'Runtime_Category', 'Language_Popularity']
ordinal_mappings = {
    'Rating_Category': {'Poor': 0, 'Below_Average': 1, 'Good': 2, 'Excellent': 3, 'Outstanding': 4},
    'Runtime_Category': {'Short': 0, 'Medium': 1, 'Long': 2, 'Very_Long': 3},
    'Language_Popularity': {'Low': 0, 'Medium': 1, 'High': 2}
}

for feature in ordinal_features:
    df_encoded[f'{feature}_Encoded'] = df_encoded[feature].map(ordinal_mappings[feature])

# One-hot encoding for categorical features
categorical_features = ['Genre', 'Language', 'Release_Season', 'Netflix_Era']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_features, prefix=categorical_features)

print("✅ Feature encoding completed")

# 3. FEATURE SELECTION
print("\n3. FEATURE SELECTION")
print("-" * 30)

# Prepare features for selection
# Select only numeric features for correlation analysis
numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col != 'IMDB_Rating']  # Exclude target

print(f"Total numeric features available: {len(numeric_features)}")

# A. Correlation-based feature selection
print("\nA. Correlation Analysis:")
correlation_with_target = df_encoded[numeric_features + ['IMDB_Rating']].corr()['IMDB_Rating'].abs().sort_values(ascending=False)
print("Top 10 features correlated with IMDB Rating:")
print(correlation_with_target.head(11)[1:])  # Exclude self-correlation

# B. Statistical feature selection
print("\nB. Statistical Feature Selection (F-test):")
X = df_encoded[numeric_features].fillna(0)  # Fill any remaining NaN values
y = df_encoded['IMDB_Rating']

# Select top k features using F-test
selector_f = SelectKBest(score_func=f_regression, k=10)
X_selected_f = selector_f.fit_transform(X, y)
selected_features_f = [numeric_features[i] for i in selector_f.get_support(indices=True)]
feature_scores_f = selector_f.scores_

print("Top 10 features by F-test scores:")
f_scores_df = pd.DataFrame({
    'Feature': selected_features_f,
    'F_Score': [feature_scores_f[numeric_features.index(feat)] for feat in selected_features_f]
}).sort_values('F_Score', ascending=False)
print(f_scores_df)

# C. Mutual Information feature selection
print("\nC. Mutual Information Feature Selection:")
selector_mi = SelectKBest(score_func=mutual_info_regression, k=10)
X_selected_mi = selector_mi.fit_transform(X, y)
selected_features_mi = [numeric_features[i] for i in selector_mi.get_support(indices=True)]
feature_scores_mi = selector_mi.scores_

print("Top 10 features by Mutual Information:")
mi_scores_df = pd.DataFrame({
    'Feature': selected_features_mi,
    'MI_Score': [feature_scores_mi[numeric_features.index(feat)] for feat in selected_features_mi]
}).sort_values('MI_Score', ascending=False)
print(mi_scores_df)

# 4. FEATURE IMPORTANCE VISUALIZATION
print("\n4. FEATURE IMPORTANCE VISUALIZATION")
print("-" * 30)

plt.figure(figsize=(15, 12))

# Correlation heatmap
plt.subplot(2, 2, 1)
top_corr_features = correlation_with_target.head(11)[1:].index  # Top 10
corr_matrix = df_encoded[list(top_corr_features) + ['IMDB_Rating']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix - Top Features')
plt.xticks(rotation=45, ha='right')

# F-test scores
plt.subplot(2, 2, 2)
plt.barh(range(len(f_scores_df)), f_scores_df['F_Score'])
plt.yticks(range(len(f_scores_df)), f_scores_df['Feature'])
plt.xlabel('F-Score')
plt.title('Top Features by F-Test Score')
plt.gca().invert_yaxis()

# Mutual Information scores
plt.subplot(2, 2, 3)
plt.barh(range(len(mi_scores_df)), mi_scores_df['MI_Score'])
plt.yticks(range(len(mi_scores_df)), mi_scores_df['Feature'])
plt.xlabel('MI Score')
plt.title('Top Features by Mutual Information')
plt.gca().invert_yaxis()

# Feature correlation with target
plt.subplot(2, 2, 4)
top_10_corr = correlation_with_target.head(11)[1:]
plt.barh(range(len(top_10_corr)), top_10_corr.values)
plt.yticks(range(len(top_10_corr)), top_10_corr.index)
plt.xlabel('Absolute Correlation with IMDB Rating')
plt.title('Top 10 Features by Correlation')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. FINAL FEATURE SET
print("\n5. FINAL FEATURE SET SELECTION")
print("-" * 30)

# Combine different selection methods to create final feature set
correlation_top = set(correlation_with_target.head(8)[1:].index)  # Top 7 by correlation
f_test_top = set(selected_features_f[:7])  # Top 7 by F-test
mi_top = set(selected_features_mi[:7])  # Top 7 by MI

# Union of top features from all methods
final_features = list(correlation_top.union(f_test_top).union(mi_top))

# Add some important engineered features manually
important_engineered = ['Release_Year', 'Content_Age_Days', 'Popularity_Score', 
                       'Rating_vs_Genre_Avg', 'Runtime_vs_Genre_Avg']
final_features.extend([f for f in important_engineered if f in numeric_features])

# Remove duplicates
final_features = list(set(final_features))

print(f"Final feature set contains {len(final_features)} features:")
for i, feature in enumerate(sorted(final_features), 1):
    print(f"{i:2d}. {feature}")

# 6. FEATURE SUMMARY STATISTICS
print("\n6. FEATURE SUMMARY STATISTICS")
print("-" * 30)

feature_summary = pd.DataFrame({
    'Feature': final_features,
    'Data_Type': [df_encoded[feat].dtype for feat in final_features],
    'Non_Null_Count': [df_encoded[feat].count() for feat in final_features],
    'Unique_Values': [df_encoded[feat].nunique() for feat in final_features],
    'Mean': [df_encoded[feat].mean() if df_encoded[feat].dtype in ['int64', 'float64'] else 'N/A' for feat in final_features],
    'Std': [df_encoded[feat].std() if df_encoded[feat].dtype in ['int64', 'float64'] else 'N/A' for feat in final_features]
})

print("Feature Summary Statistics:")
print(feature_summary.to_string(index=False))

# 7. SAVE ENGINEERED DATASET
print("\n7. SAVING ENGINEERED DATASET")
print("-" * 30)

# Save the full engineered dataset
df_encoded.to_csv('netflix_originals_engineered.csv', index=False)
print("✅ Full engineered dataset saved to 'netflix_originals_engineered.csv'")

# Save dataset with selected features only
df_final = df_encoded[['Title', 'IMDB_Rating'] + final_features].copy()
df_final.to_csv('netflix_originals_final_features.csv', index=False)
print("✅ Final feature set saved to 'netflix_originals_final_features.csv'")

# 8. FEATURE ENGINEERING SUMMARY
print("\n8. FEATURE ENGINEERING SUMMARY")
print("-" * 30)

print("Original features: 6")
print(f"Engineered features: {len(df_encoded.columns) - 6}")
print(f"Final selected features: {len(final_features)}")
print(f"Total dataset size: {df_encoded.shape}")

print("\nFeature categories created:")
print("✓ Temporal features (Release_Year, Season, Era, etc.)")
print("✓ Content-based features (Runtime_Category, Rating_Category)")
print("✓ Derived numerical features (Rating vs Genre Average, Popularity Score)")
print("✓ Interaction features (Genre_Language, Era_Genre)")
print("✓ Encoded categorical features (One-hot encoded)")

print("\n🎉 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 50)