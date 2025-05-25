# Data Visualization Project - Rubric 6: Handling Outliers and Data Transformations
# Complete implementation with multiple methods and visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class OutlierHandler:
    """
    Comprehensive outlier detection and handling class
    """
    def __init__(self, data):
        self.data = data.copy()
        self.outliers_info = {}
        
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """Detect outliers using Interquartile Range method"""
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = self.data[(self.data[column] < lower_bound) | 
                            (self.data[column] > upper_bound)]
        
        self.outliers_info[f'{column}_IQR'] = {
            'method': 'IQR',
            'outliers': outliers,
            'count': len(outliers),
            'percentage': (len(outliers) / len(self.data)) * 100,
            'bounds': (lower_bound, upper_bound)
        }
        
        return outliers
    
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(self.data[column].dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        outliers = self.data.iloc[outlier_indices]
        
        self.outliers_info[f'{column}_zscore'] = {
            'method': 'Z-score',
            'outliers': outliers,
            'count': len(outliers),
            'percentage': (len(outliers) / len(self.data)) * 100,
            'threshold': threshold
        }
        
        return outliers
    
    def detect_outliers_isolation_forest(self, columns, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.data[columns].dropna())
        outliers = self.data[outlier_labels == -1]
        
        self.outliers_info[f'isolation_forest'] = {
            'method': 'Isolation Forest',
            'outliers': outliers,
            'count': len(outliers),
            'percentage': (len(outliers) / len(self.data)) * 100,
            'contamination': contamination
        }
        
        return outliers
    
    def visualize_outliers(self, column):
        """Create comprehensive outlier visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Outlier Analysis for {column}', fontsize=16, fontweight='bold')
        
        # Box plot
        axes[0, 0].boxplot(self.data[column].dropna())
        axes[0, 0].set_title('Box Plot')
        axes[0, 0].set_ylabel(column)
        
        # Histogram with normal curve
        axes[0, 1].hist(self.data[column].dropna(), bins=30, density=True, alpha=0.7)
        mu, sigma = stats.norm.fit(self.data[column].dropna())
        x = np.linspace(self.data[column].min(), self.data[column].max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        axes[0, 1].set_title('Distribution with Normal Curve')
        axes[0, 1].legend()
        
        # Q-Q plot
        stats.probplot(self.data[column].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Scatter plot with outliers highlighted
        if f'{column}_IQR' in self.outliers_info:
            outliers = self.outliers_info[f'{column}_IQR']['outliers']
            axes[1, 1].scatter(range(len(self.data)), self.data[column], alpha=0.6, label='Normal')
            axes[1, 1].scatter(outliers.index, outliers[column], color='red', 
                             alpha=0.8, label='Outliers (IQR)')
            axes[1, 1].set_title('Data Points with Outliers')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def handle_outliers(self, column, method='cap', iqr_multiplier=1.5):
        """Handle outliers using specified method"""
        if method == 'remove':
            # Remove outliers
            outliers = self.detect_outliers_iqr(column, iqr_multiplier)
            cleaned_data = self.data.drop(outliers.index)
            return cleaned_data
        
        elif method == 'cap':
            # Cap outliers to bounds
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            capped_data = self.data.copy()
            capped_data[column] = np.clip(capped_data[column], lower_bound, upper_bound)
            return capped_data
        
        elif method == 'transform':
            # Log transformation
            transformed_data = self.data.copy()
            if (transformed_data[column] > 0).all():
                transformed_data[f'{column}_log'] = np.log(transformed_data[column])
            return transformed_data
    
    def print_outlier_summary(self):
        """Print comprehensive outlier summary"""
        print("="*60)
        print("OUTLIER DETECTION SUMMARY")
        print("="*60)
        
        for key, info in self.outliers_info.items():
            print(f"\nMethod: {info['method']}")
            print(f"Column: {key}")
            print(f"Outliers found: {info['count']} ({info['percentage']:.2f}%)")
            
            if 'bounds' in info:
                print(f"Bounds: {info['bounds'][0]:.2f} to {info['bounds'][1]:.2f}")
            
            print("-" * 40)

class DataTransformer:
    """
    Comprehensive data transformation class
    """
    def __init__(self, data):
        self.data = data.copy()
        self.transformations = {}
    
    def normalize_data(self, columns, method='standard'):
        """Normalize data using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        
        transformed_data = self.data.copy()
        transformed_data[columns] = scaler.fit_transform(self.data[columns])
        
        self.transformations[f'{method}_scaling'] = {
            'method': method,
            'columns': columns,
            'scaler': scaler
        }
        
        return transformed_data
    
    def apply_log_transform(self, columns):
        """Apply log transformation"""
        transformed_data = self.data.copy()
        
        for col in columns:
            if (self.data[col] > 0).all():
                transformed_data[f'{col}_log'] = np.log(transformed_data[col])
                print(f"✓ Log transformation applied to {col}")
            else:
                print(f"✗ Cannot apply log transformation to {col} (contains non-positive values)")
        
        return transformed_data
    
    def apply_sqrt_transform(self, columns):
        """Apply square root transformation"""
        transformed_data = self.data.copy()
        
        for col in columns:
            if (self.data[col] >= 0).all():
                transformed_data[f'{col}_sqrt'] = np.sqrt(transformed_data[col])
                print(f"✓ Square root transformation applied to {col}")
            else:
                print(f"✗ Cannot apply sqrt transformation to {col} (contains negative values)")
        
        return transformed_data
    
    def apply_box_cox_transform(self, columns):
        """Apply Box-Cox transformation"""
        transformed_data = self.data.copy()
        
        for col in columns:
            if (self.data[col] > 0).all():
                transformed_values, lambda_param = stats.boxcox(self.data[col])
                transformed_data[f'{col}_boxcox'] = transformed_values
                print(f"✓ Box-Cox transformation applied to {col} (λ = {lambda_param:.3f})")
            else:
                print(f"✗ Cannot apply Box-Cox transformation to {col} (contains non-positive values)")
        
        return transformed_data
    
    def compare_transformations(self, column):
        """Compare different transformations visually"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Transformation Comparison for {column}', fontsize=16, fontweight='bold')
        
        # Original data
        axes[0, 0].hist(self.data[column].dropna(), bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Original Data')
        axes[0, 0].set_ylabel('Frequency')
        
        # Log transformation
        if (self.data[column] > 0).all():
            log_data = np.log(self.data[column])
            axes[0, 1].hist(log_data, bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('Log Transformation')
        
        # Square root transformation
        if (self.data[column] >= 0).all():
            sqrt_data = np.sqrt(self.data[column])
            axes[0, 2].hist(sqrt_data, bins=30, alpha=0.7, color='orange')
            axes[0, 2].set_title('Square Root Transformation')
        
        # Box-Cox transformation
        if (self.data[column] > 0).all():
            boxcox_data, _ = stats.boxcox(self.data[column])
            axes[1, 0].hist(boxcox_data, bins=30, alpha=0.7, color='red')
            axes[1, 0].set_title('Box-Cox Transformation')
        
        # Standardization
        standardized_data = (self.data[column] - self.data[column].mean()) / self.data[column].std()
        axes[1, 1].hist(standardized_data, bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_title('Standardization')
        
        # Min-Max scaling
        minmax_data = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
        axes[1, 2].hist(minmax_data, bins=30, alpha=0.7, color='brown')
        axes[1, 2].set_title('Min-Max Scaling')
        
        plt.tight_layout()
        plt.show()

def main_analysis():
    """
    Main function demonstrating complete outlier and transformation analysis
    """
    print("="*60)
    print("DATA TRANSFORMATION AND OUTLIER HANDLING ANALYSIS")
    print("="*60)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample dataset with various distributions
    data = pd.DataFrame({
        'normal_data': np.random.normal(50, 15, n_samples),
        'skewed_data': np.random.exponential(2, n_samples),
        'uniform_data': np.random.uniform(0, 100, n_samples),
        'outlier_prone': np.concatenate([
            np.random.normal(25, 5, int(0.9 * n_samples)),
            np.random.normal(80, 3, int(0.1 * n_samples))  # Outliers
        ])
    })
    
    print(f"Sample dataset created with {len(data)} rows and {len(data.columns)} columns")
    print("\nDataset Info:")
    print(data.describe())
    
    # Initialize handlers
    outlier_handler = OutlierHandler(data)
    transformer = DataTransformer(data)
    
    # Analyze each column
    for column in data.columns:
        print(f"\n{'='*50}")
        print(f"ANALYSIS FOR: {column.upper()}")
        print(f"{'='*50}")
        
        # Detect outliers using multiple methods
        outlier_handler.detect_outliers_iqr(column)
        outlier_handler.detect_outliers_zscore(column)
        
        # Visualize outliers
        outlier_handler.visualize_outliers(column)
        
        # Compare transformations
        transformer.compare_transformations(column)
        
        # Apply and compare different handling methods
        print(f"\nOutlier Handling Results for {column}:")
        
        # Method 1: Capping
        capped_data = outlier_handler.handle_outliers(column, method='cap')
        print(f"Capping: {len(data)} → {len(capped_data)} rows")
        
        # Method 2: Removal
        removed_data = outlier_handler.handle_outliers(column, method='remove')
        print(f"Removal: {len(data)} → {len(removed_data)} rows")
        
        # Apply transformations
        print(f"\nTransformation Results for {column}:")
        log_transformed = transformer.apply_log_transform([column])
        sqrt_transformed = transformer.apply_sqrt_transform([column])
        boxcox_transformed = transformer.apply_box_cox_transform([column])
    
    # Print comprehensive summary
    outlier_handler.print_outlier_summary()
    
    # Demonstrate normalization
    print(f"\n{'='*50}")
    print("NORMALIZATION COMPARISON")
    print(f"{'='*50}")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    standard_scaled = transformer.normalize_data(numeric_columns, 'standard')
    minmax_scaled = transformer.normalize_data(numeric_columns, 'minmax')
    robust_scaled = transformer.normalize_data(numeric_columns, 'robust')
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Normalization Methods Comparison', fontsize=16, fontweight='bold')
    
    # Original data
    axes[0, 0].boxplot([data[col] for col in numeric_columns], labels=numeric_columns)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Standard scaling
    axes[0, 1].boxplot([standard_scaled[col] for col in numeric_columns], labels=numeric_columns)
    axes[0, 1].set_title('Standard Scaling')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Min-Max scaling
    axes[1, 0].boxplot([minmax_scaled[col] for col in numeric_columns], labels=numeric_columns)
    axes[1, 0].set_title('Min-Max Scaling')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Robust scaling
    axes[1, 1].boxplot([robust_scaled[col] for col in numeric_columns], labels=numeric_columns)
    axes[1, 1].set_title('Robust Scaling')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Analysis Complete!")
    print("Key insights and recommendations will be displayed above.")

if __name__ == "__main__":
    main_analysis()