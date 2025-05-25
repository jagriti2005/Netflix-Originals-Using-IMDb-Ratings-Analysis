import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.colors = ['#4caf50', '#2196f3', '#ff9800', '#e91e63', '#9c27b0', '#00bcd4', '#ffc107', '#795548']

    def rubric_7_outlier_and_distribution(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            print("No numeric columns available.")
            return

        fig = plt.figure(figsize=(24, 14))
        gs = GridSpec(2, 2, height_ratios=[1, 2], figure=fig)

        # -----------------------------
        # 1. Outlier Detection Summary
        # -----------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        outlier_counts = []
        outlier_percentages = []

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < Q1 - 1.5 * IQR) | (self.data[col] > Q3 + 1.5 * IQR)]
            count = len(outliers)
            pct = (count / len(self.data)) * 100
            outlier_counts.append(count)
            outlier_percentages.append(pct)

        # Color based on severity
        colors = ['red' if pct > 5 else 'orange' if pct > 2 else 'green' for pct in outlier_percentages]
        bars = ax1.bar(numeric_cols, outlier_counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_title('🚨 Outlier Detection Summary (IQR Method)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Outliers')
        ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')

        # Add labels
        for bar, count, pct in zip(bars, outlier_counts, outlier_percentages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{count}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

        # Legend
        legend_elements = [
            Patch(facecolor='green', label='Low (<2%)'),
            Patch(facecolor='orange', label='Medium (2-5%)'),
            Patch(facecolor='red', label='High (>5%)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # ---------------------------------
        # 2. Advanced Distribution Gallery
        # ---------------------------------
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        fig2, axes = plt.subplots(n_rows, 3, figsize=(22, 6 * n_rows))
        fig2.suptitle('📊 Advanced Distribution Analysis Gallery', fontsize=20, fontweight='bold')

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(numeric_cols):
            row = idx // 3
            col_pos = idx % 3
            ax = axes[row, col_pos]
            data = self.data[col].dropna()

            # Histogram
            n, bins, patches = ax.hist(data, bins=30, color=self.colors[idx % len(self.colors)],
                                       alpha=0.7, edgecolor='black', linewidth=0.5, density=True)

            # Statistical fit
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal', alpha=0.8)

            if data.min() > 0:
                try:
                    params = stats.lognorm.fit(data)
                    ax.plot(x, stats.lognorm.pdf(x, *params), 'g--', linewidth=2, label='Log-Normal', alpha=0.7)
                except:
                    pass

            # Z-score coloring
            z_scores = (bins[:-1] - mu) / sigma
            for j, patch in enumerate(patches):
                if abs(z_scores[j]) > 2:
                    patch.set_facecolor('red')
                elif abs(z_scores[j]) > 1:
                    patch.set_facecolor('orange')

            # Summary stats
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            p_val = stats.shapiro(data.sample(min(5000, len(data))))[1] if len(data) > 3 else 1

            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2, alpha=0.8)

            ax.set_title(f'{col}\nμ={mu:.2f}, σ={sigma:.2f}, Skew={skewness:.2f}, Kurt={kurtosis:.2f}\nNormality p={p_val:.3f}',
                         fontsize=9, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Interpretation box
            interp = self._interpret_distribution(skewness, kurtosis, p_val)
            ax.text(0.02, 0.98, interp, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Hide unused plots
        for i in range(n_cols, n_rows * 3):
            row, col_pos = divmod(i, 3)
            axes[row, col_pos].axis('off')

        plt.tight_layout()
        plt.show()
        return fig, fig2

    def _interpret_distribution(self, skew, kurt, p_val):
        interp = []
        # Skew
        if skew > 1:
            interp.append("Right-skewed")
        elif skew < -1:
            interp.append("Left-skewed")
        else:
            interp.append("Symmetric")
        # Kurtosis
        if kurt > 3:
            interp.append("Leptokurtic")
        elif kurt < 3:
            interp.append("Platykurtic")
        else:
            interp.append("Mesokurtic")
        # Normality
        if p_val > 0.05:
            interp.append("✅ Normal dist (Shapiro)")
        else:
            interp.append("🚨 Not normal")
        return ", ".join(interp)
