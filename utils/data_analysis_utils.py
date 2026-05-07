"""
Data Analysis Utility Functions
Reusable utility module for all 5 projects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, classification_report)
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')


class DataAnalysisUtils:
    """Reusable utility class for all data analysis projects."""

    def __init__(self):
        self._configure_style()
        np.random.seed(42)

    # ── Styling ────────────────────────────────────────────────────────────
    def _configure_style(self):
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            plt.style.use('ggplot')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })

    # ── I/O ────────────────────────────────────────────────────────────────
    def load_dataset(self, filepath, **kwargs):
        """Load CSV / Excel / Parquet with basic reporting."""
        try:
            ext = os.path.splitext(filepath)[1].lower()
            loaders = {'.csv': pd.read_csv,
                       '.xlsx': pd.read_excel,
                       '.xls': pd.read_excel,
                       '.parquet': pd.read_parquet}
            if ext not in loaders:
                raise ValueError(f"Unsupported format: {ext}")
            df = loaders[ext](filepath, **kwargs)
            print(f"✅ Loaded {filepath}  |  shape={df.shape}  |  "
                  f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB")
            return df
        except Exception as exc:
            print(f"❌ Error loading {filepath}: {exc}")
            return None

    # ── EDA ────────────────────────────────────────────────────────────────
    def comprehensive_eda(self, df, target_col=None):
        """Return a dict with shape, missing values, dtypes, and duplicates."""
        missing = df.isnull().sum()
        missing_pct = missing / len(df) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Pct': missing_pct.round(2)
        }).query('Missing_Count > 0').sort_values('Missing_Count', ascending=False)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        cat_stats = {
            col: {
                'unique': df[col].nunique(),
                'top5': df[col].value_counts().head(5).to_dict(),
                'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
            for col in cat_cols
        }

        return {
            'basic_info': {
                'shape': df.shape,
                'num_cols': num_cols,
                'cat_cols': cat_cols,
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1e6, 2),
            },
            'missing_values': missing_df,
            'numerical_stats': df[num_cols].describe().T if num_cols else pd.DataFrame(),
            'categorical_stats': cat_stats,
            'duplicates': int(df.duplicated().sum()),
        }

    # ── Cleaning ───────────────────────────────────────────────────────────
    def clean_dataframe(self, df, handle_missing='median',
                        drop_duplicates=True,
                        remove_outliers=False,
                        outlier_cols=None,
                        outlier_method='iqr'):
        """
        Flexible cleaning pipeline.
        handle_missing: 'median' | 'mean' | 'mode' | 'forward' | 'backward'
        """
        df_c = df.copy()
        num_cols = df_c.select_dtypes(include=np.number).columns
        cat_cols = df_c.select_dtypes(include=['object', 'category']).columns

        # ── missing values
        if handle_missing in ('median', 'mean'):
            for col in num_cols:
                if df_c[col].isnull().any():
                    fill_val = (df_c[col].median() if handle_missing == 'median'
                                else df_c[col].mean())
                    df_c[col] = df_c[col].fillna(fill_val)
        elif handle_missing == 'mode':
            for col in cat_cols:
                if df_c[col].isnull().any():
                    mode = df_c[col].mode()
                    df_c[col] = df_c[col].fillna(mode.iloc[0] if not mode.empty else 'Unknown')
        elif handle_missing == 'forward':
            df_c = df_c.ffill()
        elif handle_missing == 'backward':
            df_c = df_c.bfill()

        # ── duplicates
        if drop_duplicates:
            before = len(df_c)
            df_c = df_c.drop_duplicates()
            print(f"   Dropped {before - len(df_c)} duplicate rows")

        # ── outliers
        if remove_outliers and outlier_cols:
            before = len(df_c)
            for col in outlier_cols:
                if col not in df_c.columns:
                    continue
                if outlier_method == 'iqr':
                    q1, q3 = df_c[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    df_c = df_c[(df_c[col] >= q1 - 1.5 * iqr) &
                                (df_c[col] <= q3 + 1.5 * iqr)]
                elif outlier_method == 'zscore':
                    df_c = df_c[(np.abs(stats.zscore(df_c[col].dropna())) < 3)]
            print(f"   Removed {before - len(df_c)} outlier rows")

        return df_c

    # ── Feature Engineering ────────────────────────────────────────────────
    def feature_engineering(self, df, date_cols=None, text_cols=None):
        """Auto-generate date parts and text length features."""
        df_fe = df.copy()
        if date_cols:
            for col in date_cols:
                if col in df_fe.columns:
                    df_fe[col] = pd.to_datetime(df_fe[col])
                    for attr in ('year', 'month', 'day', 'dayofweek', 'quarter'):
                        df_fe[f'{col}_{attr}'] = getattr(df_fe[col].dt, attr)
        if text_cols:
            for col in text_cols:
                if col in df_fe.columns:
                    s = df_fe[col].astype(str)
                    df_fe[f'{col}_length'] = s.str.len()
                    df_fe[f'{col}_words'] = s.str.split().str.len()
        return df_fe

    # ── Visualisations ────────────────────────────────────────────────────
    def plot_correlation_matrix(self, df, method='pearson',
                                figsize=(12, 10), save_path=None):
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] < 2:
            print("⚠️  Not enough numerical columns.")
            return None
        corr = num_df.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True,
                    linewidths=0.5, ax=ax)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})',
                     fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return corr

    def plot_missing_values(self, df, save_path=None):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            print("✅ No missing values.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(missing.index, missing.values, color='coral')
        axes[0].set_title('Missing Value Counts', fontweight='bold')
        axes[1].barh(missing.index,
                     (missing / len(df) * 100).values, color='skyblue')
        axes[1].set_title('Missing Value %', fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def create_distribution_plots(self, df, cols=None,
                                  bins=30, save_path=None):
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        n_cols = min(len(cols), 3)
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        for idx, col in enumerate(cols):
            ax = axes[idx]
            ax.hist(df[col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.axvline(df[col].mean(), color='red',
                       linestyle='--', label='Mean')
            ax.axvline(df[col].median(), color='green',
                       linestyle='--', label='Median')
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.legend(fontsize=8)
        for idx in range(len(cols), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    # ── Summary ───────────────────────────────────────────────────────────
    def generate_summary_statistics(self, df):
        num_df = df.select_dtypes(include=np.number)
        cat_df = df.select_dtypes(include=['object', 'category'])
        return {
            'Numerical': num_df.describe().T if not num_df.empty else pd.DataFrame(),
            'Info': {
                'Rows': len(df),
                'Columns': len(df.columns),
                'Numerical Columns': len(num_df.columns),
                'Categorical Columns': len(cat_df.columns),
                'Missing Values': int(df.isnull().sum().sum()),
                'Duplicate Rows': int(df.duplicated().sum()),
            }
        }


# Shared instance — imported by all notebooks as:
#   from utils.data_analysis_utils import utils
utils = DataAnalysisUtils()
