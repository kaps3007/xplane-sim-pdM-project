import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed(filepath: str) -> pd.DataFrame:
    """Load the processed cleaned dataset."""
    return pd.read_csv(filepath)


def summary_stats(df: pd.DataFrame):
    """Print quick stats and missing values info."""
    print("\n📊 Dataset Overview")
    print(df.info())
    print("\n🔍 Missing Values:")
    print(df.isna().sum())
    print("\n📈 Descriptive Stats:")
    print(df.describe().T)


def plot_timeseries(df: pd.DataFrame, columns: list, time_col: str = None, out_dir="reports/figures/"):
    """
    Plot timeseries for selected engine variables.
    
    Args:
        df (pd.DataFrame): DataFrame with engine data
        columns (list): List of columns to plot
        time_col (str): Time column (if None, use index)
        out_dir (str): Where to save plots
    """
    os.makedirs(out_dir, exist_ok=True)

    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(12, 5))
            if time_col and time_col in df.columns:
                plt.plot(df[time_col], df[col], label=col)
                plt.xlabel(time_col)
            else:
                plt.plot(df.index, df[col], label=col)
                plt.xlabel("Index")
            plt.ylabel(col)
            plt.title(f"{col} over Time")
            plt.legend()
            plt.grid(True)
            out_path = os.path.join(out_dir, f"{col}_timeseries.png")
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved: {out_path}")

def correlation_heatmap(df: pd.DataFrame, out_dir=r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\reports\figures"):
    """Plot correlation heatmap of numerical variables."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap of Engine Variables")
    out_path = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    processed_file = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_cleaned.csv"
    df = load_processed(processed_file)

    summary_stats(df)

    #engine health variables to visualize
    key_vars = ["N1_1_pcnt", "N2_1_pcnt", "EGT_1_deg", "OILT1_deg", "FUEP1_psi"]

    plot_timeseries(df, key_vars, time_col="frame_time")
    correlation_heatmap(df)