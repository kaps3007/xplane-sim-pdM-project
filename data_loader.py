import os
import pandas as pd

def load_xplane_log(filepath: str) -> pd.DataFrame:
    """
    Load and clean an X-Plane log file (CSV or TXT).
    
    Args:
        filepath (str): Path to the raw X-Plane log file.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Load data
    try:
        df = pd.read_csv(filepath, sep="|", engine="python")  #X-Plane 11 logged the raw data with pipe ('|') separator
    except Exception:
        df = pd.read_csv(filepath)  #fallback to default comma
    
    #delete empty columns
    df = df.dropna(axis=1, how="all")
    
    #remove spaces, commas, strange chars
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    
    #convert numeric values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    
    return df


def save_processed(df: pd.DataFrame, filename: str, out_dir=r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed") -> str:
    """
    Save cleaned dataframe as CSV in processed folder.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        filename (str): Output file name
        out_dir (str): Output directory
    
    Returns:
        str: Path to saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    raw_file = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\raw\Data.txt"
    df = load_xplane_log(raw_file)
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    out_file = save_processed(df, "xplane_cleaned.csv")
    print(f"📂 Saved cleaned data at: {out_file}")