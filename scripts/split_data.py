import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import load_config
from src.data_split import save_splits, split_indices


def main():
    config = load_config()
    data_cfg = config["data"]
    random_state = config.get("random_state", 42)

    raw_path = PROJECT_ROOT / data_cfg["raw_path"]
    splits_dir = PROJECT_ROOT / data_cfg["splits_dir"]
    target_col = data_cfg["target_col"]
    train_ratio = data_cfg.get("train_ratio", 0.6)
    valid_ratio = data_cfg.get("valid_ratio", 0.2)
    test_ratio = data_cfg.get("test_ratio", 0.2)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_path}. "
            f"Please run 'python scripts/download_data.py' first!"
        )

    print(f"[1/3] Reading raw data from: {raw_path.relative_to(PROJECT_ROOT)}...")
    df = pd.read_csv(raw_path)
    print(f"      Rows: {len(df):,}, Columns: {len(df.columns)}")

    print(f"[2/3] Splitting dataset ({train_ratio}/{valid_ratio}/{test_ratio}, random_state={random_state})...")
    splits = split_indices(
        df=df,
        target_col=target_col,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    print(f"[3/3] Saving split index files to: {splits_dir.relative_to(PROJECT_ROOT)}...")
    save_splits(
        splits = splits,
        df = df,
        output_dir = splits_dir,
        target_col = target_col,
        random_state = random_state,
        raw_file = raw_path.name,
    )

    print("\n--- SPLIT DISTRIBUTION SUMMARY ---")
    for name, indices in splits.items():
        subset = df.iloc[indices]
        pos_ratio = subset[target_col].mean()
        print(
            f" - {name.upper():<5}: {len(indices):>5} rows ({len(indices)/len(df)*100:0.1f}%) "
            f"| Default rate (class 1): {pos_ratio*100:.2f}%"
        )
    print("----------------------------------")
    print("Done! Split index files are saved and ready for the team.")


if __name__ == "__main__":
    main()
