"""Official pipeline runner for baseline models (Logistic Regression & Decision Tree).

Usage:
    python scripts/run_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy script trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_baseline

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    train_baseline()
