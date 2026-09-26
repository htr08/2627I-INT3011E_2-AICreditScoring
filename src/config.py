"""Đọc cấu hình dùng chung từ configs/config.yaml."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG_PATH) -> dict:
    """Trả về nội dung file YAML dưới dạng dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
