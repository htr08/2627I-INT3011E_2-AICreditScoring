"""Phân tầng rủi ro và hành động đề xuất cho demo (reports/calibration_threshold.md mục 6.3).

Ranh giới dưới là ngưỡng vận hành đã khóa (`calibration.threshold`, 0.09); ranh giới trên là
`risk_tiers.alert_threshold` (0.27, ngưỡng tối ưu khi tỷ lệ chi phí FN:FP = 3:1).
"""

from dataclasses import dataclass
from typing import Optional

from src.config import load_config


@dataclass(frozen=True)
class RiskTier:
    key: str
    label: str
    action: str


NORMAL = RiskTier("normal", "🟢 Bình thường", "Duy trì hạn mức, tiếp tục theo dõi định kỳ.")
WATCH = RiskTier("watch", "🟡 Theo dõi", "Gửi nhắc nợ tự động, tạm dừng nâng hạn mức tín dụng.")
ALERT = RiskTier("alert", "🔴 Cảnh báo cao", "Giảm hạn mức khả dụng, chuyển bộ phận thu hồi nợ sớm can thiệp.")
TIERS = (NORMAL, WATCH, ALERT)


def load_thresholds(config: Optional[dict] = None) -> tuple[float, float]:
    """(ngưỡng theo dõi, ngưỡng cảnh báo cao) từ configs/config.yaml; ValueError nếu thiếu hoặc không tăng dần."""
    cfg = load_config() if config is None else config
    watch = cfg.get("calibration", {}).get("threshold")
    alert = cfg.get("risk_tiers", {}).get("alert_threshold")
    if watch is None or alert is None:
        raise ValueError("Thiếu calibration.threshold hoặc risk_tiers.alert_threshold trong configs/config.yaml.")
    if not 0 < float(watch) < float(alert) < 1:
        raise ValueError(f"Ngưỡng phải thỏa 0 < theo dõi < cảnh báo cao < 1, nhận {watch} và {alert}.")
    return float(watch), float(alert)


def classify(pd: float, watch: float, alert: float) -> RiskTier:
    """PD < watch: bình thường; watch <= PD < alert: theo dõi; PD >= alert: cảnh báo cao."""
    if pd >= alert:
        return ALERT
    if pd >= watch:
        return WATCH
    return NORMAL
