# 2627I-INT3011E_2-AICreditScoring

## Cài đặt môi trường

Yêu cầu Python 3.11 hoặc 3.12 (các version pin trong `requirements.txt` chưa hỗ trợ 3.13).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Theo dõi thí nghiệm với MLflow

Cấu hình nằm trong `configs/config.yaml` (mục `mlflow`). Mặc định log vào thư mục `mlruns/` ở gốc repo (đã gitignore),
dù chạy từ script hay notebook. Muốn dùng server chung của nhóm thì đặt biến môi trường `MLFLOW_TRACKING_URI`.

```python
import mlflow
from src.tracking import setup_mlflow

setup_mlflow()  # experiment mặc định: credit_scoring
with mlflow.start_run(run_name="logreg_baseline"):
    mlflow.log_params({"C": 1.0})
    mlflow.log_metric("auc_cv", 0.76)
```

Optuna: mỗi trial mở `mlflow.start_run(nested=True)` bên trong một run cha.

Xem kết quả:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Kiểm thử

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) tự chạy pytest trên mỗi push/Pull Request vào `main`.
