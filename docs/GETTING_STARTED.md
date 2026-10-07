# Hướng dẫn chạy pipeline — Branch `ml/baseline-logreg-dt`

Branch này huấn luyện hai mô hình baseline (Logistic Regression + Decision Tree) với 5-fold CV và log toàn bộ kết quả vào MLflow.

---

## Yêu cầu

- **Python 3.11 hoặc 3.12** (chưa hỗ trợ 3.13)
- Chạy tất cả lệnh từ **thư mục gốc repo** (nơi có `pyproject.toml`)

---

## Bước 1 — Cài môi trường

```bash
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

---

## Bước 2 — Tải dữ liệu thô

> **Bỏ qua nếu** `data/raw/UCI_Credit_Card.csv` đã tồn tại.

```bash
python scripts/download_data.py
```

Tạo ra:
```
data/raw/UCI_Credit_Card.csv                      ← dataset gốc (30 000 dòng, 25 cột)
data/raw/default-of-credit-card-clients.sha256    ← checksum để xác minh tính toàn vẹn
```

> ⚠️ Script tải từ Kaggle API — cần kết nối internet. Nếu tải thủ công, đặt file CSV vào đúng đường dẫn trên và đổi tên thành `UCI_Credit_Card.csv`.

---

## Bước 3 — Tạo file phân chia dữ liệu

> **Bỏ qua nếu** `data/splits/splits.json` đã tồn tại (đã được commit trong repo).

```bash
python scripts/split_data.py
```

Tạo ra:
```
data/splits/splits.json          ← danh sách row index cho Train / Valid / Test
data/splits/splits_summary.json  ← thống kê phân chia (xem nhanh tỷ lệ, default rate)
```

Kết quả phân chia cố định (`random_state=42`):

| Tập | Số mẫu | Tỷ lệ | Default rate |
|-----|--------|-------|--------------|
| Train | 18 000 | 60% | 22.12% |
| Valid |  6 000 | 20% | 22.12% |
| Test  |  6 000 | 20% | 22.12% |

> ℹ️ `splits.json` chỉ lưu **chỉ số dòng** (không lưu dữ liệu thực) — đã được commit và dùng chung trong toàn nhóm. **Không chạy lại** bước này nếu không có lý do, để đảm bảo mọi người dùng cùng một tập train/valid/test.

---

## Bước 4 — Huấn luyện baseline + log MLflow

```bash
python scripts/run_pipeline.py                  # baseline: LR & DT
python scripts/run_pipeline.py --mode advanced  # RF & XGBoost (tham số mặc định)
python scripts/run_pipeline.py --mode all       # cả hai
python scripts/run_pipeline.py --include-sex    # bản đối chiếu fairness (có SEX, không đăng ký model)
```
*(hoặc `python -m src.train`)*

Mô hình dùng bộ đặc trưng **feature freeze v1** (`configs/feature_freeze_v1.yaml`) qua `src.pipelines.make_pipeline`: làm sạch mã (PAY_0 → PAY_1…) → đặc trưng T2 → binning AGE theo IV → bỏ SEX/ID → one-hot + impute/scale. Input là các cột gốc của CSV.

Quá trình:
1. Đọc `data/raw/UCI_Credit_Card.csv` + `data/splits/splits.json`
2. Chạy **5-fold Stratified CV** cho Logistic Regression và Decision Tree (và RF, XGBoost với `--mode advanced`)
3. Log params, metrics (whitelist: `roc_auc`, `gini`, `ks`, `pr_auc`, `brier_score`, `precision`, `recall`, `f1`), tags phiên bản dữ liệu (checksum sha256, hash splits.json), và model artifact kèm signature vào MLflow local (`mlruns/`)

Output terminal:

```
Train size: 18000 samples, 22 input columns

=== logreg_baseline ===
  Fold 1: AUC=0.7835  KS=0.4490  Gini=0.5670
  ...
  → AUC: 0.7745 ± 0.0116
  → KS:  0.4282 ± 0.0231
  → Gini:0.5490 ± 0.0233

=== dt_baseline ===
  ...
  → AUC: 0.7672 ± 0.0120
  → KS:  0.4162 ± 0.0184
  → Gini:0.5345 ± 0.0240
```

> ℹ️ `mlruns/` bị gitignore — **mỗi người cần tự chạy lại** bước này trên máy của mình để có runs local. Không cần file `run_registry.json`. Để load model đã train:
> ```python
> import mlflow
> model = mlflow.sklearn.load_model("models:/logreg_baseline@baseline")
> ```

---

## Bước 5 — Xem kết quả MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Mở trình duyệt tại: **http://127.0.0.1:5000**

Trong UI, experiment `credit_scoring` hiển thị các run `logreg_baseline`, `dt_baseline` (và `rf_default`, `xgboost_default` nếu chạy `--mode advanced`) với đầy đủ params, metrics theo fold và model artifact.

> **Dùng server chung của nhóm:** đặt biến môi trường trước khi chạy bất kỳ script nào:
> ```bash
> # Windows
> set MLFLOW_TRACKING_URI=http://<server-ip>:5000
> # Linux / macOS
> export MLFLOW_TRACKING_URI=http://<server-ip>:5000
> ```

---

## Bước 6 (tùy chọn) — Chạy demo Streamlit

```bash
streamlit run app/streamlit_app.py
```

> App nạp baseline Logistic Regression từ MLflow Model Registry (`logreg_baseline@baseline`), được đăng ký bởi `python scripts/run_pipeline.py`. Khi registry chưa có mô hình, app chuyển sang **mock model** (hệ số đặt tay) và hiển thị cảnh báo ở sidebar.

---

## Kiểm thử

```bash
pytest
```

Hoặc kèm coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

CI tự động chạy trên mỗi push / PR vào `main` (GitHub Actions `.github/workflows/ci.yml`).

---

## Sơ đồ phụ thuộc giữa các bước

```
Bước 1: Cài môi trường (.venv)
    │
    ▼
Bước 2: download_data.py  →  data/raw/UCI_Credit_Card.csv
    │
    ▼
Bước 3: split_data.py     →  data/splits/splits.json          ← đã có trong repo, thường bỏ qua
    │
    ▼
Bước 4: python -m src.train  →  mlruns/ (local, gitignored)
    │
    ▼
Bước 5: mlflow ui             →  http://127.0.0.1:5000
```

---

## Cấu hình

Tất cả tham số nằm trong [`configs/config.yaml`](configs/config.yaml):

```yaml
random_state: 42

data:
  raw_path: data/raw/UCI_Credit_Card.csv
  target_col: default.payment.next.month
  splits_dir: data/splits
  train_ratio: 0.6
  valid_ratio: 0.2
  test_ratio: 0.2

mlflow:
  tracking_dir: mlruns
  experiment_name: credit_scoring
```
