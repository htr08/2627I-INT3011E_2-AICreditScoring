"""Demo Streamlit: nhập hồ sơ -> PD, điểm tín dụng, giải thích theo nhóm đặc trưng.

Chạy từ thư mục gốc repo:  streamlit run app/streamlit_app.py
Bố cục xem reports/demo_wireframe.md. Tuần 1 dùng mock model; đổi mô hình thật tại load_model().
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mock_model import MockModel  # noqa: E402
from app.sample_profiles import PROFILES  # noqa: E402
from src.scoring import BASE_ODDS, BASE_SCORE, PDO, SCORE_CLIP, base_points, pd_to_score, shap_to_points  # noqa: E402

MONTH_LABELS = ["T9", "T8", "T7", "T6", "T5", "T4"]  # PAY_1/BILL_AMT1 = tháng 9/2005
EDUCATION_OPTIONS = {1: "Sau đại học", 2: "Đại học", 3: "Trung học", 4: "Khác"}
MARRIAGE_OPTIONS = {1: "Đã kết hôn", 2: "Độc thân", 3: "Khác"}
PAY_OPTIONS = {
    -2: "-2: Không phát sinh chi tiêu",
    -1: "-1: Trả đủ",
    0: "0: Trả tối thiểu (quay vòng)",
    **{m: f"{m}: Trễ {m} tháng" for m in range(1, 9)},
    9: "9: Trễ ≥ 9 tháng",
}
DEFAULT_PROFILE = "Ca biên"
# Ngưỡng tạm thời; Tuần 3 thay bằng ngưỡng tối ưu theo ma trận chi phí.
DEFAULT_THRESHOLD = 0.30
# Dải điểm thực tế với PDO 600/50:1/20 (research_shap_scorecard_pdo.md, mục 4.3).
SCORE_DISPLAY_RANGE = (420, 640)


@st.cache_resource
def load_model():
    """Tải mô hình: ưu tiên BaselineModel nếu có model registry/artifact, fallback sang MockModel."""
    try:
        from app.baseline_model import load_baseline_model
        model = load_baseline_model()
        if model is not None:
            return model
    except Exception:
        pass
    return MockModel()


def apply_profile(name: str) -> None:
    st.session_state.update(PROFILES[name])


def risk_band(pd: float, threshold: float) -> str:
    if pd >= threshold:
        return "🔴 Cảnh báo"
    if pd >= threshold / 2:
        return "🟡 Theo dõi"
    return "🟢 Bình thường"


def render_form() -> None:
    with st.form("applicant"):
        tab_info, tab_pay, tab_amt = st.tabs(["Thông tin chung", "Lịch sử trả nợ", "Dư nợ & thanh toán"])
        with tab_info:
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Hạn mức tín dụng (NT$)", min_value=10_000, max_value=1_000_000, step=10_000, key="LIMIT_BAL")
            c2.number_input("Tuổi", min_value=21, max_value=79, step=1, key="AGE")
            c3.selectbox("Học vấn", EDUCATION_OPTIONS, format_func=EDUCATION_OPTIONS.get, key="EDUCATION")
            c4.selectbox("Hôn nhân", MARRIAGE_OPTIONS, format_func=MARRIAGE_OPTIONS.get, key="MARRIAGE")
        with tab_pay:
            st.caption("Trạng thái trả nợ 6 tháng gần nhất (T9 = gần nhất).")
            for col, month, i in zip(st.columns(6), MONTH_LABELS, range(1, 7)):
                col.selectbox(month, PAY_OPTIONS, format_func=PAY_OPTIONS.get, key=f"PAY_{i}")
        with tab_amt:
            st.caption("Dư nợ sao kê và số tiền đã trả mỗi tháng (NT$).")
            for col, month, i in zip(st.columns(6), MONTH_LABELS, range(1, 7)):
                col.number_input(f"Dư nợ {month}", step=1_000, key=f"BILL_AMT{i}")
                col.number_input(f"Đã trả {month}", min_value=0, step=1_000, key=f"PAY_AMT{i}")
        st.form_submit_button("Chấm điểm", type="primary")


def waterfall_chart(start: float, points: dict[str, float], end: float):
    """Waterfall theo điểm: điểm nền -> cộng/trừ từng nhóm -> điểm cuối (chưa clip)."""
    items = sorted(points.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = ["Điểm nền", *(g for g, _ in items), "Điểm cuối"]
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(labels) + 0.6))
    running = start
    visited = [start]
    ax.barh(0, start, color="#9e9e9e")
    ax.text(start, 0, f" {start:.0f}", va="center")
    for y, (_, value) in enumerate(items, start=1):
        ax.barh(y, value, left=running, color="#2e7d32" if value >= 0 else "#c62828")
        ax.text(max(running, running + value), y, f" {value:+.1f}", va="center")
        running += value
        visited.append(running)
    ax.barh(len(labels) - 1, end, color="#1565c0")
    ax.text(end, len(labels) - 1, f" {end:.0f}", va="center")
    # Phóng to quanh vùng biến động thay vì hiển thị từ 0.
    ax.set_xlim(min(visited) - 30, max(visited) + 30)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Điểm")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_result(model, threshold: float) -> None:
    record = {k: st.session_state[k] for k in PROFILES[DEFAULT_PROFILE]}
    exp = model.explain(record)
    score = float(pd_to_score(exp.pd))
    points = {g: float(shap_to_points(v)) for g, v in exp.contributions.items()}
    start = float(base_points(exp.base_logit))
    end = start + sum(points.values())

    st.subheader("Kết quả")
    m1, m2, m3 = st.columns(3)
    m1.metric("Xác suất vỡ nợ (PD)", f"{exp.pd:.1%}")
    m2.metric("Điểm tín dụng", f"{score:.0f}")
    m3.metric("Mức cảnh báo", risk_band(exp.pd, threshold), help=f"Ngưỡng PD hiện tại: {threshold:.0%}")
    lo, hi = SCORE_DISPLAY_RANGE
    st.progress(min(max((score - lo) / (hi - lo), 0.0), 1.0), text=f"Vị trí trên thang {lo}–{hi}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Đóng góp theo nhóm đặc trưng (điểm)**")
        fig = waterfall_chart(start, points, end)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Lý do chính làm giảm điểm**")
        negatives = sorted((v, g) for g, v in points.items() if v < 0)
        if negatives:
            for v, g in negatives[:3]:
                st.markdown(f"- {g}: **{v:+.0f}** điểm")
        else:
            st.markdown("- Không có nhóm nào làm giảm điểm.")
        st.markdown("**Yếu tố tích cực**")
        for v, g in sorted(((v, g) for g, v in points.items() if v > 0), reverse=True)[:3]:
            st.markdown(f"- {g}: **{v:+.0f}** điểm")

    with st.expander("Chi tiết kỹ thuật"):
        st.write(f"Điểm nền {start:.1f} + tổng đóng góp {end - start:+.1f} = {end:.1f} (trước khi clip {SCORE_CLIP[0]}–{SCORE_CLIP[1]}).")
        st.write("Đóng góp trên thang logit(PD):", {g: round(v, 4) for g, v in exp.contributions.items()})
        st.write("Dữ liệu đầu vào:", record)


def main() -> None:
    st.set_page_config(page_title="AI Credit Scoring", page_icon="💳", layout="wide")
    if "LIMIT_BAL" not in st.session_state:
        apply_profile(DEFAULT_PROFILE)
    model = load_model()

    with st.sidebar:
        st.header("Mô hình")
        st.write(model.name)
        if getattr(model, "is_mock", False):
            st.warning("Đang dùng mock model: kết quả chỉ để minh họa giao diện.")
        st.caption(f"Thang điểm PDO: {BASE_SCORE} điểm tại odds {BASE_ODDS}:1, PDO = {PDO}.")
        st.header("Hồ sơ mẫu")
        profile = st.selectbox("Chọn hồ sơ", list(PROFILES), index=list(PROFILES).index(DEFAULT_PROFILE))
        st.button("Nạp hồ sơ", on_click=apply_profile, args=(profile,))
        st.header("Ngưỡng")
        threshold = st.slider("Ngưỡng cảnh báo PD", 0.05, 0.80, DEFAULT_THRESHOLD, 0.01)

    st.title("💳 AI Credit Scoring")
    st.caption(
        "Hệ thống cảnh báo sớm rủi ro tín dụng cho danh mục thẻ hiện tại. "
        "Dữ liệu: Default of Credit Card Clients (Đài Loan, 2005)."
    )
    render_form()
    render_result(model, threshold)


main()
