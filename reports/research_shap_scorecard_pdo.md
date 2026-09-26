# Nghiên cứu: SHAP, Logistic Scorecard và phương pháp PDO

Phiên bản thư viện tham chiếu (theo `requirements.txt`): `shap==0.46.0`, `scikit-learn==1.5.1`, `xgboost==2.1.0`, `lightgbm==4.5.0`, `catboost==1.2.5`.

## Tóm tắt

- SHAP phân rã dự đoán của mô hình thành đóng góp cộng tính của từng đặc trưng. Ba explainer được khảo sát có phạm vi áp dụng khác nhau: `TreeExplainer` cho mô hình cây, `LinearExplainer` cho mô hình tuyến tính, `KernelExplainer` cho mô hình bất kỳ nhưng chậm và mang tính xấp xỉ.
- Logistic Scorecard là mô hình Logistic trên các biến đã rời rạc hóa và mã hóa WoE; điểm số của từng nhóm giá trị (bin) cộng dồn thành điểm tín dụng.
- Phương pháp PDO quy đổi log-odds sang điểm theo quan hệ tuyến tính. Với tham số của dự án (600 điểm tại odds 50:1, PDO = 20): Factor ≈ 28,854, Offset ≈ 487,12.
- Vì cả SHAP (sau hiệu chuẩn) và điểm PDO đều tuyến tính theo logit(PD), đóng góp của từng đặc trưng có thể được quy đổi trực tiếp sang số điểm cộng/trừ.

---

## 1. Nền tảng của SHAP

### 1.1. Giá trị Shapley

SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) phân rã dự đoán $f(x)$ thành tổng đóng góp của các đặc trưng:

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i, \qquad \phi_0 = \mathbb{E}[f(X)]
$$

Trong đó $\phi_i$ là giá trị Shapley của đặc trưng $i$, được tính là đóng góp biên trung bình khi $i$ được thêm vào mọi tập con $S$ của các đặc trưng còn lại:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(M-|S|-1)!}{M!}\Big[f_{S\cup\{i\}}(x) - f_S(x)\Big]
$$

với $f_S(x)$ là giá trị kỳ vọng của mô hình khi chỉ các đặc trưng trong $S$ được biết.

### 1.2. Các tính chất

| Tính chất | Nội dung | Ý nghĩa đối với dự án |
|---|---|---|
| Local accuracy (additivity) | $\phi_0 + \sum \phi_i = f(x)$ | Tổng SHAP bằng đúng log-odds của khách hàng, cho phép quy đổi trực tiếp sang điểm (mục 4.5). |
| Missingness | Đặc trưng vắng mặt có $\phi_i = 0$ | — |
| Consistency | Nếu mô hình phụ thuộc vào $i$ nhiều hơn thì $\phi_i$ không giảm | Thứ hạng tầm quan trọng nhất quán hơn so với feature importance dựa trên gain hoặc số lần split. |

Do tính cộng tính, SHAP có thể được tổng hợp theo nhóm đặc trưng: $\phi_{\text{nhóm}} = \sum_{i \in \text{nhóm}} \phi_i$. Đây là cơ sở của reason codes theo nhóm (mục 1.4).

### 1.3. Hai cách định nghĩa $f_S(x)$

- **Interventional (marginal):** đặc trưng vắng mặt được thay bằng giá trị lấy độc lập từ tập nền (background). Cách này phản ánh đúng cách mô hình tính toán ("true to the model") và cần dữ liệu nền.
- **Observational / path-dependent (conditional):** dùng kỳ vọng có điều kiện theo phân phối dữ liệu ("true to the data"). Cách này có thể gán đóng góp cho đặc trưng mà mô hình không sử dụng nếu đặc trưng đó tương quan với đặc trưng mà mô hình dùng.

Trong bộ dữ liệu Taiwan, hai nhóm `PAY_1..6` và `BILL_AMT1..6` có tương quan rất cao, nên giá trị SHAP của từng biến riêng lẻ phụ thuộc đáng kể vào cách định nghĩa trên. Kết luận ở cấp **nhóm đặc trưng** ổn định hơn cấp biến; đây là cơ sở của quy ước tại `project_plan.md` mục 3.5e.

### 1.4. Các loại biểu đồ sử dụng

| Mục đích | Biểu đồ | Ghi chú |
|---|---|---|
| Toàn cục | `shap.plots.beeswarm` (summary), `shap.plots.bar` (mean \|SHAP\|) | Lấy mẫu 2.000–5.000 dòng. |
| Tương tác | `shap.plots.scatter` (dependence), tham số `color=` chọn biến tương tác | Top 5–10 đặc trưng. |
| Cục bộ | `shap.plots.waterfall`, `shap.plots.force` | Các ca Duyệt, Từ chối và Ca biên. Waterfall phù hợp hơn khi hiển thị trên Streamlit. |
| Reason codes | Tự tính | Cộng SHAP theo nhóm; chọn 3–4 nhóm có $\phi > 0$ lớn nhất (làm tăng rủi ro). |

---

## 2. Ba Explainer

### 2.1. TreeExplainer

**Phạm vi:** Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost.

**Thuật toán.** Tree SHAP (Lundberg et al., 2020) tính giá trị Shapley chính xác trong thời gian đa thức $O(T L D^2)$ ($T$ cây, $L$ lá, độ sâu $D$), thay vì hàm mũ theo số đặc trưng. Thời gian tính cho một khách hàng ở mức mili-giây, phù hợp cho demo trực tiếp.

**Tham số chính:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `feature_perturbation` | `"tree_path_dependent"` (mặc định khi không truyền `data`) | Dùng số mẫu đi qua mỗi nhánh khi huấn luyện (cover) để xấp xỉ kỳ vọng có điều kiện; không cần dữ liệu nền; hỗ trợ `shap_interaction_values`. |
| | `"interventional"` (mặc định khi có `data`) | Cần dữ liệu nền (thường ≤ 1.000 dòng, ví dụ `shap.sample(X_train, 500)`). Bắt buộc khi dùng `model_output="probability"`. |
| `model_output` | `"raw"` (mặc định) | Với XGBoost/LightGBM/CatBoost nhị phân: log-odds (margin). Với RandomForest/DecisionTree của sklearn: xác suất (trung bình xác suất tại lá). |
| | `"probability"` | SHAP trên thang xác suất; không còn cộng tuyến tính theo log-odds nên không tương thích với hiệu chuẩn Platt của dự án. Không sử dụng. |

**Lưu ý triển khai:**

- **Hình dạng đầu ra khác nhau theo mô hình.** Với XGBoost/LightGBM/CatBoost nhị phân, `explainer(X).values` có shape `(n, M)`. Với RandomForest của sklearn (shap ≥ 0.45), shape là `(n, M, 2)` và cần lấy `[:, :, 1]` (lớp vỡ nợ); `expected_value` tương ứng là scalar hoặc mảng hai phần tử. Việc chuẩn hóa shape được đặt trong một hàm của `src/explain.py`.
- **Kiểm tra additivity.** Tham số `check_additivity=True` (mặc định) báo lỗi khi $\phi_0 + \sum\phi_i$ lệch khỏi đầu ra mô hình. Lỗi này thường do dữ liệu đầu vào khác kiểu dữ liệu hoặc thứ tự cột so với lúc huấn luyện.
- **Pipeline.** Explainer nhận mô hình cuối (`pipe[-1]`); dữ liệu đầu vào phải là dữ liệu đã qua các bước tiền xử lý (`pipe[:-1].transform(X)`), kèm tên cột từ `get_feature_names_out()`.

```python
import shap

model = pipe[-1]                               # XGBClassifier đã fit trên Train
X_t = pipe[:-1].transform(X_valid_sample)       # dữ liệu đã tiền xử lý
explainer = shap.TreeExplainer(model)           # tree_path_dependent, đầu ra log-odds
sv = explainer(X_t)                             # shap.Explanation, sv.values: (n, M)

shap.plots.beeswarm(sv)
shap.plots.waterfall(sv[0])
```

### 2.2. LinearExplainer

**Phạm vi:** Logistic Regression baseline và Logistic Scorecard (WoE).

**Công thức.** Với mô hình tuyến tính trên log-odds $f(x) = \beta_0 + \sum_j \beta_j x_j$ và giả định các đặc trưng độc lập (interventional):

$$
\phi_j = \beta_j\,(x_j - \mathbb{E}[x_j]), \qquad \phi_0 = \beta_0 + \sum_j \beta_j\,\mathbb{E}[x_j]
$$

Giá trị SHAP của mô hình tuyến tính là hệ số nhân với độ lệch so với giá trị trung bình; kết quả chính xác và tính gần như tức thời. Đại lượng được giải thích là log-odds (hàm quyết định), không phải xác suất.

**Tham số chính (shap 0.46):**

- `masker`: truyền trực tiếp `X_background` hoặc `shap.maskers.Independent(X_background)` để áp dụng công thức độc lập ở trên. `shap.maskers.Impute(X_background)` cho phép xét tương quan giữa các đặc trưng, nhưng khi đó $\phi_j$ không còn là $\beta_j(x_j - \bar x_j)$ và đóng góp có thể bị phân tán sang các biến tương quan.
- Với dự án, masker Independent được ưu tiên để giữ diễn giải trực tiếp theo hệ số, kết hợp tổng hợp theo nhóm đặc trưng để giảm ảnh hưởng của tương quan.

**Lưu ý.** Nếu pipeline có `StandardScaler`, SHAP được tính trên thang đã chuẩn hóa nhưng đơn vị của $\phi$ vẫn là log-odds; giá trị gốc của $x$ chỉ cần hiển thị kèm theo khi trình bày.

```python
logreg = pipe[-1]
X_bg = pipe[:-1].transform(X_train_sample)
explainer = shap.LinearExplainer(logreg, X_bg)   # masker Independent
sv = explainer(pipe[:-1].transform(X_valid_sample))
```

### 2.3. KernelExplainer

**Phạm vi:** mô hình bất kỳ không có explainer chuyên dụng (ví dụ MLP tùy chọn), hoặc dùng để đối chiếu kết quả của Tree/Linear trên một số mẫu.

**Thuật toán.** Kernel SHAP ước lượng giá trị Shapley bằng hồi quy tuyến tính có trọng số trên các liên minh $z' \in \{0,1\}^M$ với trọng số Shapley kernel:

$$
\pi(z') = \frac{M-1}{\binom{M}{|z'|}\,|z'|\,(M-|z'|)}
$$

Đặc trưng vắng mặt được thay bằng giá trị từ dữ liệu nền (giả định độc lập).

**Chi phí.** Mỗi dòng cần `nsamples` liên minh (mặc định `2*M + 2048`) nhân với số dòng dữ liệu nền lần gọi mô hình. Với $M \approx 30$ và nền 100 dòng, một khách hàng cần khoảng 200.000 lần `predict`, tương ứng vài giây đến hàng chục giây. Kết quả là xấp xỉ ngẫu nhiên, có phương sai.

**Cấu hình thường dùng:**

- Tóm tắt dữ liệu nền bằng `shap.kmeans(X_train_t, 50)` hoặc `shap.sample(X_train_t, 100)`.
- Dùng `link="logit"` khi hàm mô hình trả về xác suất, để SHAP nằm trên thang log-odds như hai explainer còn lại.
- Không sử dụng trong demo trực tiếp (theo quy ước dự án); kết quả cần thiết được tính trước và lưu đệm.

```python
f = lambda X: model.predict_proba(X)[:, 1]
bg = shap.kmeans(X_train_t, 50)
explainer = shap.KernelExplainer(f, bg, link="logit")
phi = explainer.shap_values(X_valid_t[:20], nsamples=500)
```

### 2.4. So sánh

| | TreeExplainer | LinearExplainer | KernelExplainer |
|---|---|---|---|
| Loại mô hình | Cây, ensemble cây | Tuyến tính (LogReg, Scorecard) | Bất kỳ |
| Độ chính xác | Chính xác | Chính xác | Xấp xỉ (lấy mẫu) |
| Thời gian / 1 dòng | Mili-giây | Micro-giây | Giây đến chục giây |
| Dữ liệu nền | Không (path-dependent) / Có (interventional) | Có (để tính $\mathbb{E}[x]$) | Có |
| Thang đầu ra mặc định | Log-odds (boosting), xác suất (RF sklearn) | Log-odds | Theo `link` |
| Dùng trong demo | Có | Có | Không |

### 2.5. SHAP sau hiệu chuẩn (Platt trên margin)

Theo quy ước dự án, PD hiển thị là $\text{PD} = \sigma(a \cdot m + b)$, với $m$ là margin thô của mô hình. Do TreeExplainer cho $m = \phi_0 + \sum \phi_i$:

$$
\text{logit}(\text{PD}) = a\,m + b = \underbrace{(a\,\phi_0 + b)}_{\phi_0^{cal}} + \sum_i \underbrace{a\,\phi_i}_{\phi_i^{cal}}
$$

Như vậy, nhân SHAP với hệ số $a$ và đặt base value mới bằng $a\phi_0 + b$ vẫn bảo toàn tính cộng tính: tổng bằng đúng logit(PD) hiển thị. Với hiệu chuẩn Isotonic, phép biến đổi phi tuyến nên tính chất này không còn; demo cần ghi chú rõ.

Đối với RandomForest của sklearn, `model_output="raw"` là xác suất chứ không phải margin. Nếu RF là mô hình cuối, Platt cần được fit trên $\text{logit}(p_{RF})$ (với $p$ được clip về $[\varepsilon, 1-\varepsilon]$) và SHAP không còn cộng tuyến tính chính xác. Vì vậy các mô hình boosting phù hợp hơn cho vai trò mô hình cuối.

---

## 3. Logistic Scorecard (WoE)

Scorecard là phương pháp chuẩn trong chấm điểm tín dụng: mô hình Logistic được huấn luyện trên các biến đã rời rạc hóa (binning) và mã hóa Weight of Evidence (WoE); mỗi bin sau đó được quy thành một số điểm cộng dồn. Ưu điểm gồm tính minh bạch, dễ kiểm toán và khả năng kiểm soát chiều ảnh hưởng (đơn điệu) của từng biến.

### 3.1. Quy trình

1. **Fine classing:** chia mỗi biến liên tục thành 10–20 bin theo phân vị.
2. **Coarse classing:** gộp các bin lân cận sao cho (i) WoE đơn điệu theo biến, hoặc có ý nghĩa nghiệp vụ như dạng chữ U của tuổi; (ii) mỗi bin chứa tối thiểu khoảng 5% quan sát và có cả khách tốt lẫn xấu; (iii) các bin khác nhau rõ rệt về tỷ lệ vỡ nợ.
3. **Giá trị đặc biệt:** các mã `PAY_x ∈ {-2, -1, 0}` (không dùng thẻ, trả đủ, trả tối thiểu), `EDUCATION ∈ {0, 5, 6}`, `MARRIAGE = 0` và `BILL_AMT ≤ 0` được tách thành bin riêng hoặc gộp có chủ đích, không để thuật toán binning trộn lẫn với giá trị thông thường.
4. **Tính WoE và IV**, sau đó lọc biến theo IV.
5. **Hồi quy Logistic** trên các cột WoE.
6. **Quy đổi điểm** cho từng bin bằng PDO (mục 4.4).

### 3.2. Weight of Evidence và Information Value

Với bin $k$ của một biến, gọi $G_k, B_k$ là số khách tốt và xấu trong bin, $G, B$ là tổng số khách tốt và xấu:

$$
\text{WoE}_k = \ln\!\left(\frac{G_k / G}{B_k / B}\right)
\qquad
\text{IV} = \sum_k \left(\frac{G_k}{G} - \frac{B_k}{B}\right)\text{WoE}_k
$$

- WoE > 0: bin có tỷ lệ khách tốt cao hơn trung bình (rủi ro thấp); WoE < 0: rủi ro cao.
- Bin có $G_k = 0$ hoặc $B_k = 0$ cho WoE vô hạn; được xử lý bằng cách gộp bin hoặc làm mịn (cộng 0,5 vào tử và mẫu).

Ngưỡng IV thường dùng (Siddiqi, 2006):

| IV | Sức dự báo |
|---|---|
| < 0,02 | Không có |
| 0,02 – 0,1 | Yếu |
| 0,1 – 0,3 | Trung bình |
| 0,3 – 0,5 | Mạnh |
| > 0,5 | Rất mạnh; cần kiểm tra khả năng leakage hoặc biến quá gần với target |

Trong bộ dữ liệu Taiwan, `PAY_1` được kỳ vọng có IV rất cao (> 0,5) do trạng thái trễ hạn của tháng gần nhất liên quan chặt với nhãn "vỡ nợ tháng kế tiếp". Đây là tín hiệu dự báo thực sự chứ không phải leakage, nhưng cần được ghi chú trong báo cáo.

### 3.3. Hồi quy trên WoE

Mô hình dự báo log-odds xấu/tốt:

$$
\ln\frac{P(\text{bad})}{P(\text{good})} = \beta_0 + \sum_j \beta_j\,\text{WoE}_j(x_j)
$$

Do WoE được định nghĩa theo $\ln(\text{good}/\text{bad})$, mọi hệ số $\beta_j$ kỳ vọng **âm**. Hệ số dương là dấu hiệu của đa cộng tuyến (thường giữa các biến `PAY_x` hoặc `BILL_AMT_x`), xử lý bằng cách loại bớt biến trong nhóm tương quan rồi fit lại. WoE đã ở thang log-odds nên không cần chuẩn hóa thêm; có thể áp dụng regularization L2 nhẹ, và chọn biến theo IV kết hợp tương quan hoặc stepwise.

### 3.4. Chống rò rỉ dữ liệu

Binning, WoE và IV đều sử dụng target, nên phải được fit trong từng fold của CV, đặt trong `sklearn.pipeline.Pipeline` dưới dạng transformer (theo `project_plan.md` mục 3.5b). Nếu IV được tính trên toàn bộ dữ liệu trước khi CV, AUC của CV bị ước lượng cao hơn thực tế.

### 3.5. Công cụ

| Lựa chọn | Ưu điểm | Hạn chế |
|---|---|---|
| **optbinning** (`BinningProcess`, `Scorecard`) | Binning tối ưu có ràng buộc đơn điệu, hỗ trợ giá trị đặc biệt, có sẵn Scorecard và PDO, tương thích sklearn Pipeline | Chưa có trong `requirements.txt`; cần bổ sung, pin phiên bản và đồng bộ với cả nhóm |
| **scorecardpy** | Đơn giản, phổ biến trong tài liệu hướng dẫn | Không theo API của sklearn, khó tích hợp vào Pipeline/CV |
| **Tự cài đặt** transformer WoE | Kiểm soát hoàn toàn, không thêm dependency | Tốn thời gian, dễ sai ở các trường hợp biên |

Phương án đề xuất: `optbinning`, do Thành viên A (phụ trách Scorecard) đánh giá và quyết định.

### 3.6. Liên hệ giữa Scorecard và SHAP

Scorecard là mô hình tuyến tính nên với LinearExplainer (Independent):

$$
\phi_j = \beta_j\,\big(\text{WoE}_j(x_j) - \overline{\text{WoE}_j}\big)
$$

Điểm của bin (mục 4.4) và SHAP chỉ khác nhau một hằng số cộng và hệ số $-\text{Factor}$. Nói cách khác, "điểm theo thuộc tính" của scorecard truyền thống là một dạng giải thích cộng tính, cho phép so sánh trực tiếp reason codes của Scorecard với các mô hình boosting.

---

## 4. Phương pháp PDO (Points to Double the Odds)

### 4.1. Nguyên lý

Điểm tín dụng là hàm tuyến tính của log-odds tốt/xấu, được xây dựng sao cho:

- tại một mức odds gốc (**Base Odds**), điểm bằng **Base Score**;
- mỗi lần điểm tăng thêm **PDO**, odds tốt/xấu tăng gấp đôi.

$$
\text{Score} = \text{Offset} + \text{Factor}\cdot\ln(\text{odds}), \qquad \text{odds} = \frac{1-\text{PD}}{\text{PD}}
$$

### 4.2. Suy ra Factor và Offset

Từ hai điều kiện trên:

$$
\begin{cases}
\text{Base Score} = \text{Offset} + \text{Factor}\cdot\ln(\text{Base Odds})\\
\text{Base Score} + \text{PDO} = \text{Offset} + \text{Factor}\cdot\ln(2\cdot\text{Base Odds})
\end{cases}
\Rightarrow
\text{Factor} = \frac{\text{PDO}}{\ln 2}, \quad
\text{Offset} = \text{Base Score} - \text{Factor}\cdot\ln(\text{Base Odds})
$$

### 4.3. Tham số của dự án

Với Base Score = 600 tại odds 50:1 và PDO = 20:

- Factor = 20 / ln 2 ≈ **28,854**
- Offset = 600 − 28,854 × ln 50 ≈ **487,123**

$$
\text{Score} \approx 487{,}123 + 28{,}854\cdot\ln\frac{1-\text{PD}}{\text{PD}}
$$

Bảng quy đổi (chưa clip):

| PD | Odds tốt:xấu | Điểm |
|---|---|---|
| 0,5% | 199 : 1 | 639,9 |
| 1% | 99 : 1 | 619,7 |
| 1,96% | 50 : 1 | **600,0** |
| 5% | 19 : 1 | 572,1 |
| 10% | 9 : 1 | 550,5 |
| 22,1% (tỷ lệ vỡ nợ trung bình của dữ liệu) | 3,5 : 1 | 523,4 |
| 30% | 2,3 : 1 | 511,6 |
| 50% | 1 : 1 | 487,1 |
| 70% | 0,43 : 1 | 462,7 |
| 90% | 0,11 : 1 | 423,7 |

**Về thang 300–850.** Điểm 300 tương ứng PD ≈ 99,85% và điểm 850 tương ứng PD ≈ 0,0003%. Với PD thực tế của bộ dữ liệu (khoảng 1%–90%), điểm nằm trong khoảng ~420–640, do đó thao tác clip hầu như không xảy ra và thang hiển thị 300–850 sẽ có phần lớn diện tích không được sử dụng. Hai hướng xử lý:

1. Giữ nguyên tham số (đúng chuẩn, dễ diễn giải) và hiển thị thanh điểm trên demo theo khoảng giá trị thực tế.
2. Tăng PDO (ví dụ 40–50) để trải rộng điểm. Đây là lựa chọn về cách hiển thị, không ảnh hưởng đến mô hình, và cần được thống nhất giữa các thành viên trước khi ghi vào `configs/config.yaml`.

### 4.4. Phân bổ điểm cho từng thuộc tính (Scorecard)

Từ mục 3.3, $\ln(\text{odds tốt}) = -(\beta_0 + \sum_j \beta_j \text{WoE}_j)$. Với $n$ biến:

$$
\text{Score} = \sum_{j=1}^{n}\underbrace{\left[-\left(\beta_j\,\text{WoE}_{j,k} + \frac{\beta_0}{n}\right)\text{Factor} + \frac{\text{Offset}}{n}\right]}_{\text{điểm của bin } k \text{ thuộc biến } j}
$$

Kết quả là bảng tra: mỗi biến, mỗi bin tương ứng một số điểm (làm tròn thành số nguyên). Điểm của khách hàng là tổng điểm của các bin mà khách hàng đó thuộc về.

### 4.5. Quy đổi SHAP sang điểm (phục vụ demo)

Do Score tuyến tính theo logit(PD) và SHAP sau hiệu chuẩn (mục 2.5) cộng đúng bằng logit(PD), đóng góp của từng đặc trưng quy đổi được trực tiếp sang điểm. Cần lưu ý logit(PD) là log-odds **xấu/tốt**, ngược dấu với odds trong công thức PDO:

$$
\text{Score} = \underbrace{\text{Offset} - \text{Factor}\cdot\phi_0^{cal}}_{\text{điểm nền}} \;+\; \sum_i \underbrace{\left(-\text{Factor}\cdot\phi_i^{cal}\right)}_{\text{điểm đóng góp của đặc trưng } i}
$$

Cách quy đổi này cho phép demo hiển thị dạng "Lịch sử trễ hạn: −35 điểm; Mức sử dụng hạn mức: −12 điểm; …", với tổng khớp điểm cuối (trước khi clip).

### 4.6. Phác thảo `src/scoring.py`

```python
import numpy as np

def pdo_params(base_score=600, base_odds=50, pdo=20):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset

def pd_to_score(pd, base_score=600, base_odds=50, pdo=20, clip=(300, 850), eps=1e-6):
    factor, offset = pdo_params(base_score, base_odds, pdo)
    pd = np.clip(pd, eps, 1 - eps)
    score = offset + factor * np.log((1 - pd) / pd)
    return np.clip(score, *clip) if clip else score

def shap_to_points(shap_logit, pdo=20):
    """SHAP trên thang logit(PD) đã hiệu chuẩn -> điểm đóng góp (dương = tăng điểm)."""
    return -shap_logit * pdo / np.log(2)
```

Các trường hợp kiểm thử đề xuất: `pd_to_score(1/51) == 600`; điểm tại odds 100:1 bằng 620; điểm giảm đơn điệu theo PD; tổng `shap_to_points` cộng điểm nền khớp `pd_to_score(..., clip=None)`.

---

## 5. Kết luận và đề xuất

1. **Explainer theo loại mô hình:** boosting/cây dùng `TreeExplainer` (đầu ra log-odds; path-dependent cho demo, interventional với nền 500 dòng cho phân tích toàn cục nhằm đối chiếu); LogReg/Scorecard dùng `LinearExplainer` (masker Independent); `KernelExplainer` chỉ dùng để kiểm tra chéo hoặc cho MLP, tính trước và không chạy trực tiếp.
2. **Thang giải thích thống nhất:** SHAP được quy về logit(PD) đã hiệu chuẩn (nhân hệ số Platt $a$), sau đó quy sang điểm bằng hệ số $-\text{Factor}$. `src/explain.py` cần chuẩn hóa shape đầu ra (RandomForest trả về `(n, M, 2)`).
3. **Reason codes theo nhóm đặc trưng** thay vì từng biến, do tương quan cao trong `PAY_x` và `BILL_AMT_x`.
4. **Scorecard:** bổ sung `optbinning` vào `requirements.txt`; binning/WoE fit trong từng fold; kiểm tra dấu hệ số (kỳ vọng âm).
5. **PDO:** giữ tham số 600 tại odds 50:1, PDO = 20 như kế hoạch; điểm thực tế nằm trong khoảng ~420–640. Cách hiển thị thang điểm trên demo cần được thống nhất (mục 4.3) trước khi đưa các tham số vào `configs/config.yaml`.

## Tài liệu tham khảo

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Lundberg, S. M., et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56–67.
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). *Feature relevance quantification in explainable AI: A causal problem.* AISTATS.
- Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). *True to the Model or True to the Data?* arXiv:2006.16234.
- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications.* SIAM.
- SHAP documentation: https://shap.readthedocs.io
- optbinning documentation: https://gnpalencia.org/optbinning/
- optbinning source code: https://github.com/guillermo-navas-palencia/optbinning
