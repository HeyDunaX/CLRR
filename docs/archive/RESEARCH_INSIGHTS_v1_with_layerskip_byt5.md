# CƠ SỞ LÝ THUYẾT, INSIGHT KHOA HỌC & LUẬN ĐIỂM BÀI BÁO (CLRR RESEARCH INSIGHTS)

> **Tài liệu nguồn lưu trữ tập trung cơ sở lý thuyết, minh chứng toán học, phân tích hình thái học định tính và luận điểm khoa học sẵn sàng phục vụ cho bản thảo bài báo ACL / ComputEL-10.**  
> *Dự án: Parameter-Neutral Cross-Layer Residual Routing and Latent Regularization for Low-Resource Polysynthetic Translation (CLRR)*  
> *Cập nhật mới nhất: Tháng 9/2026*

---

## 1. Định Vị Học Thuật & Đặt Vấn Đề (Problem Statement)

* **Tên bài báo:** *Parameter-Neutral Cross-Layer Residual Routing and Latent Regularization for Low-Resource Polysynthetic Translation*
* **Hội thảo đích:** **ComputEL-10 (2027)** / **ACL 2025-2026**
* **Cặp ngôn ngữ:** Tiếng Amis (Nam Đảo Formosan, có nguy cơ mai một theo UNESCO) $\to$ Tiếng Trung (Mandarin - Chinese, ngôn ngữ phân tích cô lập).
* **Thách thức cốt lõi:**
  1. **Thiếu hụt dữ liệu cực đoan:** Toàn bộ ngữ liệu chỉ có **5.751 cặp câu song ngữ**.
  2. **Bất đối xứng hình thái học sâu sắc:** Tiếng Amis chắp dính đa tổng hợp (polysynthetic) với hệ thống phụ tố dày đặc (*mi-*, *ma-*, *pa-*, *-en*, *-om-*), trong khi tiếng Trung là ngôn ngữ đơn lập không biến hình từ.
  3. **Hiện tượng Over-smoothing ở Transformer sâu:** Các tầng Encoder sâu dần làm mờ các phụ tố tầng nông thành vector trừu tượng chung chung, khiến Decoder bỏ sót động từ hoặc dịch sai vai trò ngữ nghĩa (Agent vs Patient).
  4. **Nghịch lý của PEFT (LoRA/Adapters):** Bổ sung thêm tham số huấn luyện ($\Delta\theta > 0$) trên dữ liệu $< 6.000$ câu gây overfitting trầm trọng.
* **Giải pháp CLRR:** Can thiệp thuần túy vào luồng truyền trạng thái ẩn ($d=2, \alpha=0.1, \text{stop\_gradient}$) kết hợp căn chỉnh tiềm ẩn LSR/JEPA mà **hoàn toàn không thêm bất kỳ tham số huấn luyện nào ($\Delta\theta = 0$)**.

---

## 2. Minh Chứng Thực Nghiệm Hiện Tượng Over-smoothing (Figure 2)

### 2.1. Phương pháp đo đạc định lượng
Đo lường bằng chỉ số **Average Pairwise Within-Sentence Token Cosine Similarity** qua từng tầng Encoder $l \in [1, 8]$ trên toàn bộ 575 câu tập Test:
$$\text{CosSim}(l) = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{h_{i,l}^\top h_{j,l}}{\|h_{i,l}\|_2 \|h_{j,l}\|_2}$$

### 2.2. Bảng số liệu thống kê chi tiết (`outputs_extra/analysis/cosine_by_layer.csv`)

| Encoder Layer | mT5 Baseline (Mean Cosine) | mT5 CLRR-Enc (Mean Cosine) | Chênh lệch ($\Delta = \text{CLRR} - \text{Base}$) | Diễn giải cơ chế hoạt động |
| :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | 0.3319 | 0.3345 | +0.0026 | Tương đương (chưa có kết nối tắt) |
| **Layer 2** | 0.3804 | 0.3807 | +0.0003 | Bắt đầu trích xuất đặc trưng cơ bản |
| **Layer 3** | 0.4485 | 0.4357 | **-0.0128** | CLRR bắt đầu nhận skip từ Layer 1 ($d=2$) |
| **Layer 4** | 0.5055 | 0.4901 | **-0.0154** | Đa dạng hóa biểu diễn, giảm bớt over-smoothing |
| **Layer 5** | 0.5361 | 0.5218 | **-0.0143** | Giữ vững thông tin hình thái |
| **Layer 6** | 0.5694 | 0.5482 | **-0.0212** | Giảm mạnh độ tương đồng token dư thừa |
| **Layer 7** | 0.5977 | 0.5756 | **-0.0221** | Ngăn chặn hiện tượng sụp đổ không gian ẩn |
| **Layer 8** | **0.6277** | **0.6033** | **-0.0244** | **Bảo tồn thành công độ phân tách token trước khi qua Decoder** |

*Ý nghĩa học thuật:* Ở baseline, cosine similarity tăng liên tục từ 0.33 lên 0.63 (các token trở nên quá giống nhau). CLRR kéo độ tương đồng này xuống liên tục ở tất cả các tầng sâu, duy trì sự khác biệt hình thái rõ nét.

---

## 3. Đạo Hàm & Chứng Minh Toán Học Của Toán Tử Stop-Gradient ($\text{sg}$)

### 3.1. Lan truyền Gradient khi KHÔNG có Stop-Gradient (Bùng nổ đường đi)
Nếu không dùng stop-gradient, công thức nối tắt là:
$$h'_i = h_i + \alpha \cdot h'_{i-d} = \mathcal{F}_i(h'_{i-1}) + \alpha \cdot h'_{i-d}$$
Theo quy tắc dây chuyền đa nhánh (multi-path chain rule), gradient của hàm mất mát $\mathcal{L}$ đối với trạng thái ẩn sớm $h'_k$ mở rộng theo tổ hợp tất cả các đường đi có hướng từ tầng $k$ đến tầng $N$:
$$\frac{\partial \mathcal{L}}{\partial h'_k} = \sum_{\pi \in \Pi(k, N)} \prod_{(u, v) \in \pi} \frac{\partial h'_v}{\partial h'_u}$$
Số lượng đường đi $|\Pi(k, N)|$ bùng nổ theo công thức tổ hợp:
$$|\Pi(k, N)| = \sum_{m=0}^{\lfloor (N-k)/d \rfloor} \binom{N-k - m(d-1)}{m}$$
* Với mô hình 12 tầng ($N=12, d=2$), $|\Pi(0, 12)| = \mathbf{233\text{ đường đi độc lập}}$.
* Chuẩn gradient bị chặn trên bởi:
$$\left\| \frac{\partial \mathcal{L}}{\partial h'_k} \right\| \le \left\| \frac{\partial \mathcal{L}}{\partial h'_N} \right\| \cdot (1 + \alpha + \|\mathcal{J}\|)^{N-k}$$
Trong môi trường siêu ít dữ liệu ($< 6.000$ câu), sự nhân đôi và bùng nổ phương sai gradient này khiến việc tối ưu hóa bị mất ổn định nghiêm trọng (đã kiểm chứng qua Task 3).

### 3.2. Lan truyền Gradient khi CÓ Stop-Gradient (Ổn định tuyệt đối)
Bằng việc áp dụng $\text{sg}(\cdot)$, ta triệt tiêu đạo hàm nhánh tắt: $\frac{\partial \text{sg}(h'_{i-d})}{\partial h'_{i-d}} = 0$. Jacobian của phép biến đổi thu gọn thành:
$$\frac{\partial h'_i}{\partial h'_{i-d}} = \frac{\partial h_i}{\partial h'_{i-d}} + 0$$
Do đó, luồng gradient chỉ truyền dọc duy nhất theo chuỗi tuần tự chuẩn:
$$\frac{\partial \mathcal{L}}{\partial h'_k} = \frac{\partial \mathcal{L}}{\partial h'_N} \prod_{j=k+1}^N \left( I + \frac{\partial f_j}{\partial h'_{j-1}} \right)$$
Trạng thái $h'_{i-d}$ được bơm vào như một hằng số biểu diễn cục bộ ở forward pass, làm phong phú tầng sâu mà hoàn toàn không gây nhiễu loạn backward pass.

---

## 4. Phân Tích Cơ Chế Các Phương Pháp Đối Chuẩn ACL Gần Đây

### 4.1. LayerSkip (Elhoushi et al., ACL 2024 Long Paper)
* **Cơ chế:** Áp dụng **Exponential Layer Dropout** trên Encoder ($p_l = p_{\max} \cdot \frac{l}{L-1}$) kết hợp các đầu phân loại thoát sớm (Early-Exit Heads) cùng hàm mất mát Curriculum Loss:
$$\mathcal{L}_{\text{total}} = \sum_{l=0}^{L-1} c_l \cdot \mathcal{L}_{\text{CE}}^{(l)}$$
* **Hạn chế bộc lộ:** Rất hiệu quả trên mô hình subword (mT5, mBART), nhưng **sụp đổ thảm họa trên ByT5** do ngắt quãng chuỗi 3-byte Hán tự.

### 4.2. Middle-Layer Alignment (Liu & Niehues, ACL 2025 Long Paper)
* **Cơ chế:** Căn chỉnh không gian tiềm ẩn tại tầng giữa (Middle Layer, tầng 4 của mT5 hoặc tầng 6 của mBART/ByT5) bằng hàm mất mát Cosine Contrastive Loss:
$$\mathcal{L}_{\text{align}} = - \frac{1}{|\mathcal{B}|} \sum_{(s, t) \in \mathcal{B}} \log \frac{\exp(\text{sim}(h_s^i, h_t^i) / \tau)}{\sum_{v \in \mathcal{B}} \exp(\text{sim}(h_s^i, h_v^i) / \tau)}$$
* **Ưu điểm:** Tăng tốc hội tụ nhanh ở những epoch đầu và hoạt động an toàn trên cả mô hình byte-level.
* **So với CLRR:** Mid-Align chỉ căn chỉnh biểu diễn tại tầng giữa mà không thay đổi cấu trúc kết nối trạng thái ẩn, trong khi CLRR liên tục dẫn truyền phụ tố tầng nông tới thẳng Decoder.

---

## 5. Cơ Chế Suy Biến Byte UTF-8 Của Đầu Thoát Sớm Trên ByT5

* **Cơ chế mã hóa UTF-8:** Trong tiếng Trung, mỗi chữ Hán cấu thành từ đúng **3 byte liên tiếp** (ví dụ chữ 人 là `0xE4 0xBA 0xBA`, gồm 1 lead byte `0xE4` và 2 continuation bytes `0xBA 0xBA`).
* **Tại sao LayerSkip sụp đổ (-5.03 BLEU):** 
  * Khi áp dụng layer dropout và bắt các tầng nông phải dự đoán ký tự đầu ra, tầng nông với trường tiếp nhận (receptive field) còn hạn chế bị ép phải phát ra từng byte rời rạc.
  * Việc này làm đứt gãy sự liên kết 3-byte, khiến Decoder bị rối loạn không gian nhúng và rơi vào **vòng lặp vô tận (Repetition Loops)**: tần suất lặp 4-gram tăng vọt từ 0.36 lên 0.62 (+72%) và độ dài chuỗi sinh tăng +13.6%.
* **Tại sao CLRR an toàn:** CLRR duy trì đồ thị liên tục không ngắt tầng, trạng thái ẩn được dẫn truyền nguyên vẹn giúp ByT5 giữ vững mạch sinh ký tự tự nhiên (tỷ lệ lặp 0.3688, tương đương baseline 0.3621).

---

## 6. Insight Bóc Tách Thành Phần: Hiệu Ứng Hiệp Đồng Bắt Buộc (Compounding Synergy)

Bộ thực nghiệm bóc tách mBART-50 (Ablation Suite) đã mang lại phát hiện mang tính quy luật:
1. **CLRR-only ($\alpha=0.1, \lambda=0$ - 18.47 BLEU / 13.24 chrF++):** Việc bơm tín hiệu tầng nông lên tầng sâu trong một mô hình khổng lồ 611M tham số mà không có hàm mất mát định hướng không gian ngữ nghĩa chung sẽ khiến biểu diễn bị phân tán nhẹ.
2. **LSR-only ($\alpha=0, \lambda=0.1$ - 20.08 BLEU / 16.07 chrF++):** Đóng vai trò là chiếc mỏ neo ngữ nghĩa vững chắc, giúp cải thiện +0.47 BLEU và +2.02 chrF++.
3. **Full CLRR-Enc + LSR ($\alpha=0.1, \lambda=0.1$ - 20.39 BLEU / 19.08 chrF++):**
   * **Hiệp đồng hoàn hảo:** LSR neo giữ không gian ngữ nghĩa toàn cục, tạo điều kiện cho các luồng tắt của CLRR truyền trọn vẹn các phụ tố hình thái tầng nông lên Decoder mà không sợ bị trôi dạt ngữ nghĩa.
   * Kết quả kích hoạt mức nhảy vọt **+5.03 chrF++ ($p < 0.001$)**, đặc biệt bứt phá **+8.81 chrF++ trên các câu chứa tiền tố *ma-***.

---

## 7. Nghiên Cứu Điển Hình Định Tính (Qualitative Translation Case Studies)

| STT | Câu Nguồn (Amis) & Phân Tích Hình Thái | Bản Dịch Tham Chiếu (Reference) | Vanilla Baseline | CLRR + LSR (Đề xuất) | Phân Tích Ngôn Ngữ Học |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | ***Mi*nengneng koya wawa to cudad.**<br>(*mi-*: Actor Voice; look child that book) | 那個小孩在看書。<br>*(That child is reading a book.)* | 那裡的小孩書。<br>*(Bỏ sót vị ngữ: "there child book")* | **那個小孩在看書。**<br>*(Đầy đủ vị ngữ tiến hành & chủ thể)* | Baseline làm mất tiền tố chủ động *mi-*, CLRR giữ đúng động từ "đang đọc sách". |
| **2** | ***Ma*futi' ko wawa i sasingalan.**<br>(*ma-*: Stative Voice; sleep child at window) | 小孩在窗邊睡著了。<br>*(Child fell asleep by window.)* | 小孩子窗戶。<br>*(Mất vị ngữ: "child window")* | **小孩在窗邊睡覺。**<br>*(Bắt đúng vị ngữ trạng thái ngủ)* | Baseline làm rơi rụng tiền tố trạng thái *ma-*, CLRR chuyển hóa chính xác thành "ngủ bên cửa sổ". |
| **3** | ***Pa*nanom ko ina to wawa.**<br>(*pa-*: Causative Prefix; water mother child) | 媽媽給小孩喝水。<br>*(Mother gives child water to drink.)* | 媽媽水小孩。<br>*(Trật tự danh từ gãy đổ)* | **媽媽讓小孩喝水。**<br>*(Cấu trúc vị ngữ cầu khiến chuẩn)* | Tiền tố sai khiến *pa-* được CLRR dịch chuẩn xác sang cấu trúc "để/cho ai làm gì". |
| **4** | **Ira ko kaka i *pi*kikayan.**<br>(*pi-*: Locative base; machine-LOC = airport/station) | 哥哥在車站。<br>*(Elder brother is at station.)* | 哥哥機器。<br>*(Dịch nhầm từ căn thành "máy móc")*| **哥哥在車站裡。**<br>*(Dịch chuẩn danh từ hóa vị trí)* | Baseline dịch từ căn thô, CLRR nhận diện tiền tố danh từ hóa chỉ địa điểm *pi-*. |
| **5** | ***Ta-tosa* kami a tayra i riyar.**<br>(*ta-tosa*: Collective numeral; two together) | 我們兩個人一起去海邊。<br>*(Two of us go to beach together.)* | 我們去海邊。<br>*(Mất tiền tố tập hợp số lượng)* | **我們兩人一起去海邊。**<br>*(Giữ trọn vẹn lượng từ hai người)* | CLRR bảo tồn tiền tố số từ tập hợp *ta-* trong khi baseline bỏ sót hoàn toàn. |
