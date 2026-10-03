# TÀI LIỆU KỸ THUẬT THAM KHẢO & ĐẶC TẢ HAI PHƯƠNG PHÁP PEFT: LORA & BITFIT

Tài liệu này tổng hợp toàn bộ thông tin học thuật, đường dẫn file PDF đã tải về, trích dẫn BibTeX chính thức và công thức toán học/code chuẩn bị cho việc triển khai bộ đối chuẩn PEFT (**Ưu tiên 1**) trên `mBART-large-50`.

---

## 1. Danh Mục File PDF Đã Tải Về Trong Thư Mục `docs/`

| Phương pháp | Hội nghị / Năm | File PDF Trong Kho | Kích thước | Nguồn gốc chính thức |
| :--- | :---: | :--- | :---: | :--- |
| **LoRA** | **ICLR 2022** | [`docs/baseline-paper/hu2022_lora_2106.09685.pdf`](file:///d:/Code/CLRR/docs/baseline-paper/hu2022_lora_2106.09685.pdf) | 1.6 MB | arXiv:2106.09685 / OpenReview |
| **BitFit** | **ACL 2022 (Short)** | [`docs/baseline-paper/benzaken2022_bitfit_2022.acl-short.1.pdf`](file:///d:/Code/CLRR/docs/baseline-paper/benzaken2022_bitfit_2022.acl-short.1.pdf) | 328 KB | ACL Anthology: 2022.acl-short.1 |

---

## 2. Thông Tin Trích Dẫn BibTeX Chính Thức (Đã cập nhật vào `clrr_references.bib`)

### 2.1. LoRA (Hu et al., ICLR 2022)
```bibtex
@inproceedings{hu-etal-2022-lora,
  title = "Lo{RA}: Low-Rank Adaptation of Large Language Models",
  author = "Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu",
  booktitle = "International Conference on Learning Representations (ICLR 2022)",
  year = "2022",
  url = "https://openreview.net/forum?id=nZeVKeeFYf9"
}
```

### 2.2. BitFit (Ben-Zaken et al., ACL 2022)
```bibtex
@inproceedings{ben-zaken-etal-2022-bitfit,
  title = "{B}it{F}it: Simple Parameter-efficient Fine-tuning for {T}ransformer-based Masked Language-models",
  author = "Ben-Zaken, Elad and Ravfogel, Shauli and Goldberg, Yoav",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
  month = may,
  year = "2022",
  address = "Dublin, Ireland",
  publisher = "Association for Computational Linguistics",
  pages = "1--9",
  doi = "10.18653/v1/2022.acl-short.1",
  url = "https://aclanthology.org/2022.acl-short.1"
}
```

---

## 3. Đặc Tả Toán Học & Thiết Kế Kiến Trúc

### 3.1. LoRA: Low-Rank Adaptation (ICLR 2022)
* **Bản chất toán học:**
  Giả sử ma trận trọng số tiền huấn luyện là $W_0 \in \mathbb{R}^{d \times k}$. Thay vì cập nhật trực tiếp toàn bộ $W_0$, LoRA phân rã ma trận cập nhật $\Delta W$ thành tích của hai ma trận hạng thấp (low-rank matrices):
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
  trong đó $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$ với rank $r \ll \min(d, k)$.
* **Quy tắc khởi tạo:**
  * Ma trận $A$ được khởi tạo ngẫu nhiên theo phân phối chuẩn Gaussian: $A \sim \mathcal{N}(0, \sigma^2)$.
  * Ma trận $B$ được khởi tạo bằng **0**: $B = 0$.
  * Hệ quả: Tại bước huấn luyện đầu tiên ($t=0$), $\Delta W = B \cdot A = 0$, mô hình hoàn toàn tương đương với mô hình tiền huấn luyện gốc ban đầu.
* **Hệ số co giãn (Scaling factor $\frac{\alpha}{r}$):**
  * Thường chọn $r = 8$ hoặc $r = 4$, $\alpha = 16$ hoặc $2r$. Hệ số này giúp ổn định gradient khi thử nghiệm các giá trị rank $r$ khác nhau.
* **Module can thiệp trên mBART-50:**
  * Áp dụng trên các ma trận Query và Value chú ý (`q_proj`, `v_proj`) của cả Self-Attention và Cross-Attention (Encoder và Decoder).
  * **Số tham số huấn luyện thêm ($\Delta\theta$):** $\approx 1.54\text{ M}$ tham số (chiếm khoảng **0.25%** tổng số 611M tham số của `mBART-large-50`).

### 3.2. BitFit: Bias-Term Fine-Tuning (ACL 2022)
* **Bản chất toán học:**
  Đóng băng 100% tất cả các ma trận trọng số $W \in \mathbb{R}^{d \times k}$, **chỉ cho phép cập nhật các vector bias $b \in \mathbb{R}^d$**:
  $$y = W_0 x + b, \quad \theta_{\text{trainable}} = \{b\}$$
* **Các vector bias được cập nhật trong Transformer:**
  * Biases của Multi-Head Attention: Query, Key, Value, Output projection.
  * Biases của Feed-Forward Networks: Intermediate dense layer (FC1) và Output dense layer (FC2).
  * Biases của Layer Normalization.
* **Đặc điểm nổi bật:**
  * Không thêm bất kỳ module ngoại lai nào vào đồ thị tính toán (không tăng inference latency, không cần tensor product phụ).
  * **Số tham số huấn luyện thêm ($\Delta\theta$):** $\approx 0.12\text{ M}$ tham số (chiếm khoảng **0.02%** tổng số tham số của `mBART-large-50`).

---

## 4. Code Mẫu Triển Khai Chuẩn Xác (Hugging Face / PyTorch Reference)

### 4.1. Cấu hình LoRA qua thư viện `peft` chính thức:
```python
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_to_mbart(model, r=8, lora_alpha=16, lora_dropout=0.05):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"], # hoặc thêm ["k_proj", "out_proj"]
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
```

### 4.2. Cấu hình BitFit thuần túy bằng PyTorch:
```python
def apply_bitfit_to_model(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    print(f"BitFit Active: Trainable {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.3f}%)")
    return model
```

---

## 5. Ý Nghĩa Học Thuật Khi Đối Chuẩn Với CLRR
1. **BitFit ($\Delta\theta \approx 0.12\text{M}$):** Chỉ tinh chỉnh bias nên dung lượng biểu diễn (capacity) bị giới hạn, khó có thể nắn dòng đặc trưng phức tạp của hiện tượng chắp dính phụ tố Amis (*mi-*, *ma-*).
2. **LoRA ($\Delta\theta \approx 1.54\text{M}$):** Mặc dù giảm đáng kể tham số so với Full Fine-Tuning (611M), nhưng việc học 1.54M tham số mới trên tập huấn luyện cực nhỏ (**4.600 câu**) vẫn rất dễ dẫn đến hiện tượng quá khớp (overfitting) hoặc học mẹo bề mặt.
3. **CLRR-Enc + LSR ($\Delta\theta = 0$):** Không thêm bất kỳ tham số nào cần học mới, mà chỉ **tái định tuyến luồng thông tin còn nguyên vẹn** từ tầng nông lên tầng sâu để chống suy biến biểu diễn (over-smoothing). Đây chính là lý do vì sao CLRR đạt hiệu năng vượt trội (+5.03 chrF++) trên ngữ liệu cực hiếm.
