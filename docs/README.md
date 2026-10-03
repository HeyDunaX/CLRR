# CLRR Documentation Hub

> **Trung tâm tra cứu và điều hướng toàn bộ tài liệu nghiên cứu, bản thảo bài báo, thực nghiệm và vận hành của dự án Cross-Layer Residual Rewiring (CLRR).**

---

## 🗺️ Bản đồ cấu trúc thư mục `docs/`

```text
docs/
├── paper/                      # Bản thảo bài báo, mã nguồn LaTeX, styles ACL & file nộp bài
│   ├── clrr_main.tex           # Bản thảo LaTeX chính thức hiện hành
│   ├── clrr_main.pdf           # Bản PDF biên dịch mới nhất
│   ├── clrr_references.bib     # Danh mục tài liệu trích dẫn BibTeX
│   ├── acl.sty                 # Style chính thức ACL
│   ├── acl_natbib.bst          # Style bibliography ACL
│   ├── clrr_pdflatex_upload.zip# Bundle nộp bài / compile Overleaf/ArXiv
│   ├── baseline-paper/         # Tài liệu bài báo baseline tham chiếu (TADA)
│   ├── proposal.pdf            # Đề cương nghiên cứu ban đầu
│   └── proposal_extracted.txt  # Văn bản trích xuất từ proposal
│
├── experiments/                # Kết quả thực nghiệm, số liệu kiểm toán & đối chuẩn
│   ├── EXPERIMENTS_UPDATED.md  # Sổ kết quả thực nghiệm hiện hành duy nhất (A/B/C)
│   ├── METRICS.md              # Chuẩn hóa quy chuẩn đo đạc (SacreBLEU, chrF++, Bootstrap)
│   └── PEFT_PAPERS_REFERENCE.md# Tổng hợp đối chứng các tài liệu về LoRA / BitFit
│
├── research/                   # Cơ sở lý thuyết, phân tích ngôn ngữ học & kế hoạch
│   ├── RESEARCH_INSIGHTS.md    # Lý thuyết dừng gradient, phân tích hình thái Amis & insights
│   └── CLRR_GO_NOGO_PLAN.md    # Kế hoạch đánh giá ngưỡng Go/No-Go khoa học
│
├── operations/                 # Hướng dẫn vận hành, runbooks & kế hoạch kỹ thuật
│   ├── CONFIGURATION_SEARCH_HANDOFF.md # Runbook bàn giao tối ưu cấu hình mBART Amis
│   ├── COLAB_SSH_GUIDE.md      # Hướng dẫn kết nối SSH và quản trị GPU Colab
│   └── implementation_plan.md  # Kế hoạch triển khai kỹ thuật tổng thể
│
└── archive/                    # Bản lưu trữ lịch sử các phiên bản cũ trước revision
```

---

## 🔍 Bảng tra cứu nhanh theo mục đích sử dụng

| Nhu cầu của bạn | Thư mục trỏ tới | Tài liệu khuyến nghị đọc |
| :--- | :---: | :--- |
| **Xem bản thảo bài báo / Nộp bài** | [`paper/`](paper/) | • [`clrr_main.tex`](paper/clrr_main.tex): Bản thảo LaTeX mới nhất.<br>• [`clrr_main.pdf`](paper/clrr_main.pdf): Bản xem trước PDF hoàn chỉnh.<br>• [`clrr_references.bib`](paper/clrr_references.bib): Danh mục trích dẫn. |
| **Tra cứu bảng số liệu & kết quả** | [`experiments/`](experiments/) | • [`EXPERIMENTS_UPDATED.md`](experiments/EXPERIMENTS_UPDATED.md): Bảng số liệu chuẩn hóa 36 runs & 42 paired bootstrap.<br>• [`METRICS.md`](experiments/METRICS.md): Phân tích chi tiết độ nhạy metric chrF++ và protocol. |
| **Đọc cơ sở lý thuyết & insight** | [`research/`](research/) | • [`RESEARCH_INSIGHTS.md`](research/RESEARCH_INSIGHTS.md): Đạo hàm stop-gradient, phân tích hình thái Amis, hiện tượng vocab dispersion.<br>• [`CLRR_GO_NOGO_PLAN.md`](research/CLRR_GO_NOGO_PLAN.md): Tiêu chuẩn định lượng quyết định hướng đi. |
| **Vận hành Colab, Train & Handoff** | [`operations/`](operations/) | • [`CONFIGURATION_SEARCH_HANDOFF.md`](operations/CONFIGURATION_SEARCH_HANDOFF.md): Runbook bàn giao các trial mBART.<br>• [`COLAB_SSH_GUIDE.md`](operations/COLAB_SSH_GUIDE.md): Thiết lập SSH ngrok/cloudflared với Colab A100.<br>• [`implementation_plan.md`](operations/implementation_plan.md): Lộ trình kỹ thuật tổng thể. |
| **Xem lại tài liệu & số liệu cũ** | [`archive/`](archive/) | • Chứa các phiên bản báo cáo, bản nháp lịch sử đã được lưu trữ bảo toàn. |

---

## 📌 Ghi chú quan trọng
* Toàn bộ các script biên dịch bài báo (`compile_paper_pdftex.py`, `package_paper_pdftex.py`, `build_paper_tables.py`) đều tự động nhận diện bản thảo tại [`docs/paper/clrr_main.tex`](paper/clrr_main.tex).
* Tất cả dữ liệu thử nghiệm thô và checkpoints lớn đã được loại trừ khỏi git để đảm bảo kho lưu trữ gọn nhẹ và sạch sẽ.
