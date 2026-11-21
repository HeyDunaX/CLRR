# NotebookLM: literature review qua CLI và MCP

## Notebook của project

- **CLRR — Literature Review & Method Development**
- URL: https://notebook.google.com/notebook/bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890
- ID: `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`
- Danh mục nguồn và receipts: `NOTEBOOKLM_SOURCES.json` cùng thư mục.
- Xác thực và thao tác đọc `notebook_list` đã thành công ngày 13/11/2026.

## Cài đặt trên máy hiện tại

Package cộng đồng `notebooklm-mcp-cli==0.16.5`, cài dependencies riêng tại
`C:\Users\Lenovo\.codex\tools\notebooklm-mcp\site-packages-0.16.5`.
Chạy bằng Python conda `clrr`; không cài dependencies này vào môi trường ML.
MCP server `notebooklm` được cấu hình trong `C:\Users\Lenovo\.codex\config.toml`.
Codex có thể cần khởi động lại để nạp server mới vào danh mục tools.
CLI đã hoạt động và dùng được ngay, độc lập với việc nạp lại tools của chat.

Credentials được lưu bằng chế độ `protected` của package, sử dụng Windows
keystore. Không đưa credentials vào repo, logs hoặc bundle tái lập.
Đây là client cộng đồng sử dụng API nội bộ; không phải MCP chính thức của Google.

## Lệnh PowerShell

```powershell
$clrrPython = 'C:\Users\Lenovo\miniconda3\envs\clrr\python.exe'
$nlmLauncher = 'C:\Users\Lenovo\.codex\tools\notebooklm-mcp\run.py'
$clrrNotebook = 'bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890'

& $clrrPython -I -X utf8 $nlmLauncher nlm source list $clrrNotebook --json
& $clrrPython -I -X utf8 $nlmLauncher nlm notebook query $clrrNotebook 'Compare the closest related methods, cite source evidence, and distinguish established facts from hypotheses.' --json

# Thêm paper mới, sau khi kiểm tra chưa có trong manifest/source list:
& $clrrPython -I -X utf8 $nlmLauncher nlm source add $clrrNotebook --url 'PAPER_URL' --wait --json
```

## Khi cần đăng nhập lại bằng Chrome đang dùng

Trong Chrome, bật quyền tại `chrome://inspect/#remote-debugging` rồi chạy:

```powershell
& $clrrPython -I -X utf8 'C:\Users\Lenovo\.codex\tools\notebooklm-mcp\login_existing_chrome.py'
```

Adapter cục bộ hỗ trợ kết nối WebSocket mới của Chrome và mở tab NotebookLM
riêng trong phiên đang đăng nhập. Bấm Allow nếu Chrome yêu cầu. Adapter không
đóng Chrome hoặc sao chép profile; chỉ yêu cầu cookies áp dụng cho NotebookLM.
Sau khi xác thực, CLI dùng phiên đã lưu; có thể đóng tab NotebookLM trên web.

## Cách dùng cho nghiên cứu

1. Nhập PDF/URL paper gốc; ghi source ID và đường dẫn/URL trong manifest.
2. Hỏi công thức, assumptions, gradient paths và matched controls kèm dẫn nguồn.
3. Đọc lại công thức/section trong PDF và code chính thức trước khi chốt novelty.
4. Phân biệt prior công bố với giả thuyết của project. NotebookLM không chứng
   nhận novelty hoặc thay thế kiểm chứng thực nghiệm.

Notebook mới dùng cho literature review. Không tự tải checkpoint, dữ liệu nghiên
cứu hoặc số thực nghiệm nội bộ lên notebook khi chưa có yêu cầu tương ứng.

## Quy trình research cho CLRR: Codex chủ trì, NotebookLM hỗ trợ

### Phân công đã chốt với tác giả

**Codex chịu trách nhiệm nghĩ phương pháp**, dựa trên code, kết quả và insight
thực nghiệm của project. NotebookLM hỗ trợ đọc và đối chiếu literature, tìm
prior gần, phản biện giả thuyết và cung cấp dẫn nguồn. Codex phải đọc lại bằng
chứng gốc rồi tự tổng hợp quyết định; không chuyển câu trả lời NotebookLM thành
phương pháp được duyệt hoặc verdict novelty tự động.

| Công việc | Codex chủ trì | NotebookLM hỗ trợ | Đầu ra cần giữ |
|---|---|---|---|
| Chẩn đoán vấn đề | Replay metrics, kiểm tra code/provenance, tách quan sát khỏi suy luận | Tìm các cơ chế đã được nghiên cứu có thể giải thích hiện tượng | Một vấn đề cụ thể và các giải thích cạnh tranh |
| Đề xuất phương pháp | Viết giả thuyết cơ chế, công thức, gradient paths và dự đoán có thể bác bỏ | Tìm công trình dùng operator/objective tương tự, chỉ ra assumptions khác nhau | Candidate card, chưa code/train |
| Literature review | Search web và chọn paper gốc; theo citation của prior gần | Fast/Deep Research, đọc nhóm nguồn, lập bảng so sánh | Danh mục paper và bằng chứng có source IDs |
| Audit novelty | Đọc PDF/code chính thức, chuẩn hóa ký hiệu để so công thức | Tìm phản ví dụ cho claim mới; trích section/equation khi xác định được | Established / proposed / unclear, cùng lý do |
| Thiết kế ablation | Chọn matched controls và can thiệp phân biệt các cơ chế | Đối chiếu controls, confounds, cost và failure modes từng prior | Kế hoạch pilot khóa trước chạy |
| Code và gates | Implement, kiểm tra đại số, gradient, save/load, CPU/GPU gates | Kiểm tra checklist và diễn giải paper; không chứng nhận code đúng | Receipts thực thi bằng code |
| Đọc kết quả | Tính số/CI/p bằng scripts, phân tích mọi seeds và quyết định GO/NO-GO | Đối chiếu kết quả với dự đoán của literature nếu đã cung cấp tóm tắt được phép | Quyết định có dẫn bằng chứng, không sửa số theo narrative |
| Viết paper | Chọn claim đúng phạm vi, viết phương pháp và lập luận | Citation audit, so related work và mô phỏng phản biện | Claim–evidence table đã kiểm chứng |

### Vòng làm việc cho một candidate

1. **Đọc evidence local trước.** Dùng docs hiện hành, không lấy số archive làm
   kết quả hiện tại. Viết rõ hiện tượng, độ chắc của bằng chứng, và tối đa vài
   giải thích cạnh tranh. Không dùng tương quan representation làm nguyên nhân.
2. **Codex nghĩ thiết kế tối thiểu.** Nêu vì sao cơ chế đó có thể giải quyết vấn
   đề; viết loss/forward, views, masks, detach, parameters và train/inference cost.
   Ghi điều gì phải quan sát được nếu giả thuyết đúng.
3. **Tìm prior theo cơ chế và công thức.** Search cả tên gọi tương đương, không
   chỉ tên candidate. NotebookLM tìm nguồn bổ sung; Codex search web độc lập và
   theo backward/forward citations của các prior gần. Chưa có kết quả tìm kiếm
   không có nghĩa chưa từng có công trình.
4. **Review nguồn chọn lọc.** Query riêng 2–5 prior gần với `--source-ids`; yêu
   cầu objective, gradient path và giới hạn bằng chứng. Đọc lại PDF và code ở
   revision cụ thể. Citation trỏ đúng paper vẫn có thể không hỗ trợ đúng claim.
5. **Codex phân loại candidate.** (a) Reproduction/engineering được ghi đúng
   nguồn; (b) giả thuyết đóng góp mới cần kiểm chứng; (c) chưa rõ, còn thiếu prior;
   (d) bỏ vì trùng hoặc không có cơ sở. Không bảo đảm novelty hay acceptance ACL.
   Engineering vẫn có thể được chạy với nhãn đúng nếu phù hợp mục tiêu tác giả.
6. **Khóa pilot trước code/train dài.** Candidate card có nearest prior, claim
   có thể kiểm chứng, matched controls, data/seeds, selector, metric chính,
   practical threshold, cost và stop rule. Không đổi baseline đã khóa. Không
   gọi một thay đổi hệ số hoặc thêm module là novelty chỉ vì implementation khác.
7. **Correctness → smoke → train.** Chỉ thực hiện sau audit ở trên và đúng quyền
   chạy hiện hành. Gates kiểm chứng implementation, không chứng minh chất lượng.
8. **Quay lại giả thuyết sau kết quả.** Đọc cả kết quả bất lợi, so với controls,
   giữ evidence cũ. Một pilot có điểm cao chưa là bằng chứng cơ chế hoặc ổn định.
   Không tune tiếp bằng test đã xem mà gọi đó là confirmation độc lập.

### Cách tương tác hiệu quả qua CLI

- **Discovery và reasoning là hai bước riêng.** `notebook query` đọc tập nguồn
  đã chọn; muốn mở rộng coverage thì dùng web search hoặc `research start`.
- **Một truy vấn, một câu hỏi.** Lượt đầu trích cơ chế; lượt sau so sánh; lượt
  cuối tìm phản chứng. Không yêu cầu vừa tìm literature vừa chốt phương pháp
  và bảo đảm novelty trong một prompt.
- **Giới hạn nguồn khi so công thức.** Dùng source IDs của nearest priors; dùng
  `--conversation-id` cho follow-up cùng chủ đề. Dùng `--new-conversation` cho
  phản biện độc lập hoặc khi đổi giả thuyết; đây không bảo đảm kết luận độc lập.
- **Giữ receipts nhỏ.** Lưu prompt, response JSON, citations/references, source
  IDs và verdict kiểm chứng vào mục evidence của candidate. Không sinh hàng
  loạt file research hoặc tải weights cho literature review.
- **Không tạo vòng tự xác nhận.** AI summaries không thay paper gốc. Nếu lưu
  summary vào notebook, gắn nhãn AI-generated; loại khỏi nhóm prior dùng để
  chứng minh claim. Không coi bản kế hoạch của project là bằng chứng đã công bố.
- **Nguồn mới phải qua kiểm tra.** Xác nhận title/authors/year/revision, đọc PDF
  gốc; bỏ bản trùng arXiv/conference khi cùng nội dung. Cập nhật manifest khi
  thêm nguồn và xác minh source đã xử lý xong trước query.
- **Không tự nhập toàn bộ kết quả discovery.** Chọn paper liên quan; search hit
  hoặc deep-research report chưa là nguồn đã được kiểm chứng. Không lặp deep
  research mỗi lượt; chỉ mở rộng khi có khoảng trống cụ thể hoặc candidate mới.
- **Corpus hiện tại chỉ có literature.** NotebookLM chưa biết toàn bộ code,
  dữ liệu và kết quả CLRR. Khi cần, cung cấp mô tả tối thiểu trong prompt hoặc
  tóm tắt được phép; không tự tải cả repo/checkpoint lên notebook.

Ví dụ lệnh đã kiểm tra cú pháp với phiên bản cài trên máy:

```powershell
# Tìm nguồn mới; ghi lại task ID trả về. Không tự import toàn bộ.
& $clrrPython -I -X utf8 $nlmLauncher nlm research start 'SPECIFIC_MECHANISM_QUERY' --notebook-id $clrrNotebook --source web --mode fast
& $clrrPython -I -X utf8 $nlmLauncher nlm research status $clrrNotebook --task-id 'TASK_ID' --max-wait 0 --full

# Sau khi kiểm tra danh sách và chỉ số trả về:
& $clrrPython -I -X utf8 $nlmLauncher nlm research import $clrrNotebook 'TASK_ID' --indices 'SELECTED_INDICES'

# Hỏi riêng nearest priors, trả JSON có citations/references:
& $clrrPython -I -X utf8 $nlmLauncher nlm notebook query $clrrNotebook 'Extract exact objectives, gradient paths, and assumptions. Cite evidence; mark missing details unknown.' --source-ids 'SOURCE_ID_1,SOURCE_ID_2' --new-conversation --json
```

Chỉ dùng `--mode deep` khi fast/web search bỏ ngỏ một câu hỏi literature quan
trọng. Không dùng `--force` để chồng research task. Kiểm tra `--help` nếu nâng
version; không giả định CLI hỗ trợ mọi tính năng mới trên web.

### Prompt mẫu theo vai trò hỗ trợ

**Đọc prior:**

> For each selected paper, extract the research setting, exact objective,
> representation/views, trainable parameters, stop-gradient placement, inference
> changes, and essential controls. Cite the supporting passage and section or
> equation only when available. Mark absent details unknown. Separate the
> authors' claims from their experimental evidence.

**Phản biện novelty của thiết kế do Codex viết:**

> Proposed mechanism: [formula and assumptions]. This is a proposal, not
> established evidence. Search the selected papers for the closest existing
> operator and objective, including equivalent formulations under different
> notation. Which parts already exist? Which differences are substantive,
> setting-only, or unclear? Cite evidence for both overlap and differences.
> Do not certify novelty from absence in this source collection.

**Hỗ trợ thiết kế ablation:**

> Given hypothesis H and alternative explanations H1/H2, use the selected
> papers to identify the controls that would distinguish them. State expected
> outcomes if each explanation is true, confounds, additional compute, and what
> cannot be inferred. Label your suggested experiments as proposals.

**Kiểm tra lập luận paper:**

> Audit the following proposed claims against the selected original papers.
> Return claim, supporting or contradicting evidence, citation, scope mismatch,
> and conservative wording. Flag unsupported priority, causal, and generality
> claims. Do not invent missing numbers, citations, or experimental results.

### Áp dụng ngay cho trạng thái CLRR hiện tại

- Đọc cùng `docs/research/POST_PAIRED_ROUTE_METHOD_PLAN.md`. A–D gợi ý consistency
  có tín hiệu ở một seed; chưa xác lập cơ chế hay ưu thế ổn định. Codex phải giữ
  giới hạn đó khi nghĩ hướng tiếp theo.
- Các prior gần cần so kỹ: R-Drop, LR-Drop, DKD; bổ sung alignment/skip priors
  nếu thiết kế dùng lại CLRR/LSR. Không dùng riêng tên CLRR+LSR để miễn audit cho
  một objective mới.
- Hai phân rã gold/rest trong plan là ablation engineering có dẫn DKD, không
  tự trở thành phương pháp novel. Dùng chúng để kiểm chứng giả thuyết nếu được
  triển khai đúng plan; không ép chúng thành đóng góp chính để đủ narrative.
- Bước tiếp theo của nghiên cứu vẫn do Codex tổng hợp từ evidence, đề xuất
  candidate và giải thích cho tác giả; NotebookLM cung cấp prior và phản biện.

### Tài liệu đã tra cứu

Kiểm tra ngày 13/11/2026; cú pháp CLI đối chiếu thêm `--help` của bản0.16.2.

- Google, sources/discovery và giới hạn nhập nguồn:
  https://support.google.com/gemininotebook/answer/16215270?hl=en
- Google, chat, chọn nguồn và citations:
  https://support.google.com/gemininotebook/answer/16179559
- CLI của package cộng đồng:
  https://github.com/jacob-bd/gemini-notebook-mcp-cli/blob/main/docs/CLI_GUIDE.md
- Workflow tham khảo của package; quy trình CLRR ở trên là thiết kế của project:
  https://github.com/jacob-bd/gemini-notebook-mcp-cli/blob/main/src/notebooklm_tools/data/references/workflows.md
