# CLRR–LSR với module residual và gradient alignment có giới hạn

## Material Passport

- Ngày: 28/10/2026.
- Skill/mode: academic-research-suite / experiment-agent / plan.
- Phiên bản: design-v1; tên mô tả nội bộ, chưa phải tên phương pháp trong paper.
- Trạng thái: **IMPLEMENTED / LOCAL CPU PASS / GPU GATES PENDING / NO TRAINING SCORES**.
- Novelty: **HOLD cho toàn bộ thiết kế; NO-GO cho claim “giới hạn gradient alignment vào adapter là nguyên lý mới”.**
- Tác giả đã yêu cầu triển khai code và chuẩn bị pilot RTX; trạng thái thực thi ở [runbook](../operations/ALIGNMENT_RESIDUAL_RTX.md).
- Thiết kế gốc được viết trước code. Cập nhật triển khai:8scientific CPU tests và4bundle cases PASS; chưa có quality results. Method cũ/lock/scores giữ nguyên.

## 1. Quyết định và câu hỏi nghiên cứu

RQ: Khi tái sử dụng biểu diễn xuyên tầng, việc giới hạn gradient LSR vào module
residual có cải thiện quyết định dịch hơn alignment cập nhật toàn encoder,
sau khi kiểm soát cùng kiến trúc và số tham số không?

Đây là giả thuyết cần kiểm chứng, chưa phải nguyên nhân đã xác định.

| Bằng chứng nội bộ | Dùng để định hướng | Không được suy ra |
|---|---|---|
| Full NLL tốt hơn baseline nhưng content top-1 thấp hơn; sửa 227, làm mất 272 quyết định reference-top1 trong probe seed42 | Phân biệt surrogate alignment/NLL với quyết định token | LSR đã được chứng minh là nguyên nhân làm hỏng token |
| Route ON hữu ích ở checkpoint đã co-adapt; CLRR chưa thắng baseline ổn định | Kiểm tra thông tin donor được dùng như thế nào | Bật route làm phương pháp vượt baseline |
| P1 SDPA seeds42–44: mean ΔBLEU −0.1228; thắng 1/3 | Dừng mở rộng ordinary depth fusion hiện tại | Mọi kiến trúc dùng nhiều độ sâu đều vô ích |
| CE continuation cũng giảm so với initialization đã chọn | Tránh dùng continuation làm pilot chính | Đã xác định nguyên nhân là reset optimizer hoặc overfitting |

Nguồn: [insight](RESEARCH_INSIGHTS.md), [token diagnosis](TOKEN_DECODING_DIAGNOSIS.md),
[continuation audit](CONTINUATION_REVIEW_AND_NEXT_DESIGN.md).
Các bằng chứng thuộc nhiều cohort; không lấy chênh số giữa cohort làm gain.

## 2. Ký hiệu

| Ký hiệu | Định nghĩa |
|---|---|
| \((x_i,y_i)\) | Cặp dịch trong training, gồm source IDs, target labels và masks |
| \(\theta\) | Tất cả tham số backbone gốc: embedding dùng chung, encoder, decoder, LM head và các alias tied weights |
| \(\phi=\{\phi_l:l\in\mathcal S\}\) | Các module residual mới, độc lập giữa các layer |
| \(F_l\) | Toàn bộ encoder block gốc, gồm residual và normalization bên trong block |
| \(h_l,u_l\in\mathbb R^{n\times D}\) | Output sau can thiệp và output block gốc, tại layer l |
| \(d,\alpha\) | Khoảng cách donor và hệ số cộng; pilot dự kiến d=2, α=0.10 |
| \(\mathcal S\) | Layer nhận, cố định \(\{3,\ldots,12\}\) cho encoder 12 layer |
| \(q\in\{0,1\}\) | Quyền truy cập donor: q=1 có donor; q=0 là local-adapter control |
| \(\operatorname{sg}\) | Giữ nguyên giá trị forward, đạo hàm bằng 0 |
| \(\xi\) | Toàn bộ randomness của một forward, gồm dropout/layerdrop nếu có |
| \(\rho=10^{-6}\) | Epsilon của phép chuẩn hóa đầu vào module |
| \(\lambda=0.10\) | Trọng số LSR dự kiến, kế thừa Full; chưa phải optimum |

Mỗi biểu thức token bên dưới là vector cột; áp dụng độc lập từng token.
Attention masks, stack-level final normalization và vị trí can thiệp giữ theo
backbone. Layer đếm từ 1; không dùng embedding h0 làm donor.

## 3. Nhắc lại CLRR gốc để phân biệt

\[
u_l=F_l(h_{l-1};\theta_l),\qquad
h_l=
\begin{cases}
u_l,&l\le d,\\
u_l+\alpha\operatorname{sg}(h_{l-d}),&l>d.
\end{cases}
\tag{1}
\]

Donor là **post-rewiring** output, detach ở từng lần được sử dụng.
Chỉ encoder được rewire. Decoder nguyên bản.

Định nghĩa đối chiếu: [original/clrr.py](../../src/methods/original/clrr.py),
[original/lsr.py](../../src/methods/original/lsr.py).

## 4. Forward của kiến trúc dự kiến

### 4.1 Module

Chuẩn hóa RMS không có tham số, cho vector v:

\[
\nu(v)=\frac{v}{s(v)},\qquad
s(v)=\sqrt{D^{-1}v^\top v+\rho}.
\tag{2}
\]

Mỗi layer có hai ma trận, không bias, không dropout riêng:

\[
W_{\downarrow,l}\in\mathbb R^{r\times D},\qquad
W_{\uparrow,l}\in\mathbb R^{D\times r},\qquad r=64.
\tag{3}
\]

\[
A_l(v;\phi_l)=W_{\uparrow,l}
\,\operatorname{GELU}\!\left(W_{\downarrow,l}\nu(v)\right).
\tag{4}
\]

Đây là bottleneck adapter thông thường về cấu trúc; RMS/GELU/rank64 không
được nhận là novelty. Chọn rank cố định để có thiết kế cụ thể, không suy từ
kết quả pilot và không mở grid rank.

### 4.2 Donor và hiệu chỉnh

Cho \(l\in\mathcal S\):

\[
b_l=q\,\operatorname{sg}(h_{l-d}),\qquad
e_l=b_l-u_l,
\tag{5}
\]

\[
a_l=\alpha A_l(e_l;\phi_l),\qquad
h_l=u_l+\alpha b_l+a_l.
\tag{6}
\]

Với \(l\notin\mathcal S\), \(h_l=u_l\).

Ý nghĩa: giữ fixed skip CLRR và học một hiệu chỉnh phụ thuộc phần chênh donor
với output tầng nhận. Không khẳng định hiệu chỉnh này là “thông tin ngôn ngữ
còn thiếu”: đó chỉ là cách tham số hóa để kiểm tra.

Toàn encoder là \(H=E_{\theta,\phi}^{q}(x;\xi)\), sau mọi final normalization
gốc. Chính H này đi vào cross-attention decoder. Không có projection head
riêng chỉ dùng để cải thiện điểm alignment.

### 4.3 Khởi tạo và các trường hợp đặc biệt

- \(W_{\uparrow,l}=0\); các phần tử \(W_{\downarrow,l}\) lấy độc lập từ
  \(\mathcal U[-1/\sqrt D,1/\sqrt D]\), tương ứng Kaiming-uniform a=sqrt(5)
  cho ma trận này; không bias. Dùng generator riêng có module_seed=seed+10000,
  sinh các layer theo thứ tự tăng dần; không tiêu thụ RNG của backbone/CE.
- q=1 và mọi up-projection bằng 0: forward chính xác bằng CLRR gốc ở cùng θ,
  masks và RNG.
- q=0 và up-projection bằng 0: forward chính xác bằng baseline gốc.
- q=0 sau học: \(e_l=-u_l\), \(h_l=u_l+\alpha A_l(-u_l)\).
  Đây là local-adapter control, có cùng số tham số và kích thước bottleneck.
- α=0 loại cả fixed skip và hiệu chỉnh module, trở về baseline; **không dùng
  α=0 làm đối chứng capacity**, vì khi đó module mất tác dụng.
- λ=0 vẫn giữ kiến trúc/module, chỉ bỏ auxiliary.

Module không tự sinh thông tin mới, không bảo đảm giữ lexical distinctions,
không bảo đảm tránh collapse hoặc tăng BLEU.

### 4.4 Số tham số và chi phí

\[
N_\phi=|\mathcal S|\,2Dr
=10\times2\times1024\times64=1{,}310{,}720.
\tag{7}
\]

Khoảng 0.21% của backbone cỡ 610M, chỉ là tỷ lệ gần đúng; số backbone chính xác
phải đếm từ model khi triển khai. Tất cả backbone vẫn được fine-tune bằng CE:
đây **không phải PEFT training**.

Mỗi token mỗi layer module thêm khoảng \(2Dr\) multiply-accumulates, chưa tính
GELU/RMS. Không suy FLOPs này thành latency hoặc VRAM đo được.
Inference không cần target encoder nhưng vẫn chạy module và giữ donor cache.
Bản triển khai tham chiếu auxiliary có thể cần thêm source encoder forward;
phải báo wall-clock và peak VRAM, không gọi overhead là miễn phí.

## 5. CE và LSR: giá trị loss

### 5.1 CE theo đúng reduction hiện tại

Đặt \(\mathcal T_B=\{(i,t):\text{label}_{i,t}\ne-100\}\) trong một microbatch:

\[
\mathcal L_{\rm CE}
=-\frac{1}{|\mathcal T_B|}
\sum_{(i,t)\in\mathcal T_B}
\log p_{\theta,\phi}^{q}(y_{i,t}\mid x_i,y_{i,<t};\xi_{\rm CE}).
\tag{8}
\]

Bao gồm các special labels hợp lệ như code cũ; không tự đổi sang chỉ content
tokens hoặc label smoothing. Content-token metrics ở §9 là diagnostics riêng.

### 5.2 Pooling/anchor giữ LSR gốc

\[
P(H,m)=\frac{\sum_t m_tH_t}{\max(\sum_t m_t,10^{-6})},
\quad
N(v)=\frac{v}{\max(\|v\|_2,10^{-12})}.
\tag{9}
\]

Source mask là attention mask gốc, target IDs thay label −100 thành PAD;
target mask bằng 1 khi ID khác PAD. Không tự loại language IDs/EOS/special
tokens khỏi LSR, vì LSR gốc chỉ loại padding.

\[
z_{x,i}=N(P(E_{\theta,\phi}^{q}(x_i;\xi_x),m_{x,i})),
\]
\[
z_{y,i}=\operatorname{sg}
\left[N(P(E_{\theta,\phi}^{q}(y_i;\xi_y),m_{y,i}))\right],
\]
\[
\mathcal L_{\rm LSR}
=B^{-1}\sum_{i=1}^B(1-z_{x,i}^{\top}z_{y,i}).
\tag{10}
\]

Target encoder dùng cùng θ, φ, q, tokenizer setup và model train/eval flags,
nhưng toàn bộ target forward no_grad. Không EMA teacher, frozen reference
baseline, InfoNCE, token alignment hoặc target labels lúc inference.
Anchor có thể đổi giữa optimizer steps do CE/module updates; stop-gradient
không biến nó thành teacher cố định.

Loss value để log:

\[
\mathcal L_{\rm reported}=\mathcal L_{\rm CE}
+\lambda\mathcal L_{\rm LSR}.
\tag{11}
\]

Phải log CE, LSR raw và weighted LSR riêng. C2 dùng quy tắc gradient ở §6,
không được mô tả như backward thông thường của Eq.(11).

## 6. Gradient: định nghĩa chính xác candidate và controls

### 6.1 C1: alignment thông thường

Với cùng state và forward:

\[
g_\theta^{C1}=\nabla_\theta\mathcal L_{\rm CE}
+\lambda\nabla_\theta\mathcal L_{\rm LSR},
\quad
g_\phi^{C1}=\nabla_\phi\mathcal L_{\rm CE}
+\lambda\nabla_\phi\mathcal L_{\rm LSR}.
\tag{12}
\]

Target branch không góp đạo hàm. Decoder không có LSR gradient trực tiếp;
shared embeddings thuộc θ vẫn có thể nhận LSR từ source encoder.

### 6.2 C2: chỉ module nhận gradient alignment

Định nghĩa auxiliary bằng parameter-frozen view:

\[
\widetilde E(x)=E_{\operatorname{sg}(\theta),\phi}^{q}(x;\xi_x).
\tag{13}
\]

Dùng \(\widetilde E(x)\) thay source encoder trong Eq.(10), target vẫn toàn
stop-gradient, ta được \(\widetilde{\mathcal L}_{\rm LSR}\).
Giá trị forward bằng LSR C1 tại cùng state và cùng RNG, nhưng:

\[
\nabla_\theta\widetilde{\mathcal L}_{\rm LSR}=0,\qquad
\nabla_\phi\widetilde{\mathcal L}_{\rm LSR}
=\nabla_\phi\mathcal L_{\rm LSR}.
\tag{14}
\]

\[
\boxed{
g_\theta^{C2}=\nabla_\theta\mathcal L_{\rm CE},\qquad
g_\phi^{C2}=\nabla_\phi\mathcal L_{\rm CE}
+\lambda\nabla_\phi\widetilde{\mathcal L}_{\rm LSR}
}
\tag{15}
\]

Eq.(14) yêu cầu **chỉ detach tham số θ**, không detach mọi intermediate hidden
states. Đạo hàm vẫn phải xuyên qua các block θ cố định để truyền tới những
module φ ở trước. Donor detach của Eq.(5) vẫn được giữ nguyên.

Không bọc toàn bộ source auxiliary trong no_grad; làm vậy sẽ xóa cả φ gradient.
Không đặt requires_grad=False toàn backbone xuyên cả CE pass.
Không nhầm việc detach riêng target z_y với chặn LSR source gradient vào θ.

### 6.3 Jacobian cho thấy điều được và không được bảo đảm

Cho một token, đặt \(k=W_\downarrow\nu(e)\), \(g=\partial\mathcal L/\partial h_l\).

\[
J_\nu(e)=\frac{I}{s(e)}-\frac{ee^\top}{D\,s(e)^3},
\]
\[
\frac{\partial h_l}{\partial u_l}
=I-\alpha W_\uparrow
\operatorname{diag}(\operatorname{GELU}'(k))W_\downarrow J_\nu(e).
\tag{16}
\]

Đạo hàm trực tiếp về donor trong Eq.(5) bằng 0 do stop-gradient.
Các đường tuần tự qua \(F_l,F_{l-1},\ldots\) vẫn tồn tại.

\[
\frac{\partial\mathcal L}{\partial W_\uparrow}
=\alpha g\,\operatorname{GELU}(k)^\top,
\]
\[
\frac{\partial\mathcal L}{\partial W_\downarrow}
=\alpha\left[(W_\uparrow^\top g)\odot
\operatorname{GELU}'(k)\right]\nu(e)^\top.
\tag{17}
\]

Tại zero up-projection, down-projection gradient bằng 0 ở bước đầu; up-projection
vẫn có thể học. Không yêu cầu mọi parameter gradient phải khác 0 tại initialization.

Giới hạn: sau khi φ học, nó thay forward và Jacobian CE; C2 không bảo đảm
giữ nguyên θ trajectory, baseline predictions, rank hoặc token margins.
“Không có LSR gradient trực tiếp vào backbone” là claim hẹp chính xác.

## 7. Accumulation, clipping và AdamW

Với K=32 microbatches trong một optimizer window:

\[
\bar g_\theta=K^{-1}\sum_{k=1}^{K}g_{\theta,k},
\qquad
\bar g_\phi=K^{-1}\sum_{k=1}^{K}g_{\phi,k}.
\tag{18}
\]

Đây là mean của microbatch means như lock; không đổi sang mean theo tổng số
tokens toàn window. Với window cuối không đủ K, phải replay đúng shared runner
đã khóa và lưu actual scaling; không tự sửa reduction trong candidate.

Joint clipping trên mọi trainable parameters:

\[
c=\min\left(1,\frac{1}{\sqrt{\|\bar g_\theta\|_2^2+
\|\bar g_\phi\|_2^2}+\epsilon_{\rm clip}}\right),\qquad
\hat g=(c\bar g_\theta,c\bar g_\phi).
\tag{19}
\]

\(\epsilon_{\rm clip}\) là hằng thực tế của pinned torch, phải ghi lại lúc
triển khai. Giữ global max_grad_norm=1 như baseline, không đổi sang group clipping.

**Hệ quả:** auxiliary có thể thay clipping factor thông qua φ. Vì vậy ngay
tại cùng θ, φ và optimizer state, θ update C2 chưa chắc bằng C0 nếu joint clip
hoạt động. Phải log preclip norm, c và actual θ update difference.

Cho từng parameter p thuộc θ hoặc φ:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)\hat g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)\hat g_t^{\odot2},
\]
\[
\widehat m_t=m_t/(1-\beta_1^t),\quad
\widehat v_t=v_t/(1-\beta_2^t),
\]
\[
p_{t+1}=(1-\eta_t w)p_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon_{\rm Adam}}.
\tag{20}
\]

Giữ β1=.9, β2=.999, epsilonAdam=1e−8, w=0, fused AdamW, LR schedule của lock.
θ và φ cùng LR5e−5; chưa có module-specific LR hoặc optimizer riêng.
LR scheduler, optimizer moments, clipping và accumulation là một phần method
execution; không đánh giá ảnh hưởng chỉ bằng tỷ số norm hai gradients.

## 8. Bốn nhánh pilot: công thức đầy đủ và estimand

| ID nội bộ | q | Auxiliary | θ gradients | φ gradients |
|---|---:|---|---|---|
| C0 — donor/module capacity | 1 | λ=0 | CE | CE |
| C1 — unrestricted alignment | 1 | Eq.(10), λ=.10 | CE+LSR | CE+LSR |
| C2 — parameter-confined alignment | 1 | Eq.(13–15), λ=.10 | CE trực tiếp | CE+LSR |
| C3 — local-module control | 0 | Eq.(13–15), λ=.10 | CE trực tiếp | CE+LSR |

Tất cả dùng Eq.(2–6), cùng rank, layer set, α, parameter count và optimizer.
Các module ở C0/C1/C2/C3 có initial tensors giống hệt cho cùng seed.
Tải xong pretrained θ rồi mới khởi tạo φ bằng RNG riêng để không làm thay đổi
initial θ hoặc data shuffle seed.

| Contrast | Câu hỏi được trả lời | Giới hạn |
|---|---|---|
| C2−C1 | Giới hạn nơi auxiliary gradient tác động có lợi không? | Bao gồm ảnh hưởng downstream của training và clipping; không chỉ một gradient snapshot |
| C2−C0 | Auxiliary có ích ngoài cùng donor/module capacity không? | Không phải so với no-adapter |
| C2−C3 | Donor xuyên tầng có ích hơn local adapter không? | Cùng lúc bỏ fixed skip và donor input; chỉ xác định hiệu ứng cả gói donor |
| C2−baseline | Phương pháp có vượt đối sánh thực tế không? | Chỉ matched khi baseline reuse audit PASS |

**Sửa sự mơ hồ của phác thảo trong chat:** C3 ở đây bỏ toàn bộ donor bằng q=0.
Chỉ bỏ αb nhưng vẫn cho module đọc h_(l−d) không phải “không CLRR/donor”.
Muốn tách fixed skip với donor-input cần thêm một control, không lén suy từ
bốn runs này. Chưa đưa control thứ năm vào ngân sách.

Baseline, CLRR-only, LSR-only, Full cũ là references nếu cùng protocol và seed;
giữ số cũ, không retrain chỉ để điền đủ bảng. Các số A100/eager không được
trộn làm matched controls RTX SDPA.

## 9. Phép đo cơ chế trước khi cân nhắc pilot dài

### 9.1 So sánh cập nhật từ cùng trạng thái

Mỗi fork lấy cùng \((\theta,\phi,m,v,\text{scheduler},\text{RNG})\), cùng training
window K microbatches. Thực hiện một optimizer update C0/C1/C2 trong các bản
copy độc lập; C3 không tham gia same-state contrast vì forward khác.

Đánh giá trước/sau trên batch Q khác batch cập nhật U, chỉ lấy từ train.
Danh sách row U/Q khóa trước, không overlap trong từng fork; không dùng
validation/test làm meta-training labels hoặc chọn auxiliary weights.

\[
\Delta_j M=M(Q;p_j^+)-M(Q;p),\qquad j\in\{C0,C1,C2\},
\]
\[
\delta_j M=M(Q;p_j^+)-M(Q;p_{C0}^+).
\tag{21}
\]

M gồm CE, content top1, mean gold-vs-best-other margin và repair/damage.
Tại cùng state, C1/C2 phải có CE và LSR forward values khớp khi replay RNG;
khác biệt có chủ đích là parameter-gradient policy.

Phải report riêng:

- Initialization: φ up=0; update hành vi đặc biệt theo Eq.(17).
- Nonzero φ states nếu có short controlled warmup được duyệt sau này.
- Snapshot có optimizer thật so với probe optimizer mới.

Không load model-only weights rồi gọi optimizer mới là “khôi phục lịch sử
AdamW”. Nếu thiếu optimizer/RNG, ghi rõ fresh-optimizer sensitivity probe.
Không gọi 8 hoặc 16 probe batches là training seeds độc lập.

### 9.2 Token repair/damage

Content mask C loại tokenizer special IDs và ignored/PAD labels; cùng
row/position cho mọi nhánh. Nó vẫn chứa punctuation/subwords, không tương
đương tập “từ mang nghĩa”.

\[
m_{i,t}=z_{i,t,y_{i,t}}-\max_{v\ne y_{i,t}}z_{i,t,v}.
\tag{22}
\]

Gọi a là prediction reference trước update (hoặc baseline khi so checkpoint),
b là prediction sau update/candidate:

\[
R=\sum_{(i,t)\in C}\mathbf1[a\ne y,\ b=y],\quad
D_{\rm damage}=\sum_{(i,t)\in C}\mathbf1[a=y,\ b\ne y],
\]
\[
\Delta{\rm top1}=(R-D_{\rm damage})/|C|.
\tag{23}
\]

Không dùng margin>0 thay argmax ở ties; báo tie handling.
Tần suất token tính từ train tokenizer IDs; bins 0,1–5,6–50,>50 như probe cũ.
Teacher-forced gains không chứng minh free-generation gains.

### 9.3 Auxiliary geometry và module usage

- Alignment cosine, matched vs shuffled/centered cosine nếu dùng; không xem
  raw cosine tăng là quality gate.
- Residual ratios:
  \(R_{\rm fixed,l}=\|\alpha b_l\|_F/(\|u_l\|_F+\epsilon)\),
  \(R_{\rm learned,l}=\|a_l\|_F/(\|u_l\|_F+\epsilon)\);
  norms chỉ tính valid source tokens, kèm mean/quantiles.
- Centered pooled singular values \(s_j\):
  \(p_j=s_j/\sum_k s_k\),
  \({\rm erank}=\exp(-\sum_jp_j\log p_j)\).
  Trường hợp tổng singular values=0 báo degenerate, không gán rank đẹp.
- Actual parameter updates \(U_\theta=\theta^+-\theta\),
  \(U_\phi=\phi^+-\phi\), và difference so CE-only; không đồng nhất gradient
  norm với actual update norm.

### 9.4 Falsification

1. Nếu actual-update probes không tái hiện sự đánh đổi alignment–decision,
   không kể câu chuyện “LSR làm hỏng backbone” như cơ chế đã chứng minh.
2. Nếu C2 chỉ tăng cosine/rank nhưng không cải thiện quyết định/generation,
   không mở rộng vì representation đẹp.
3. Nếu C0 giải thích gain của C2, đóng góp là capacity/architecture, chưa phải
   gradient confinement.
4. Nếu C3 bằng hoặc hơn C2, chưa có cơ sở giữ cross-layer donor làm đóng góp.
5. Một update không dự đoán được toàn trajectory; short probe chỉ quyết định
   giả thuyết còn đáng kiểm tiếp, không phải gate bảo đảm BLEU.

## 10. Recipe giữ cố định

Nguồn duy nhất cho phần shared recipe:
[amis_rtx_baseline_sdpa.json](../../configs/protocols/amis_rtx_baseline_sdpa.json).
SHA256 đọc ngày 28/10/2026:

~~~text
7e0edd270019b83813c33fba69f3f8b3724b70fdbd513770c6e4dfc91f31baf0
~~~

| Trường | Giá trị |
|---|---|
| Data | Amis→Mandarin, train4600 / validation576 / test575; hashes từ lock |
| Model | facebook/mbart-large-50-many-to-many-mmt; revision từ lock |
| Language | source tl_XX; target zh_CN; không thêm prefix |
| Backend/precision | SDPA, FP32, TF32 off |
| Batch | microbatch4 × accumulation32 |
| Epoch/early stop | cap20, patience10 |
| LR/optimizer | 5e−5, fused AdamW, linear schedule, warmup.06 |
| Clipping/decay | global norm1, weight decay0 |
| Length | source256 / target256 |
| Checkpointing | ON, use_reentrant=True theo lock |
| Selector | original-reference validation chrF++, beam1 FP32, earliest tie |
| Diagnostics | BLEU zh, chrF++, chrF0 cùng predictions |
| Test | Không mở trong pilot; beam4 chỉ sau method/analysis lock |
| New architecture | layers3–12, d2, α.10, rank64, no module bias/dropout |
| Auxiliary | cosine LSR λ.10; gradient policy theo từng nhánh |

RMS epsilon và module init là phần thiết kế mới, không phải “đã chạy theo lock”.
Checkpoint selector giữ nguyên dù chrF++ Mandarin nhạy sparse word bigrams;
BLEU/chrF0 dùng làm cross-check. Không đổi selector sau khi nhìn số.

## 11. RNG, checkpointing và hợp đồng triển khai tương lai

### 11.1 Forward tham chiếu và RNG giữa các controls

1. CE forward bình thường, source và decoder dùng θ,φ live.
2. Source auxiliary forward có cùng kiến trúc; C1 dùng θ live, C2 dùng θ như
   constants nhưng φ live. Replay source RNG từ CE để source hidden values
   khớp; replay không làm tiến global RNG lần thứ hai.
3. Target auxiliary no_grad, cùng train/eval flags và fresh routing cache.
   Target pass có RNG draw riêng như nguyên tắc LSR gốc. C0 vẫn thực hiện
   target forward no_grad nhưng không dùng loss của nó, để C0/C1/C2/C3 có cùng
   thứ tự tiêu thụ RNG auxiliary; thiếu bước này sẽ trộn ảnh hưởng gradient
   với thay đổi dropout RNG ở CE steps sau.
4. Restore đúng RNG/global mode/cache kể cả exception và checkpoint recompute.
5. Backward theo Eq.(12)/(15), accumulation/clipping/optimizer theo §7.

Đây là reference có thể tốn thêm encoder pass; chỉ tối ưu khi kiểm chứng gradient
equivalence và RNG-state equivalence. C0 có thể bỏ source auxiliary replay,
vì replay không làm tiến RNG và λ=0; target dummy ở bước3 vẫn cần giữ hoặc
được thay bằng cách advance RNG đã chứng minh tương đương. Không giả định seed
giống nhau là tự động có CE dropout giống nhau. Ghi auxiliary-forward count,
dropout draw policy và overhead.

### 11.2 Bẫy cần tránh

- Parameter-frozen view phải giữ mọi tied aliases của shared embeddings; không
  vô tình detach φ hoặc để alias θ khác nhận gradient.
- Frozen θ không có nghĩa encoder auxiliary được chạy toàn no_grad.
- Reentrant checkpointing không tương thích mọi cách gọi autograd.grad(inputs=...).
  Không tự đổi use_reentrant=False để né lỗi. Nếu cần graph riêng, phải giữ
  exact gradient definition và vượt gates.
- Cache donor theo từng forward/graph; target hoặc auxiliary không được ghi đè
  cache mà CE backward recomputation cần dùng. Không dựa global mutable cache
  chung giữa các graph.
- Thêm parameters phải lưu cùng checkpoint. Registry/engine hiện có nhánh
  metadata extra_parameters=0 cho một số wrappers: khi code phải xử lý riêng
  family mới, không tái sử dụng metadata sai.
- Tách weights-only export phục vụ inference với resumable optimizer checkpoint.
- Chỉ load adapter weights lên backbone trống rồi quên module/gradient policy
  không được xem là restore thành công.

### 11.3 Correctness gates bắt buộc, hiện CHƯA CHẠY

| Gate | Điều kiện |
|---|---|
| Shape/mask | Ragged batches, PAD, length1 và d boundary đúng |
| Initialization | q1 zero-up khớp CLRR; q0 zero-up khớp baseline, cùng RNG |
| Parameters | 1,310,720 extra params cho D1024/r64/10layers; không duplicate tied θ |
| Donor | post-rewiring d2; direct donor gradient0; sequential gradients tồn tại |
| Gradient C2 | LSR-only θ grads0/None, φ up grads hữu hạn và không mất đường tới module sớm |
| Gradient equivalence | C1/C2 LSR values và φ gradients khớp tại cùng state/RNG; chỉ θ contribution khác |
| Zero-up behavior | Down grads0 ở init là đúng; dùng nonzero-up fixture để kiểm down-path |
| Checkpointing | Reentrant on/off fixture cho logits và gradients khớp trong tolerance khóa trước |
| Accumulation | K microbatch means đúng một lần scaling; kiểm cả window cuối |
| Clipping | Đo c; chỉ yêu cầu θ C2=CE-only update khi same state và clipping factors bằng nhau |
| Save/load/resume | θ,φ,backend,metadata,optimizer,scheduler,RNG; uninterrupted/resume khớp |
| Inference | Không labels/target auxiliary; beam/cache logits/outputs hợp lệ |
| Real GPU | Locked SDPA/FP32/GA32; check VRAM, không dừng job người khác |

Không có PASS receipt nào cho thiết kế này tại thời điểm tạo tài liệu.

Tolerance FP32 đề xuất cho các gates so hai cách tính cùng toán tử:
logits/loss rtol=1e−5, atol=2e−5; gradients và actual updates rtol=2e−4,
atol=5e−5. θ auxiliary-only ở C2 phải None hoặc đúng tensor0; các gradient
khác phải finite. Fixtures dùng cùng thiết bị/runtime, có cả zero-up và nonzero-up.
Đây chưa phải tolerance được đo; nếu gate fail phải điều tra, không tự nới
ngưỡng để tạo PASS. Không đòi bitwise equality giữa hai GPU khác loại.

## 12. Pilot, thống kê và GO/NO-GO dự kiến

### 12.1 Thứ tự và số runs

| Giai đoạn | Việc | Training models mới |
|---|---|---:|
| Hiện tại | Hoàn tất tài liệu/công thức và novelty card | 0 |
| Novelty/technical | Đối chiếu prior + correctness + controlled update probes, khi được duyệt | 0 full runs |
| Screening | C0,C1,C2,C3, seed42, từ pretrained revision gốc | 4 |
| Replication nếu screening đạt | Cả4 nhánh, seeds43/44 | Thêm8, tổng12 |
| Confirmation/Ê Đê | Thiết kế riêng sau method lock | Chưa mở/ngân sách chưa chốt |

Không tiếp tục train từ best baseline R1 cho main pilot. Baseline42/43/44
chỉ reuse sau metadata/data/backend/selector audit PASS; nếu không PASS,
báo tác giả, không tự mở baseline retraining ngoài bảng.

**Novelty hiện chưa PASS, nên bảng này không phải queue đã được duyệt chạy.**
Đặc biệt, generic confinement không được mở 4 runs chỉ vì đã viết xong công thức.

### 12.2 Screening thresholds đề xuất trước khi xem kết quả

Ngưỡng sau là lựa chọn thực dụng cho development, không suy ra từ power và
không phải bảo đảm ACL. Cần đóng băng bằng design receipt trước training:

- C2−baseline ≥+0.30 BLEU.
- C2−C0, C2−C1 và C2−C3 mỗi contrast ≥+0.20 BLEU.
- chrF0 không giảm so cả4 đối sánh; chrF++ không giảm so baseline và C0.
- Repair−damage không âm so baseline và C1 trên cùng validation content mask.
- Không chỉ gain từ special tokens hoặc một nhóm word-bigram hiếm; báo
  full-corpus scores và influence diagnostics, không xóa rows.

Nếu thiếu một điều kiện: HOLD replication; phân loại phần thất bại thay vì
đổi λ/rank/layer để cứu số trong cùng pilot.
Nếu quality tăng nhưng mechanism metrics không ủng hộ, chỉ nhận quality
observation, không nhận causal mechanism.

### 12.3 Thống kê

Khóa 4 contrasts: C2−C1, C2−C0, C2−C3, C2−baseline.
Mỗi contrast đánh giá BLEU, chrF++, chrF0 trên toàn576 validation rows.
10,000 paired sentence bootstrap draws, RNG42; resample cùng row indices cho
hai systems, recompute corpus scores, không bootstrap token độc lập.

Family screening cố định 4×3=12 tests, Holm12, percentile CI95% và exact
implementation/convention của native SacreBLEU p phải được ghi trong receipt.
Không giả định p và percentile CI là hai dạng đảo chính xác của cùng kiểm định.
Nếu bỏ runs thì không tự thu nhỏ family sau khi thấy kết quả.

Một seed là development; sentence bootstrap không đo seed uncertainty.
Không bắt p<.05 để quyết định mọi pilot, nhưng không gọi gain là significant
khi chưa đạt corrected p. CI chứa0 không chứng minh equivalence.

Nếu replication được duyệt: báo mean, SD, paired seed deltas, win counts của
4 contrasts; nếu chạy cùng 12 per-seed tests trên3 seeds thì family per-seed
là36, không tái dùng Holm12 cho toàn cohort. Seed-level n3 không đủ exact
two-sided sign-flip p<.05 (minimum .25); confirmation cần design/power riêng.
Amis validation/test đã được xem trong development, fresh seeds không biến nó
thành language holdout sạch.

## 13. Novelty audit và điều kiện chưa giải quyết

### 13.1 Prior chính đã đối chiếu

| Prior | Phần đã có | Quan hệ với thiết kế này |
|---|---|---|
| [Houlsby et al., ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html) | Bottleneck adapters | Module nhỏ là thành phần đã biết |
| [MAD-X, EMNLP2020](https://aclanthology.org/2020.emnlp-main.617.pdf) | Modular language/task adapters | Modular cross-lingual transfer đã có |
| [Modular Sentence Encoders, ACL2025, §3.2](https://aclanthology.org/2025.acl-long.108.pdf) | Cross-lingual alignment adapters, giữ các modules trước cố định | Tách alignment khỏi specialization không phải nguyên lý mới |
| [CAFE, arXiv preprint, §5 Eq.8](https://arxiv.org/html/2610.19134v1) | Chặn feature-distillation gradient khỏi shared encoder, chỉ cập nhật adapters | Prior rất gần của gradient policy C2; không nhận confinement là mới |
| [Auxiliary Task Update Decomposition, ICLR2021](https://arxiv.org/abs/2108.11346) | Phân tích/thay đổi auxiliary update directions | “Auxiliary có phần giúp/hại main task” là động cơ đã biết |

Đọc CAFE ngày28/10/2026, bản v1 đăng13/10/2026. Đó là preprint, không gán
nhãn đã được ACL chấp nhận. Paper xét multilingual ASR/feature KD; bài hiện tại
xét MT/cross-layer donor/shared-encoder cosine. Khác task/loss không tự đủ novelty.

Tóm tắt prior ở đây là paraphrase, không phải claim tái lập implementation.
Đối chiếu này có phạm vi, không phải chứng minh không tồn tại mọi prior khác.

### 13.2 Những khác biệt cụ thể còn phải đánh giá

1. Hiệu chỉnh phụ thuộc donor−recipient thay vì local hidden adapter.
2. LSR source→detached target trong cùng model dịch, cùng lúc backbone học CE.
3. Gradient phi vẫn truyền qua các block θ cố định và các module phía trước;
   không đơn giản detach local feature rồi xem đó là cùng Jacobian.
4. Control C3 kiểm donor bundle và C1 kiểm full vs confined gradient.

Đây là **các điểm để so công thức**, chưa phải đóng góp đã được xác nhận.
Nếu Eq.(4–15) chỉ tương đương một adapter/confinement prior sau đổi ký hiệu,
phân loại là adapted control và NO-GO làm phương pháp chính.
Không thêm một phép toán tùy ý chỉ để khác công thức.

Novelty GO cần: công thức phân biệt có ý nghĩa, giải thích được dự đoán kiểm
chứng nào khác prior, matched prior comparator và giới hạn claim rõ.
Không có điều kiện “BLEU cao thì novelty được xem là PASS”.

## 14. Lưu trữ và bàn giao nếu triển khai sau này

Đường source **đã tạo**:

~~~text
src/methods/alignment_residual/v1/
  README.md
  module.py
  model.py
  __init__.py
~~~

CLI/config đi qua shared experiment runner; không viết worker riêng cho từng run.
Method IDs đề xuất:
alignment-residual-capacity-v1,
alignment-residual-fullgrad-v1,
alignment-residual-confined-v1,
alignment-local-confined-v1.
Không dùng C0/C1/C2/C3 làm method name lâu dài hoặc ghi đè IDs cũ.

Metadata bắt buộc: θ revision, module spec/init seed, gradient policy, q/d/α/r/λ,
layer indices/counting, masks/reductions, target-mode/dropout/RNG policy,
all library versions, data/lock/bundle hashes, resolved optimizer args,
selected/final step và stop reason, save-load/resume checks, extra/trainable params.

HF tương lai, chỉ khi có chạy và được phép upload:

~~~text
FiveC/amis-rewire-checkpoints/
  checkpoints/amis_mandarin/mbart-large-50/<method-id>/seed-42/<recipe>/
  analysis/amis_mandarin/mbart-large-50/alignment-residual/seed-42/<recipe>/
~~~

Lịch sử lượt formulation không tạo folder public, upload, tải weights hoặc xóa checkpoint.
Lượt triển khai theo yêu cầu mới dùng runbook riêng; policy frozen-parameter nằm
trong model.py, không cần module gradient_policy riêng.
Chạy sau này chỉ trong project RTX đã được phép, gồm conda/cache/output:
/workspace/ura/thanhdat/projects/CLRR/.
Weights chỉ dọn sau khi backup HF đã checksum verified.

## 15. Kết luận thiết kế

Đã có forward, loss values, gradient policy, Jacobians, optimizer/accumulation,
controls, diagnostics, correctness gates và screening/replication plan đủ cụ thể
để review trước khi code. Hiện đã có CPU correctness results, chưa có training quality results.

**Ưu tiên hiện tại là giải quyết novelty overlap với CAFE/modular alignment.**
Tài liệu này giữ giả thuyết nghiên cứu có thể kiểm chứng; không tự phê duyệt
candidate làm đóng góp chính. Phạm vi queue theo yêu cầu tác giả và runbook triển khai;
quality GO và novelty GO là hai quyết định riêng.
