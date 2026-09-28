# Plan: Quantum measurement cho DualPrompt (task-state PGM + class-state PGM)

Repo đích: fork của https://github.com/JH-LEE-KR/dualprompt-pytorch. Trong tài liệu, đường dẫn file và dòng code là của repo đó (bản `main`). Mọi phần thêm vào **chỉ chạy lúc inference hoặc cuối task**. Training của DualPrompt giữ nguyên từng bit, nên mọi head/router đều được so sánh cặp trên cùng một model, cùng một run.

## 0. Vì sao chọn hướng này

Đã thử nhiều hướng "quantum" trên L2P (Split-CIFAR100, seed 10961, 5 epoch/task; L2P = 84.48 / F 6.30):

- Can thiệp vào prompt không giúp: QSD router 84.08; unitary prompt circuit 80.60 (F 12.4); compositional PGM prompt 80.34; circuit bỏ phase 84.46.
- Kết quả dương duy nhất: **density class head + PGM, fusion với linear head** 85.48, rank 16 đạt 85.61. Head này không đổi training, chỉ đọc feature cuối task.
- Head có feature đóng băng, không drift vẫn quên khoảng 6–9. Nghĩa là forgetting chủ yếu đến từ cạnh tranh giữa các lớp và từ việc chọn sai, không phải từ drift.

DualPrompt khác L2P ở ba điểm, cả ba đều có lợi cho hướng này:

1. **Chọn E-prompt thực chất là đoán task.** Có 10 E-prompt (bằng số task), `top_k=1`, và E-prompt của task t có index t.
2. **Router không tham gia training.** Khi train, `use_prompt_mask=True` ép dùng E-prompt của task hiện tại (`vision_transformer.py:530-534`). Key chỉ được học qua pull constraint. Vì vậy thay router lúc test không làm đổi training.
3. **E-prompt và key của task cũ không bao giờ bị sửa lại.** Bởi vậy trạng thái task tính trên feature ViT đóng băng không drift, và không cần retention loss.

Như vậy "chọn E-prompt" đúng là một bài toán **phân biệt trạng thái lượng tử (QSD)** giữa T trạng thái task. Còn "chọn lớp" là QSD giữa các trạng thái lớp, như ở L2P.

## 1. Phần "lượng tử" ở đây là gì (và không là gì)

| Khái niệm | Dùng ở đâu |
|---|---|
| Density matrix (mixed state) | Mỗi lớp là σ_c; mỗi task là ρ_t, hỗn hợp đều các σ_c của nó |
| Pure state + Born rule | Ảnh test x (đã unit-normalize) là \|x⟩⟨x\|; xác suất = xᵀ E x |
| POVM, pretty-good measurement (PGM, Hausladen–Wootters) | Phép đo đầy đủ trên mọi task/lớp đã thấy, với Σ E = I |
| Coarse-graining POVM | E_t = Σ_{c∈t} E_c; kiểm tra đồng nhất với task-PGM ở §3.3 |
| Đo tuần tự (task → lớp) | Phase 3 |

Mọi thứ được mô phỏng chính xác bằng đại số tuyến tính thực trong PyTorch: không có mạch, không có QPU. Đóng góp có thể bảo vệ là "phân biệt trạng thái task/lớp bằng complete measurement", **không phải lợi thế lượng tử**. Chỉ được nói phần quantum có ích khi PGM thắng các đối chứng classical cùng thống kê (NCM, whitened NCM, LDA) trên nhiều seed.

## 2. Phase 0: sửa hạ tầng và đo headroom (làm trước, rẻ)

### 2.1 Sửa flag bool

Mọi `type=bool` trong `configs/*.py` đều coi chuỗi "False" là True. Thêm hàm sau và thay `type=bool` bằng `type=str2bool` ít nhất cho `--batchwise_prompt`, `--task_inc`, `--train_mask`:

```python
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('1', 'true', 'yes', 'y'): return True
    if v.lower() in ('0', 'false', 'no', 'n'): return False
    raise argparse.ArgumentTypeError('boolean expected')
```

### 2.2 Cho phép ép E-prompt lúc eval

Trong `vision_transformer.py`, thêm tham số `prompt_idx=None` (LongTensor [B], là task id được chọn cho từng ảnh) vào `forward` và `forward_features`:

```python
# forward_features, thay khối chọn prompt_mask:
if prompt_idx is not None:
    k = self.e_prompt.top_k
    prompt_mask = prompt_idx.view(-1, 1) * k + torch.arange(k, device=x.device)  # [B, k]
elif self.use_prompt_mask and train:
    ... (giữ nguyên)
else:
    prompt_mask = None
```

Truyền `prompt_idx` xuống từ `forward(x, task_id, cls_features, train, prompt_idx=None)`. Khi `prompt_idx=None` thì hành vi phải giống hệt code gốc; cần có test cho điều này.

### 2.3 Chế độ chọn prompt theo từng ảnh

Mặc định `batchwise_prompt=True` và mỗi batch test chỉ chứa ảnh của một task. Vote đa số trong batch vì vậy gần như bằng oracle task id. **Mọi so sánh router chạy với `--batchwise_prompt false`.** Chỉ báo cáo thêm bản `true` để đối chiếu với số của README (86.13 / F 5.17).

### 2.4 Đo headroom

Trong `evaluate(... task_id=i ...)`, mọi ảnh thuộc task i (vì loader tách theo task), nên task thật là `i`. Log thêm:

- `route_acc[router][i]`: tỉ lệ ảnh của task i được router chọn đúng E-prompt i.
- `route_confusion[router]`: ma trận T×T.
- Accuracy CIL khi dùng E-prompt oracle (`prompt_idx = i`), gọi là head `oracle_linear`.
- Accuracy TIL (`--task_inc`) với E-prompt oracle, gọi là `oracle_til`.

**Go/no-go:** nếu `oracle_linear − cosine_linear` (per-image) < 1 điểm, router không còn chỗ để thắng. Khi đó bỏ Phase 1–3 và chỉ làm Phase 2A (class head).

## 3. Phase 1: Task-state PGM router

### 3.1 Bank trạng thái lớp (dùng chung cho Phase 1 và 2)

Port nguyên `DensityClassBank` (xem Phụ lục A) vào file mới `quantum_measurement.py`. Cuối task t, với mỗi lớp c ∈ C_t:

```text
X_c   = các feature đã unit-normalize của ảnh TRAIN lớp c, đọc qua EVAL transform  [n_c × D]
Σ̂_c   = X_cᵀ X_c / n_c                        (second moment, trace = 1)
σ_c   = U_c diag(λ_c) U_cᵀ                     top-r eigenpairs của Σ̂_c, λ_c chuẩn hóa Σλ = 1
μ_c   = mean(X_c)            (raw_means, KHÔNG renormalize; dùng cho LDA/centering)
m_c   = normalize(μ_c)       (means; dùng cho NCM)
scatter += (X_c − μ_c)ᵀ(X_c − μ_c)             (within-class dùng chung, float64)
```

Mặc định `r = 32` và `eps = 1e-4`. Tính bằng SVD của `X_c / sqrt(n_c)`: các right singular vector là U_c và λ = s². Mỗi lớp chỉ được ghi một lần, sau đó bất biến.

Có hai nguồn feature, mỗi nguồn một bank:

- `frozen`: `original_model(input)['pre_logits']`, tức CLS của ViT đóng băng, cùng feature với query của E-prompt.
- `prompted`: `model(input, prompt_idx=t)['pre_logits']`, dùng **E-prompt của đúng task t**, khớp với lúc train.

Router ở Phase 1 **chỉ dùng bank `frozen`**.

### 3.2 Trạng thái task là hỗn hợp các trạng thái lớp

```text
ρ_t = (1/|C_t|) Σ_{c∈C_t} σ_c              trace = 1, rank ≤ |C_t|·r
W_t = [ U_c · diag(sqrt(λ_c / |C_t|)) ]_{c∈C_t}   ∈ R^{D × |C_t| r},   ρ_t = W_t W_tᵀ
```

Không cần lưu gì thêm, vì ρ_t là một view trên bank lớp.

### 3.3 PGM trên các task đã thấy

Gọi 𝒯 là tập task đã thấy, T' = |𝒯|, prior đều:

```text
A_t = (ρ_t + eps·I) / T'
S   = Σ_{t∈𝒯} A_t = mean_t ρ_t + eps·I
E_t = S^{-1/2} A_t S^{-1/2}                  ⇒ Σ_t E_t = I   (POVM đầy đủ)
y   = S^{-1/2} x                             (x unit-norm, S^{-1/2} đối xứng)
p_t(x) = xᵀ E_t x = ( ‖W_tᵀ y‖² + eps·‖y‖² ) / T'   ⇒ Σ_t p_t(x) = ‖x‖² = 1
```

Ridge `eps·I` phải có ở **mỗi** A_t, không chỉ ở S; nếu thiếu thì mất tính completeness. Tính S^{-1/2} bằng `eigh` ở float64 (clamp trị riêng ≥ 0 rồi cộng eps), cache lại theo tập task đã thấy, và xóa cache khi thêm lớp hoặc load checkpoint.

**Đồng nhất coarse-graining (dùng làm unit test):** khi mọi task có cùng số lớp, PGM lớp với prior đều có cùng S, và Σ_{c∈t} A_c^{class} = (ρ_t + eps I)/T'. Suy ra:

```text
p_t^{task-PGM}(x) = Σ_{c∈C_t} p_c^{class-PGM}(x)
```

Test phải khẳng định đẳng thức này tới sai số float. Vì vậy chỉ cần cài một hàm PGM, và task-PGM là tổng các outcome lớp theo task.

### 3.4 Các router (đều top-1, chỉ trên task đã thấy)

| Router | Công thức chọn t̂ | Vai trò |
|---|---|---|
| `cosine` | argmax_{t∈0..9} cos(q, k_t), gồm cả key chưa train | Baseline DualPrompt gốc |
| `cosine_seen` | argmax_{t∈𝒯} cos(q, k_t) | Tách lợi ích của mask task chưa thấy |
| `task_ncm` | argmax_t cos(q, normalize(mean_{c∈t} μ_c)) | Prototype classical |
| `task_ncm_white` | argmax_t cos(S^{-1/2}q, S^{-1/2} mean_{c∈t} μ_c) | Whitening mà không dùng Born rule |
| `class_ncm_task` | task của argmax_c cos(q, m_c) | Coarse-graining bản classical cứng |
| `task_lda` | argmax_t Σ_{c∈t} softmax_c(LDA_c(q)) | **Đối chứng khớp với PGM** (cùng thống kê bậc hai) |
| `task_fidelity` | argmax_t qᵀ ρ_t q | Born rule nhưng không thiết kế phép đo |
| `task_pgm` | argmax_t p_t(q) | **Phương pháp** |
| `task_pgm_r8`, `task_pgm_r16` | như trên, σ_c cắt còn 8/16 eigenpair rồi renormalize | Ablation về rank |
| `oracle` | t̂ = task thật | Cận trên |

LDA (giống `density_head.py` bên L2P):

```text
Σ_w = scatter / Σ n_c ;  ridge = eps · tr(Σ_w)
w_c = (Σ_w + ridge·I)^{-1} μ_c ;  b_c = −½ μ_cᵀ w_c ;  LDA_c(x) = xᵀ w_c + b_c
```

q ở mọi router là `cls_features` đã unit-normalize (chính query của DualPrompt).

### 3.5 Đánh giá nhiều router với chi phí thấp

Mỗi router cho ra `idx_r ∈ [B]`. Với mỗi batch, lấy tập U các task được ít nhất một router chọn. Với mỗi t ∈ U, chỉ forward các ảnh có router nào đó chọn t, dùng `prompt_idx = t`, và lưu `logits`, `pre_logits` theo (ảnh, t). Sau đó lắp kết quả cho từng router. Phần lớn router trùng lựa chọn nên chi phí chỉ khoảng 1.2–2 lần forward.

Head mặc định dưới mỗi router là `<router>_linear`: argmax logits trên các lớp đã thấy. Logit của lớp chưa thấy được gán `-inf` ở mọi head, kể cả `cosine_linear`. Nên log thêm `cosine_linear_nomask`, đúng như code gốc, để tách lợi ích của việc mask.

## 4. Phase 2: Class-state PGM head + fusion

### 4.1 Class PGM (M = số lớp đã thấy)

```text
A_c = (σ_c + eps·I)/M,  S = Σ_c A_c,  E_c = S^{-1/2} A_c S^{-1/2}
p_c(x) = ( Σ_k λ_ck (u_ckᵀ y)² + eps‖y‖² ) / M,   y = S^{-1/2} x
```

### 4.2 Fusion (product of experts, không có tham số học)

```text
score_c = log_softmax_{c∈seen}(z(x))_c + w · log p_c(x)      w = 1.0
ŷ = argmax_c score_c
```

`z(x)` là logits của DualPrompt dưới router đang xét, và `x` là feature của nguồn tương ứng.

### 4.3 Các head cần đánh giá

Phase 2A đánh giá dưới router `cosine` (bắt buộc) và `oracle`. Phase 2B đánh giá thêm dưới `task_pgm` và `task_lda`, cấu hình qua `--qm_head_routers`. Với mỗi nguồn `frozen`/`prompted`:

- `ncm`, `ncm_white`, `fidelity`, `pgm`, `pgm_r8`, `pgm_r16`, `lda`
- `fusion` (dùng pgm), `fusion_r8`, `fusion_r16`, `lda_fusion` (đối chứng khớp)

Với nguồn `prompted`, feature lúc test được tính dưới E-prompt mà router chọn, còn bank được xây dưới E-prompt đúng task. Khi router chọn sai, feature và bank lệch nhau; đó là một phần của hiện tượng cần đo.

## 5. Phase 3: Đo tuần tự task → lớp (chỉ làm nếu Phase 1 có headroom)

E-prompt t được train với `train_mask` trên C_t, nên softmax trong C_t đúng là thứ nó được học để làm. Phân tách:

```text
P(c | x) = P(t(c) | x) · P(c | x, t(c)),    P(c | x, t) = softmax_{c'∈C_t}( z^{(t)}(x) )_c
```

- **Hard (1 forward):** `<router>_til`: t̂ lấy từ router, ŷ = argmax_{c∈C_{t̂}} z^{(t̂)}(x).
- **Soft top-m (m forward, mặc định m = 2):** chọn m task có P(t|x) cao nhất, rồi

  ```text
  P(c|x) ∝ P(t|x) · softmax_{C_t}(z^{(t)}(x))_c,   t ∈ top-m,   ŷ = argmax_c
  ```

  P(t|x) lấy từ `task_pgm` (phương pháp) hoặc `task_lda` (đối chứng). Cả hai đều là xác suất đã chuẩn hóa, không cần nhiệt độ.
- **Soft + class PGM:** nhân thêm `p_c(x)^w` của class PGM `frozen`, rồi chuẩn hóa lại.

## 6. Tích hợp vào code

### 6.1 File mới `quantum_measurement.py`

- `DensityClassBank`: port nguyên từ Phụ lục A. Thêm `task_of_class: LongTensor[num_classes]` (buffer) để gom lớp theo task.
- `task_scores(features, bank, task_of_class) -> dict[str, Tensor[B, num_tasks]]`: log-prob hoặc score cho mọi router ở §3.4, task chưa thấy nhận `-inf`. Task-PGM = `logsumexp` của `pgm` lớp theo task (§3.3).
- `route(...) -> dict[str, LongTensor[B]]`.
- `add_qm_args(parser)` với các cờ: `--qm_eval` (bật mọi thứ), `--qm_sources frozen prompted`, `--qm_rank 32`, `--qm_eps 1e-4`, `--qm_fusion_weight 1.0`, `--qm_pgm_ranks 8 16`, `--qm_head_routers cosine oracle`, `--qm_soft_topm 2`.

### 6.2 `engine.py`

1. Cuối mỗi task, **sau vòng epoch và trước `evaluate_till_now`**, gọi `consolidate_qm_bank(model, original_model, data_loader[task_id], class_mask[task_id], task_id, args)`:

   ```python
   @torch.no_grad()
   def consolidate_qm_bank(...):
       rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
       ds = _with_transform(loaders['train'].dataset, loaders['val'].dataset.transform)  # eval transform
       loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
       model.eval(); original_model.eval()
       feats = {src: {c: [] for c in class_mask_t} for src in args.qm_sources}
       for x, y in loader:
           q = original_model(x)['pre_logits']
           p = model(x, cls_features=q, prompt_idx=torch.full_like(y, task_id))['pre_logits']
           # gom theo y vào feats['frozen'], feats['prompted']
       for src, per_class in feats.items():
           for c, f in per_class.items(): banks[src].add_class(c, torch.cat(f), task=task_id)
       torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
   ```

   `_with_transform` đệ quy qua `Subset`, `copy.copy` dataset gốc rồi gán `.transform = transform_val`. Không dùng test hay validation để fit. Khôi phục RNG là bắt buộc, vì tạo iterator của DataLoader rút một seed, làm đổi shuffle của task sau.
2. `evaluate(...)` khi bật `--qm_eval`: tính `cls_features`, lấy mọi router (§3.4), forward khử trùng (§3.5), rồi tính mọi head (§4.3, §5). Trả về `{head_name: acc1}` và `route_acc`.
3. `evaluate_till_now`: một `acc_matrix` riêng cho mỗi head, đi qua cùng công thức Acc/Forgetting/Backward như code gốc.

### 6.3 Checkpoint

Không gắn bank vào `model`, để giữ nguyên `state_dict` và optimizer. Lưu thêm `state_dict['qm_banks'] = {src: bank.state_dict()}` và xóa cache sau khi load.

### 6.4 DDP

Phiên bản đầu chỉ chạy một GPU (Kaggle). Nếu chạy DDP, phải `all_gather` feature theo lớp trước `add_class` và gộp metric; chưa làm được thì assert `world_size == 1` khi bật `--qm_eval`.

### 6.5 `results_summary.json`

Ghi sau mỗi task vào `--output_dir`. Với mọi head: final acc, average incremental acc, forgetting, backward, acc_matrix. Với mọi router: `route_acc` theo task và ma trận confusion. Thêm runtime, và in ra một bảng ngắn khi chạy xong.

## 7. Thí nghiệm

- Split-CIFAR100, 10 task, config mặc định (5 epoch, batch 24, lr 0.03). Mọi so sánh dùng `--batchwise_prompt false`.
- **Ít nhất 3 seed.** Siêu tham số chốt trước và không tune trên test: rank 32, eps 1e-4, w = 1.0, rank cắt 8/16, m = 2.
- Mỗi seed là một run với `--qm_eval`, cho ra mọi router và head. Chạy thêm một run không có cờ để xác nhận training trùng khớp.
- Nếu còn thời gian: ImageNet-R (`train_imr_dualprompt.sh`). Domain ở đó khác xa pretrain nên đoán task khó hơn, và headroom có thể lớn hơn.

**Bảng cần báo cáo:**

1. Router: `route_acc` trung bình, và `<router>_linear` Acc/Forgetting cho mọi router ở §3.4.
2. Class head dưới `cosine` và `oracle`: linear, ncm_white, pgm, lda, fusion, fusion_r8, fusion_r16, lda_fusion.
3. Phase 3: `<router>_til` và soft top-2 cho `task_pgm` so với `task_lda`.

**Tiêu chí để kết luận:**

- Phần quantum ở router có ích khi `task_pgm > task_lda`, `task_ncm_white` và `cosine_seen` về route_acc và Acc, nhất quán trên các seed.
- Phần quantum ở head có ích khi `fusion > lda_fusion` và `fusion > linear`.
- `task_fidelity < task_pgm` cho thấy phần thiết kế phép đo (PGM) mới là thứ quan trọng, không chỉ Born rule.
- Nếu `task_lda ≈ task_pgm` thì kết luận trung thực là: lợi ích đến từ thống kê bậc hai của lớp, không riêng gì phép đo lượng tử.

## 8. Test tối thiểu (`tests/test_quantum_measurement.py`)

1. **Completeness:** Σ_t E_t = I và Σ_c E_c = I (dựng ma trận đầy đủ với D nhỏ), Σ p = 1 cho x ngẫu nhiên.
2. **Coarse-graining:** task-PGM bằng tổng class-PGM theo task khi số lớp mỗi task bằng nhau (§3.3).
3. Công thức eigenpair khớp với dựng ma trận đầy đủ; rank cắt được renormalize.
4. Task và lớp chưa thấy không bao giờ thắng; `oracle` trả về task thật.
5. `prompt_idx=None` cho logits giống hệt code gốc; `prompt_idx = task_id` khớp với `use_prompt_mask` lúc train.
6. **Training không đổi:** smoke test 2 task, cùng seed, bật và tắt `--qm_eval` cho trọng số giống hệt sau task 2.
7. Checkpoint round-trip giữ nguyên prediction của mọi head.
8. `str2bool("false") is False`.

## 9. Thứ tự làm và điểm dừng

1. Phase 0 (§2), rồi chạy 1 seed để có `route_acc(cosine)` và khoảng cách oracle. **Quyết định go/no-go.**
2. Phase 2A (§4) dưới `cosine`/`oracle`: độc lập với kết quả go/no-go, vì đã có tín hiệu dương từ L2P.
3. Nếu go: Phase 1 (§3), rồi Phase 2B, rồi Phase 3 (§5).
4. Chạy 3 seed, lập các bảng ở §7.

Chưa làm trong plan này: loss mới lúc train (margin, calibration, relational distillation), mạch lượng tử có tham số, can thiệp vào G-prompt. Chỉ cân nhắc sau khi §7 có kết quả.

## Phụ lục A: `DensityClassBank` tham chiếu (từ l2p-pytorch, nhánh `complete-measurement`)

```python
class DensityClassBank(nn.Module):
    def __init__(self, num_classes, dim, rank=32, eps=1e-4, pgm_ranks=(), lda_ridge=None):
        super().__init__()
        self.rank, self.eps, self.pgm_ranks = rank, eps, tuple(sorted(set(pgm_ranks)))
        self.lda_ridge = eps if lda_ridge is None else lda_ridge
        self.register_buffer('vectors', torch.zeros(num_classes, dim, rank))
        self.register_buffer('values', torch.zeros(num_classes, rank))
        self.register_buffer('means', torch.zeros(num_classes, dim))
        self.register_buffer('raw_means', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('scatter', torch.zeros(dim, dim, dtype=torch.float64))
        self.register_buffer('scatter_count', torch.zeros((), dtype=torch.long))
        self._cache = {}
        self.register_load_state_dict_post_hook(lambda *a: self._cache.clear())

    @torch.no_grad()
    def add_class(self, label, features):
        assert not self.valid[label] and len(features) > 0
        x = F.normalize(features.to(self.vectors.device, torch.float64), dim=-1)
        _, s, vh = torch.linalg.svd(x / math.sqrt(len(x)), full_matrices=False)
        r = min(self.rank, len(s)); lam = s[:r].square()
        self.vectors[label].zero_(); self.values[label].zero_()
        self.vectors[label, :, :r] = vh[:r].T.to(self.vectors.dtype)
        self.values[label, :r] = (lam / lam.sum()).to(self.values.dtype)
        mean = x.mean(0)
        self.raw_means[label] = mean.to(self.raw_means.dtype)
        self.means[label] = F.normalize(mean, dim=0).to(self.means.dtype)
        c = x - mean
        self.scatter += c.T @ c; self.scatter_count += len(x)
        self.valid[label] = True; self._cache.clear()

    def _states(self, classes, rank=None):
        v, lam = self.vectors[classes], self.values[classes]
        if rank is not None:
            v, lam = v[..., :rank], lam[:, :rank]
            lam = lam / lam.sum(-1, keepdim=True).clamp_min(1e-12)
        return v, lam

    def _pgm_root(self, classes, rank=None):          # S^{-1/2}, cache theo (rank, classes)
        key = ('pgm', rank, tuple(classes.tolist()))
        if key not in self._cache:
            v, lam = self._states(classes, rank)
            w = (v.double() * lam.double().sqrt().unsqueeze(1)).permute(1, 0, 2).flatten(1)
            evals, evecs = torch.linalg.eigh(w @ w.T / len(classes))
            evals = evals.clamp_min(0) + self.eps
            self._cache[key] = ((evecs * evals.rsqrt()) @ evecs.T).to(self.vectors.dtype)
        return self._cache[key]

    @staticmethod
    def _energy(x, v, lam):                            # <x|σ_c|x> cho mọi c
        return (torch.einsum('bd,cdk->bck', x, v).square() * lam).sum(-1)

    def pgm_log_probs(self, x, classes, rank=None):    # x đã unit-normalize, [B, D]
        v, lam = self._states(classes, rank)
        y = x @ self._pgm_root(classes, rank)
        born = (self._energy(y, v, lam) + self.eps * y.square().sum(-1, keepdim=True)) / len(classes)
        born = born.clamp_min(1e-12)
        return (born / born.sum(-1, keepdim=True)).log()   # [B, |classes|]

    def lda_log_probs(self, x, classes):
        cov = self.scatter / self.scatter_count.clamp_min(1)
        ridge = self.lda_ridge * cov.diagonal().sum().clamp_min(1e-6)
        mu = self.raw_means[classes].double()
        w = torch.linalg.solve(cov + ridge * torch.eye(len(cov), dtype=cov.dtype, device=cov.device), mu.T)
        b = -0.5 * (mu * w.T).sum(-1)
        return (x.double() @ w + b).log_softmax(-1).to(x.dtype)
```

Các readout còn lại:

- `ncm = x @ means[classes].T`
- `ncm_white = normalize(x @ R) @ normalize(raw_means[classes] @ R).T` với `R = _pgm_root(classes)`
- `fidelity = log normalize(_energy(x, v, lam))`

Task-PGM là `logsumexp` của `pgm_log_probs` theo nhóm lớp của từng task (§3.3). `task_lda` làm tương tự với `lda_log_probs`.
