# Quantum PGM heads cho Class-Incremental Learning

Nhánh `density-pgm-heads` của RainbowPrompt, commit `c627ebc`.
Code gốc: <https://github.com/thanhphu25/RainbowPrompt/blob/c627ebc/density_head.py>.

## 1. Ý tưởng

- Mỗi class được lưu thành một **density matrix** `ρ_c` low-rank, lấy từ feature của ViT. Lưu một lần
  sau khi học class đó, không cập nhật lại và không lưu exemplar.
- Lúc test, dùng **PGM (Pretty Good Measurement)**, một phép đo lượng tử có công thức đóng, để ra
  `p(c|x)` với tổng đúng bằng 1.
- Xác suất này được dùng làm head mới, hoặc fusion với linear head.
- Không train thêm gì. Thêm class mới chỉ cần tính lại `S^{-1/2}` một lần.

## 2. Vị trí trong kiến trúc và pipeline

```
ảnh x ─► ViT frozen ───────────► f_frozen ─────────────┐
      └► ViT + prompt (method) ─► f_prompted ─► linear ─► logits (baseline)
                                      │                 │
                                      ▼                 ▼
                          PGM(f_prompted)     PGM(f_frozen) + fusion với logits   ← MỚI
```

Density heads nằm **song song với linear head, sau backbone**, và không đụng vào prompt, loss hay
optimizer. Chỉ cần 3 hook trong engine:

1. **Sau khi train xong task t:** forward train split của task t với eval transform và đường
   inference, rồi `density.add_task(feats, labels)`. Phải lưu và khôi phục RNG quanh bước này để
   quá trình train không bị lệch.
2. **Trong `evaluate`:** `density.predict({'frozen': f_frozen, 'prompted': f_prompted}, logits)`, rồi
   đếm đúng cho từng head.
3. **Checkpoint:** lưu thêm `density.state_dict()`.

`f_frozen = original_model(x)['pre_logits']`, `f_prompted = model(x)['pre_logits']` (đúng input của
linear head). `logits` phải có chỉ số cột bằng global class id.

## 3. Công thức cốt lõi

```
Class c, feature đã L2-norm X̂ (n×D):
    ρ_c = Σ_j λ_j u_j u_jᵀ     top-r eigen của X̂ᵀX̂/n (KHÔNG trừ mean), Σλ = 1

S     = (1/M) Σ_c ρ_c + εI                     M = số class đã thấy, ε = 1e-4
y     = S^{-1/2} x̂
p_c   = (1/M) [ Σ_j λ_cj (u_cjᵀ y)² + ε‖y‖² ]    → Σ_c p_c = 1  (POVM đầy đủ nhờ εI)

Fusion:  score_c = log_softmax(logits[:, seen])_c + w · log p_c      (w = 1)
```

Code tham chiếu: `ClassBank.add_class`, `ClassBank.measurement`, `ClassBank.pgm_probs` và
`DensityHeads.predict` trong `density_head.py`. Rank đọc `k < r` lấy k cột đầu rồi chuẩn hoá lại λ.

## 4. Kết quả hiện có

CUB-200, 20 task, ViT-B/16, 20 epoch, 1 seed (10961). Baseline là linear head của RainbowPrompt.

| Head | Acc | vs linear | IncAcc | Forget |
|---|---|---|---|---|
| linear (baseline) | 82.17 | 0.00 | 84.31 | 5.11 |
| **prompted_pgm_r8** | **85.42** | **+3.26** | 85.74 | 4.65 |
| prompted_pgm_r16 | 85.22 | +3.05 | 85.68 | 4.83 |
| prompted_pgm (r32) | 84.99 | +2.82 | 85.62 | 4.98 |
| **frozen_pgm_fusion_r8** | **84.85** | **+2.69** | 86.02 | 4.52 |
| frozen_pgm_fusion_r16 | 84.70 | +2.53 | 85.98 | 4.54 |
| frozen_pgm_fusion (r32) | 84.65 | +2.48 | 85.93 | 4.60 |
| *tham chiếu classical:* frozen_lda_fusion | 83.81 | +1.65 | 84.95 | 4.02 |
| *tham chiếu classical:* prompted_ncm_whitened | 85.85 | +3.68 | 86.04 | 4.97 |

Head classical tốt nhất vẫn là `prompted_ncm_whitened`. Mục tiêu của các hướng dưới đây là vượt
được nó.

## 5. Ba hướng chính

### Hướng 1: fusion linear với PGM trên feature frozen (bằng chứng mạnh nhất)

- **Là gì:** `log_softmax(linear trên feature prompted) + w·log p_PGM(f_frozen)`.
- **Vì sao mới:** PGM chưa từng được dùng trên feature sâu hay trong CIL. Chưa ai fusion xác suất PGM
  với linear head ở mức class.
- **Số hiện tại:**
  - +2.69 so với linear, và **+1.04 so với LDA fusion** dùng cùng feature frozen. Đây là chỗ duy
    nhất PGM thắng rõ classical.
  - PGM frozen đứng một mình thì thua linear (79.17). Nó chỉ mạnh khi bổ sung cho logits prompted,
    vì hai nguồn mang thông tin khác nhau.
- **Việc tiếp:**
  - Tune w.
  - Thử rank 1, 2, 4.
  - So thêm với FeCAM-fusion và RanPAC-fusion.
  - Chạy CIFAR-100 và ImageNet-R, 3 seed.

### Hướng 2: PGM low-rank thay linear head trên feature prompted

- **Là gì:** dùng trực tiếp `argmax p_PGM(f_prompted)`.
- **Số hiện tại:** +3.26 ở r8. Rank càng thấp càng tốt (r8 > r16 > r32), nhưng vẫn kém NCM whitened
  0.43 điểm, nằm trong mức nhiễu của 1 seed.
- **Việc tiếp:**
  - Rank 1, 2, 4. Lưu ý: khi r → 1 thì PGM tiến gần NCM whitened, nên cần một cải tiến *mang tính
    lượng tử* mới vượt được.
  - Ứng viên chính là **nhiều bản sao `ρ^{⊗k}`** (Giuntini et al. 2023, Cruzeiro et al. 2024),
    tương đương kernel đa thức bậc k trên feature. Cần kernel trick hoặc Nyström vì D^k lớn. Đọc
    định nghĩa chính xác trong hai paper trước khi cài.

### Hướng 3: kết hợp hai nguồn PGM (chưa chạy)

- **Là gì:** `log_softmax(linear) + w1·log p_PGM(frozen) + w2·log p_PGM(prompted)`.
- **Vì sao:** ghép hai head đang tăng lại với nhau. Nguồn frozen bổ sung thông tin, nguồn prompted
  chính xác hơn. Chưa ai làm.
- **Chi phí:** chỉ cần thêm vài dòng trong `DensityHeads.predict`. **Không phải train lại**: chạy
  `--eval` trên checkpoint từng task của lần chạy CUB. Checkpoint đã chứa `density_bank`.
- **Lưu ý khi eval offline:** `DensityHeads.load_state_dict` hiện **ghi đè** `ranks` và `weight` bằng
  giá trị trong checkpoint. Phải sửa để ưu tiên args, nếu không `--density_ranks 1 2 4` và w mới sẽ
  bị bỏ qua.

## 6. Điểm mới so với prior art

- **Đã có, phải trích dẫn:**
  - PGM làm classifier: Giuntini et al., ASC 2023 (arXiv:2104.00971). Zambrini Cruzeiro et al.,
    QMI 2024 (arXiv:2303.15353). Sergioli et al. 2026 (arXiv:2603.00223).
  - Density matrix làm prototype trong CIL: Wu et al. 2026 (arXiv:2608.10464). Bài này train
    prototype, đo bằng khoảng cách Hilbert–Schmidt trên mạch lượng tử mô phỏng, có replay và chạy
    ở quy mô nhỏ.
- **Mới (chưa tìm thấy):** PGM đầy đủ trên feature ViT pre-trained hoặc prompted, dùng cho CIL
  exemplar-free, fusion mức class với linear head, và phân tích hai nguồn frozen/prompted.
- **Baseline bắt buộc để claim:** LDA, NCM whitened, FeCAM, RanPAC trên cùng model đã train.

## 7. Triển khai ở repo khác

Nguyên tắc: `density_head.py` **không phụ thuộc phương pháp nền**. Mọi phần phụ thuộc repo đích
nằm ở 5 điểm tích hợp (§7.3–§7.7). Code mẫu dưới đây lấy từ commit `c627ebc`, cấu trúc theo
codebase DualPrompt của JH-LEE-KR. Repo khác thì giữ nguyên logic và đổi tên biến cho khớp.

### 7.1 Lấy code

```bash
SHA=c627ebcf349a0f8ed652e5df4c7a6a56a1e8a408
curl -L -o density_head.py https://raw.githubusercontent.com/thanhphu25/RainbowPrompt/$SHA/density_head.py
mkdir -p tests && curl -L -o tests/test_density_head.py \
  https://raw.githubusercontent.com/thanhphu25/RainbowPrompt/$SHA/tests/test_density_head.py
```

- `density_head.py` chỉ cần `torch`, `numpy`, `loguru`. Nếu repo đích không có `loguru` thì
  `pip install loguru`, hoặc đổi dòng `from loguru import logger` thành
  `import logging; logger = logging.getLogger(__name__)`.
- Test 1–7 trong `tests/test_density_head.py` chạy được ngay. Test 8
  (`test_engine_consolidate_and_evaluate`) gọi `engine.consolidate_density_heads` và
  `engine.evaluate` của repo này, nên phải sửa theo engine của repo đích (xem §7.8).

### 7.2 Hợp đồng dữ liệu (kiểm tra trước khi viết code)

| Thứ cần có | Shape | Lấy từ đâu | Điều kiện bắt buộc |
|---|---|---|---|
| `f_prompted` | `(B, D)` | Output của model phương pháp | **Đúng vector đi vào linear head**: sau `norm`/pool, trước `head`. Trong repo JH-LEE-KR là `output['pre_logits']` |
| `f_frozen` | `(B, D)` | ViT pre-trained chưa adapt | Prompt-CIL thường đã có sẵn làm query: `original_model(x)['pre_logits']`. Nếu không có thì bỏ nguồn này (`--density_sources prompted`) |
| `logits` | `(B, C_total)` | Linear head | **Cột thứ j ứng với global class id j.** Nếu repo chỉ trả logits của task hiện tại thì phải ghép lại thành đủ C_total cột |
| `target` | `(B,)` | Loader | Cùng không gian nhãn với cột của `logits` |
| Train split của task t với **eval transform** | loader | Dataset | Không augmentation. Chỉ gồm class của task t |
| Forward lúc test | hàm | Engine | Consolidate phải gọi **đúng lời gọi model trong `evaluate()`** (cùng tham số, `train=False`, cùng cách chọn prompt/task) |

### 7.3 Bước 1: thêm args và khởi tạo

Với repo dùng argparse (JH-LEE-KR, HiDe-Prompt):

```python
# main.py
from density_head import DensityHeads, HeadMetrics, add_density_args
...
get_args_parser(config_parser)
add_density_args(config_parser)          # thêm SAU parser của config
args = parser.parse_args()
...
density, head_metrics = None, None
if args.density_heads:
    if args.distributed:
        logger.warning('density heads do not aggregate statistics across processes; disabled under DDP')
    else:
        density = DensityHeads(args, dim=model.num_features)   # D = 768 với ViT-B/16
        head_metrics = HeadMetrics(args.num_tasks)
```

Với repo dùng config JSON/dict (LAMDA-PILOT), tạo một `Namespace` có đủ 6 trường mà
`DensityHeads` đọc:

```python
from argparse import Namespace
dargs = Namespace(density_sources=cfg.get('density_sources', ['frozen', 'prompted']),
                  density_rank=cfg.get('density_rank', 32), density_ranks=cfg.get('density_ranks', [8, 16]),
                  density_eps=cfg.get('density_eps', 1e-4), density_fusion_weight=cfg.get('density_fusion_weight', 1.0),
                  density_lda_shrink=cfg.get('density_lda_shrink', 0.1))
density = DensityHeads(dargs, dim=768)
head_metrics = HeadMetrics(num_tasks)
```

**DDP:** bank chỉ thấy dữ liệu của process hiện tại, nên phải tắt, hoặc tự `all_gather` feature và
nhãn trước `add_task`. `DataParallel` một process (PILOT dùng `self._network.module`) thì không bị
vấn đề này.

### 7.4 Bước 2: loader train split với eval transform

**Không được** gán `dataset_train.transform = transform_val` trực tiếp, vì như vậy sẽ mất
augmentation của quá trình train. Phải tạo một bản copy:

```python
# datasets.py
import copy
from torch.utils.data import Subset

def _with_transform(dataset, transform, cache):
    # shallow copy of the (underlying) dataset with another transform; Subset indices are kept
    if isinstance(dataset, Subset):
        return Subset(_with_transform(dataset.dataset, transform, cache), dataset.indices)
    if id(dataset) not in cache:
        ds = copy.copy(dataset)
        ds.transform = transform
        cache[id(dataset)] = ds
    return cache[id(dataset)]

# trong build_continual_dataloader, eval_copies = {} khai báo MỘT lần trước vòng lặp task,
# rồi với mỗi task:
if getattr(args, 'density_heads', False):
    loaders['train_eval'] = torch.utils.data.DataLoader(
        _with_transform(dataset_train, transform_val, eval_copies),
        sampler=torch.utils.data.SequentialSampler(dataset_train),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem)
```

- Nếu dataset gốc lưu transform ở thuộc tính khác (ví dụ `transforms`, hoặc transform nằm trong
  wrapper), sửa `_with_transform` cho khớp.
- **LAMDA-PILOT / PyCIL:** data manager có sẵn chế độ này, ví dụ
  `data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")`.
  Đây là mẫu của PyCIL, cần kiểm tra lại chữ ký hàm trong repo.

### 7.5 Bước 3: consolidate sau khi train xong mỗi task, TRƯỚC khi evaluate

```python
# engine.py
import random
import numpy as np

def _get_rng_state():
    state = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def _set_rng_state(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

@torch.no_grad()
def consolidate_density_heads(model, original_model, data_loader, device, task_id, density, args=None):
    """Add class states of the current task, using its train split with the eval transform.
    RNG state is restored afterwards so enabling the heads does not change later training."""
    rng_state = _get_rng_state()
    model.eval()
    original_model.eval()
    feats = {s: [] for s in density.sources}
    labels = []
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        cls_features = original_model(input)['pre_logits']
        # ↓ PHẢI giống hệt lời gọi model trong evaluate() của repo đích
        output = model(input, task_id=task_id, learned_id=task_id, cls_features=cls_features)
        batch_feats = {'frozen': cls_features, 'prompted': output['pre_logits']}
        for s in density.sources:
            feats[s].append(batch_feats[s].float().cpu())
        labels.append(target.cpu())
    density.add_task({s: torch.cat(v) for s, v in feats.items()}, torch.cat(labels))
    _set_rng_state(rng_state)
    logger.info(f'Density heads: stored {len(density)} classes after task {task_id + 1}')
```

Gọi trong vòng lặp task, **sau epoch cuối và trước `evaluate_till_now`**:

```python
for task_id in range(args.num_tasks):
    ...  # train các epoch như cũ
    if density is not None:
        consolidate_density_heads(model, original_model, data_loader[task_id]['train_eval'],
                                  device, task_id, density, args)
    test_stats = evaluate_till_now(..., density=density, head_metrics=head_metrics)
```

- **Vì sao phải khôi phục RNG:** vòng lặp DataLoader rút base seed từ torch RNG. Nếu không khôi
  phục, các task sau sẽ train khác đi, và `linear` khi bật heads không còn so sánh được với baseline.
- **Phương pháp có nhiều pha train mỗi task** (ví dụ HiDe-Prompt có pha task-identity rồi pha
  sửa/TAP): consolidate **sau pha cuối cùng** của task.
- **LAMDA-PILOT:** vòng lặp là `incremental_train → eval_task → after_task`. `after_task` chạy
  **sau** eval, nên **không** được đặt consolidate ở đó. Đặt ở cuối `incremental_train`, hoặc trong
  `trainer.py` giữa hai lời gọi. Lấy feature bằng `self._network.extract_vector(x)` (đang được
  `_extract_vectors` dùng), nhưng phải kiểm tra nó trả đúng vector đi vào head của phương pháp
  prompt.

### 7.6 Bước 4: hook trong evaluate

```python
def evaluate(model, original_model, data_loader, device, task_id=-1, cur_id=-1,
             class_mask=None, args=None, density=None):
    use_density = density is not None and len(density) > 0
    head_correct, head_total = {}, 0
    ...
    for input, target in data_loader:
        ...
        cls_features = original_model(input)['pre_logits']
        output = model(input, task_id=task_id, learned_id=cur_id, cls_features=cls_features)
        logits = output['logits']          # sau khi áp mask task-inc nếu có, như code gốc
        ...
        if use_density:
            feats = {'frozen': cls_features, 'prompted': output['pre_logits']}
            for head, pred in density.predict(feats, logits).items():
                head_correct[head] = head_correct.get(head, 0) + (pred == target).sum().item()
            head_total += target.shape[0]
    ...
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if use_density:
        stats['density'] = {h: 100.0 * c / head_total for h, c in head_correct.items()}
    return stats
```

Trong `evaluate_till_now`, với mỗi task i ≤ t đã được evaluate:

```python
if head_metrics is not None and 'density' in test_stats:
    for head, acc in test_stats['density'].items():
        head_metrics.update(head, i, task_id, acc)
# sau vòng lặp i:
if head_metrics is not None and head_metrics.acc:
    head_stats = head_metrics.end_task(task_id)
    head_metrics.log_table(task_id, head_stats)
    if args.output_dir and utils.is_main_process():
        head_metrics.save(args.output_dir, task_id, head_stats)
```

Nếu có ghi log `test_*` ra JSON thì bỏ khoá `density`:
`{f'test_{k}': v for k, v in test_stats.items() if k != 'density'}`.

**Repo evaluate mọi class đã thấy trong một loader** (LAMDA-PILOT làm vậy): gom `pred` và `target`
của từng head qua toàn bộ tập test, rồi tính accuracy **theo từng task** bằng khoảng class id. Với
PILOT, nhãn đã được đánh lại theo class order, nên task i là
`[sum(increments[:i]), sum(increments[:i+1]))`. Sau đó gọi `head_metrics.update(head, i, t, acc_i)`
cho từng i. Không làm vậy thì Forgetting sẽ sai.

### 7.7 Bước 5: checkpoint và eval offline

```python
if density is not None:
    state_dict['density_bank'] = density.state_dict()
...
# chế độ --eval, với mỗi task t:
if density is not None:
    if 'density_bank' in checkpoint:
        density.load_state_dict(checkpoint['density_bank'])
    else:
        density, head_metrics = None, None
```

**Bản vá cần có khi eval offline với rank hoặc w mới.** Hiện tại `load_state_dict` ghi đè `ranks` và
`weight` bằng giá trị lưu trong checkpoint. Thêm tham số và mặc định giữ hành vi cũ, để
`test_checkpoint_round_trip` vẫn qua:

```python
# density_head.py, DensityHeads
def load_state_dict(self, state, use_saved_config=True):
    for s in self.sources:
        self.banks[s].load_state_dict(state['banks'][s])
    self.rank = self.banks[self.sources[0]].rank
    if use_saved_config:
        self.ranks = list(state['ranks'])
        self.weight = state['weight']
    else:  # giữ ranks/weight từ args hiện tại, chỉ bỏ rank >= rank đã lưu
        self.ranks = sorted(k for k in self.ranks if 0 < k < self.rank)
```

Trong chế độ `--eval` thì gọi `density.load_state_dict(ckpt['density_bank'], use_saved_config=False)`.
Rank đọc phải **nhỏ hơn** rank đã lưu (32). Muốn thử rank lớn hơn thì phải consolidate lại.

### 7.8 Bước 6 (tùy chọn): head fusion 3 nguồn cho Hướng 3

Trong `DensityHeads.predict`, lưu `log p` của PGM theo `(nguồn, rank)` trong vòng lặp, rồi ghép lại
sau vòng lặp:

```python
logp = {}
for s in self.sources:
    ...
    for k, suffix in [(self.rank, '')] + [(k, f'_r{k}') for k in self.ranks]:
        p = bank.pgm_probs(xn, k)
        logp[(s, k)] = torch.log(p.clamp_min(_TINY))
        ...  # các head cũ giữ nguyên
if set(self.sources) == {'frozen', 'prompted'}:
    for k, suffix in [(self.rank, '')] + [(k, f'_r{k}') for k in self.ranks]:
        preds[f'dual_pgm_fusion{suffix}'] = pick(
            log_lin + self.weight * (logp[('frozen', k)] + logp[('prompted', k)]))
```

- Thêm các tên `dual_pgm_fusion{suffix}` vào `head_names()`, nếu không test
  `test_unseen_classes_never_predicted` sẽ fail.
- Muốn tách trọng số `w1` / `w2` cho hai nguồn thì thêm một arg mới, ví dụ
  `--density_fusion_weight2`.

### 7.9 Ghi chú theo từng codebase

| Codebase | Điểm cần chú ý |
|---|---|
| **DualPrompt / L2P** (JH-LEE-KR) | Cùng cấu trúc với repo này, áp §7.3–7.7 gần như nguyên văn (diff thật: `git diff 0b84d3e c627ebc -- engine.py main.py datasets.py` trong repo RainbowPrompt). Lời gọi model trong consolidate phải copy từ `evaluate()` của repo đó. **L2P:** pool prompt dùng chung và được cập nhật qua mọi task, nên thống kê prompted của class cũ lỗi thời dần. Kiểm tra `batchwise_prompt` trong config, vì nó chọn prompt theo đa số trong batch lúc eval. **DualPrompt:** G-prompt vẫn được train qua các task, nên có trôi nhẹ |
| **HiDe-Prompt** (thu-ml) | README ghi rõ code dựa trên DualPrompt-pytorch. Engine nằm trong `engines/` và `trainers/`, có hai pha train. Consolidate sau pha cuối của mỗi task. Forward lúc test dùng task id dự đoán từ task-identity head, nên consolidate phải dùng đúng đường đó. `logits` cho fusion là output cuối (sau TAP). Thống kê Gaussian riêng của HiDe là phần tách biệt, đừng gộp với bank |
| **LAMDA-PILOT** | Vòng lặp là `incremental_train → eval_task → after_task` (đã kiểm tra `trainer.py`). `_eval_cnn` dùng `self._network(inputs)["logits"]`, `_extract_vectors` dùng `self._network.extract_vector(x)` (đã kiểm tra `models/base.py`). Có FeCAM và RanPAC để làm baseline. Backbone mặc định có thể là `vit_base_patch16_224_in21k`, khác repo này, nên đừng mong số `frozen_*` khớp tuyệt đối |
| **RainbowPrompt** (repo gốc) | Lúc test chọn một task chung cho cả batch trong khi val loader tách theo task. Khi so với phương pháp chọn prompt theo từng mẫu, cần eval với batch trộn nhiều task |

### 7.10 Kiểm tra sau khi port

1. `python -m pytest tests -q`: test 1–7 phải pass. Test 8 cần sửa `FakeModel` / `FakeOriginal` cho
   khớp interface.
2. **`linear` == `Acc@1`** ở mọi task, sai khác dưới 1e-6. Nếu lệch thì logits truyền vào
   `predict` khác với logits dùng tính Acc@1 (ví dụ mask khác nhau).
3. **Bật và tắt heads cho Acc@1 từng task giống hệt nhau** khi cùng seed. Nếu lệch thì RNG chưa
   được khôi phục, hoặc transform của dataset train bị sửa trực tiếp.
4. `bank.pgm_probs(x, k).sum(1)` ≈ 1 với mọi k.
5. Sau task 1, log phải có dòng `Density heads: stored N classes after task 1`, với N = số class của
   task 1.
6. **Kiểm tra chéo:** cùng backbone `vit_base_patch16_224`, CUB 20 task, cùng thứ tự class thì
   `frozen_ncm_cos` ≈ 80.2 và `frozen_lda` ≈ 83.5 sau task 20. Các head frozen không phụ thuộc
   phương pháp nền, nên lệch nhiều thì khả năng cao là transform, split hoặc nhãn bị sai.

### 7.11 Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `IndexError` ở `logits[:, cls]` | Logits chỉ phủ class của task hiện tại, hoặc nhãn không phải global id | Ghép logits đủ `C_total` cột, hoặc ánh xạ nhãn về global id |
| Acc@1 của baseline đổi khi bật heads | RNG không khôi phục, hoặc sửa `transform` trực tiếp trên dataset train | Dùng `_get_rng_state` / `_set_rng_state` và `_with_transform` |
| Mọi head prompted đều thấp bất thường | Consolidate dùng đường train (ground-truth task id, augmentation), khác đường test | Copy đúng lời gọi model của `evaluate()`, dùng eval transform |
| PGM ra toàn một class hoặc NaN | Feature chưa L2-norm (tự viết lại thay vì dùng `predict`), hoặc feature fp16 | Dùng `density.predict`. Bank tự ép sang float64 |
| Forgetting của head sai | Evaluate gộp mọi task mà không tách accuracy theo task | Tách theo khoảng class id như §7.6 |
| Heads tự tắt không báo gì | Chạy DDP | Chạy 1 process, hoặc all-gather trước `add_task` |
| `--density_ranks` hay w mới không có tác dụng khi `--eval` | `load_state_dict` ghi đè bằng giá trị trong checkpoint | Áp bản vá §7.7 |
| Thống kê của class cũ lỗi thời (L2P, CODA) | Feature prompted trôi vì prompt dùng chung vẫn được train | Đây không phải lỗi code. Đo bằng ablation tính lại thống kê của mọi class cũ mỗi task (oracle) |
