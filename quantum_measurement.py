# ------------------------------------------
# Quantum-measurement readouts for DualPrompt (see plan-quantum.md)
# ------------------------------------------
"""
Eval-time density-matrix readouts for DualPrompt. Nothing here touches training.

Each seen class c is a mixed state sigma_c (top-r eigenpairs of the second moment of its
unit-normalised train features, trace 1). Each task t is the uniform mixture rho_t of its
classes. A test image is the pure state |x><x| of its unit-normalised feature.

Task routing and class prediction use the pretty-good measurement (PGM, Hausladen-Wootters)

    A_i = pi_i (sigma_i + eps I),  S = sum_i A_i,  E_i = S^-1/2 A_i S^-1/2,  sum_i E_i = I

so p_i(x) = x^T E_i x is a normalised distribution over every seen outcome (Born rule).
The task-PGM is the coarse-graining E_t = sum_{c in t} E_c of the class-PGM with a
task-uniform prior. Everything is exact real linear algebra; there is no circuit.
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from timm.utils import accuracy

import utils


ROUTERS = ('cosine', 'cosine_batch', 'cosine_seen', 'task_ncm', 'task_ncm_white', 'class_ncm_task',
           'task_lda', 'task_fidelity', 'task_pgm', 'oracle')
SOFT_ROUTERS = ('task_pgm', 'task_lda')
SOURCES = ('frozen', 'prompted')


def add_qm_args(parser):
    parser.add_argument('--qm_eval', action='store_true',
                        help='build density-matrix banks at the end of each task and evaluate every router/head')
    parser.add_argument('--qm_sources', default=['frozen', 'prompted'], nargs='+', choices=SOURCES,
                        help='feature sources for the class heads (the frozen bank is always built for routing)')
    parser.add_argument('--qm_rank', default=32, type=int, help='eigenpairs kept per class state')
    parser.add_argument('--qm_eps', default=1e-4, type=float, help='ridge added to every PGM element')
    parser.add_argument('--qm_fusion_weight', default=1.0, type=float, help='w in log_softmax(z) + w log p_c')
    parser.add_argument('--qm_pgm_ranks', default=[8, 16], type=int, nargs='*', help='truncated-rank PGM ablations')
    parser.add_argument('--qm_head_routers', default=['cosine', 'oracle'], nargs='+',
                        help='routers whose E-prompt feeds the class heads (Phase 2A: cosine oracle; 2B: + task_pgm task_lda)')
    parser.add_argument('--qm_soft_topm', default=2, type=int, help='number of tasks kept by the soft task->class measurement')


class DensityClassBank(nn.Module):
    """Per-class density matrices sigma_c = U_c diag(lambda_c) U_c^T plus first/second-order classical statistics."""

    def __init__(self, num_classes, dim, rank=32, eps=1e-4, lda_ridge=None):
        super().__init__()
        self.rank, self.eps = rank, eps
        self.lda_ridge = eps if lda_ridge is None else lda_ridge
        self.register_buffer('vectors', torch.zeros(num_classes, dim, rank))
        self.register_buffer('values', torch.zeros(num_classes, rank))
        self.register_buffer('means', torch.zeros(num_classes, dim))
        self.register_buffer('raw_means', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('task_of_class', torch.full((num_classes,), -1, dtype=torch.long))
        self.register_buffer('scatter', torch.zeros(dim, dim, dtype=torch.float64))
        self.register_buffer('scatter_count', torch.zeros((), dtype=torch.long))
        self._cache = {}

    def load_state_dict(self, *args, **kwargs):
        # register_load_state_dict_post_hook does not exist in torch 1.12
        out = super().load_state_dict(*args, **kwargs)
        self._cache.clear()
        return out

    @torch.no_grad()
    def add_class(self, label, features, task):
        assert not bool(self.valid[label]) and len(features) > 0, f'class {label} already stored or empty'
        x = F.normalize(features.to(self.vectors.device, torch.float64), dim=-1)
        # right singular vectors of X/sqrt(n) are the eigenvectors of X^T X / n, with eigenvalues s^2
        _, s, vh = torch.linalg.svd(x / math.sqrt(len(x)), full_matrices=False)
        r = min(self.rank, len(s))
        lam = s[:r].square()
        self.vectors[label].zero_()
        self.values[label].zero_()
        self.vectors[label, :, :r] = vh[:r].T.to(self.vectors.dtype)
        self.values[label, :r] = (lam / lam.sum()).to(self.values.dtype)
        mean = x.mean(0)
        self.raw_means[label] = mean.to(self.raw_means.dtype)
        self.means[label] = F.normalize(mean, dim=0).to(self.means.dtype)
        c = x - mean
        self.scatter += c.T @ c
        self.scatter_count += len(x)
        self.task_of_class[label] = task
        self.valid[label] = True
        self._cache.clear()

    def _states(self, classes, rank=None):
        key = ('states', rank, tuple(classes.tolist()))
        if key not in self._cache:
            v, lam = self.vectors[classes].double(), self.values[classes].double()
            if rank is not None and rank < v.shape[-1]:
                v, lam = v[..., :rank], lam[:, :rank]
                lam = lam / lam.sum(-1, keepdim=True).clamp_min(1e-12)
            self._cache[key] = (v, lam)
        return self._cache[key]

    def prior(self, classes, prior='class'):
        """pi_c: uniform over classes, or uniform over tasks then uniform over the classes of each task."""
        if prior == 'class':
            return torch.full((len(classes),), 1.0 / len(classes), dtype=torch.float64, device=classes.device)
        _, inverse, counts = torch.unique(self.task_of_class[classes], return_inverse=True, return_counts=True)
        return 1.0 / (len(counts) * counts[inverse].double())

    def pgm_root(self, classes, rank=None, prior='class'):
        """S^-1/2 with S = sum_c pi_c sigma_c + eps I (float64, cached per seen set)."""
        key = ('pgm', rank, prior, tuple(classes.tolist()))
        if key not in self._cache:
            v, lam = self._states(classes, rank)
            pi = self.prior(classes, prior)
            w = (v * (lam * pi[:, None]).sqrt().unsqueeze(1)).permute(1, 0, 2).flatten(1)  # D x (C r)
            evals, evecs = torch.linalg.eigh(w @ w.T)
            evals = evals.clamp_min(0) + self.eps
            self._cache[key] = (evecs * evals.rsqrt()) @ evecs.T
        return self._cache[key]

    @staticmethod
    def _energy(x, v, lam):
        """<x|sigma_c|x> for every class c: [B, C]."""
        return (torch.einsum('bd,cdk->bck', x, v).square() * lam).sum(-1)

    def energy(self, x, classes, rank=None):
        v, lam = self._states(classes, rank)
        return self._energy(x.double(), v, lam)

    def pgm_log_probs(self, x, classes, rank=None, prior='class'):
        """log p_c(x) = log x^T E_c x for unit-norm x [B, D]; sums to one over `classes`."""
        v, lam = self._states(classes, rank)
        y = x.double() @ self.pgm_root(classes, rank, prior)
        born = self.prior(classes, prior) * (self._energy(y, v, lam) + self.eps * y.square().sum(-1, keepdim=True))
        born = born.clamp_min(1e-300)
        return (born / born.sum(-1, keepdim=True)).log()

    def povm(self, classes, rank=None, prior='class'):
        """Full POVM elements E_c [C, D, D] (for tests only)."""
        v, lam = self._states(classes, rank)
        root = self.pgm_root(classes, rank, prior)
        eye = torch.eye(v.shape[1], dtype=torch.float64, device=v.device)
        a = torch.einsum('cdk,ck,cek->cde', v, lam, v) + self.eps * eye
        a = a * self.prior(classes, prior)[:, None, None]
        return root @ a @ root

    def fidelity_log_probs(self, x, classes):
        e = self.energy(x, classes).clamp_min(1e-300)
        return (e / e.sum(-1, keepdim=True)).log()

    def ncm(self, x, classes):
        return x.double() @ self.means[classes].double().T

    def ncm_white(self, x, classes, prior='class'):
        root = self.pgm_root(classes, None, prior)
        return F.normalize(x.double() @ root, dim=-1) @ F.normalize(self.raw_means[classes].double() @ root, dim=-1).T

    def _lda(self, classes):
        key = ('lda', tuple(classes.tolist()))
        if key not in self._cache:
            cov = self.scatter / self.scatter_count.clamp_min(1)
            ridge = self.lda_ridge * cov.diagonal().sum().clamp_min(1e-6)
            mu = self.raw_means[classes].double()
            eye = torch.eye(len(cov), dtype=cov.dtype, device=cov.device)
            w = torch.linalg.solve(cov + ridge * eye, mu.T)
            b = -0.5 * (mu * w.T).sum(-1)
            self._cache[key] = (w, b)
        return self._cache[key]

    def lda_log_probs(self, x, classes):
        w, b = self._lda(classes)
        return (x.double() @ w + b).log_softmax(-1)


def group_by_task(scores, class_task, num_tasks, reduce='logsumexp'):
    """Reduce class scores [B, C] to task scores [B, T]; tasks without classes get -inf."""
    out = scores.new_full((scores.shape[0], num_tasks), float('-inf'))
    for t in class_task.unique().tolist():
        s = scores[:, class_task == t]
        if reduce == 'logsumexp':
            out[:, t] = s.logsumexp(-1)
        elif reduce == 'logmeanexp':
            out[:, t] = s.logsumexp(-1) - math.log(s.shape[1])
        elif reduce == 'max':
            out[:, t] = s.max(-1).values
        else:
            raise ValueError(reduce)
    return out


def class_readouts(bank, x, classes, pgm_ranks):
    """Every class head of plan section 4.3 on unit-norm features x: name -> [B, C] (argmax = prediction)."""
    out = {
        'ncm': bank.ncm(x, classes),
        'ncm_white': bank.ncm_white(x, classes),
        'fidelity': bank.fidelity_log_probs(x, classes),
        'pgm': bank.pgm_log_probs(x, classes),
        'lda': bank.lda_log_probs(x, classes),
    }
    for r in pgm_ranks:
        out[f'pgm_r{r}'] = bank.pgm_log_probs(x, classes, rank=r)
    return out


def task_scores(bank, q, classes, num_tasks, pgm_ranks, readouts):
    """Router scores [B, T] on the frozen query (plan section 3.4); unseen tasks are -inf.
    `readouts` are the class-prior readouts of the same bank on the same q (reused, not recomputed)."""
    class_task = bank.task_of_class[classes]
    seen_tasks = class_task.unique()
    out = {}

    task_mu = torch.zeros(num_tasks, q.shape[1], dtype=torch.float64, device=q.device)
    for t in seen_tasks.tolist():
        task_mu[t] = bank.raw_means[classes[class_task == t]].double().mean(0)
    unseen = torch.ones(num_tasks, dtype=torch.bool, device=q.device)
    unseen[seen_tasks] = False

    out['task_ncm'] = (q.double() @ F.normalize(task_mu, dim=-1).T).masked_fill(unseen, float('-inf'))
    root = bank.pgm_root(classes, None, 'task')
    white = F.normalize(q.double() @ root, dim=-1) @ F.normalize(task_mu @ root, dim=-1).T
    out['task_ncm_white'] = white.masked_fill(unseen, float('-inf'))
    out['class_ncm_task'] = group_by_task(readouts['ncm'], class_task, num_tasks, 'max')
    out['task_lda'] = group_by_task(readouts['lda'], class_task, num_tasks, 'logsumexp')
    # q^T rho_t q with rho_t the uniform mixture of the task's class states
    out['task_fidelity'] = group_by_task(bank.energy(q, classes).clamp_min(1e-300).log(), class_task, num_tasks, 'logmeanexp')
    # task-PGM = coarse-graining of the class-PGM under a task-uniform prior (log-probs over seen tasks)
    out['task_pgm'] = group_by_task(bank.pgm_log_probs(q, classes, prior='task'), class_task, num_tasks, 'logsumexp')
    for r in pgm_ranks:
        out[f'task_pgm_r{r}'] = group_by_task(bank.pgm_log_probs(q, classes, rank=r, prior='task'),
                                              class_task, num_tasks, 'logsumexp')
    return out


def batchwise_vote(idx, pool_size, top_k):
    """Verbatim copy of the batchwise_prompt branch of EPrompt.forward (majority vote over the batch)."""
    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
    if prompt_id.shape[0] < pool_size:
        prompt_id = torch.cat([prompt_id, torch.full((pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
        id_counts = torch.cat([id_counts, torch.full((pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
    _, major_idx = torch.topk(id_counts, k=top_k)
    major_prompt_id = prompt_id[major_idx]
    return major_prompt_id.expand(idx.shape[0], -1).contiguous()


def _base_transform(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds.transform


def _with_transform(ds, transform):
    """Shallow copy of a (possibly nested) Subset whose base dataset uses `transform`."""
    if isinstance(ds, Subset):
        return Subset(_with_transform(ds.dataset, transform), ds.indices)
    ds = copy.copy(ds)
    ds.transform = transform
    return ds


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _summarize(acc_matrix, stage):
    """Same Acc / Forgetting / Backward formulas as engine.evaluate_till_now, plus average incremental acc."""
    avgs = [float(np.mean(acc_matrix[:s + 1, s])) for s in range(stage + 1)]
    out = {'final_acc': avgs[-1], 'avg_incremental_acc': float(np.mean(avgs)), 'forgetting': None, 'backward': None}
    if stage > 0:
        diagonal = np.diag(acc_matrix)
        out['forgetting'] = float(np.mean((np.max(acc_matrix[:, :stage + 1], axis=1) - acc_matrix[:, stage])[:stage]))
        out['backward'] = float(np.mean((acc_matrix[:, stage] - diagonal)[:stage]))
    return out


class QuantumMeasurement:
    """Holds the class banks and every per-head accuracy matrix of one run."""

    def __init__(self, args, dim, device):
        assert utils.get_world_size() == 1, '--qm_eval supports a single process only (see plan section 6.4)'
        assert args.top_k == 1 and args.size == args.num_tasks, '--qm_eval assumes one E-prompt per task (top_k=1, size=num_tasks)'
        self.args = args
        self.num_tasks = args.num_tasks
        self.pgm_ranks = sorted(set(r for r in args.qm_pgm_ranks if r < args.qm_rank))
        self.sources = [s for s in SOURCES if s in args.qm_sources]
        routers = list(ROUTERS) + [f'task_pgm_r{r}' for r in self.pgm_ranks]
        self.routers = routers
        for r in args.qm_head_routers:
            assert r in routers, f'unknown router {r} in --qm_head_routers (choose from {routers})'
        self.head_routers = list(dict.fromkeys(args.qm_head_routers))
        banks = ['frozen'] + (['prompted'] if 'prompted' in self.sources else [])
        self.banks = {s: DensityClassBank(args.nb_classes, dim, args.qm_rank, args.qm_eps).to(device) for s in banks}
        self.acc = {}          # head -> [T, T] acc matrix (row = task evaluated, column = stage)
        self.route_acc = {}    # router -> [T, T] fraction routed to the true E-prompt
        self.confusion = {}    # router -> {stage: [T, T] counts (true, chosen)}
        self.start_time = time.time()

    # ----- checkpoint -----
    def state_dict(self):
        return {s: b.state_dict() for s, b in self.banks.items()}

    def load_state_dict(self, state):
        if not all(s in state for s in self.banks):
            return False
        for s, b in self.banks.items():
            b.load_state_dict(state[s])
        return True

    # ----- end of task: store class states -----
    @torch.no_grad()
    def consolidate(self, model, original_model, loaders, classes_t, task_id, device):
        """Add every class of task `task_id` to the banks, from its TRAIN images read with the EVAL transform."""
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model = _unwrap(model)
        was_training = model.training
        model.eval()
        original_model.eval()

        ds = _with_transform(loaders['train'].dataset, _base_transform(loaders['val'].dataset))
        loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=False,
                            num_workers=self.args.num_workers, pin_memory=self.args.pin_mem)
        feats = {s: [] for s in self.banks}
        labels = []
        for input, target in loader:
            input = input.to(device, non_blocking=True)
            q = original_model(input)['pre_logits']
            feats['frozen'].append(q.float())
            if 'prompted' in self.banks:
                idx = torch.full((len(input),), task_id, dtype=torch.long, device=device)
                feats['prompted'].append(model(input, cls_features=q, prompt_idx=idx)['pre_logits'].float())
            labels.append(target.to(device))
        labels = torch.cat(labels)
        for s, f in feats.items():
            f = torch.cat(f)
            for c in classes_t:
                self.banks[s].add_class(int(c), f[labels == c], task=task_id)

        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        print(f'[QM] task {task_id + 1}: stored {len(classes_t)} class states ({len(labels)} train images) '
              f'in banks {list(self.banks)}')

    # ----- evaluation -----
    @torch.no_grad()
    def evaluate(self, model, original_model, data_loader, device, task_id, stage, class_mask, args):
        """Evaluate every router/head on the test split of task `task_id` after training task `stage`.
        Returns the stats of the original DualPrompt readout (same keys as engine.evaluate)."""
        model = _unwrap(model)
        model.eval()
        original_model.eval()
        T, w = self.num_tasks, args.qm_fusion_weight
        classes = torch.tensor(sum((list(class_mask[t]) for t in range(stage + 1)), []), dtype=torch.long, device=device)
        frozen = self.banks['frozen']
        assert bool(frozen.valid[classes].all()), 'bank is missing seen classes; was consolidate() called?'
        class_task = frozen.task_of_class[classes]
        owner = torch.full((args.nb_classes,), -1, dtype=torch.long, device=device)
        owner[classes] = class_task
        seen_class = owner >= 0
        seen_task = torch.zeros(T, dtype=torch.bool, device=device)
        seen_task[class_task] = True
        m_soft = min(args.qm_soft_topm, int(seen_task.sum()))  # head names keep the configured m

        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)
        correct, route_hits, n_total = {}, {r: 0 for r in self.routers}, 0
        confusion = {r: torch.zeros(T, T, dtype=torch.long) for r in self.routers}

        def hit(name, pred):
            correct[name] = correct.get(name, 0) + int((pred == target).sum())

        def seen_argmax(z):
            return z.masked_fill(~seen_class, float('-inf')).argmax(-1)

        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            B, arange = len(input), torch.arange(len(input), device=device)

            q_raw = original_model(input)['pre_logits']
            q = F.normalize(q_raw.double(), dim=-1)

            # ---- routers (plan section 3.4) ----
            e_prompt = model.e_prompt
            key_norm = e_prompt.l2_normalize(e_prompt.prompt_key, dim=-1)
            sim = torch.matmul(key_norm, e_prompt.l2_normalize(q_raw, dim=-1).t()).t()  # same ops as EPrompt.forward
            topk_idx = torch.topk(sim, k=e_prompt.top_k, dim=1)[1]
            frozen_ro = class_readouts(frozen, q, classes, self.pgm_ranks)
            scores = task_scores(frozen, q, classes, T, self.pgm_ranks, frozen_ro)
            routes = {
                'cosine': topk_idx[:, 0],
                'cosine_batch': batchwise_vote(topk_idx, e_prompt.pool_size, e_prompt.top_k)[:, 0],
                'cosine_seen': sim.masked_fill(~seen_task, float('-inf')).argmax(-1),
            }
            for r, s in scores.items():
                routes[r] = s.argmax(-1)
            routes['oracle'] = torch.full((B,), task_id, dtype=torch.long, device=device)
            soft_top = {r: scores[r].topk(m_soft, dim=-1).indices for r in SOFT_ROUTERS}

            # ---- deduplicated forwards: one pass per E-prompt that some router asked for (section 3.5) ----
            need = torch.zeros(B, T, dtype=torch.bool, device=device)
            for r in self.routers:
                need[arange, routes[r]] = True
            for top in soft_top.values():
                need.scatter_(1, top, True)
            logits = torch.full((T, B, args.nb_classes), float('nan'), device=device)
            feats = torch.full((T, B, model.embed_dim), float('nan'), device=device)
            for t in need.any(0).nonzero()[:, 0].tolist():
                sel = need[:, t].nonzero()[:, 0]
                out = model(input[sel], cls_features=q_raw[sel], prompt_idx=torch.full_like(sel, t))
                logits[t, sel] = out['logits'].float()
                feats[t, sel] = out['pre_logits'].float()

            # ---- original DualPrompt readout, kept for the usual log line ----
            z_legacy = logits[routes['cosine_batch' if e_prompt.batchwise_prompt else 'cosine'], arange]
            if args.task_inc:
                til_mask = torch.full_like(z_legacy, float('-inf'))
                til_mask[:, torch.as_tensor(class_mask[task_id], device=device)] = 0.0
                z_legacy = z_legacy + til_mask
            loss = criterion(z_legacy, target)
            acc1, acc5 = accuracy(z_legacy, target, topk=(1, 5))
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=B)

            # ---- router heads: linear (seen-masked), TIL inside the chosen task ----
            for r in self.routers:
                idx = routes[r]
                route_hits[r] += int((idx == task_id).sum())
                confusion[r][task_id] += torch.bincount(idx.cpu(), minlength=T)
                z = logits[idx, arange]
                hit(f'{r}_linear', seen_argmax(z))
                hit(f'{r}_til', z.masked_fill(owner[None, :] != idx[:, None], float('-inf')).argmax(-1))
            hit('cosine_linear_nomask', logits[routes['cosine'], arange].argmax(-1))
            hit('cosine_batch_linear_nomask', logits[routes['cosine_batch'], arange].argmax(-1))

            # ---- class heads (section 4) ----
            if 'frozen' in self.sources:
                for h, s in frozen_ro.items():
                    hit(f'frozen_{h}', classes[s.argmax(-1)])
            for r in self.head_routers:
                idx = routes[r]
                z_seen = logits[idx, arange][:, classes].double().log_softmax(-1)
                ro_by_src = {}
                if 'frozen' in self.sources:
                    ro_by_src['frozen'] = frozen_ro
                if 'prompted' in self.sources:
                    xp = F.normalize(feats[idx, arange].double(), dim=-1)
                    ro_by_src['prompted'] = class_readouts(self.banks['prompted'], xp, classes, self.pgm_ranks)
                    for h, s in ro_by_src['prompted'].items():
                        hit(f'{r}_prompted_{h}', classes[s.argmax(-1)])
                for src, ro in ro_by_src.items():
                    hit(f'{r}_{src}_fusion', classes[(z_seen + w * ro['pgm']).argmax(-1)])
                    for k in self.pgm_ranks:
                        hit(f'{r}_{src}_fusion_r{k}', classes[(z_seen + w * ro[f'pgm_r{k}']).argmax(-1)])
                    hit(f'{r}_{src}_lda_fusion', classes[(z_seen + w * ro['lda']).argmax(-1)])

            # ---- sequential task -> class measurement (section 5) ----
            for r in SOFT_ROUTERS:
                log_pt = scores[r]  # log P(t|x), normalised over seen tasks
                allowed = torch.zeros(B, T, dtype=torch.bool, device=device)
                allowed.scatter_(1, soft_top[r], True)
                joint = torch.full((B, len(classes)), float('-inf'), dtype=torch.float64, device=device)
                for t in class_task.unique().tolist():
                    cols = (class_task == t).nonzero()[:, 0]
                    within = logits[t][:, classes[cols]].double().log_softmax(-1)  # log P(c | x, t)
                    val = log_pt[:, t:t + 1] + within
                    joint[:, cols] = torch.where(allowed[:, t:t + 1], val, torch.full_like(val, float('-inf')))
                name = f'{r}_soft{args.qm_soft_topm}'
                hit(name, classes[joint.argmax(-1)])
                if 'frozen' in self.sources:
                    hit(f'{name}_cpgm', classes[(joint + w * frozen_ro['pgm']).argmax(-1)])
                    if r == 'task_lda':
                        hit(f'{name}_clda', classes[(joint + w * frozen_ro['lda']).argmax(-1)])
            n_total += B

        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

        for name, c in correct.items():
            self.acc.setdefault(name, np.zeros((T, T)))[task_id, stage] = 100.0 * c / n_total
        for r in self.routers:
            self.route_acc.setdefault(r, np.zeros((T, T)))[task_id, stage] = 100.0 * route_hits[r] / n_total
            self.confusion.setdefault(r, {}).setdefault(stage, np.zeros((T, T), dtype=np.int64))
            self.confusion[r][stage] += confusion[r].numpy()
        print('[QM] route acc: ' + '  '.join(f'{r} {self.route_acc[r][task_id, stage]:.1f}'
                                              for r in ('cosine', 'task_lda', 'task_pgm')))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # ----- reporting -----
    def summary(self, stage):
        heads = {}
        for name, a in self.acc.items():
            heads[name] = dict(_summarize(a, stage), acc_matrix=a[:stage + 1, :stage + 1].tolist())
        routers = {}
        for r in self.routers:
            a = self.route_acc[r]
            routers[r] = {
                'route_acc_mean': float(np.mean(a[:stage + 1, stage])),
                'route_acc': a[:stage + 1, stage].tolist(),
                'route_acc_matrix': a[:stage + 1, :stage + 1].tolist(),
                'confusion': self.confusion[r][stage][:stage + 1, :stage + 1].tolist(),
            }
        return {'stage': stage + 1, 'runtime_sec': time.time() - self.start_time, 'heads': heads, 'routers': routers}

    def end_stage(self, stage):
        summary = self.summary(stage)
        if self.args.output_dir and utils.is_main_process():
            keep = ('dataset', 'seed', 'epochs', 'batch_size', 'batchwise_prompt', 'qm_rank', 'qm_eps',
                    'qm_fusion_weight', 'qm_pgm_ranks', 'qm_head_routers', 'qm_soft_topm', 'qm_sources')
            summary['args'] = {k: getattr(self.args, k) for k in keep if hasattr(self.args, k)}
            with open(os.path.join(self.args.output_dir, 'results_summary.json'), 'w') as f:
                json.dump(summary, f, indent=1)
        h = summary['heads']
        gap = h['oracle_linear']['final_acc'] - h['cosine_linear']['final_acc']
        print(f'[QM] stage {stage + 1}: ' + '  '.join(
            f'{n} {h[n]["final_acc"]:.2f}' for n in ('cosine_linear', 'task_pgm_linear', 'task_lda_linear', 'oracle_linear',
                                                     'frozen_pgm', 'cosine_frozen_fusion') if n in h)
              + f'  | oracle-cosine gap {gap:.2f}')
        return summary

    def print_table(self, stage):
        s = self.summary(stage)
        heads, routers = s['heads'], s['routers']

        def fmt(v):
            return '   -  ' if v is None else f'{v:6.2f}'

        def row(name):
            if name not in heads:
                return
            d = heads[name]
            print(f'  {name:<34} {fmt(d["final_acc"])} {fmt(d["avg_incremental_acc"])} {fmt(d["forgetting"])} {fmt(d["backward"])}')

        print('\n' + '=' * 78)
        print(f'[QM] summary after task {stage + 1}   (Acc = final avg acc, AvgInc = mean over stages)')
        print('-' * 78)
        print(f'  {"router":<18} {"route":>6} {"Acc":>6} {"F":>6} {"TIL":>6}')
        for r in self.routers:
            lin, til = heads[f'{r}_linear'], heads[f'{r}_til']
            print(f'  {r:<18} {routers[r]["route_acc_mean"]:6.2f} {fmt(lin["final_acc"])} {fmt(lin["forgetting"])} {fmt(til["final_acc"])}')
        print('-' * 78)
        print(f'  {"head":<34} {"Acc":>6} {"AvgInc":>6} {"F":>6} {"BWT":>6}')
        for n in ('cosine_linear_nomask', 'cosine_batch_linear_nomask'):
            row(n)
        for n in [n for n in heads if n.startswith('frozen_')]:
            row(n)
        for r in self.head_routers:
            row(f'{r}_linear')
            for n in [n for n in heads if n.startswith(f'{r}_frozen_') or n.startswith(f'{r}_prompted_')]:
                row(n)
        for n in [n for n in heads if '_soft' in n]:
            row(n)
        for r in self.routers:
            row(f'{r}_til')
        print('=' * 78 + '\n')
