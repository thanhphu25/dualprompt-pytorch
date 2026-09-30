"""
Density class heads for prompt-based class-incremental learning
(ported from RainbowPrompt commit c627ebc, see plan-head.md).

Each seen class is stored as a low-rank density matrix built from unit-normalized
features (top-r eigenpairs of X^T X / n, eigenvalues renormalized to sum 1).
At inference these class states give:
  - a complete pretty-good measurement (PGM): E_c = S^-1/2 A_c S^-1/2 with
    A_c = (sigma_c + eps I) / M and S = sum_c A_c, so sum_c p_c(x) = 1 for unit x;
  - log-probability fusion with the linear classifier over seen classes;
  - controls: NCM cosine / centered / whitened, fidelity, shared-covariance LDA
    and linear+LDA fusion;
  - dual-source fusion: log_softmax(linear) + w1 log p_pgm(frozen) + w2 log p_pgm(prompted).
Nothing here is trained and nothing touches the training loss.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

SOURCES = ('frozen', 'prompted')
_TINY = 1e-300


def str2bool(x):
    return str(x).lower() in ('1', 'true', 'yes', 'y', 't')


def add_density_args(parser):
    parser.add_argument('--density_heads', type=str2bool, default=False,
                        help='collect class statistics after each task and evaluate density heads')
    parser.add_argument('--density_sources', default=list(SOURCES), nargs='+', choices=SOURCES,
                        help='frozen: pre_logits of the frozen ViT, prompted: pre_logits of the prompted model')
    parser.add_argument('--density_rank', default=32, type=int, help='stored rank per class')
    parser.add_argument('--density_ranks', default=[8, 16], type=int, nargs='*',
                        help='extra truncated ranks for the PGM readout ablation')
    parser.add_argument('--density_eps', default=1e-4, type=float, help='ridge added to every PGM outcome')
    parser.add_argument('--density_fusion_weight', default=1.0, type=float, help='weight of log p_pgm in fusion')
    parser.add_argument('--density_fusion_weight2', default=None, type=float,
                        help='weight of log p_pgm(prompted) in dual_pgm_fusion (default: --density_fusion_weight)')
    parser.add_argument('--density_lda_shrink', default=0.1, type=float,
                        help='LDA covariance shrinkage towards (trace/D) I')
    parser.add_argument('--density_dump_dir', default='', type=str,
                        help='with --eval: dump train/val/test features of every task checkpoint here '
                             'for density_sweep.py (the evaluation itself is skipped)')
    parser.add_argument('--density_val_every', default=10, type=int,
                        help='every k-th train sample of a task is held out as validation in density_sweep.py')


class ClassBank:
    """Class states of one feature source. Old classes are written once and never updated."""

    def __init__(self, dim, rank=32, eps=1e-4, lda_shrink=0.1):
        self.dim = dim
        self.rank = rank
        self.eps = eps
        self.lda_shrink = lda_shrink
        self.classes = []
        self.U = torch.zeros(0, dim, rank)
        self.lam = torch.zeros(0, rank)
        self.mu_norm = torch.zeros(0, dim)
        self.mu_raw = torch.zeros(0, dim)
        self.counts = torch.zeros(0, dtype=torch.long)
        self.scatter = torch.zeros(dim, dim, dtype=torch.float64)
        self.n_scatter = 0
        self._cache = {}

    def __len__(self):
        return len(self.classes)

    def add_class(self, c, feats):
        assert c not in self.classes, f'class {c} already in bank'
        X = feats.detach().double().cpu()
        n = X.shape[0]
        Xn = F.normalize(X, dim=1)
        _, s, Vh = torch.linalg.svd(Xn / math.sqrt(n), full_matrices=False)
        ev = s ** 2
        k = min(self.rank, ev.numel())
        U = torch.zeros(self.dim, self.rank, dtype=torch.float64)
        lam = torch.zeros(self.rank, dtype=torch.float64)
        U[:, :k] = Vh[:k].T
        lam[:k] = ev[:k]
        lam = lam / lam.sum()

        mu_raw = X.mean(0)
        Xc = X - mu_raw
        self.scatter += Xc.T @ Xc
        self.n_scatter += n

        self.classes.append(int(c))
        self.U = torch.cat([self.U, U.float()[None]])
        self.lam = torch.cat([self.lam, lam.float()[None]])
        self.mu_norm = torch.cat([self.mu_norm, Xn.mean(0).float()[None]])
        self.mu_raw = torch.cat([self.mu_raw, mu_raw.float()[None]])
        self.counts = torch.cat([self.counts, torch.tensor([n])])
        self._cache.clear()

    def measurement(self, rank, device):
        """Complete PGM built from the first `rank` eigenpairs (renormalized)."""
        key = ('pgm', rank, str(device))
        if key not in self._cache:
            M = len(self.classes)
            U = self.U[:, :, :rank].to(device, torch.float64)
            lam = self.lam[:, :rank].to(device, torch.float64)
            lam = lam / lam.sum(1, keepdim=True).clamp_min(_TINY)
            U_flat = U.permute(1, 0, 2).reshape(self.dim, -1)
            W = U_flat * lam.sqrt().reshape(1, -1)
            S = W @ W.T / M + self.eps * torch.eye(self.dim, device=device, dtype=torch.float64)
            e, V = torch.linalg.eigh(S)
            S_mh = (V * e.clamp_min(_TINY).rsqrt()) @ V.T
            self._cache[key] = dict(U_flat=U_flat, lam=lam, S_mh=S_mh, M=M, r=rank)
        return self._cache[key]

    def pgm_probs(self, xn, rank):
        m = self.measurement(rank, xn.device)
        y = xn @ m['S_mh']
        proj = (y @ m['U_flat']).view(xn.shape[0], m['M'], m['r'])
        p = (proj ** 2 * m['lam']).sum(-1) + self.eps * (y ** 2).sum(-1, keepdim=True)
        return p / m['M']

    def fidelity(self, xn):
        m = self.measurement(self.rank, xn.device)
        proj = (xn @ m['U_flat']).view(xn.shape[0], m['M'], m['r'])
        return (proj ** 2 * m['lam']).sum(-1)

    def lda(self, device):
        key = ('lda', str(device))
        if key not in self._cache:
            cov = self.scatter.to(device) / max(self.n_scatter, 1)
            iso = torch.trace(cov) / self.dim
            eye = torch.eye(self.dim, device=device, dtype=torch.float64)
            cov = (1 - self.lda_shrink) * cov + self.lda_shrink * iso * eye
            P = torch.linalg.inv(cov)
            m = self.mu_raw.to(device, torch.float64)
            Pm = m @ P
            self._cache[key] = dict(Pm=Pm, bias=-0.5 * (Pm * m).sum(1))
        return self._cache[key]

    def state_dict(self):
        return dict(dim=self.dim, rank=self.rank, eps=self.eps, lda_shrink=self.lda_shrink,
                    classes=list(self.classes), U=self.U, lam=self.lam, mu_norm=self.mu_norm,
                    mu_raw=self.mu_raw, counts=self.counts, scatter=self.scatter, n_scatter=self.n_scatter)

    def load_state_dict(self, state):
        for k in ('dim', 'rank', 'eps', 'lda_shrink', 'n_scatter'):
            setattr(self, k, state[k])
        self.classes = list(state['classes'])
        for k in ('U', 'lam', 'mu_norm', 'mu_raw', 'counts', 'scatter'):
            setattr(self, k, state[k].cpu())
        self._cache.clear()


class DensityHeads:
    def __init__(self, args, dim):
        self.sources = list(dict.fromkeys(args.density_sources))
        self.rank = args.density_rank
        self.ranks = sorted({k for k in args.density_ranks if 0 < k < self.rank})
        self.weight = args.density_fusion_weight
        w2 = getattr(args, 'density_fusion_weight2', None)
        self.weight2 = self.weight if w2 is None else w2
        self.banks = {s: ClassBank(dim, self.rank, args.density_eps, args.density_lda_shrink) for s in self.sources}

    @property
    def classes(self):
        return self.banks[self.sources[0]].classes

    def __len__(self):
        return len(self.classes)

    def head_names(self):
        names = ['linear', 'linear_seen']
        for s in self.sources:
            names += [f'{s}_ncm_cos', f'{s}_ncm_centered', f'{s}_ncm_whitened', f'{s}_fidelity',
                      f'{s}_pgm', f'{s}_pgm_fusion']
            for k in self.ranks:
                names += [f'{s}_pgm_r{k}', f'{s}_pgm_fusion_r{k}']
            names += [f'{s}_lda', f'{s}_lda_fusion']
        if self._dual:
            names += ['dual_pgm_fusion'] + [f'dual_pgm_fusion_r{k}' for k in self.ranks]
        return names

    @property
    def _dual(self):
        return set(self.sources) == set(SOURCES)

    def _pgm_ranks(self):
        return [(self.rank, '')] + [(k, f'_r{k}') for k in self.ranks]

    def add_task(self, feats, labels):
        """feats: source -> (N, D) features of the current task's train split, labels: (N,)"""
        labels = labels.cpu()
        for c in torch.unique(labels).tolist():
            idx = labels == c
            for s in self.sources:
                self.banks[s].add_class(c, feats[s][idx])

    @torch.no_grad()
    def predict(self, feats, logits):
        """Returns head name -> predicted global class ids (B,). Only seen classes can be predicted."""
        dev = logits.device
        cls = torch.tensor(self.classes, device=dev)
        logits_seen = logits[:, cls].double()
        log_lin = F.log_softmax(logits_seen, dim=1)
        preds = {'linear': logits.argmax(1), 'linear_seen': cls[logits_seen.argmax(1)]}

        def pick(scores):
            return cls[scores.argmax(1)]

        logp = {}
        for s in self.sources:
            bank = self.banks[s]
            f = feats[s].to(dev, torch.float64)
            xn = F.normalize(f, dim=1)
            mu = bank.mu_norm.to(dev, torch.float64)
            preds[f'{s}_ncm_cos'] = pick(xn @ F.normalize(mu, dim=1).T)
            g = mu.mean(0, keepdim=True)
            preds[f'{s}_ncm_centered'] = pick(F.normalize(xn - g, dim=1) @ F.normalize(mu - g, dim=1).T)
            S_mh = bank.measurement(self.rank, dev)['S_mh']
            preds[f'{s}_ncm_whitened'] = pick(F.normalize(xn @ S_mh, dim=1) @ F.normalize(mu @ S_mh, dim=1).T)
            preds[f'{s}_fidelity'] = pick(bank.fidelity(xn))

            for k, suffix in self._pgm_ranks():
                p = bank.pgm_probs(xn, k)
                logp[(s, k)] = torch.log(p.clamp_min(_TINY))
                preds[f'{s}_pgm{suffix}'] = pick(p)
                preds[f'{s}_pgm_fusion{suffix}'] = pick(log_lin + self.weight * logp[(s, k)])

            lda = bank.lda(dev)
            lda_scores = f @ lda['Pm'].T + lda['bias']
            preds[f'{s}_lda'] = pick(lda_scores)
            preds[f'{s}_lda_fusion'] = pick(log_lin + self.weight * F.log_softmax(lda_scores, dim=1))

        if self._dual:
            for k, suffix in self._pgm_ranks():
                preds[f'dual_pgm_fusion{suffix}'] = pick(
                    log_lin + self.weight * logp[('frozen', k)] + self.weight2 * logp[('prompted', k)])
        return preds

    def state_dict(self):
        return dict(sources=self.sources, rank=self.rank, ranks=self.ranks, weight=self.weight, weight2=self.weight2,
                    banks={s: b.state_dict() for s, b in self.banks.items()})

    def load_state_dict(self, state, use_saved_config=True):
        """use_saved_config=False keeps ranks / weights from the current args (offline --eval sweeps);
        only ranks below the stored rank can be read back. eps and lda_shrink are read-time settings too."""
        for s in self.sources:
            bank = self.banks[s]
            eps, lda_shrink = bank.eps, bank.lda_shrink
            bank.load_state_dict(state['banks'][s])
            if not use_saved_config:
                bank.eps, bank.lda_shrink = eps, lda_shrink
        self.rank = self.banks[self.sources[0]].rank
        if use_saved_config:
            self.ranks = list(state['ranks'])
            self.weight = state['weight']
            self.weight2 = state.get('weight2', self.weight)
        else:
            self.ranks = sorted(k for k in self.ranks if 0 < k < self.rank)

    def config(self):
        bank = self.banks[self.sources[0]]
        return dict(sources=self.sources, rank=self.rank, ranks=self.ranks, weight=self.weight,
                    weight2=self.weight2, eps=bank.eps, lda_shrink=bank.lda_shrink)


class HeadMetrics:
    """Per-head accuracy matrices, same conventions as evaluate_till_now."""

    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.acc = {}
        self.avg_history = {}

    def update(self, head, test_task, train_task, acc):
        if head not in self.acc:
            self.acc[head] = np.zeros((self.num_tasks, self.num_tasks))
        self.acc[head][test_task, train_task] = acc

    def end_task(self, task_id):
        stats = {}
        for head, m in self.acc.items():
            avg = float(np.mean(m[:task_id + 1, task_id]))
            forgetting = float(np.mean((np.max(m, axis=1) - m[:, task_id])[:task_id])) if task_id > 0 else 0.0
            hist = self.avg_history.setdefault(head, [])
            del hist[task_id:]
            hist.append(avg)
            stats[head] = dict(acc=avg, incremental_acc=float(np.mean(hist)), forgetting=forgetting)
        return stats

    def log_table(self, task_id, stats):
        base = stats.get('linear', {}).get('acc', 0.0)
        lines = [f'[Density heads after task {task_id + 1}]',
                 f'{"head":<28}{"Acc":>9}{"vs linear":>11}{"IncAcc":>9}{"Forget":>9}']
        for head, st in stats.items():
            lines.append(f'{head:<28}{st["acc"]:>9.2f}{st["acc"] - base:>+11.2f}'
                         f'{st["incremental_acc"]:>9.2f}{st["forgetting"]:>9.2f}')
        print('\n'.join(lines))

    def save(self, output_dir, task_id, stats, prefix='density_heads', config=None):
        if not output_dir:
            return
        out = dict(task=task_id + 1, config=config, heads=stats,
                   acc_matrix={h: m[:task_id + 1, :task_id + 1].tolist() for h, m in self.acc.items()})
        with open(os.path.join(output_dir, f'{prefix}_summary.json'), 'w') as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(output_dir, f'{prefix}_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(task=task_id + 1, config=config, heads=stats)) + '\n')
