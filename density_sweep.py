"""
Offline sweep of training-free class heads on features dumped by
    python main.py <config> --eval --output_dir OUT --density_dump_dir DUMP ...
Run:
    python density_sweep.py DUMP [--device cuda]

Protocol (no hyper-parameter is chosen on the test set):
  * val:  class statistics from the train split minus every `val_every`-th sample of each task; those
          held-out samples (re-encoded by every later checkpoint) form the validation split.
  * test: class statistics from the full train split, evaluated on the test split.
For every head family the configuration with the best final validation accuracy is reported on test.
`oracle` is the best final test accuracy inside the family (optimistic, for reference only).

Heads, per feature source s in {frozen, prompted} (all class statistics are written once, when the class
is learned, and never updated):
  s_ncm              cosine to the normalized class mean
  s_lda              shared-covariance LDA with shrinkage towards (trace/D) I          (SLDA / FSA)
  s_fecam            per-class Mahalanobis, shrunk and correlation-normalized covariance (FeCAM, no Tukey)
  s_ranpac           ReLU random projection + ridge classifier on the Gram matrix       (RanPAC head)
  s_pgm / s_pgmc     E_c = S^-a A_c S^-a with A_c = (rho_c^beta + eps I) / M; a=0.5, beta=1 is the PGM
                     of density_head.py. pgmc builds the states from features centered at the task-1 mean.
  s_opt / s_optc     minimum-error measurement from the iterative fixed point
                     Pi_c <- G^-1/2 rho_c Pi_c rho_c G^-1/2, G = sum_c rho_c Pi_c rho_c (Jezek, Rehacek,
                     Fiurasek, PRA 65 060301, 2002), started from the PGM; only the stored rho_c are used.
  s_<head>_fusion    log_softmax(linear over seen classes) + w * score
  dual_<head>_fusion log_softmax(linear) + w1 * score(frozen) + w2 * score(prompted)
"""
import argparse
import itertools
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

SOURCES = ('frozen', 'prompted')
KINDS = ('pgm', 'pgmc', 'opt', 'optc', 'lda', 'fecam', 'ranpac')
_TINY = 1e-300


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dump_dir')
    p.add_argument('--out', default=None, help='JSON with every configuration (default: DUMP_DIR/sweep.json)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--sources', nargs='+', default=list(SOURCES), choices=SOURCES)
    p.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    p.add_argument('--weights', nargs='+', type=float, default=[0.01, 0.03, 0.1, 0.3, 1, 2, 4, 8],
                   help='fusion weights (w, w1 and w2)')
    p.add_argument('--eps', type=float, default=1e-4)
    p.add_argument('--pgm_ranks', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--pgm_powers', nargs='+', type=float, default=[0.25, 0.5, 0.75],
                   help='a in E_c = S^-a A_c S^-a (0.5 = PGM)')
    p.add_argument('--pgm_betas', nargs='+', type=float, default=[0.5, 1.0], help='eigenvalue tempering')
    p.add_argument('--opt_ranks', nargs='+', type=int, default=[2, 4, 8, 32])
    p.add_argument('--opt_iters', nargs='+', type=int, default=[1, 2, 5, 10, 20])
    p.add_argument('--lda_shrinks', nargs='+', type=float, default=[0.01, 0.1, 0.3])
    p.add_argument('--fecam_gammas', nargs='+', type=float, default=[1.0])
    p.add_argument('--ranpac_dim', type=int, default=5000)
    p.add_argument('--ranpac_lambdas', nargs='+', type=float, default=[1e-4, 1e-3, 1e-2, 1e-1],
                   help='ridge, relative to the mean eigenvalue of the Gram matrix')
    p.add_argument('--max_tasks', type=int, default=None)
    return p.parse_args(argv)


def load_dumps(dump_dir, max_tasks=None):
    files = sorted((f for f in os.listdir(dump_dir) if re.fullmatch(r'task\d+\.pt', f)), key=lambda f: int(f[4:-3]))
    dumps = []
    for t, f in enumerate(files[:max_tasks]):
        d = torch.load(os.path.join(dump_dir, f), map_location='cpu')
        assert d['task'] == t, f'{f}: expected dump of task {t + 1}, missing an earlier task?'
        dumps.append(d)
    assert dumps, f'no task*.pt in {dump_dir}'
    return dumps


def eval_sets(dumps, t, split):
    """Evaluation samples at checkpoint t, one dict per task i <= t."""
    every = dumps[0]['val_every']
    sets = []
    for i in range(t + 1):
        if split == 'test':
            e = dict(dumps[t]['test'][i], frozen=dumps[i]['test'][i]['frozen'])
        elif i < t:
            e = dict(dumps[t]['val'][i], frozen=dumps[i]['train']['frozen'][::every])
        else:
            e = {k: v[::every] for k, v in dumps[t]['train'].items()}
        sets.append(e)
    return sets


def inv_pow(S, a, rel_floor=1e-12):
    e, V = torch.linalg.eigh(S)
    e = e.clamp_min(rel_floor * e.max().clamp_min(_TINY))
    return (V * e.pow(-a)) @ V.T


def top_eig(X, rank):
    """Top-`rank` eigenpairs of X^T X / n, eigenvalues renormalized to sum 1 (zero-padded)."""
    n, D = X.shape
    _, s, Vh = torch.linalg.svd(X / math.sqrt(n), full_matrices=False)
    k = min(rank, s.numel())
    U = torch.zeros(D, rank, dtype=X.dtype)
    lam = torch.zeros(rank, dtype=X.dtype)
    U[:, :k] = Vh[:k].T
    lam[:k] = s[:k] ** 2
    return U, lam / lam.sum()


class Stats:
    """Class statistics of the seen classes for one run (val or test), grown task by task."""

    def __init__(self, args, num_classes, dim):
        self.args, self.dev, self.D = args, torch.device(args.device), dim
        self.rank = max(args.pgm_ranks + args.opt_ranks)
        self.classes = []
        self.center = {}
        self.src = {s: dict(U={'none': [], 'task1': []}, lam={'none': [], 'task1': []}, mu_norm=[], mu_raw=[],
                            scatter=torch.zeros(dim, dim, dtype=torch.float64), n=0, fecam={g: [] for g in args.fecam_gammas})
                    for s in args.sources}
        if 'ranpac' in args.kinds:
            g = torch.Generator().manual_seed(0)
            self.rp = torch.randn(dim, args.ranpac_dim, generator=g).to(self.dev)
            for s in args.sources:
                self.src[s]['G'] = torch.zeros(args.ranpac_dim, args.ranpac_dim, dtype=torch.float64, device=self.dev)
                self.src[s]['Q'] = torch.zeros(args.ranpac_dim, num_classes, dtype=torch.float64, device=self.dev)
        self._cache = {}

    def add_task(self, train, keep):
        y = train['target'][keep]
        new = torch.unique(y).tolist()
        assert not set(new) & set(self.classes), 'a class appears in two tasks'
        centers = ['none', 'task1'] if {'pgmc', 'optc'} & set(self.args.kinds) else ['none']
        for s in self.args.sources:
            st = self.src[s]
            X_all = train[s][keep].double()
            if s not in self.center:  # fixed once, from the first task only
                self.center[s] = X_all.mean(0)
            for c in new:
                X = X_all[y == c]
                assert len(X) > 0, f'class {c} has no statistics samples'
                Xn = F.normalize(X, dim=1)
                for cen in centers:
                    Z = Xn if cen == 'none' else F.normalize(X - self.center[s], dim=1)
                    U, lam = top_eig(Z, self.rank)
                    st['U'][cen].append(U)
                    st['lam'][cen].append(lam)
                st['mu_norm'].append(Xn.mean(0))
                mu = X.mean(0)
                st['mu_raw'].append(mu)
                st['scatter'] += (X - mu).T @ (X - mu)
                st['n'] += len(X)
                if 'fecam' in self.args.kinds:
                    Xc = Xn - Xn.mean(0)
                    cov = Xc.T @ Xc / max(len(X) - 1, 1)
                    diag = torch.diagonal(cov)
                    off = (cov.sum() - diag.sum()) / (self.D * (self.D - 1))
                    eye = torch.eye(self.D, dtype=cov.dtype)
                    for g in self.args.fecam_gammas:
                        cs = cov + g * (diag.mean() * eye + off * (1 - eye))
                        d = torch.diagonal(cs).sqrt()
                        st['fecam'][g].append(torch.linalg.inv(cs / d[:, None] / d[None, :]).float())
            if 'ranpac' in self.args.kinds:
                H = torch.relu(X_all.float().to(self.dev) @ self.rp).double()
                st['G'] += H.T @ H
                st['Q'].index_add_(1, y.to(self.dev), H.T)
        self.classes += new
        self._cache.clear()

    def stack(self, s, key, cen=None):
        k = (s, key, cen)
        if k not in self._cache:
            v = self.src[s][key] if cen is None else self.src[s][key][cen]
            self._cache[k] = torch.stack(v).to(self.dev)
        return self._cache[k]


# ---------------------------------------------------------------------------------------------- heads

def _chunks(n, size=4096):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def pgm_logp(xn, U, lam, eps, a):
    """log p_c(x) for E_c = S^-a A_c S^-a, A_c = (rho_c + eps I) / M, renormalized over classes."""
    M, D, k = U.shape
    Uf = U.permute(1, 0, 2).reshape(D, M * k)
    W = Uf * lam.reshape(1, -1).sqrt()
    S = W @ W.T / M + eps * torch.eye(D, dtype=U.dtype, device=U.device)
    S_ma = inv_pow(S, a)
    out = []
    for sl in _chunks(len(xn)):
        y = xn[sl] @ S_ma
        p = ((y @ Uf).view(-1, M, k) ** 2 * lam).sum(-1) + eps * (y ** 2).sum(-1, keepdim=True)
        out.append(p)
    p = torch.cat(out).clamp_min(_TINY)
    return p.log() - p.sum(1, keepdim=True).log()


def optimal_measurements(U, lam, eps, iters):
    """Iterates Pi_c <- G^-1/2 rho_c Pi_c rho_c G^-1/2 from the PGM. With rho_c = U_c L_c U_c^T every iterate
    has the form Pi_c = Gmh U_c B_c U_c^T Gmh. Returns {n: (Gmh, B)} and the success probability
    (1/M) sum_c tr(rho_c Pi_c) of every iterate (index 0 = PGM)."""
    M, D, k = U.shape
    Uf = U.permute(1, 0, 2).reshape(D, M * k)
    eye = torch.eye(D, dtype=U.dtype, device=U.device)
    S = (Uf * lam.reshape(1, -1).sqrt()) @ (Uf * lam.reshape(1, -1).sqrt()).T / M + eps * eye
    Gmh = inv_pow(S, 0.5)
    Z = (Gmh @ Uf).view(D, M, k).permute(1, 0, 2)
    K = torch.einsum('mdi,mdj->mij', U, Z)
    T = (K * lam[:, None, :]) @ K + eps * torch.einsum('mdi,mdj->mij', Z, Z)
    T = T / M  # T_c = U_c^T Pi_c U_c for the PGM Pi_c = Gmh (rho_c + eps I) Gmh / M
    psucc = [(torch.diagonal(T, dim1=1, dim2=2) * lam).sum().item() / M]
    snaps = {}
    for n in range(1, max(iters) + 1):
        B = lam[:, :, None] * T * lam[:, None, :]  # L_c T_c L_c
        G = (U @ B).permute(1, 0, 2).reshape(D, M * k) @ Uf.T
        Gmh = inv_pow((G + G.T) / 2, 0.5)
        K = torch.einsum('mdi,mdj->mij', U, (Gmh @ Uf).view(D, M, k).permute(1, 0, 2))
        T = K @ B @ K.transpose(1, 2)
        psucc.append((torch.diagonal(T, dim1=1, dim2=2) * lam).sum().item() / M)
        if n in iters:
            snaps[n] = (Gmh, B)
    return snaps, psucc


def measurement_logp(xn, U, Gmh, B):
    M, D, k = U.shape
    Uf = U.permute(1, 0, 2).reshape(D, M * k)
    out = []
    for sl in _chunks(len(xn)):
        z = ((xn[sl] @ Gmh) @ Uf).view(-1, M, k)
        out.append(torch.einsum('bmi,mij,bmj->bm', z, B, z))
    p = torch.cat(out).clamp_min(_TINY)
    return p.log() - p.sum(1, keepdim=True).log()


def head_groups(stats, feats, info):
    """Yields (kind, config, {source: score (B, M)}); fusion and dual heads are derived by the caller."""
    a, dev = stats.args, stats.dev
    srcs = a.sources

    def trunc(U, lam, k, beta=1.0):
        lam = lam[:, :k] ** beta
        return U[:, :, :k], lam / lam.sum(1, keepdim=True).clamp_min(_TINY)

    xn = {(s, 'none'): F.normalize(feats[s], dim=1) for s in srcs}
    if {'pgmc', 'optc'} & set(a.kinds):
        for s in srcs:
            xn[(s, 'task1')] = F.normalize(feats[s] - stats.center[s].to(dev), dim=1)

    yield 'ncm', '', {s: xn[(s, 'none')] @ F.normalize(stats.stack(s, 'mu_norm'), dim=1).T for s in srcs}

    for kind, cen in (('pgm', 'none'), ('pgmc', 'task1')):
        if kind not in a.kinds:
            continue
        for k, pw, beta in itertools.product(a.pgm_ranks, a.pgm_powers, a.pgm_betas):
            sc = {}
            for s in srcs:
                U, lam = trunc(stats.stack(s, 'U', cen), stats.stack(s, 'lam', cen), k, beta)
                sc[s] = pgm_logp(xn[(s, cen)], U, lam, a.eps, pw)
            yield kind, f'k={k},a={pw},b={beta}', sc

    for kind, cen in (('opt', 'none'), ('optc', 'task1')):
        if kind not in a.kinds:
            continue
        for k in a.opt_ranks:
            snaps = {}
            for s in srcs:
                U, lam = trunc(stats.stack(s, 'U', cen), stats.stack(s, 'lam', cen), k)
                snaps[s], psucc = optimal_measurements(U, lam, a.eps, a.opt_iters)
                info[f'{s}_{kind}|k={k}'] = [round(v, 6) for v in psucc]
            for n in a.opt_iters:
                yield kind, f'k={k},it={n}', {
                    s: measurement_logp(xn[(s, cen)], trunc(stats.stack(s, 'U', cen), stats.stack(s, 'lam', cen), k)[0],
                                        *snaps[s][n]) for s in srcs}

    if 'lda' in a.kinds:
        for shrink in a.lda_shrinks:
            sc = {}
            for s in srcs:
                st = stats.src[s]
                cov = st['scatter'].to(dev) / max(st['n'], 1)
                eye = torch.eye(stats.D, dtype=cov.dtype, device=dev)
                cov = (1 - shrink) * cov + shrink * torch.trace(cov) / stats.D * eye
                mu = stats.stack(s, 'mu_raw')
                Pm = mu @ torch.linalg.inv(cov)
                sc[s] = feats[s] @ Pm.T - 0.5 * (Pm * mu).sum(1)
            yield 'lda', f'shrink={shrink}', sc

    if 'fecam' in a.kinds:
        for g in a.fecam_gammas:
            sc = {}
            for s in srcs:
                x = xn[(s, 'none')].float()
                mu = stats.stack(s, 'mu_norm').float()
                P = stats.stack(s, 'fecam', g)
                d = torch.empty(len(x), len(mu), device=dev)
                for c in range(len(mu)):
                    diff = x - mu[c]
                    d[:, c] = ((diff @ P[c]) * diff).sum(1)
                sc[s] = -0.5 * d.double()
            yield 'fecam', f'gamma={g}', sc

    if 'ranpac' in a.kinds:
        cls = torch.tensor(stats.classes, device=dev)
        eig = {}
        for s in srcs:
            e, V = torch.linalg.eigh(stats.src[s]['G'])
            eig[s] = (e, V, V.T @ stats.src[s]['Q'][:, cls])
        for lr in a.ranpac_lambdas:
            sc = {}
            for s in srcs:
                e, V, VtQ = eig[s]
                Wout = V @ (VtQ / (e.clamp_min(0) + lr * e.mean())[:, None])
                sc[s] = torch.cat([torch.relu(feats[s][sl].float() @ stats.rp).double() @ Wout
                                   for sl in _chunks(len(feats[s]))])
            yield 'ranpac', f'lambda={lr}', sc


# ------------------------------------------------------------------------------------------ evaluation

class Recorder:
    def __init__(self, T):
        self.T = T
        self.acc = {}

    def begin(self, t, target_local, tid):
        self.t, self.y, self.tid = t, target_local, tid
        self.n = torch.bincount(tid, minlength=t + 1).double()

    def add(self, name, cfg, score):
        correct = score.argmax(1) == self.y
        hit = torch.bincount(self.tid[correct], minlength=self.t + 1).double()
        m = self.acc.setdefault((name, cfg), np.zeros((self.T, self.T)))
        m[:self.t + 1, self.t] = (100.0 * hit / self.n).cpu().numpy()


def evaluate_checkpoint(stats, sets, t, rec, info):
    a, dev = stats.args, stats.dev
    cls = torch.tensor(stats.classes, device=dev)
    local = torch.full((int(cls.max()) + 1,), -1, dtype=torch.long, device=dev)
    local[cls] = torch.arange(len(cls), device=dev)
    target = torch.cat([e['target'] for e in sets]).to(dev)
    y = local[target]
    assert (y >= 0).all(), 'an evaluation label is not a seen class'
    tid = torch.cat([torch.full((len(e['target']),), i, dtype=torch.long) for i, e in enumerate(sets)]).to(dev)
    rec.begin(t, y, tid)
    log_lin = F.log_softmax(torch.cat([e['logits'] for e in sets]).to(dev)[:, cls].double(), dim=1)
    rec.add('linear', '', log_lin)
    feats = {s: torch.cat([e[s] for e in sets]).to(dev, torch.float64) for s in a.sources}
    dual = set(a.sources) == set(SOURCES)
    for kind, cfg, sc in head_groups(stats, feats, info):
        sep = ',' if cfg else ''
        for s in a.sources:
            rec.add(f'{s}_{kind}', cfg, sc[s])
            for w in a.weights:
                rec.add(f'{s}_{kind}_fusion', f'{cfg}{sep}w={w}', log_lin + w * sc[s])
        if dual:
            for w1, w2 in itertools.product(a.weights, a.weights):
                rec.add(f'dual_{kind}_fusion', f'{cfg}{sep}w1={w1},w2={w2}',
                        log_lin + w1 * sc['frozen'] + w2 * sc['prompted'])


def summary(m):
    """Final Acc, IncAcc and Forgetting of an acc[test_task, train_task] matrix (as HeadMetrics)."""
    T = m.shape[0]
    acc = float(m[:, T - 1].mean())
    inc = float(np.mean([m[:t + 1, t].mean() for t in range(T)]))
    forget = float(np.mean(m[:T - 1].max(1) - m[:T - 1, T - 1])) if T > 1 else 0.0
    return dict(acc=acc, incremental_acc=inc, forgetting=forget)


def select(val, test):
    """Per head family: configuration with the best final val accuracy, reported on test."""
    families = {}
    for (name, cfg) in test:
        families.setdefault(name, []).append(cfg)
    rows = []
    for name, cfgs in families.items():
        best = max(cfgs, key=lambda c: val[(name, c)][:, -1].mean())
        oracle = max(cfgs, key=lambda c: test[(name, c)][:, -1].mean())
        rows.append(dict(head=name, config=best, val_acc=float(val[(name, best)][:, -1].mean()),
                         **summary(test[(name, best)]), oracle_acc=float(test[(name, oracle)][:, -1].mean()),
                         oracle_config=oracle))
    return sorted(rows, key=lambda r: -r['acc'])


def print_table(rows, base):
    print(f'{"head":<26}{"val":>7}{"Acc":>8}{"vs lin":>8}{"IncAcc":>8}{"Forget":>8}{"oracle":>8}  config')
    for r in rows:
        print(f'{r["head"]:<26}{r["val_acc"]:>7.2f}{r["acc"]:>8.2f}{r["acc"] - base:>+8.2f}'
              f'{r["incremental_acc"]:>8.2f}{r["forgetting"]:>8.2f}{r["oracle_acc"]:>8.2f}  {r["config"]}')


def run(args):
    dumps = load_dumps(args.dump_dir, args.max_tasks)
    T = len(dumps)
    every = dumps[0]['val_every']
    num_classes = dumps[0]['train']['logits'].shape[1]
    dim = dumps[0]['train']['frozen'].shape[1]
    stats = {'val': Stats(args, num_classes, dim), 'test': Stats(args, num_classes, dim)}
    recs = {'val': Recorder(T), 'test': Recorder(T)}
    info = {}
    start = time.time()
    for t in range(T):
        train = dumps[t]['train']
        n = len(train['target'])
        pos = torch.arange(n)
        keep = {'val': pos % every != 0, 'test': torch.ones(n, dtype=torch.bool)}
        for split in ('val', 'test'):
            stats[split].add_task(train, keep[split])
            evaluate_checkpoint(stats[split], eval_sets(dumps, t, split), t, recs[split], info)
        lin = recs['test'].acc[('linear', '')][:t + 1, t].mean()
        print(f'task {t + 1}/{T}: {len(stats["test"].classes)} classes, linear {lin:.2f}, '
              f'{time.time() - start:.0f}s', flush=True)

    val, test = recs['val'].acc, recs['test'].acc
    rows = select(val, test)
    base = summary(test[('linear', '')])['acc']
    print(f'\n{args.dump_dir} | {T} tasks | config selected on val (every {every}th train sample)')
    print_table(rows, base)

    ref = {'frozen_pgm_fusion': 'k=32,a=0.5,b=1.0,w=1.0', 'prompted_pgm_fusion': 'k=32,a=0.5,b=1.0,w=1.0',
           'dual_pgm_fusion': 'k=32,a=0.5,b=1.0,w1=1.0,w2=1.0', 'prompted_lda_fusion': 'shrink=0.1,w=1.0',
           'frozen_lda_fusion': 'shrink=0.1,w=1.0'}
    ref_rows = [dict(head=h, config=c, val_acc=float(val[(h, c)][:, -1].mean()), **summary(test[(h, c)]),
                     oracle_acc=float('nan')) for h, c in ref.items() if (h, c) in test]
    if ref_rows:
        print('\nreference: density_head.py defaults (should match density_heads_eval_*.json with w=1)')
        print_table(ref_rows, base)

    out = args.out or os.path.join(args.dump_dir, 'sweep.json')
    with open(out, 'w') as f:
        json.dump(dict(args=vars(args), tasks=T, val_every=every, selected=rows, opt_success_prob=info,
                       test={f'{h}|{c}': summary(m) for (h, c), m in test.items()},
                       val={f'{h}|{c}': float(m[:, -1].mean()) for (h, c), m in val.items()}), f, indent=1)
    print(f'\nsaved {out}')
    return rows, val, test, info


if __name__ == '__main__':
    run(get_args())
