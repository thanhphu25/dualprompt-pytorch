"""Checks for density_sweep.py and the feature dump. Run: python -m pytest tests -q"""
import os
import sys
import tempfile
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import density_sweep as ds  # noqa: E402
from density_head import DensityHeads  # noqa: E402
from test_density_head import D, FakeModel, FakeOriginal, _import_engine, built_heads, clustered  # noqa: E402

SMALL = ['--weights', '0.1', '1', '--pgm_ranks', '2', '8', '--pgm_powers', '0.5', '--pgm_betas', '1.0',
         '--opt_ranks', '8', '--opt_iters', '1', '3', '--lda_shrinks', '0.1', '--ranpac_dim', '64',
         '--device', 'cpu']


def full_measurement(U, Gmh, B):
    return torch.stack([Gmh @ U[c] @ B[c] @ U[c].T @ Gmh for c in range(len(U))])


def test_pgm_logp_matches_density_head():
    heads, X, _ = built_heads()
    bank = heads.banks['frozen']
    xn = F.normalize(X.double(), dim=1)
    for k in (2, 8):
        lam = bank.lam[:, :k].double()
        p = ds.pgm_logp(xn, bank.U[:, :, :k].double(), lam / lam.sum(1, keepdim=True), bank.eps, 0.5).exp()
        assert torch.allclose(p, bank.pgm_probs(xn, k), atol=1e-6)


def test_optimal_measurement_is_complete_and_improves_on_pgm():
    X, y = clustered([0, 1, 2, 3, 4, 5], 12, spread=1.5)
    Us, lams = zip(*[ds.top_eig(F.normalize(X[y == c].double(), dim=1), 8) for c in range(6)])
    U, lam = torch.stack(Us), torch.stack(lams)  # 6 classes x rank 8 = 48 >= D, so sum_c Pi_c = I
    snaps, psucc = ds.optimal_measurements(U, lam, 1e-4, [1, 5, 30])
    for n in (1, 5, 30):
        Pi = full_measurement(U, *snaps[n])
        assert torch.allclose(Pi.sum(0), torch.eye(D, dtype=torch.float64), atol=1e-6)
        assert torch.linalg.eigvalsh(Pi).min() > -1e-9
    assert psucc[30] >= psucc[0] - 1e-9, psucc


def test_optimal_measurement_reaches_helstrom_bound():
    # two qubit states that are not symmetric, so the PGM is not optimal
    plus, minus = torch.tensor([1., 1.]) / 2 ** 0.5, torch.tensor([1., -1.]) / 2 ** 0.5
    U = torch.stack([torch.eye(2), torch.stack([plus, minus], 1)]).double()
    lam = torch.tensor([[1.0, 0.0], [0.7, 0.3]], dtype=torch.float64)
    rho = torch.einsum('mdi,mi,mei->mde', U, lam, U)
    helstrom = 0.5 * (1 + 0.5 * torch.linalg.eigvalsh(rho[0] - rho[1]).abs().sum().item())
    _, psucc = ds.optimal_measurements(U, lam, 1e-12, [500])
    assert psucc[0] < helstrom - 1e-3
    assert abs(psucc[-1] - helstrom) < 1e-4, (psucc[0], psucc[-1], helstrom)


def fake_dumps(out, tasks=((0, 1), (2, 3), (4, 5)), every=3, n=15):
    g = torch.Generator().manual_seed(1)
    centers = torch.randn(6, D, generator=g) * 3
    W = torch.randn(D, 6, generator=g)

    def split(classes, m):
        f = torch.cat([centers[c] + torch.randn(m, D, generator=g) for c in classes])
        return dict(frozen=f, prompted=2 * f + 1, logits=f @ W, target=torch.tensor(classes).repeat_interleave(m))

    test = [split(c, 10) for c in tasks]
    train = [split(c, n) for c in tasks]
    for t in range(len(tasks)):
        val = {i: {k: v[::every] for k, v in train[i].items() if k != 'frozen'} for i in range(t)}
        te = {i: {k: v for k, v in test[i].items() if k != 'frozen' or i == t} for i in range(t + 1)}
        torch.save(dict(task=t, val_every=every, train=train[t], val=val, test=te), os.path.join(out, f'task{t + 1}.pt'))
    return train, test


def test_sweep_end_to_end():
    with tempfile.TemporaryDirectory() as out:
        _, test = fake_dumps(out)
        rows, val, res, info = ds.run(ds.get_args([out] + SMALL))
        heads = {r['head'] for r in rows}
        for s in ('frozen', 'prompted', 'dual'):
            for kind in ('pgm', 'pgmc', 'opt', 'optc', 'lda', 'fecam', 'ranpac', 'ncm'):
                assert f'{s}_{kind}_fusion' in heads, (s, kind)
        # linear head: accuracy of the dumped logits restricted to the seen classes
        seen = [0, 1, 2, 3, 4, 5]
        expect = [100.0 * (test[i]['logits'][:, seen].argmax(1) == test[i]['target']).float().mean().item()
                  for i in range(3)]
        assert np.allclose(res[('linear', '')][:, 2], expect)
        assert res[('prompted_pgm', 'k=8,a=0.5,b=1.0')][:, 2].mean() > 90
        assert os.path.exists(os.path.join(out, 'sweep.json'))
        assert 'prompted_opt|k=8' in info


def test_dump_density_features_feeds_sweep():
    engine = _import_engine()
    torch.manual_seed(0)
    loaders = []
    for i, classes in enumerate([[0, 1, 2], [3, 4, 5]]):
        X, y = clustered(classes, 10, seed=i)
        dset = torch.utils.data.TensorDataset(X, y)
        loaders.append({'val': torch.utils.data.DataLoader(dset, batch_size=8),
                        'train_eval': torch.utils.data.DataLoader(dset, batch_size=8)})
    model, original = FakeModel(), FakeOriginal()
    with tempfile.TemporaryDirectory() as out:
        args = Namespace(density_val_every=4, density_dump_dir=out, task_inc=False)
        rng = torch.get_rng_state()
        for t in range(2):
            engine.dump_density_features(model, original, loaders, 'cpu', t, None, args)
        assert torch.equal(torch.get_rng_state(), rng)
        assert not any(model.calls), 'dump must use the eval forward (train=False)'
        d = torch.load(os.path.join(out, 'task2.pt'))
        assert len(d['train']['target']) == 30 and 'frozen' in d['train']
        assert len(d['val'][0]['target']) == 8 and 'frozen' not in d['val'][0]  # positions 0, 4, ..., 28
        assert 'frozen' not in d['test'][0] and 'frozen' in d['test'][1]
        X0 = loaders[0]['train_eval'].dataset.tensors[0]
        with torch.no_grad():
            assert torch.allclose(d['val'][0]['prompted'], model.proj(X0[::4]), atol=1e-6)
        rows, _, res, _ = ds.run(ds.get_args([out] + SMALL + ['--kinds', 'pgm', 'lda']))
        with torch.no_grad():
            lin = [(model(l['val'].dataset.tensors[0])['logits'].argmax(1) == l['val'].dataset.tensors[1])
                   .float().mean().item() * 100 for l in loaders]
        assert np.allclose(res[('linear', '')][:, 1], lin)
