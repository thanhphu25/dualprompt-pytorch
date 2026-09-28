"""Tests for quantum_measurement.py (plan-quantum.md section 8). Run: python -m pytest tests -q"""
import argparse
import copy
import io
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
from configs.cifar100_dualprompt import get_args_parser
from quantum_measurement import DensityClassBank, QuantumMeasurement, group_by_task, task_scores, class_readouts
from utils import str2bool
from vision_transformer import VisionTransformer


# ---------------------------------------------------------------- bank maths

def make_bank(counts=(2, 2, 2), dim=12, rank=4, n=40, eps=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = []
    bank = DensityClassBank(sum(counts) + 2, dim, rank=rank, eps=eps)  # two extra never-seen classes
    c = 0
    for t, k in enumerate(counts):
        for _ in range(k):
            centre = torch.randn(dim, generator=g)
            basis = torch.randn(3, dim, generator=g)
            x = centre + 0.5 * torch.randn(n, 3, generator=g) @ basis + 0.1 * torch.randn(n, dim, generator=g)
            bank.add_class(c, x, task=t)
            labels.append(c)
            c += 1
    return bank, torch.tensor(labels)


def unit(n, dim, seed=1):
    return torch.nn.functional.normalize(torch.randn(n, dim, generator=torch.Generator().manual_seed(seed)), dim=-1)


@pytest.mark.parametrize('prior', ['class', 'task'])
@pytest.mark.parametrize('rank', [None, 2])
def test_completeness(prior, rank):
    bank, classes = make_bank(counts=(2, 3, 1))
    E = bank.povm(classes, rank=rank, prior=prior)
    assert torch.allclose(E.sum(0), torch.eye(E.shape[1], dtype=E.dtype), atol=1e-8)
    assert all(torch.linalg.eigvalsh(e).min() > -1e-10 for e in E)  # every element is PSD
    x = unit(16, E.shape[1]).double()
    p = torch.einsum('bd,cde,be->bc', x, E, x)
    assert torch.allclose(p.sum(-1), torch.ones(16, dtype=p.dtype), atol=1e-8)
    # the eigenpair formula gives the same Born probabilities as the explicit matrices (no renormalisation needed)
    assert torch.allclose(bank.pgm_log_probs(x, classes, rank=rank, prior=prior).exp(), p, atol=1e-8)


def test_eigenpairs_match_second_moment():
    dim, n = 10, 50
    x = torch.randn(n, dim)
    bank = DensityClassBank(1, dim, rank=dim, eps=1e-4)
    bank.add_class(0, x, task=0)
    xn = torch.nn.functional.normalize(x.double(), dim=-1)
    second = xn.T @ xn / n
    v, lam = bank.vectors[0].double(), bank.values[0].double()
    assert torch.allclose((v * lam) @ v.T, second / second.trace(), atol=1e-5)
    assert abs(float(lam.sum()) - 1) < 1e-6
    # truncated rank is renormalised to trace one
    _, lam_r = bank._states(torch.tensor([0]), rank=3)
    assert lam_r.shape[-1] == 3 and abs(float(lam_r.sum()) - 1) < 1e-9


def test_coarse_graining_equal_tasks():
    bank, classes = make_bank(counts=(2, 2, 2))
    x = unit(20, 12).double()
    ct = bank.task_of_class[classes]
    task = group_by_task(bank.pgm_log_probs(x, classes, prior='task'), ct, 4)
    via_class = group_by_task(bank.pgm_log_probs(x, classes, prior='class'), ct, 4)
    assert torch.allclose(task[:, :3], via_class[:, :3], atol=1e-10)
    assert torch.isinf(task[:, 3]).all()


def test_task_pgm_is_pgm_on_task_states_unequal_tasks():
    bank, classes = make_bank(counts=(1, 3, 2))
    ct = bank.task_of_class[classes]
    v, lam = bank._states(classes)
    sig = torch.einsum('cdk,ck,cek->cde', v, lam, v)
    eye = torch.eye(12, dtype=torch.float64)
    rho = torch.stack([sig[ct == t].mean(0) for t in range(3)])
    A = (rho + bank.eps * eye) / 3
    ev, U = torch.linalg.eigh(A.sum(0))
    R = (U * ev.rsqrt()) @ U.T
    E = R @ A @ R
    x = unit(20, 12).double()
    p_direct = torch.einsum('bd,tde,be->bt', x, E, x)
    p_bank = group_by_task(bank.pgm_log_probs(x, classes, prior='task'), ct, 3).exp()
    assert torch.allclose(p_direct, p_bank, atol=1e-8)


def test_unseen_never_wins():
    bank, _ = make_bank(counts=(2, 2, 2))
    classes = torch.tensor([0, 1, 2, 3])  # only tasks 0 and 1 seen
    x = unit(64, 12)
    ro = class_readouts(bank, x, classes, [2])
    scores = task_scores(bank, x, classes, 5, [2], ro)
    for name, s in scores.items():
        assert s.argmax(-1).max() <= 1, name
        assert torch.isinf(s[:, 2:]).all(), name


def test_bank_checkpoint_round_trip():
    bank, classes = make_bank()
    x = unit(8, 12)
    before = bank.pgm_log_probs(x, classes)
    buf = io.BytesIO()
    torch.save(bank.state_dict(), buf)
    buf.seek(0)
    other = DensityClassBank(8, 12, rank=4, eps=1e-3)
    other.load_state_dict(torch.load(buf))
    assert torch.equal(other.pgm_log_probs(x, classes), before)


def test_str2bool():
    assert str2bool('false') is False and str2bool('False') is False and str2bool('0') is False
    assert str2bool('true') is True and str2bool(True) is True
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool('maybe')
    p = argparse.ArgumentParser()
    get_args_parser(p)
    a = p.parse_args(['--batchwise_prompt', 'false', '--task_inc', 'False'])
    assert a.batchwise_prompt is False and a.task_inc is False and a.train_mask is True


# ---------------------------------------------------------------- tiny DualPrompt end to end

N_CLASSES, N_TASKS, DIM = 6, 2, 48


def tiny_args(**over):
    p = argparse.ArgumentParser()
    get_args_parser(p)
    args = p.parse_args(['--num_tasks', str(N_TASKS), '--size', str(N_TASKS), '--batch-size', '8', '--epochs', '1',
                         '--num_workers', '0', '--no-pin-mem', '--output_dir', '', '--print_freq', '1000',
                         '--qm_rank', '6', '--qm_pgm_ranks', '2', '4',
                         '--qm_head_routers', 'cosine', 'oracle', 'task_pgm', 'task_lda'])
    args.distributed, args.nb_classes, args.lr = False, N_CLASSES, 0.01
    for k, v in over.items():
        setattr(args, k, v)
    return args


class FakeImages(Dataset):
    def __init__(self, n_per_class, seed, transform):
        g = torch.Generator().manual_seed(seed)
        protos = torch.randn(N_CLASSES, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        self.targets = torch.arange(N_CLASSES).repeat_interleave(n_per_class)
        self.data = protos[self.targets] + 0.8 * torch.randn(len(self.targets), 3, 32, 32, generator=g)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(self.data[i]), int(self.targets[i])


def train_tf(x):  # draws from the global torch RNG, like RandomResizedCrop/Flip
    return x.flip(-1) if torch.rand(()) < 0.5 else x


def eval_tf(x):
    return x


def tiny_data(args):
    train, val = FakeImages(12, 0, train_tf), FakeImages(4, 1, eval_tf)
    class_mask = [[0, 1, 2], [3, 4, 5]]
    loaders = []
    for cm in class_mask:
        tr = Subset(train, [i for i, t in enumerate(train.targets.tolist()) if t in cm])
        va = Subset(val, [i for i, t in enumerate(val.targets.tolist()) if t in cm])
        loaders.append({'train': DataLoader(tr, sampler=torch.utils.data.RandomSampler(tr), batch_size=args.batch_size),
                        'val': DataLoader(va, sampler=torch.utils.data.SequentialSampler(va), batch_size=args.batch_size)})
    return loaders, class_mask


def tiny_models(args, seed=0):
    torch.manual_seed(seed)
    common = dict(img_size=32, patch_size=8, embed_dim=DIM, depth=6, num_heads=4, num_classes=N_CLASSES)
    original = VisionTransformer(**common)
    model = VisionTransformer(
        **common, prompt_length=args.length, embedding_key=args.embedding_key, prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool, prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask, use_g_prompt=args.use_g_prompt, g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx, use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt, e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt, same_key_value=args.same_key_value)
    for p in original.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith(tuple(args.freeze)):
            p.requires_grad = False
    return model, original


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    if not torch.cuda.is_available():  # train_one_epoch calls torch.cuda.synchronize()
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: None)


def run(args, with_qm):
    loaders, class_mask = tiny_data(args)
    model, original = tiny_models(args)
    qm = QuantumMeasurement(args, DIM, torch.device('cpu')) if with_qm else None
    torch.manual_seed(1)
    optimizer = engine.create_optimizer(args, model)
    engine.train_and_evaluate(model, model, original, torch.nn.CrossEntropyLoss(), loaders, optimizer, None,
                              torch.device('cpu'), class_mask, args, qm=qm)
    return model, original, qm, loaders, class_mask


def test_prompt_idx_matches_native_paths():
    args = tiny_args()
    model, original = tiny_models(args)
    model.eval()
    x = torch.randn(5, 3, 32, 32)
    q = original(x)['pre_logits']
    native = model(x, cls_features=q)
    forced = model(x, cls_features=q, prompt_idx=native['prompt_idx'][:, 0])
    assert torch.equal(native['logits'], forced['logits'])
    t = 1
    train_path = model(x, task_id=t, cls_features=q, train=True)  # use_prompt_mask during training
    forced_t = model(x, cls_features=q, prompt_idx=torch.full((5,), t))
    assert torch.equal(train_path['logits'], forced_t['logits'])
    assert torch.equal(train_path['reduce_sim'], forced_t['reduce_sim'])


def test_training_unchanged_and_heads(capsys):
    args = tiny_args()
    m_off, *_ = run(args, with_qm=False)
    m_on, original, qm, loaders, class_mask = run(args, with_qm=True)
    for (n, a), (_, b) in zip(m_off.state_dict().items(), m_on.state_dict().items()):
        assert torch.equal(a, b), n

    s = qm.summary(N_TASKS - 1)
    heads, routers = s['heads'], s['routers']
    assert routers['oracle']['route_acc_mean'] == 100.0
    for name in ('cosine_linear', 'cosine_linear_nomask', 'oracle_linear', 'oracle_til', 'task_pgm_linear',
                 'task_pgm_r2_linear', 'frozen_pgm', 'frozen_lda', 'frozen_ncm_white', 'cosine_frozen_fusion',
                 'oracle_prompted_fusion_r4', 'task_lda_prompted_lda_fusion', 'task_pgm_soft2', 'task_pgm_soft2_cpgm',
                 'task_lda_soft2_clda'):
        assert name in heads, name
    for d in heads.values():
        assert 0.0 <= d['final_acc'] <= 100.0

    # the legacy log line of the QM path is the original DualPrompt readout
    for i in range(N_TASKS):
        ref = engine.evaluate(m_on, original, loaders[i]['val'], torch.device('cpu'), task_id=i,
                              class_mask=class_mask, args=args)
        new = qm.evaluate(m_on, original, loaders[i]['val'], torch.device('cpu'), i, N_TASKS - 1, class_mask, args)
        assert abs(ref['Acc@1'] - new['Acc@1']) < 1e-9 and abs(ref['Loss'] - new['Loss']) < 1e-4

    # checkpoint round trip keeps every head's prediction
    buf = io.BytesIO()
    torch.save({'qm_banks': qm.state_dict()}, buf)
    buf.seek(0)
    qm2 = QuantumMeasurement(args, DIM, torch.device('cpu'))
    assert qm2.load_state_dict(torch.load(buf)['qm_banks'])
    for i in range(N_TASKS):
        qm2.evaluate(m_on, original, loaders[i]['val'], torch.device('cpu'), i, N_TASKS - 1, class_mask, args)
    for name, a in qm.acc.items():
        assert np.array_equal(a[:, -1], qm2.acc[name][:, -1]), name
    capsys.readouterr()


def test_batchwise_legacy_matches_native():
    args = tiny_args(batchwise_prompt=True)
    model, original, qm, loaders, class_mask = run(args, with_qm=True)
    for i in range(N_TASKS):
        ref = engine.evaluate(model, original, loaders[i]['val'], torch.device('cpu'), task_id=i,
                              class_mask=class_mask, args=args)
        new = qm.evaluate(model, original, loaders[i]['val'], torch.device('cpu'), i, N_TASKS - 1, class_mask, args)
        assert abs(ref['Acc@1'] - new['Acc@1']) < 1e-9
