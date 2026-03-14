import os, copy, time, logging, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_MAP:   Dict[str, int] = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}
CLASS_NAMES: List[str]      = ["Normal", "DoS", "Probe", "R2L", "U2R"]
NUM_CLASSES: int             = 5

CONTINUOUS_FEATURES: List[str] = [
    "ct_srv_src", "ct_srv_dst", "sloss", "sbytes", "dur", "ct_dst_ltm",
    "dloss", "is_ftp_login", "ct_dst_src_ltm", "dbytes", "total_bytes",
    "packet_rate", "host_interaction", "connection_density",
    "service_diversity", "flow_balance",
]

NSL_KDD_TAXONOMY: Dict[str, int] = {
    "normal": 0,
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "apache2": 1, "udpstorm": 1, "processtable": 1,
    "worm": 1, "mailbomb": 1, "snmpgetattack": 1, "snmpguess": 1, "mscan": 1,
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "saint": 2, "portscan": 2,
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "xlock": 3, "xsnoop": 3,
    "sendmail": 3, "named": 3, "httptunnel": 3,
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "xterm": 4, "ps": 4, "sqlattack": 4,
}

UNSW_NB15_TAXONOMY: Dict[str, int] = {
    "normal": 0,
    "dos": 1, "generic": 1,
    "reconnaissance": 2, "fuzzers": 2, "analysis": 2,
    "backdoors": 3, "exploits": 3,
    "shellcode": 4, "worms": 4,
}


@dataclass
class MultiClassNIDSConfig:
    input_dim:           int        = 248
    encoder_dims:        List[int]  = field(default_factory=lambda: [512, 256, 128])
    latent_dim:          int        = 64
    classifier_hidden:   int        = 128
    dropout_rate:        float      = 0.30
    noise_std:           float      = 0.10
    sparsity_weight:     float      = 1e-3
    sparsity_target:     float      = 0.05
    learning_rate:       float      = 1e-3
    weight_decay:        float      = 1e-4
    batch_size:          int        = 512
    pretrain_epochs:     int        = 100
    finetune_epochs:     int        = 180
    early_stop_patience: int        = 15
    focal_gamma:         float      = 2.0
    label_smoothing:     float      = 0.05
    attn_num_heads:      int        = 4
    skip_proj_dim:       int        = 16
    t0_epochs:           int        = 20
    eta_min:             float      = 1e-6
    device:              str        = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers:         int        = 0
    seed:                int        = 42


def load_hybrid_csv(csv_path: str, label_col: str = "label",
                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df            = pd.read_csv(csv_path)
    feature_names = [c for c in df.columns if c != label_col]
    X             = df[feature_names].values.astype(np.float32)
    y             = df[label_col].values.astype(np.int64)
    if verbose:
        log.info("─" * 60)
        log.info(f"Loaded: {Path(csv_path).name}  shape={X.shape}")
        for i, name in enumerate(CLASS_NAMES):
            n = (y == i).sum()
            log.info(f"  [{i}] {name:<8}: {n:>8,}  ({n/len(y)*100:.2f}%)")
        log.info("─" * 60)
    return X, y, feature_names


def apply_targeted_smote(X: np.ndarray, y: np.ndarray,
                         minority_classes: Tuple[int, ...] = (3, 4),
                         target_ratio: float = 0.15,
                         k_neighbors: int = 5,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    counts   = Counter(y.tolist())
    max_cnt  = max(counts.values())
    target   = int(max_cnt * target_ratio)
    strategy = {c: target for c in minority_classes if counts.get(c, 0) < target}
    if not strategy:
        log.info("  SMOTE: all minority classes already meet target — skipped.")
        return X, y
    for c, t in strategy.items():
        log.info(f"  SMOTE [{CLASS_NAMES[c]}]: {counts.get(c,0)} → {t}")
    k = max(1, min(k_neighbors, min(counts.get(c, 1) for c in strategy) - 1))
    X_r, y_r = SMOTE(sampling_strategy=strategy, k_neighbors=k,
                     random_state=random_state).fit_resample(X, y)
    nc = len(CONTINUOUS_FEATURES)
    X_r[:, nc:] = np.clip(np.round(X_r[:, nc:]), 0, 1)
    perm = np.random.default_rng(random_state).permutation(len(X_r))
    return X_r[perm], y_r[perm]


class _NIDSDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.y[i]


def build_multiclass_dataloader(X, y, batch_size=512,
                                 use_class_aware_sampler=True,
                                 shuffle=True, num_workers=0) -> DataLoader:
    sampler = None
    if use_class_aware_sampler:
        counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
        cw      = 1.0 / (counts + 1e-6)
        cw      = cw / cw.sum() * NUM_CLASSES
        sampler = WeightedRandomSampler(
            torch.from_numpy(cw[y].astype(np.float32)),
            num_samples=len(y), replacement=True,
        )
        shuffle = False
    return DataLoader(_NIDSDS(X, y), batch_size=batch_size,
                      shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=False)


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, num_classes=NUM_CLASSES,
                 reduction="mean", label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.num_classes = num_classes
        self.reduction = reduction; self.label_smoothing = label_smoothing
        self.register_buffer("alpha",
            alpha.float() if alpha is not None else torch.ones(num_classes))

    @classmethod
    def from_class_counts(cls, counts: np.ndarray, gamma=2.0,
                           beta=0.9999, label_smoothing=0.05):
        eff = 1.0 - np.power(beta, counts.astype(np.float64))
        w   = (1.0 - beta) / (eff + 1e-8)
        w   = w / (w.sum() / len(w))
        log.info("  Focal Loss alpha (Class-Balanced):")
        for i, (n, v) in enumerate(zip(CLASS_NAMES, w)):
            log.info(f"    [{i}] {n:<8}: {v:.4f}")
        return cls(gamma=gamma, alpha=torch.from_numpy(w.astype(np.float32)),
                   label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        B, C  = logits.shape
        log_p = F.log_softmax(logits, -1)
        p_t   = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t).pow(self.gamma)
        a_t   = self.alpha[targets]
        if self.label_smoothing > 0 and self.training:
            eps = self.label_smoothing
            sm  = torch.full((B, C), eps/(C-1), device=logits.device)
            sm.scatter_(1, targets.unsqueeze(1), 1.0 - eps)
            ce = -(sm * log_p).sum(-1)
        else:
            ce = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = a_t * focal * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class ResidualEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(in_dim,  out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.fc2  = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.proj = (nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
                     if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        res = self.proj(x)
        h   = self.act(self.bn1(self.fc1(x)))
        h   = self.drop(h)
        h   = self.bn2(self.fc2(h))
        return self.act(h + res)


class BottleneckSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.1):
        super().__init__()
        while latent_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.token_embed = nn.Linear(1, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ln   = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, 1)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, z):
        tokens  = self.token_embed(z.unsqueeze(-1))
        out, _  = self.attn(tokens, tokens, tokens)
        refined = self.proj(self.drop(out)).squeeze(-1)
        return self.ln(z + refined)


class DenseSkipClassifier(nn.Module):
    def __init__(self, latent_dim, encoder_dims, hidden_dim=128,
                 num_classes=NUM_CLASSES, dropout=0.3, skip_proj_dim=16):
        super().__init__()
        self.skip_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, skip_proj_dim), nn.GELU())
            for d in encoder_dims
        ])
        concat_dim = latent_dim + len(encoder_dims) * skip_proj_dim
        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z_attn, skips):
        parts = [z_attn] + [p(s) for p, s in zip(self.skip_projs, skips)]
        return self.net(torch.cat(parts, -1))


class MultiClassNIDSDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dims, input_dim, dropout=0.3):
        super().__init__()
        dims = [latent_dim] + list(reversed(encoder_dims))
        self.blocks   = nn.Sequential(*[ResidualEncoderBlock(dims[i], dims[i+1], dropout)
                                        for i in range(len(dims)-1)])
        self.out_proj = nn.Linear(dims[-1], input_dim)

    def forward(self, z):
        return self.out_proj(self.blocks(z))


class MultiClassNIDSModel(nn.Module):
    def __init__(self, cfg: MultiClassNIDSConfig):
        super().__init__()
        self.cfg = cfg; self._mode = "pretrain"
        dims = [cfg.input_dim] + cfg.encoder_dims
        self.encoder_blocks = nn.ModuleList([
            ResidualEncoderBlock(dims[i], dims[i+1], cfg.dropout_rate)
            for i in range(len(dims)-1)
        ])
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim),
            nn.BatchNorm1d(cfg.latent_dim), nn.GELU(),
        )
        n_heads = cfg.attn_num_heads
        while cfg.latent_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.bottleneck_attn = BottleneckSelfAttention(cfg.latent_dim, n_heads,
                                                        cfg.dropout_rate * 0.3)
        self.classifier = DenseSkipClassifier(cfg.latent_dim, cfg.encoder_dims,
                                               cfg.classifier_hidden, NUM_CLASSES,
                                               cfg.dropout_rate, cfg.skip_proj_dim)
        self.decoder = MultiClassNIDSDecoder(cfg.latent_dim, cfg.encoder_dims,
                                              cfg.input_dim, cfg.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def set_mode(self, mode):
        assert mode in ("pretrain", "finetune"); self._mode = mode
        for p in self.decoder.parameters():
            p.requires_grad = (mode == "pretrain")

    def freeze_encoder(self):
        for p in (list(self.encoder_blocks.parameters()) +
                  list(self.bottleneck.parameters()) +
                  list(self.bottleneck_attn.parameters())):
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.parameters(): p.requires_grad = True

    def encode(self, x, add_noise=False):
        if add_noise and self.training:
            x = x + torch.randn_like(x) * self.cfg.noise_std
        h, skips = x, []
        for blk in self.encoder_blocks:
            h = blk(h); skips.append(h)
        z = self.bottleneck(h)
        return self.bottleneck_attn(z), skips

    def forward(self, x):
        z_attn, skips = self.encode(x, add_noise=(self._mode == "pretrain"))
        if self._mode == "pretrain":
            return self.decoder(z_attn), z_attn
        return self.classifier(z_attn, skips), z_attn

    def extract_latent(self, X: np.ndarray, device, batch_size=512) -> np.ndarray:
        self.eval(); self.to(device)
        ds     = DataLoader(_NIDSDS(X, np.zeros(len(X), dtype=np.int64)),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        out = []
        with torch.no_grad():
            for x, _ in ds:
                z, _ = self.encode(x.to(device), add_noise=False)
                out.append(z.cpu().numpy())
        return np.concatenate(out)


class SparseAELoss(nn.Module):
    def __init__(self, sw=1e-3, st=0.05):
        super().__init__(); self.lam = sw; self.rho = st

    def kl(self, z):
        rh  = torch.sigmoid(z).mean(0).clamp(1e-6, 1-1e-6)
        rho = torch.full_like(rh, self.rho).clamp(1e-6, 1-1e-6)
        return (rho*(rho/rh).log() + (1-rho)*((1-rho)/(1-rh)).log()).sum()

    def forward(self, xh, x, z):
        r = F.mse_loss(xh, x); k = self.kl(z)
        return r + self.lam * k, {"recon": r.item(), "kl": k.item()}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_score = None
        self.best_state = None; self.stop = False

    def step(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True
        return self.stop

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            log.info("  ✓ Best weights restored.")


class MultiClassNIDSTrainer:
    def __init__(self, model, cfg, y_train):
        self.model  = model.to(cfg.device)
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.ae_loss  = SparseAELoss(cfg.sparsity_weight, cfg.sparsity_target)
        counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
        self.cls_loss = MultiClassFocalLoss.from_class_counts(
            counts, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing,
        ).to(self.device)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        log.info(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
                 f"device: {cfg.device}")

    def pretrain(self, train_loader, val_loader):
        log.info("=" * 65)
        log.info("PHASE 1 — Sparse Denoising Autoencoder Pre-training")
        log.info("=" * 65)
        self.model.set_mode("pretrain")
        opt     = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sched   = CosineAnnealingWarmRestarts(opt, T_0=self.cfg.t0_epochs,
                                              T_mult=2, eta_min=self.cfg.eta_min)
        stopper = EarlyStopping(self.cfg.early_stop_patience)
        n_b     = len(train_loader)
        for epoch in range(1, self.cfg.pretrain_epochs + 1):
            t0 = time.time()
            tl = self._p1_train(train_loader, opt, sched, epoch, n_b)
            vl = self._p1_val(val_loader)
            log.info(f"  Ep {epoch:03d}/{self.cfg.pretrain_epochs} "
                     f"tr={tl:.5f} val={vl:.5f} "
                     f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.1f}s")
            if stopper.step(-vl, self.model):
                log.info(f"  Early stop @ {epoch}"); break
        stopper.restore(self.model); log.info("Phase 1 complete.\n")

    def _p1_train(self, loader, opt, sched, epoch, n_b):
        self.model.train(); total = n = 0
        for i, (x, _) in enumerate(loader):
            x = x.to(self.device)
            opt.zero_grad(set_to_none=True)
            xh, z   = self.model(x)
            loss, _ = self.ae_loss(xh, x, z)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step(); sched.step(epoch - 1 + i / n_b)
            total += loss.item() * len(x); n += len(x)
        return total / n

    @torch.no_grad()
    def _p1_val(self, loader):
        self.model.eval(); total = n = 0
        for x, _ in loader:
            x = x.to(self.device)
            xh, z   = self.model(x)
            loss, _ = self.ae_loss(xh, x, z)
            total += loss.item() * len(x); n += len(x)
        return total / n

    def finetune(self, train_loader, val_loader, freeze_encoder_epochs=5):
        log.info("=" * 65)
        log.info("PHASE 2 — Multi-class Focal Loss Fine-tuning")
        log.info("=" * 65)
        self.model.set_mode("finetune"); self.model.freeze_encoder()
        opt     = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sched   = CosineAnnealingWarmRestarts(opt, T_0=self.cfg.t0_epochs,
                                              T_mult=2, eta_min=self.cfg.eta_min)
        stopper = EarlyStopping(self.cfg.early_stop_patience); best = 0.0
        n_b     = len(train_loader)
        for epoch in range(1, self.cfg.finetune_epochs + 1):
            if epoch == freeze_encoder_epochs + 1:
                self.model.unfreeze_encoder()
                opt = AdamW(self.model.parameters(),
                            lr=self.cfg.learning_rate * 0.1,
                            weight_decay=self.cfg.weight_decay)
                sched = CosineAnnealingWarmRestarts(opt, T_0=self.cfg.t0_epochs,
                                                    T_mult=2, eta_min=self.cfg.eta_min)
                log.info("  ✓ Encoder unfrozen.")
            t0 = time.time()
            tl = self._p2_train(train_loader, opt, sched, epoch, n_b)
            mt = self._p2_val(val_loader)
            if mt["macro_f1"] > best: best = mt["macro_f1"]
            log.info(f"  Ep {epoch:03d}/{self.cfg.finetune_epochs} "
                     f"loss={tl:.4f} vloss={mt['loss']:.4f} "
                     f"mF1={mt['macro_f1']:.4f} acc={mt['acc']:.4f} "
                     f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.1f}s")
            if stopper.step(mt["macro_f1"], self.model):
                log.info(f"  Early stop @ {epoch}. Best={best:.4f}"); break
        stopper.restore(self.model)
        log.info(f"Phase 2 complete. Best Val MacroF1={best:.4f}\n")

    def _p2_train(self, loader, opt, sched, epoch, n_b):
        self.model.train(); total = n = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad(set_to_none=True)
            logits, _ = self.model(x)
            loss      = self.cls_loss(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step(); sched.step(epoch - 1 + i / n_b)
            total += loss.item() * len(x); n += len(x)
        return total / n

    @torch.no_grad()
    def _p2_val(self, loader):
        self.model.eval(); total = n = 0; pa, la = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            total += self.cls_loss(logits, y).item() * len(x); n += len(x)
            pa.append(logits.argmax(-1).cpu()); la.append(y.cpu())
        p = torch.cat(pa).numpy(); l = torch.cat(la).numpy()
        return {"loss": total/n, "acc": (p==l).mean(),
                "macro_f1": f1_score(l, p, average="macro", zero_division=0)}


def evaluate_multiclass(model, X_test, y_test, cfg, output_dir="multiclass_outputs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    model.set_mode("finetune"); model.eval(); model.to(device)
    loader = DataLoader(_NIDSDS(X_test, y_test), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0)
    pa, pb, la = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(device))
            pb.append(F.softmax(logits, -1).cpu().numpy())
            pa.append(logits.argmax(-1).cpu().numpy())
            la.append(y.numpy())
    y_pred = np.concatenate(pa); y_prob = np.concatenate(pb); y_true = np.concatenate(la)
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    rows   = []
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp
        tn = cm.sum()-tp-fp-fn
        dr   = tp/(tp+fn+1e-9); far = fp/(fp+tn+1e-9)
        prec = tp/(tp+fp+1e-9); f1  = 2*prec*dr/(prec+dr+1e-9)
        rows.append({"Class": name,
                     "TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn),
                     "DR % (Recall)": round(dr*100,3),
                     "FAR % (FPR)":   round(far*100,3),
                     "Precision":     round(prec,4),
                     "F1-Score":      round(f1,4),
                     "Support":       int(tp+fn)})
    mf1 = f1_score(y_true,y_pred,average="macro",    zero_division=0)
    wf1 = f1_score(y_true,y_pred,average="weighted", zero_division=0)
    mp  = precision_score(y_true,y_pred,average="macro", zero_division=0)
    mr  = recall_score(y_true,y_pred,average="macro",    zero_division=0)
    acc = (y_pred==y_true).mean()
    rows.append({"Class":"Macro Avg","TP":"","FP":"","FN":"","TN":"",
                 "DR % (Recall)":round(mr*100,3),"FAR % (FPR)":"",
                 "Precision":round(mp,4),"F1-Score":round(mf1,4),"Support":len(y_true)})
    rows.append({"Class":"Wtd. Avg","TP":"","FP":"","FN":"","TN":"",
                 "DR % (Recall)":"","FAR % (FPR)":"",
                 "Precision":"","F1-Score":round(wf1,4),"Support":len(y_true)})
    df = pd.DataFrame(rows)
    _print_table(df, acc)
    _plot_cm(cm, CLASS_NAMES, output_dir)
    _plot_bars(df, output_dir)
    return {"metrics_df":df,"confusion_matrix":cm,"y_pred":y_pred,
            "y_prob":y_prob,"overall_accuracy":acc,"macro_f1":mf1}


def _print_table(df, acc):
    sep = "─" * 98
    log.info("\n" + "═"*98)
    log.info("  GRANULAR PER-CLASS CYBERSECURITY METRICS — 5-Class NIDS")
    log.info("═"*98)
    log.info(f"  {'Class':<12} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8} "
             f"{'DR % ↑':>10} {'FAR % ↓':>10} {'Precision':>10} {'F1':>10} {'Support':>9}")
    log.info(sep)
    for _, r in df.iterrows():
        cls = str(r["Class"])
        if cls in ("Macro Avg","Wtd. Avg"): log.info(sep)
        dr  = f"{r['DR % (Recall)']:.3f}%" if r["DR % (Recall)"] != "" else ""
        far = f"{r['FAR % (FPR)']:.3f}%"   if r["FAR % (FPR)"]   != "" else ""
        p   = f"{r['Precision']:.4f}"        if r["Precision"]     != "" else ""
        f1  = f"{r['F1-Score']:.4f}"          if r["F1-Score"]      != "" else ""
        log.info(f"  {cls:<12} {str(r['TP']):>8} {str(r['FP']):>8} "
                 f"{str(r['FN']):>8} {str(r['TN']):>8} "
                 f"{dr:>10} {far:>10} {p:>10} {f1:>10} {str(r['Support']):>9}")
    log.info("═"*98)
    log.info(f"  Overall Accuracy: {acc*100:.3f}%")
    log.info("═"*98+"\n")


def _plot_cm(cm, class_names, out_dir):
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True)+1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("5-Class NIDS Confusion Matrix", fontsize=14, fontweight="bold")
    for ax, (data, fmt, title, cmap) in zip(axes, [
        (cm,   "d",    "Raw Counts",                     "Blues"),
        (cm_n, ".3f",  "Row-Normalised (DR on diagonal)", "RdYlGn"),
    ]):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=(1.0 if fmt==".3f" else None))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n = len(class_names)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11, fontweight="bold")
        t = data.max() / 2
        for i in range(n):
            for j in range(n):
                v = data[i,j]
                ax.text(j, i, f"{int(v):,}" if fmt=="d" else f"{v:.3f}",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if v>t else "black")
    plt.tight_layout()
    fig.savefig(str(Path(out_dir)/"confusion_matrix_5class.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(df, out_dir):
    pf = df[~df["Class"].isin(["Macro Avg","Wtd. Avg"])].copy()
    pf["DR"]  = pd.to_numeric(pf["DR % (Recall)"], errors="coerce").fillna(0)
    pf["FAR"] = pd.to_numeric(pf["FAR % (FPR)"],   errors="coerce").fillna(0)
    pf["F1"]  = pd.to_numeric(pf["F1-Score"],       errors="coerce").fillna(0)
    yp = np.arange(len(pf)); h = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(yp-h, pf["DR"],      h, label="DR % ↑",   color="#43A047", alpha=0.88)
    ax.barh(yp,   pf["FAR"],     h, label="FAR % ↓",  color="#E53935", alpha=0.88)
    ax.barh(yp+h, pf["F1"]*100, h, label="F1×100 ↑", color="#1E88E5", alpha=0.85)
    ax.set_yticks(yp); ax.set_yticklabels(pf["Class"], fontsize=11)
    ax.set_xlabel("Score (%)"); ax.set_title("Per-Class DR / FAR / F1", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_facecolor("#FAFAFA"); ax.grid(axis="x", color="#EEE", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(str(Path(out_dir)/"per_class_dr_far_f1.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_multiclass_pipeline(
    train_csv:   str,
    test_csv:    str,
    cfg:         Optional[MultiClassNIDSConfig] = None,
    output_dir:  str   = "multiclass_outputs",
    apply_smote: bool  = True,
    val_split:   float = 0.15,
) -> Tuple[MultiClassNIDSModel, Dict[str, Any]]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if cfg is None: cfg = MultiClassNIDSConfig()

    X_tr_all, y_tr_all, feats = load_hybrid_csv(train_csv)
    X_test,   y_test,   _     = load_hybrid_csv(test_csv, verbose=True)
    cfg.input_dim = X_tr_all.shape[1]

    sss = StratifiedShuffleSplit(1, test_size=val_split, random_state=cfg.seed)
    ti, vi = next(sss.split(X_tr_all, y_tr_all))
    X_train, y_train = X_tr_all[ti], y_tr_all[ti]
    X_val,   y_val   = X_tr_all[vi], y_tr_all[vi]
    log.info(f"Split → train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    if apply_smote:
        log.info("Applying targeted SMOTE (R2L + U2R)...")
        X_train, y_train = apply_targeted_smote(X_train, y_train,
                                                  minority_classes=(3,4), target_ratio=0.15)

    tl = build_multiclass_dataloader(X_train, y_train, cfg.batch_size,
                                      use_class_aware_sampler=True, num_workers=cfg.num_workers)
    vl = build_multiclass_dataloader(X_val,   y_val,   cfg.batch_size,
                                      use_class_aware_sampler=False, shuffle=False,
                                      num_workers=cfg.num_workers)

    model   = MultiClassNIDSModel(cfg)
    trainer = MultiClassNIDSTrainer(model, cfg, y_train)
    trainer.pretrain(tl, vl)
    trainer.finetune(tl, vl, freeze_encoder_epochs=5)

    results = evaluate_multiclass(model, X_test, y_test, cfg, output_dir)

    ckpt = str(Path(output_dir) / "nids_multiclass_model.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "config": cfg.__dict__, "feature_names": feats}, ckpt)
    log.info(f"Checkpoint saved → {ckpt}")
    return model, results
