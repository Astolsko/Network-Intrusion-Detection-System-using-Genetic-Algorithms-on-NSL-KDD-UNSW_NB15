import os, time, copy, logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
@dataclass
class NIDSConfig:
    input_dim:           int        = 248
    encoder_dims:        List[int]  = field(default_factory=lambda: [256, 128, 64])
    latent_dim:          int        = 32
    dropout_rate:        float      = 0.3
    noise_std:           float      = 0.05
    sparsity_weight:     float      = 1e-3
    sparsity_target:     float      = 0.05
    classifier_hidden:   int        = 64
    learning_rate:       float      = 1e-3
    weight_decay:        float      = 1e-4
    batch_size:          int        = 512
    pretrain_epochs:     int        = 30
    finetune_epochs:     int        = 50
    early_stop_patience: int        = 8
    device:              str        = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers:         int        = 0
    pin_memory:          bool       = True
    seed:                int        = 42
    pos_class_weight:    Optional[float] = None


class NIDSTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)) if y is not None else None
        self.has_labels = y is not None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.has_labels else self.X[idx]


def build_dataloader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    cfg: NIDSConfig,
    shuffle: bool = True,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    dataset = NIDSTabularDataset(X, y)
    sampler = None
    if use_weighted_sampler and y is not None:
        counts  = np.bincount(y.astype(int))
        weights = 1.0 / (counts + 1e-6)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights[y.astype(int)].astype(np.float32)),
            num_samples=len(dataset), replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and cfg.device == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )


class DenseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(in_dim,  out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.fc2  = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.proj = (
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        res = self.proj(x)
        h   = self.act(self.bn1(self.fc1(x)))
        h   = self.drop(h)
        h   = self.bn2(self.fc2(h))
        return self.act(h + res)


class FeatureAttentionGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid(),
        )

    def forward(self, h):
        w = self.gate(h)
        return h * w, w


class GatedResidualBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1      = nn.Linear(in_dim,  hidden)
        self.fc2      = nn.Linear(hidden,  out_dim)
        self.gate_fc  = nn.Linear(in_dim,  out_dim)
        self.ln       = nn.LayerNorm(out_dim)
        self.drop     = nn.Dropout(p=dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h    = self.drop(F.elu(self.fc1(x)))
        h    = self.fc2(h)
        gate = torch.sigmoid(self.gate_fc(x))
        return self.ln(gate * h + self.shortcut(x))


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.noise_std = cfg.noise_std
        dims = [cfg.input_dim] + cfg.encoder_dims
        self.layers   = nn.Sequential(*[
            DenseBlock(dims[i], dims[i+1], cfg.dropout_rate)
            for i in range(len(dims)-1)
        ])
        self.attn_gate = FeatureAttentionGate(cfg.encoder_dims[-1])
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim),
            nn.BatchNorm1d(cfg.latent_dim),
            nn.GELU(),
        )

    def forward(self, x, add_noise=False):
        if add_noise and self.training:
            x = x + torch.randn_like(x) * self.noise_std
        h, attn = self.attn_gate(self.layers(x))
        return self.bottleneck(h), attn


class Decoder(nn.Module):
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        dims = [cfg.latent_dim] + list(reversed(cfg.encoder_dims))
        self.proj = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.BatchNorm1d(dims[1]), nn.GELU()
        )
        self.layers  = nn.Sequential(*[
            DenseBlock(dims[i], dims[i+1], cfg.dropout_rate)
            for i in range(1, len(dims)-1)
        ])
        self.out = nn.Linear(dims[-1], cfg.input_dim)

    def forward(self, z):
        return self.out(self.layers(self.proj(z)))


class ClassificationHead(nn.Module):
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        h  = cfg.classifier_hidden
        ld = cfg.latent_dim
        self.drop1  = nn.Dropout(cfg.dropout_rate)
        self.grn1   = GatedResidualBlock(ld, h,    h,    cfg.dropout_rate)
        self.drop2  = nn.Dropout(cfg.dropout_rate / 2)
        self.grn2   = GatedResidualBlock(h,  h//2, h//2, cfg.dropout_rate / 2)
        self.output = nn.Linear(h // 2, 1)

    def forward(self, z):
        return self.output(self.grn2(self.drop2(self.grn1(self.drop1(z))))).squeeze(-1)


class NIDSModel(nn.Module):
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.cfg               = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        self.decoder           = Decoder(cfg)
        self.classifier        = ClassificationHead(cfg)
        self._mode             = "pretrain"
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def set_mode(self, mode: str):
        assert mode in ("pretrain", "finetune")
        self._mode = mode
        for p in self.decoder.parameters():
            p.requires_grad = (mode == "pretrain")

    def freeze_extractor(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_extractor(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def forward(self, x):
        z, attn = self.feature_extractor(x, add_noise=(self._mode == "pretrain"))
        if self._mode == "pretrain":
            return self.decoder(z), z, attn
        return self.classifier(z), z, attn


class SparseAutoencoderLoss(nn.Module):
    def __init__(self, sparsity_weight=1e-3, sparsity_target=0.05):
        super().__init__()
        self.lam = sparsity_weight
        self.rho = sparsity_target

    def kl(self, z):
        rho_hat = torch.sigmoid(z).mean(0).clamp(1e-6, 1-1e-6)
        rho     = torch.full_like(rho_hat, self.rho).clamp(1e-6, 1-1e-6)
        return (rho*(rho/rho_hat).log() + (1-rho)*((1-rho)/(1-rho_hat)).log()).sum()

    def forward(self, x_hat, x, z):
        r = F.mse_loss(x_hat, x)
        k = self.kl(z)
        return r + self.lam * k, {"recon": r.item(), "kl": k.item()}


def build_classification_loss(cfg: NIDSConfig, y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
    if cfg.pos_class_weight is not None:
        pw = cfg.pos_class_weight
    else:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pw    = n_neg / max(n_pos, 1)
        logger.info(f"  pos_weight={pw:.4f}  (N_neg={n_neg}, N_pos={n_pos})")
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32)
    )


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            logger.info("  ✓ Best weights restored.")


class NIDSTrainer:
    def __init__(self, model: NIDSModel, cfg: NIDSConfig, y_train: np.ndarray):
        self.model   = model.to(cfg.device)
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)
        self.ae_loss = SparseAutoencoderLoss(cfg.sparsity_weight, cfg.sparsity_target)
        self.cls_loss = build_classification_loss(cfg, y_train).to(self.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    def pretrain(self, train_loader, val_loader):
        logger.info("=" * 60)
        logger.info("PHASE 1 — Autoencoder Pre-training")
        logger.info("=" * 60)
        self.model.set_mode("pretrain")
        opt     = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sched   = CosineAnnealingLR(opt, T_max=self.cfg.pretrain_epochs)
        stopper = EarlyStopping(self.cfg.early_stop_patience)

        for epoch in range(1, self.cfg.pretrain_epochs + 1):
            t0  = time.time()
            trl = self._pretrain_epoch(train_loader, opt)
            vl  = self._pretrain_val(val_loader)
            sched.step()
            logger.info(f"  Ep {epoch:03d}/{self.cfg.pretrain_epochs} "
                        f"| tr={trl:.5f} val={vl:.5f} "
                        f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.1f}s")
            if stopper.step(-vl, self.model):
                logger.info(f"  Early stop @ epoch {epoch}"); break
        stopper.restore_best(self.model)
        logger.info("Phase 1 complete.\n")

    def _pretrain_epoch(self, loader, opt):
        self.model.train()
        total = n = 0
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            opt.zero_grad(set_to_none=True)
            xh, z, _ = self.model(x)
            loss, _  = self.ae_loss(xh, x, z)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x); n += len(x)
        return total / n

    @torch.no_grad()
    def _pretrain_val(self, loader):
        self.model.eval()
        total = n = 0
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            xh, z, _ = self.model(x)
            loss, _  = self.ae_loss(xh, x, z)
            total += loss.item() * len(x); n += len(x)
        return total / n

    def finetune(self, train_loader, val_loader, freeze_extractor_epochs=5):
        logger.info("=" * 60)
        logger.info("PHASE 2 — Supervised Fine-tuning")
        logger.info("=" * 60)
        self.model.set_mode("finetune")
        self.model.freeze_extractor()
        opt     = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sched   = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3,
                                    min_lr=1e-6)
        stopper = EarlyStopping(self.cfg.early_stop_patience)
        best_f1 = 0.0

        for epoch in range(1, self.cfg.finetune_epochs + 1):
            if epoch == freeze_extractor_epochs + 1:
                self.model.unfreeze_extractor()
                opt = AdamW(self.model.parameters(),
                            lr=self.cfg.learning_rate * 0.1,
                            weight_decay=self.cfg.weight_decay)
                logger.info("  ✓ Encoder unfrozen.")

            t0  = time.time()
            trl = self._finetune_epoch(train_loader, opt)
            met = self._finetune_val(val_loader)
            sched.step(met["f1"])
            if met["f1"] > best_f1:
                best_f1 = met["f1"]
            logger.info(f"  Ep {epoch:03d}/{self.cfg.finetune_epochs} "
                        f"| loss={trl:.4f} vloss={met['loss']:.4f} "
                        f"f1={met['f1']:.4f} auc={met['auc']:.4f} "
                        f"acc={met['acc']:.4f} {time.time()-t0:.1f}s")
            if stopper.step(met["f1"], self.model):
                logger.info(f"  Early stop @ epoch {epoch}. Best F1={best_f1:.4f}"); break
        stopper.restore_best(self.model)
        logger.info(f"Phase 2 complete. Best F1={best_f1:.4f}\n")

    def _finetune_epoch(self, loader, opt):
        self.model.train()
        total = n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad(set_to_none=True)
            logit, _, _ = self.model(x)
            loss        = self.cls_loss(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x); n += len(x)
        return total / n

    @torch.no_grad()
    def _finetune_val(self, loader):
        self.model.eval()
        total = n = 0
        logits_all, labels_all = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logit, _, _ = self.model(x)
            total += self.cls_loss(logit, y).item() * len(x); n += len(x)
            logits_all.append(logit.cpu()); labels_all.append(y.cpu())
        lo = torch.cat(logits_all).numpy()
        la = torch.cat(labels_all).numpy().astype(int)
        pr = 1 / (1 + np.exp(-lo))
        pd_ = (pr > 0.5).astype(int)
        return {
            "loss": total / n,
            "acc":  accuracy_score(la, pd_),
            "f1":   f1_score(la, pd_, zero_division=0),
            "auc":  roc_auc_score(la, pr) if len(np.unique(la)) > 1 else 0.5,
        }


def extract_latent_features(
    model:            NIDSModel,
    X:                np.ndarray,
    cfg:              NIDSConfig,
    return_attention: bool = False,
    batch_size:       Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    device = torch.device(cfg.device)
    bs     = batch_size or cfg.batch_size
    X      = X.astype(np.float32)
    loader = DataLoader(NIDSTabularDataset(X), batch_size=bs,
                        shuffle=False, num_workers=0)
    model.set_mode("finetune"); model.eval(); model.to(device)
    all_z, all_a = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            z, attn = model.feature_extractor(x, add_noise=False)
            all_z.append(z.cpu().numpy())
            if return_attention:
                all_a.append(attn.cpu().numpy())
    Z = np.concatenate(all_z)
    A = np.concatenate(all_a) if return_attention else None
    logger.info(f"Latent features extracted: Z={Z.shape}"
                + (f" A={A.shape}" if A is not None else ""))
    return Z, A


def evaluate_model(
    model:     NIDSModel,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    cfg:       NIDSConfig,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.set_mode("finetune"); model.eval(); model.to(device)
    loader = DataLoader(NIDSTabularDataset(X_test, y_test),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logit, _, _ = model(x.to(device))
            logits_all.append(logit.cpu().numpy())
            labels_all.append(y.numpy())
    lo = np.concatenate(logits_all)
    la = np.concatenate(labels_all).astype(int)
    pr = 1 / (1 + np.exp(-lo))
    pd_ = (pr > threshold).astype(int)
    res = {
        "accuracy":         accuracy_score(la, pd_),
        "f1":               f1_score(la, pd_, zero_division=0),
        "precision":        precision_score(la, pd_, zero_division=0),
        "recall":           recall_score(la, pd_, zero_division=0),
        "auc_roc":          roc_auc_score(la, pr) if len(np.unique(la)) > 1 else 0.5,
        "confusion_matrix": confusion_matrix(la, pd_).tolist(),
        "threshold":        threshold,
    }
    logger.info("─" * 50)
    logger.info("BINARY TEST RESULTS")
    for k, v in res.items():
        if k != "confusion_matrix":
            logger.info(f"  {k:<18}: {v:.4f}" if isinstance(v, float) else f"  {k:<18}: {v}")
    logger.info(f"  confusion_matrix  : {res['confusion_matrix']}")
    logger.info("─" * 50)
    return res


def save_model(model: NIDSModel, cfg: NIDSConfig, path: str):
    torch.save({"model_state_dict": model.state_dict(),
                "config": cfg.__dict__, "architecture": "AG-SDAE-v1"}, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str) -> Tuple[NIDSModel, NIDSConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = NIDSConfig(**ckpt["config"])
    m    = NIDSModel(cfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.set_mode("finetune"); m.eval()
    logger.info(f"Model loaded ← {path}")
    return m, cfg


class GAChromosome:
    GENE_SPACE = {
        "encoder_dim_0":   [64, 128, 256, 512],
        "encoder_dim_1":   [32, 64,  128, 256],
        "encoder_dim_2":   [16, 32,  64,  128],
        "latent_dim":      [8,  16,  32,  48, 64],
        "dropout_rate":    (0.1, 0.5),
        "sparsity_weight": (1e-5, 1e-1),
        "sparsity_target": (0.01, 0.2),
        "noise_std":       (0.01, 0.15),
        "learning_rate":   (1e-5, 1e-2),
        "batch_size":      [128, 256, 512, 1024],
    }

    @staticmethod
    def decode(chromosome: List[Any], input_dim: int) -> NIDSConfig:
        assert len(chromosome) == 10
        return NIDSConfig(
            input_dim    = input_dim,
            encoder_dims = [int(chromosome[0]), int(chromosome[1]), int(chromosome[2])],
            latent_dim   = int(chromosome[3]),
            dropout_rate = float(chromosome[4]),
            sparsity_weight = float(chromosome[5]),
            sparsity_target = float(chromosome[6]),
            noise_std    = float(chromosome[7]),
            learning_rate = float(chromosome[8]),
            batch_size   = int(chromosome[9]),
        )

    @staticmethod
    def fitness(chromosome, X_train, y_train, X_val, y_val) -> float:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            cfg = GAChromosome.decode(chromosome, X_train.shape[1])
            cfg.pretrain_epochs = 10; cfg.finetune_epochs = 0
            model = NIDSModel(cfg)
            tl = build_dataloader(X_train, y_train, cfg, shuffle=True)
            vl = build_dataloader(X_val,   y_val,   cfg, shuffle=False)
            NIDSTrainer(model, cfg, y_train).pretrain(tl, vl)
            Z_tr, _ = extract_latent_features(model, X_train, cfg)
            Z_va, _ = extract_latent_features(model, X_val,   cfg)
            clf = GradientBoostingClassifier(n_estimators=50)
            clf.fit(Z_tr, y_train)
            return float(f1_score(y_val, clf.predict(Z_va), zero_division=0))
        except Exception as e:
            logger.warning(f"GA fitness failed: {e}")
            return 0.0


def run_full_pipeline(
    X_train, y_train, X_val, y_val, X_test, y_test,
    cfg=None, save_path="nids_binary_model.pt",
):
    if cfg is None:
        cfg = NIDSConfig(input_dim=X_train.shape[1])
    logger.info("=" * 60)
    logger.info(f"Binary AG-SDAE | train={X_train.shape} test={X_test.shape} device={cfg.device}")
    logger.info("=" * 60)
    tl = build_dataloader(X_train, y_train, cfg, shuffle=True, use_weighted_sampler=True)
    vl = build_dataloader(X_val,   y_val,   cfg, shuffle=False)
    model   = NIDSModel(cfg)
    trainer = NIDSTrainer(model, cfg, y_train)
    trainer.pretrain(tl, vl)
    trainer.finetune(tl, vl, freeze_extractor_epochs=5)
    results = evaluate_model(model, X_test, y_test, cfg)
    Z_test, A_test = extract_latent_features(model, X_test, cfg, return_attention=True)
    save_model(model, cfg, save_path)
    return model, results, Z_test
