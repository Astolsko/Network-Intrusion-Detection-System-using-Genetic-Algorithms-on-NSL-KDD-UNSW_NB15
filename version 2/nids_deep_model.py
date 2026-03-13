import os
import time
import copy
import logging
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
    recall_score, roc_auc_score, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



@dataclass
class NIDSConfig:

    input_dim: int = 50
    encoder_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 32
    dropout_rate: float = 0.3
    noise_std: float = 0.05
    sparsity_weight: float = 1e-3
    sparsity_target: float = 0.05
    classifier_hidden: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    pretrain_epochs: int = 30
    finetune_epochs: int = 50
    early_stop_patience: int = 8

    device: str = "cuda" if torch.cuda.is_available() else "cpu" 
    num_workers: int = 4
    pin_memory: bool = True

    seed: int = 42


    pos_class_weight: Optional[float] = None



class NIDSTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        super().__init__()
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.X = torch.from_numpy(X)  # (N, D)
        self.y = (
            torch.from_numpy(y.astype(np.float32))
            if y is not None else None
        )
        self.has_labels = y is not None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.has_labels:
            return self.X[idx], self.y[idx]
        return self.X[idx]


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
        # Compute per-sample weights inversely proportional to class frequency
        class_counts = np.bincount(y.astype(int))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[y.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (cfg.device == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )



class DenseBlock(nn.Module):
   
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # Shortcut projection: maps in_dim → out_dim when they differ
        self.shortcut = (
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.linear1(x)))
        out = self.drop(out)
        out = self.bn2(self.linear2(out))
        return self.act(out + residual)


class FeatureAttentionGate(nn.Module):
    
    def __init__(self, dim: int, reduction: int = 4):
        
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),  # Outputs ∈ (0, 1) — soft feature mask
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        attention_weights = self.gate(h)
        z = h * attention_weights
        return z, attention_weights


class GatedResidualBlock(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.gate_fc = nn.Linear(in_dim, out_dim)
  
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout)
        # Shortcut
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
 
        gate = torch.sigmoid(self.gate_fc(x))
        # Gated output + residual + normalisation
        out = self.layer_norm(gate * h + self.shortcut(x))
        return out


class FeatureExtractor(nn.Module):
  
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.cfg = cfg


        dims = [cfg.input_dim] + cfg.encoder_dims  
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(DenseBlock(dims[i], dims[i + 1], cfg.dropout_rate))
        self.encoder_layers = nn.Sequential(*encoder_layers)


        self.attention_gate = FeatureAttentionGate(dim=cfg.encoder_dims[-1])
        self.bottleneck = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)
        self.bottleneck_bn = nn.BatchNorm1d(cfg.latent_dim)
        self.bottleneck_act = nn.GELU()
        self.noise_std = cfg.noise_std

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        h = self.encoder_layers(x)              # stacked Dense Blocks
        h, attn_weights = self.attention_gate(h) # attention gating
        z = self.bottleneck_act(
            self.bottleneck_bn(self.bottleneck(h))
        )                                         # bottleneck z
        return z, attn_weights


class Decoder(nn.Module):

    def __init__(self, cfg: NIDSConfig):
        super().__init__()

        dims = [cfg.latent_dim] + list(reversed(cfg.encoder_dims))
        # [32, 64, 128, 256] → then project to input_dim

        # Initial expansion from latent space
        self.input_proj = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.GELU(),
        )
        decoder_layers = []
        for i in range(1, len(dims) - 1):
            decoder_layers.append(DenseBlock(dims[i], dims[i + 1], cfg.dropout_rate))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self.output_proj = nn.Linear(dims[-1], cfg.input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        h = self.input_proj(z)
        h = self.decoder_layers(h)
        x_hat = self.output_proj(h)
        return x_hat


class ClassificationHead(nn.Module):

    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        h = cfg.classifier_hidden
        ld = cfg.latent_dim

        self.drop1 = nn.Dropout(p=cfg.dropout_rate)
        self.grn1 = GatedResidualBlock(ld, h, h, dropout=cfg.dropout_rate)
        self.drop2 = nn.Dropout(p=cfg.dropout_rate / 2)
        self.grn2 = GatedResidualBlock(h, h // 2, h // 2, dropout=cfg.dropout_rate / 2)
        self.output = nn.Linear(h // 2, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        h = self.grn1(self.drop1(z))
        h = self.grn2(self.drop2(h))
        return self.output(h).squeeze(-1)  # (B,)



class NIDSModel(nn.Module):
    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        self.decoder = Decoder(cfg)
        self.classifier = ClassificationHead(cfg)
        self._mode = "pretrain"

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_mode(self, mode: str):

        assert mode in ("pretrain", "finetune"), \
            f"mode must be 'pretrain' or 'finetune', got '{mode}'"
        self._mode = mode
        # Freeze/unfreeze decoder to avoid wasting GPU memory in finetune
        for p in self.decoder.parameters():
            p.requires_grad = (mode == "pretrain")

    def freeze_extractor(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        logger.info("FeatureExtractor (Block A) weights frozen.")

    def unfreeze_extractor(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
        logger.info("FeatureExtractor (Block A) weights unfrozen.")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        add_noise = (self._mode == "pretrain")
        z, attn_weights = self.feature_extractor(x, add_noise=add_noise)

        if self._mode == "pretrain":
            x_hat = self.decoder(z)
            return x_hat, z, attn_weights
        else:  # finetune / inference
            logit = self.classifier(z)
            return logit, z, attn_weights


class SparseAutoencoderLoss(nn.Module):
    def __init__(self, sparsity_weight: float = 1e-3, sparsity_target: float = 0.05):
        super().__init__()
        self.lam = sparsity_weight
        self.rho = sparsity_target
        self.mse = nn.MSELoss()

    def kl_divergence(self, z: torch.Tensor) -> torch.Tensor:

        rho_hat = torch.sigmoid(z).mean(dim=0)          # (latent_dim,)
        rho = torch.full_like(rho_hat, self.rho).clamp(1e-6, 1 - 1e-6)
        rho_hat = rho_hat.clamp(1e-6, 1 - 1e-6)
        kl = rho * torch.log(rho / rho_hat) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return kl.sum()

    def forward(
        self,
        x_hat: torch.Tensor,
        x_original: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        recon_loss = self.mse(x_hat, x_original)
        sparsity_loss = self.kl_divergence(z)
        total = recon_loss + self.lam * sparsity_loss
        return total, {
            "recon": recon_loss.item(),
            "sparsity": sparsity_loss.item(),
        }


def build_classification_loss(cfg: NIDSConfig, y_train: np.ndarray) -> nn.BCEWithLogitsLoss:

    if cfg.pos_class_weight is not None:
        pos_weight = cfg.pos_class_weight
    else:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = n_neg / max(n_pos, 1)
        logger.info(f"  Computed pos_weight = {pos_weight:.4f} "
                    f"(N_neg={n_neg}, N_pos={n_pos})")

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
@dataclass
class TrainingHistory:
    pretrain_losses: List[float] = field(default_factory=list)
    pretrain_val_losses: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    best_val_f1: float = 0.0
    best_epoch: int = 0


class EarlyStopping:

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict] = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> bool:

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(" Restored best model state.")


class NIDSTrainer:

    def __init__(self, model: NIDSModel, cfg: NIDSConfig, y_train: np.ndarray):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.history = TrainingHistory()
        self.ae_loss_fn = SparseAutoencoderLoss(
            sparsity_weight=cfg.sparsity_weight,
            sparsity_target=cfg.sparsity_target,
        )
        self.cls_loss_fn = build_classification_loss(cfg, y_train).to(self.device)

        self._set_seed(cfg.seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        logger.info("=" * 60)
        logger.info("PHASE 1 — Autoencoder Pre-training")
        logger.info("=" * 60)

        self.model.set_mode("pretrain")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.pretrain_epochs)
        early_stop = EarlyStopping(patience=self.cfg.early_stop_patience)

        for epoch in range(1, self.cfg.pretrain_epochs + 1):
            t0 = time.time()
            train_loss = self._pretrain_epoch(train_loader, optimizer)
            val_loss = self._pretrain_val_epoch(val_loader)
            scheduler.step()

            self.history.pretrain_losses.append(train_loss)
            self.history.pretrain_val_losses.append(val_loss)

            logger.info(
                f"  Epoch {epoch:03d}/{self.cfg.pretrain_epochs} "
                f"| Train Loss: {train_loss:.5f} "
                f"| Val Loss: {val_loss:.5f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e} "
                f"| {time.time()-t0:.1f}s"
            )

            if early_stop.step(-val_loss, self.model):
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

        early_stop.restore_best(self.model)
        logger.info("Phase 1 complete.\n")

    def _pretrain_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
          
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            x_hat, z, _ = self.model(x)
            loss, _ = self.ae_loss_fn(x_hat, x, z)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _pretrain_val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            x_hat, z, _ = self.model(x)
            loss, _ = self.ae_loss_fn(x_hat, x, z)
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)



    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        freeze_extractor_epochs: int = 0,
    ):

        logger.info("=" * 60)
        logger.info("PHASE 2 — Supervised Fine-tuning")
        logger.info("=" * 60)

        self.model.set_mode("finetune")

        if freeze_extractor_epochs > 0:
            self.model.freeze_extractor()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=3, min_lr=1e-6,
        )
        early_stop = EarlyStopping(patience=self.cfg.early_stop_patience)

        for epoch in range(1, self.cfg.finetune_epochs + 1):
            # Unfreeze after warm-up
            if epoch == freeze_extractor_epochs + 1 and freeze_extractor_epochs > 0:
                self.model.unfreeze_extractor()
                # Re-build optimizer to include all params
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.cfg.learning_rate * 0.1,  # lower LR for encoder
                    weight_decay=self.cfg.weight_decay,
                )

            t0 = time.time()
            train_loss = self._finetune_epoch(train_loader, optimizer)
            metrics = self._finetune_val_epoch(val_loader)

            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(metrics["loss"])
            self.history.val_f1.append(metrics["f1"])
            self.history.val_auc.append(metrics["auc"])

            if metrics["f1"] > self.history.best_val_f1:
                self.history.best_val_f1 = metrics["f1"]
                self.history.best_epoch = epoch

            scheduler.step(metrics["f1"])
            logger.info(
                f"  Epoch {epoch:03d}/{self.cfg.finetune_epochs} "
                f"| TLoss: {train_loss:.4f} "
                f"| VLoss: {metrics['loss']:.4f} "
                f"| F1: {metrics['f1']:.4f} "
                f"| AUC: {metrics['auc']:.4f} "
                f"| Acc: {metrics['acc']:.4f} "
                f"| {time.time()-t0:.1f}s"
            )

            if early_stop.step(metrics["f1"], self.model):
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

        early_stop.restore_best(self.model)
        logger.info(
            f"Phase 2 complete. Best Val F1: {self.history.best_val_f1:.4f} "
            f"at epoch {self.history.best_epoch}.\n"
        )

    def _finetune_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logit, _, _ = self.model(x)
            loss = self.cls_loss_fn(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _finetune_val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logit, _, _ = self.model(x)
            loss = self.cls_loss_fn(logit, y)
            total_loss += loss.item() * len(x)
            all_logits.append(logit.cpu())
            all_labels.append(y.cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy().astype(int)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs > 0.5).astype(int)

        return {
            "loss": total_loss / len(loader.dataset),
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "auc": roc_auc_score(labels, probs),
        }




def extract_latent_features(
    model: NIDSModel,
    X: np.ndarray,
    cfg: NIDSConfig,
    return_attention: bool = False,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    device = torch.device(cfg.device)
    bs = batch_size or cfg.batch_size

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    dataset = NIDSTabularDataset(X) 
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,  # serial for stability during inference
        pin_memory=False,
    )

    model.set_mode("finetune")  # ensures decoder is bypassed
    model.eval()
    model.to(device)

    all_z, all_a = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            # Directly call the encoder 
            z, attn_weights = model.feature_extractor(x, add_noise=False)
            all_z.append(z.cpu().numpy())
            if return_attention:
                all_a.append(attn_weights.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)   # (N, latent_dim)
    A = np.concatenate(all_a, axis=0) if return_attention else None

    logger.info(
        f"Latent features extracted: Z.shape={Z.shape}"
        + (f", A.shape={A.shape}" if A is not None else "")
    )
    return Z, A


def evaluate_model(
    model: NIDSModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: NIDSConfig,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.set_mode("finetune")
    model.eval()
    model.to(device)

    dataset = NIDSTabularDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logit, _, _ = model(x)
            all_logits.append(logit.cpu().numpy())
            all_labels.append(y.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(int)
    probs = 1 / (1 + np.exp(-logits))
    logger.info("─" * 50)
    logger.info("THRESHOLD ANALYSIS (Find best balance for DR vs FAR)")
    logger.info(f"{'Threshold':<10} {'DR (Recall)':<12} {'FAR':<10} {'F1':<10} {'Acc':<10}")
    logger.info("-" * 50)
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        p_tmp = (probs > t).astype(int)
        dr_t = recall_score(labels, p_tmp, zero_division=0)
        # FAR = FP / (FP + TN)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(labels, p_tmp).ravel()
        far_t = fp_t / (fp_t + tn_t + 1e-9)
        f1_t = f1_score(labels, p_tmp, zero_division=0)
        acc_t = accuracy_score(labels, p_tmp)
        logger.info(f"{t:<10.2f} {dr_t:<12.4f} {far_t:<10.4f} {f1_t:<10.4f} {acc_t:<10.4f}")
    logger.info("─" * 50)

    preds = (probs > threshold).astype(int)

    results = {
        "accuracy":         accuracy_score(labels, preds),
        "f1":               f1_score(labels, preds, zero_division=0),
        "precision":        precision_score(labels, preds, zero_division=0),
        "recall":           recall_score(labels, preds, zero_division=0),
        "auc_roc":          roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "threshold":        threshold,
    }

    logger.info("─" * 50)
    logger.info("TEST RESULTS")
    logger.info(f"  Accuracy  : {results['accuracy']:.4f}")
    logger.info(f"  F1 Score  : {results['f1']:.4f}")
    logger.info(f"  Precision : {results['precision']:.4f}")
    logger.info(f"  Recall    : {results['recall']:.4f}")
    logger.info(f"  AUC-ROC   : {results['auc_roc']:.4f}")
    logger.info(f"  ConfMatrix: {results['confusion_matrix']}")
    logger.info("─" * 50)

    return results

def save_model(model: NIDSModel, cfg: NIDSConfig, path: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "architecture": "AG-SDAE-v1",
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved to: {path}")
def load_model(path: str) -> Tuple[NIDSModel, NIDSConfig]:

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint["config"]
    cfg = NIDSConfig(**cfg_dict)
    model = NIDSModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_mode("finetune")
    model.eval()
    logger.info(f"Model loaded from: {path}")
    return model, cfg



class GAChromosome:
    GENE_SPACE = {
        "encoder_dim_0":    [64, 128, 256, 512],
        "encoder_dim_1":    [32, 64, 128, 256],
        "encoder_dim_2":    [16, 32, 64, 128],
        "latent_dim":       [8, 16, 24, 32, 48, 64],
        "dropout_rate":     (0.1, 0.5),      # continuous
        "sparsity_weight":  (1e-5, 1e-1),    # log-uniform
        "sparsity_target":  (0.01, 0.2),     # continuous
        "noise_std":        (0.01, 0.15),    # continuous
        "learning_rate":    (1e-5, 1e-2),    # log-uniform
        "batch_size":       [128, 256, 512, 1024],
    }

    @staticmethod
    def decode(chromosome: List[Any], input_dim: int) -> NIDSConfig:

        assert len(chromosome) == 10, "Chromosome must have 10 genes."
        return NIDSConfig(
            input_dim=input_dim,
            encoder_dims=[int(chromosome[0]), int(chromosome[1]), int(chromosome[2])],
            latent_dim=int(chromosome[3]),
            dropout_rate=float(chromosome[4]),
            sparsity_weight=float(chromosome[5]),
            sparsity_target=float(chromosome[6]),
            noise_std=float(chromosome[7]),
            learning_rate=float(chromosome[8]),
            batch_size=int(chromosome[9]),
        )

    @staticmethod
    def fitness(
        chromosome: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:

        try:
            from xgboost import XGBClassifier

            cfg = GAChromosome.decode(chromosome, input_dim=X_train.shape[1])
            cfg.pretrain_epochs = 10   # reduce for GA speed
            cfg.finetune_epochs = 0    # skip DL fine-tuning — use XGBoost
            cfg.early_stop_patience = 3

            model = NIDSModel(cfg)
            train_loader = build_dataloader(X_train, y_train, cfg, shuffle=True)
            val_loader = build_dataloader(X_val, y_val, cfg, shuffle=False)

            trainer = NIDSTrainer(model, cfg, y_train)
            trainer.pretrain(train_loader, val_loader)

            # Extract latent features
            Z_train, _ = extract_latent_features(model, X_train, cfg)
            Z_val, _ = extract_latent_features(model, X_val, cfg)

            # Lightweight XGBoost as Block B evaluator
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            )
            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_val)
            fitness = f1_score(y_val, y_pred, zero_division=0)
            return float(fitness)

        except Exception as e:
            logger.warning(f"Chromosome evaluation failed: {e}")
            return 0.0
def run_full_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Optional[NIDSConfig] = None,
    save_path: str = "nids_model.pt",
) -> Tuple[NIDSModel, Dict[str, Any], np.ndarray]:
    if cfg is None:
        cfg = NIDSConfig(input_dim=X_train.shape[1])

    logger.info("=" * 60)
    logger.info("AG-SDAE NIDS Pipeline — Full Run")
    logger.info(f"  Dataset shape  : X_train={X_train.shape}, X_test={X_test.shape}")
    logger.info(f"  Feature dim    : {cfg.input_dim}")
    logger.info(f"  Latent dim     : {cfg.latent_dim}")
    logger.info(f"  Encoder dims   : {cfg.encoder_dims}")
    logger.info(f"  Device         : {cfg.device}")
    logger.info("=" * 60)
    train_loader = build_dataloader(X_train, y_train, cfg, shuffle=True, use_weighted_sampler=False)
    val_loader   = build_dataloader(X_val,   y_val,   cfg, shuffle=False)
    test_loader  = build_dataloader(X_test,  y_test,  cfg, shuffle=False)
    model = NIDSModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {n_params:,}")
    trainer = NIDSTrainer(model, cfg, y_train)
    trainer.pretrain(train_loader, val_loader)

    trainer.finetune(train_loader, val_loader, freeze_extractor_epochs=5)

    results = evaluate_model(model, X_test, y_test, cfg)

    Z_test, A_test = extract_latent_features(model, X_test, cfg, return_attention=True)
    logger.info(f"  Latent Z_test shape  : {Z_test.shape}")
    logger.info(f"  Attention A_test shape: {A_test.shape}")


    save_model(model, cfg, save_path)

    return model, results, Z_test


if __name__ == "__main__":

    logger.info("Running smoke test with synthetic data...")

    rng = np.random.default_rng(42)
    N_TRAIN, N_VAL, N_TEST, D = 5_000, 1_000, 2_000, 50  # small scale for demo

    X_tr = rng.standard_normal((N_TRAIN, D)).astype(np.float32)
    y_tr = rng.integers(0, 2, N_TRAIN).astype(np.float32)
    X_va = rng.standard_normal((N_VAL,   D)).astype(np.float32)
    y_va = rng.integers(0, 2, N_VAL).astype(np.float32)
    X_te = rng.standard_normal((N_TEST,  D)).astype(np.float32)
    y_te = rng.integers(0, 2, N_TEST).astype(np.float32)

    cfg = NIDSConfig(
        input_dim=D,
        encoder_dims=[128, 64, 32],
        latent_dim=16,
        pretrain_epochs=5,
        finetune_epochs=5,
        early_stop_patience=3,
        batch_size=256,
        num_workers=0,
        pin_memory=False,
    )

    model, results, Z_test = run_full_pipeline(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        cfg=cfg,
        save_path="nids_smoke_test.pt",
    )

    logger.info(f"Smoke test complete. F1={results['f1']:.4f}, AUC={results['auc_roc']:.4f}")
    logger.info(f"Latent feature shape for GA: {Z_test.shape}")
