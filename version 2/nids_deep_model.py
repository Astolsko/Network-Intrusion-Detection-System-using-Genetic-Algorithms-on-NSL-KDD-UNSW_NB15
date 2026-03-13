"""
=============================================================================
GA-NIDS: Genetic Algorithm–Based Network Intrusion Detection System
Deep Learning Core — Attention-Gated Sparse Denoising Autoencoder (AG-SDAE)
=============================================================================

ARCHITECTURE OVERVIEW
─────────────────────
This module implements a two-block deep learning pipeline for binary
Network Intrusion Detection on pre-processed tabular data (NSL-KDD,
UNSW-NB15, or their Hybrid variant).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Block A — Attention-Gated Sparse Denoising AutoEncoder         │
  │  (AG-SDAE Feature Extractor)                                    │
  │                                                                 │
  │   Input x ──► GaussianNoise ──► Encoder Layers (Linear +       │
  │   BatchNorm + GELU + Dropout) ──► Feature Attention Gate ──►   │
  │   Bottleneck z  (latent representation)                         │
  │                   │                                             │
  │                   └──► Decoder (pre-training only)              │
  ├─────────────────────────────────────────────────────────────────┤
  │  Block B — Gated Residual MLP Classifier                        │
  │                                                                 │
  │   z ──► Gated Linear Units ──► Residual Block ──► Sigmoid ──►  │
  │   ŷ ∈ {0 = Normal, 1 = Attack}                                  │
  └─────────────────────────────────────────────────────────────────┘

TRAINING STRATEGY
─────────────────
  Phase 1 (Unsupervised Pre-training):
    Train the full autoencoder (Encoder + Decoder) on the training set using
    a composite loss: MSE reconstruction + KL-divergence sparsity penalty.
    This forces the encoder to learn a sparse, information-rich latent space
    without any label bias.

  Phase 2 (Supervised Fine-tuning):
    Freeze/unfreeze the encoder, attach Block B, and train end-to-end using
    Binary Cross-Entropy. The latent representation is now shaped both by
    reconstruction fidelity and discriminative supervision.

GA INTERFACE
────────────
  The `extract_latent_features()` utility exposes the bottleneck vector z
  as a clean NumPy array. This array is the natural chromosome substrate for
  a Genetic Algorithm that may:
    (a) Evolve hyperparameters (encoder dims, sparsity, dropout, LR) where
        each candidate is evaluated by training a lightweight classifier on z.
    (b) Perform GA-based feature selection directly on z dimensions.
    (c) Feed z into an XGBoost/LightGBM booster as Block B for hybrid evaluation.

DATASET COMPATIBILITY (from preprocessing PDF)
───────────────────────────────────────────────
  - NSL-KDD    : ~55–65 features after one-hot, ~40–50 after selection
  - UNSW-NB15  : ~55–65 features after one-hot, ~40–50 after selection
  - Hybrid     : ~55–65 features after one-hot, ~40–50 after selection
  - Training   : ~300 k samples → ~350 k after SMOTE
  - Testing    : ~100 k samples
  - Labels     : Binary  {0 = Normal, 1 = Attack}

REFERENCES
──────────
  [1] Al-Qatf et al. (2018) — Sparse AE + SVM on NSL-KDD, F1 = 85.28%
  [2] Song et al. (2021)    — Systematic study of AE latent size for NIDS
  [3] Springer (2024)       — Stacked Sparse AE + LightGBM; 99.24% on NSL-KDD
  [4] Springer (2024)       — ADSAE-CNN: Attention + Deep Sparse AE for NIDS
  [5] Khudhur (2026)        — GA-optimised DNN for NIDS; 98.54% accuracy
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import time
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

# ─── Numerical / Data ────────────────────────────────────────────────────────
import numpy as np

# ─── PyTorch Core ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ─── Optimisation & Scheduling ───────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─── Metrics ─────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NIDSConfig:
    """
    Central configuration object for the AG-SDAE pipeline.

    All hyperparameters exposed here are candidates for Genetic Algorithm
    chromosome encoding. A GA chromosome could be a flat vector:

        [encoder_dims[0], encoder_dims[1], encoder_dims[2],
         latent_dim, dropout_rate, sparsity_weight,
         sparsity_target, noise_std, learning_rate, batch_size]

    where each gene is drawn from a bounded search space defined below.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    input_dim: int = 50
    """Feature dimension after preprocessing & feature selection.
    Expected range: 40–65 (see preprocessing PDF §3.5–3.7)."""

    # ── Block A — Encoder Layer Dimensions ────────────────────────────────
    encoder_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    """Hidden layer widths for the encoder (and mirrored decoder).
    GA search space: each dim ∈ {32, 64, 128, 256, 512}."""

    latent_dim: int = 32
    """Bottleneck (latent space) dimension.
    GA search space: {8, 16, 32, 48, 64}.
    Research finding [2]: latent_dim has the strongest impact on NIDS F1."""

    # ── Block A — Regularisation ──────────────────────────────────────────
    dropout_rate: float = 0.3
    """Dropout probability applied after each encoder/decoder layer.
    GA search space: [0.1, 0.5]."""

    noise_std: float = 0.05
    """Standard deviation of Gaussian noise injected to the input during
    pre-training. Implements the Denoising AE principle.
    GA search space: [0.01, 0.15]."""

    sparsity_weight: float = 1e-3
    """λ: weight of KL-divergence sparsity penalty on bottleneck activations.
    GA search space: [1e-5, 1e-1] (log-uniform)."""

    sparsity_target: float = 0.05
    """ρ: target average activation of bottleneck neurons (sparsity level).
    GA search space: [0.01, 0.2]."""

    # ── Block B — Classifier ─────────────────────────────────────────────
    classifier_hidden: int = 64
    """Hidden dimension inside the Gated Residual Block of Block B.
    GA search space: {32, 64, 128}."""

    # ── Training ─────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    """Initial learning rate for AdamW.
    GA search space: [1e-5, 1e-2] (log-uniform)."""

    weight_decay: float = 1e-4
    """L2 regularisation coefficient for AdamW."""

    batch_size: int = 512
    """Mini-batch size.
    GA search space: {128, 256, 512, 1024}."""

    pretrain_epochs: int = 30
    """Maximum epochs for Phase 1 (autoencoder pre-training)."""

    finetune_epochs: int = 50
    """Maximum epochs for Phase 2 (supervised fine-tuning)."""

    early_stop_patience: int = 8
    """Epochs without validation improvement before early stopping fires."""

    # ── Hardware ─────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Target compute device."""

    num_workers: int = 4
    """DataLoader worker threads."""

    pin_memory: bool = True
    """Pin memory for faster GPU transfers."""

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for full reproducibility."""

    pos_class_weight: Optional[float] = None
    """If set, overrides the auto-computed positive class weight used in
    BCEWithLogitsLoss to handle class imbalance residual after SMOTE."""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET & DATALOADER
# ═════════════════════════════════════════════════════════════════════════════

class NIDSTabularDataset(Dataset):
    """
    PyTorch Dataset for pre-processed tabular NIDS data.

    Accepts NumPy arrays directly, avoiding any file I/O overhead during
    training. Converts to float32 tensors on access, which is the expected
    dtype for all PyTorch layers below.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Pre-processed feature matrix (Z-score normalised, one-hot encoded).
    y : np.ndarray, shape (N,), optional
        Binary labels {0, 1}. If None, the dataset operates in inference-only
        mode (e.g., for latent feature extraction on unlabelled data).

    Example
    -------
    >>> X_train = np.load("X_train_hybrid.npy")
    >>> y_train = np.load("y_train_hybrid.npy")
    >>> dataset = NIDSTabularDataset(X_train, y_train)
    >>> print(len(dataset), dataset[0][0].shape)
    350000 torch.Size([50])
    """

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
    """
    Construct an optimised DataLoader for large-scale tabular NIDS data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray or None
        Labels. Required if use_weighted_sampler=True.
    cfg : NIDSConfig
        Hyperparameter configuration.
    shuffle : bool
        Standard shuffling. Ignored when use_weighted_sampler=True.
    use_weighted_sampler : bool
        When True, uses a WeightedRandomSampler to handle residual class
        imbalance (relevant for the training set after SMOTE is already
        applied but minor imbalance may still exist).

    Returns
    -------
    DataLoader
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILDING BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class DenseBlock(nn.Module):
    """
    A single fully-connected residual block used in the encoder / decoder.

    Architecture:
        Input → Linear → BatchNorm1d → GELU → Dropout → Linear → BatchNorm1d
              ↘─────────────── Projection (if dims differ) ──────────────────↗
                                         ⊕
                                       GELU

    GELU (Gaussian Error Linear Unit) is preferred over ReLU because:
      - It is smooth and differentiable everywhere (better gradient flow).
      - It has been shown to outperform ReLU in self-attention architectures
        and is standard in modern transformers / tabular networks.
      - Formula: GELU(x) = x · Φ(x), where Φ is the cumulative normal CDF.
    """

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
    """
    Soft Feature Attention Gate applied to the pre-bottleneck representation.

    Motivation
    ──────────
    Standard autoencoders treat all hidden activations equally. In NIDS data,
    many features (especially one-hot encoded categoricals) carry redundant
    or noisy signal. The Feature Attention Gate learns a per-neuron importance
    weight vector a ∈ (0, 1)^D via a lightweight 2-layer gating network:

        a = σ(W₂ · GELU(W₁ · h))     # gating probabilities
        z = h ⊙ a                      # element-wise masking

    This is architecturally analogous to the squeeze-and-excitation (SE)
    mechanism (Hu et al., 2018) adapted for tabular 1-D feature vectors.
    The bottleneck then receives a contextually filtered, attention-weighted
    representation rather than raw activations — improving sparsity and
    discriminative power simultaneously.
    """

    def __init__(self, dim: int, reduction: int = 4):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the incoming activation vector.
        reduction : int
            Bottleneck reduction factor inside the gate (default: 4×).
        """
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),  # Outputs ∈ (0, 1) — soft feature mask
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z : torch.Tensor
            Attention-weighted hidden representation.
        attention_weights : torch.Tensor
            The learned per-feature importance scores (useful for
            interpretability and GA-based feature selection).
        """
        attention_weights = self.gate(h)
        z = h * attention_weights
        return z, attention_weights


class GatedResidualBlock(nn.Module):
    """
    Gated Residual Block (GRN) — used as the core building block of Block B.

    Inspired by the Variable Selection Network in TFT (Lim et al., 2021),
    adapted for non-temporal binary classification:

        ELU(W₁·x + b₁) ──► Dropout
        W₂·[above]        → gate projection
        σ(W₃·x + b₃)     → gating signal
        GRN(x) = LayerNorm( gate ⊙ gated_projection + shortcut )

    The gating mechanism allows the network to suppress irrelevant latent
    dimensions adaptively, which is especially useful when the latent space z
    is high-dimensional relative to the separability of the problem.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        # Primary pathway
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Gating pathway
        self.gate_fc = nn.Linear(in_dim, out_dim)
        # Normalisation
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout)
        # Shortcut
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primary
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        # Gate
        gate = torch.sigmoid(self.gate_fc(x))
        # Gated output + residual + normalisation
        out = self.layer_norm(gate * h + self.shortcut(x))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BLOCK A: FEATURE EXTRACTOR (AG-SDAE Encoder + Decoder)
# ═════════════════════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    """
    Block A — Attention-Gated Sparse Denoising AutoEncoder Encoder.

    This module is the heart of the feature extraction pipeline. It produces
    a low-dimensional latent vector z that:
      (a) Minimises reconstruction loss  →  preserves data structure
      (b) Is regularised by KL-sparsity  →  encourages disentanglement
      (c) Is filtered by attention gate  →  emphasises informative dimensions

    The combination of denoising + sparsity + attention has been validated
    on both NSL-KDD and UNSW-NB15 in the ADSAE-CNN paper (Springer, 2024),
    achieving >99% accuracy on both datasets.

    Parameters
    ----------
    cfg : NIDSConfig
        Pipeline configuration.

    Architecture (for default cfg.encoder_dims = [256, 128, 64], latent=32):

        Input(50) → DenseBlock(50→256) → DenseBlock(256→128)
                  → DenseBlock(128→64) → FeatureAttentionGate(64)
                  → Linear(64→32)      → Bottleneck z ∈ ℝ³²
    """

    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.cfg = cfg

        # ── Build stacked encoder layers ──────────────────────────────────
        dims = [cfg.input_dim] + cfg.encoder_dims  # e.g. [50, 256, 128, 64]
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(DenseBlock(dims[i], dims[i + 1], cfg.dropout_rate))
        self.encoder_layers = nn.Sequential(*encoder_layers)

        # ── Feature Attention Gate (applied to pre-bottleneck repr.) ──────
        self.attention_gate = FeatureAttentionGate(dim=cfg.encoder_dims[-1])

        # ── Bottleneck projection ─────────────────────────────────────────
        self.bottleneck = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)
        self.bottleneck_bn = nn.BatchNorm1d(cfg.latent_dim)
        self.bottleneck_act = nn.GELU()

        # ── Gaussian noise injector (active during pre-training only) ─────
        self.noise_std = cfg.noise_std

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)
            Pre-processed input batch.
        add_noise : bool
            When True, injects Gaussian noise (used during pre-training).

        Returns
        -------
        z : torch.Tensor, shape (B, latent_dim)
            Latent representation — the output passed to Block B.
        attn_weights : torch.Tensor, shape (B, encoder_dims[-1])
            Per-feature attention scores from the gate layer.
        """
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
    """
    Decoder for Phase 1 (autoencoder pre-training).

    Mirrors the encoder's architecture in reverse. Only used during
    pre-training; detached from the model in Phase 2.

    Parameters
    ----------
    cfg : NIDSConfig

    Architecture (mirroring encoder):
        z(32) → Linear(32→64) → DenseBlock(64→128)
              → DenseBlock(128→256) → Linear(256→50) = x̂
    """

    def __init__(self, cfg: NIDSConfig):
        super().__init__()

        # Reverse the encoder dimension sequence
        dims = [cfg.latent_dim] + list(reversed(cfg.encoder_dims))
        # e.g. [32, 64, 128, 256] → then project to input_dim

        # Initial expansion from latent space
        self.input_proj = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.GELU(),
        )

        # Stacked decoder blocks
        decoder_layers = []
        for i in range(1, len(dims) - 1):
            decoder_layers.append(DenseBlock(dims[i], dims[i + 1], cfg.dropout_rate))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        # Final reconstruction projection (no activation — output is raw)
        self.output_proj = nn.Linear(dims[-1], cfg.input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original input from latent z.

        Returns
        -------
        x_hat : torch.Tensor, shape (B, input_dim)
            Reconstructed feature vector.
        """
        h = self.input_proj(z)
        h = self.decoder_layers(h)
        x_hat = self.output_proj(h)
        return x_hat


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BLOCK B: CLASSIFICATION HEAD
# ═════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    Block B — Gated Residual MLP Binary Classifier.

    Takes the latent representation z from Block A and produces a binary
    logit for normal (0) / attack (1) classification.

    Architecture:
        z(latent_dim) → Dropout → GatedResidualBlock(latent→hidden→hidden)
                      → Dropout → GatedResidualBlock(hidden→hidden/2→hidden/2)
                      → Linear(hidden/2 → 1)
                      → Logit (no sigmoid — BCEWithLogitsLoss is used)

    The Gated Residual Block is chosen over a plain MLP because:
      - The gating mechanism allows learned suppression of irrelevant
        latent dimensions (soft feature selection at inference time).
      - LayerNorm inside GRN provides stable gradients when the latent
        space has varying activation magnitudes across training samples.
      - Empirically superior on tabular data compared to vanilla FFNs
        (TFT paper, Lim et al., 2021; validated on NIDS in [4]).

    Parameters
    ----------
    cfg : NIDSConfig
    """

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
        """
        Parameters
        ----------
        z : torch.Tensor, shape (B, latent_dim)

        Returns
        -------
        logit : torch.Tensor, shape (B,)
            Raw logit (apply sigmoid for probability, threshold at 0.5).
        """
        h = self.grn1(self.drop1(z))
        h = self.grn2(self.drop2(h))
        return self.output(h).squeeze(-1)  # (B,)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FULL NIDS MODEL (WRAPPER)
# ═════════════════════════════════════════════════════════════════════════════

class NIDSModel(nn.Module):
    """
    Complete GA-NIDS Model: Block A (AG-SDAE) + Block B (Gated Residual Classifier).

    This wrapper class unifies both blocks and exposes separate forward modes
    for pre-training (autoencoder) and fine-tuning (classification).

    Usage
    ─────
    Pre-training (Phase 1):
        model.set_mode("pretrain")
        x_hat, z, attn = model(x_batch)
        loss = reconstruction_loss(x_hat, x_original) + sparsity_loss(z)

    Fine-tuning (Phase 2):
        model.set_mode("finetune")
        logit, z, attn = model(x_batch)
        loss = bce_loss(logit, labels)

    Inference:
        model.set_mode("finetune")
        model.eval()
        with torch.no_grad():
            logit, z, attn = model(x_batch)
            predictions = (torch.sigmoid(logit) > 0.5).long()
    """

    def __init__(self, cfg: NIDSConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        self.decoder = Decoder(cfg)
        self.classifier = ClassificationHead(cfg)
        self._mode = "pretrain"

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming (He) initialisation for Linear layers with GELU.
        Ensures variance is preserved through deep encoder stacks,
        preventing vanishing gradients at initialisation.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_mode(self, mode: str):
        """
        Switch the model between "pretrain" and "finetune" modes.

        In "finetune" mode, the decoder is detached from the computational
        graph (no gradients flow through it), saving ~40% memory.
        """
        assert mode in ("pretrain", "finetune"), \
            f"mode must be 'pretrain' or 'finetune', got '{mode}'"
        self._mode = mode
        # Freeze/unfreeze decoder to avoid wasting GPU memory in finetune
        for p in self.decoder.parameters():
            p.requires_grad = (mode == "pretrain")

    def freeze_extractor(self):
        """
        Freeze Block A weights — useful for ablation studies or when
        fine-tuning only Block B after pre-training convergence.
        """
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        logger.info("FeatureExtractor (Block A) weights frozen.")

    def unfreeze_extractor(self):
        """Re-enable gradient updates for Block A."""
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
        logger.info("FeatureExtractor (Block A) weights unfrozen.")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)

        Returns
        -------
        output : torch.Tensor
            Pre-train mode  → x_hat  (B, input_dim)  reconstruction
            Fine-tune mode  → logit  (B,)             classification logit
        z : torch.Tensor, shape (B, latent_dim)
            Latent representation (always returned for GA access).
        attn_weights : torch.Tensor, shape (B, encoder_dims[-1])
            Feature attention scores.
        """
        add_noise = (self._mode == "pretrain")
        z, attn_weights = self.feature_extractor(x, add_noise=add_noise)

        if self._mode == "pretrain":
            x_hat = self.decoder(z)
            return x_hat, z, attn_weights
        else:  # finetune / inference
            logit = self.classifier(z)
            return logit, z, attn_weights


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

class SparseAutoencoderLoss(nn.Module):
    """
    Composite loss for Phase 1 (autoencoder pre-training).

    L_total = L_recon + λ · L_sparsity

    Where:
      L_recon    = MSELoss(x̂, x)          — reconstruction fidelity
      L_sparsity = KL(ρ || ρ̂)             — sparsity regularisation
                 = ρ·log(ρ/ρ̂) + (1-ρ)·log((1-ρ)/(1-ρ̂))

    The KL-divergence sparsity term forces the mean activation of each
    bottleneck neuron (ρ̂) towards the target ρ (typically 0.05). This is
    the standard sparse autoencoder formulation from Ng (2011), proven
    effective for learning disentangled representations in tabular NIDS data.

    Parameters
    ----------
    sparsity_weight : float
        λ — multiplier for the KL term.
    sparsity_target : float
        ρ — target average activation probability.
    """

    def __init__(self, sparsity_weight: float = 1e-3, sparsity_target: float = 0.05):
        super().__init__()
        self.lam = sparsity_weight
        self.rho = sparsity_target
        self.mse = nn.MSELoss()

    def kl_divergence(self, z: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between target sparsity ρ and mean activation ρ̂.

        The bottleneck activations z are first passed through sigmoid to
        produce pseudo-probabilities ρ̂_j = mean_batch(σ(z_j)).
        """
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
        """
        Returns
        -------
        total_loss : torch.Tensor (scalar)
        breakdown : dict with 'recon' and 'sparsity' components
        """
        recon_loss = self.mse(x_hat, x_original)
        sparsity_loss = self.kl_divergence(z)
        total = recon_loss + self.lam * sparsity_loss
        return total, {
            "recon": recon_loss.item(),
            "sparsity": sparsity_loss.item(),
        }


def build_classification_loss(cfg: NIDSConfig, y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
    """
    Build a class-weighted BCEWithLogitsLoss for Phase 2.

    pos_weight = N_negative / N_positive ensures the loss penalises
    missed attacks (False Negatives) more heavily than missed normals —
    a critical requirement for NIDS where FN cost >> FP cost.

    Parameters
    ----------
    cfg : NIDSConfig
    y_train : np.ndarray
        Binary training labels used to compute the weight.

    Returns
    -------
    nn.BCEWithLogitsLoss with appropriate pos_weight tensor.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINER
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingHistory:
    """Tracks all loss and metric values across training epochs."""
    pretrain_losses: List[float] = field(default_factory=list)
    pretrain_val_losses: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    best_val_f1: float = 0.0
    best_epoch: int = 0


class EarlyStopping:
    """
    Standard early stopping with optional model state checkpointing.

    Monitors a validation metric (higher = better, e.g. F1 or neg-loss)
    and signals stop after `patience` epochs without improvement.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict] = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Parameters
        ----------
        score : float
            Current epoch's validation metric (higher is better).
        model : nn.Module

        Returns
        -------
        bool : True if training should stop.
        """
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
        """Load the best-checkpoint weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("  ✓ Restored best model state.")


class NIDSTrainer:
    """
    Two-phase trainer for the AG-SDAE NIDS pipeline.

    Phase 1 — Unsupervised pre-training:
        Trains the full autoencoder (Block A + Decoder) using the composite
        SparseAutoencoderLoss. Labels are NOT used in this phase.

    Phase 2 — Supervised fine-tuning:
        Freezes / unfreezes the encoder, trains Block A + Block B end-to-end
        using class-weighted BCE loss.

    Parameters
    ----------
    model : NIDSModel
    cfg : NIDSConfig
    y_train : np.ndarray
        Required for computing the positive class weight in Phase 2.
    """

    def __init__(self, model: NIDSModel, cfg: NIDSConfig, y_train: np.ndarray):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.history = TrainingHistory()

        # ── Phase 1 Loss ──────────────────────────────────────────────────
        self.ae_loss_fn = SparseAutoencoderLoss(
            sparsity_weight=cfg.sparsity_weight,
            sparsity_target=cfg.sparsity_target,
        )

        # ── Phase 2 Loss ──────────────────────────────────────────────────
        self.cls_loss_fn = build_classification_loss(cfg, y_train).to(self.device)

        self._set_seed(cfg.seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: Pre-training
    # ─────────────────────────────────────────────────────────────────────

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """
        Phase 1: Train the autoencoder with composite sparse reconstruction loss.

        The decoder and encoder are trained jointly. No labels are used.
        An AdamW optimiser with CosineAnnealingLR schedule is used for
        stable convergence on large datasets (~350k samples).
        """
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

            # Monitor val loss (negate since EarlyStopping expects higher=better)
            if early_stop.step(-val_loss, self.model):
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

        early_stop.restore_best(self.model)
        logger.info("Phase 1 complete.\n")

    def _pretrain_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            # Unpack; may or may not have labels (labels are ignored here)
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

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: Fine-tuning
    # ─────────────────────────────────────────────────────────────────────

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        freeze_extractor_epochs: int = 0,
    ):
        """
        Phase 2: End-to-end supervised fine-tuning.

        Parameters
        ----------
        train_loader : DataLoader (must contain labels)
        val_loader   : DataLoader (must contain labels)
        freeze_extractor_epochs : int
            If > 0, Block A is frozen for this many epochs first (warm-up
            Block B), then unfrozen for the remainder. This prevents the
            pre-trained encoder from being destroyed early in fine-tuning.
        """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — LATENT FEATURE EXTRACTION (GA INTERFACE)
# ═════════════════════════════════════════════════════════════════════════════

def extract_latent_features(
    model: NIDSModel,
    X: np.ndarray,
    cfg: NIDSConfig,
    return_attention: bool = False,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract the bottleneck latent representation z for all samples in X.

    This is the primary interface between Block A and:
      (a) An external ML classifier (XGBoost / LightGBM / SVM), which
          takes z as input features for a fully traditional classifier.
      (b) A Genetic Algorithm, which may use z for:
              · GA-based feature selection (which latent dims to use)
              · Hyperparameter chromosome evaluation (fitness = F1 on z)
              · Population initialisation based on cluster structure of z

    Parameters
    ----------
    model : NIDSModel
        Trained model (both phases should ideally be complete).
    X : np.ndarray, shape (N, input_dim)
        Pre-processed feature matrix (same format as training data).
    cfg : NIDSConfig
    return_attention : bool
        If True, also returns the (N, encoder_dims[-1]) attention weight matrix,
        which indicates which input features were most attended to per sample.
        This can be used by the GA for feature importance-guided mutation.
    batch_size : int, optional
        Override batch size for extraction (uses cfg.batch_size by default).

    Returns
    -------
    Z : np.ndarray, shape (N, latent_dim)
        Latent feature matrix — ready for XGBoost/GA consumption.
    A : np.ndarray or None, shape (N, encoder_dims[-1])
        Attention weight matrix (returned only if return_attention=True).

    Example
    -------
    >>> Z_train = extract_latent_features(model, X_train, cfg)[0]
    >>> # Now feed Z_train to XGBoost:
    >>> from xgboost import XGBClassifier
    >>> xgb = XGBClassifier(n_estimators=500, tree_method="hist")
    >>> xgb.fit(Z_train, y_train)
    >>>
    >>> # Or use Z_train as the chromosome phenotype in a GA:
    >>> # Each GA individual specifies which latent dims to keep
    >>> # fitness(chromosome) = f1_score(xgb.predict(Z_train[:, chromosome]))
    """
    device = torch.device(cfg.device)
    bs = batch_size or cfg.batch_size

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    dataset = NIDSTabularDataset(X)  # no labels needed
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
            # Directly call the encoder (avoid going through full model.forward)
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — EVALUATION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: NIDSModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: NIDSConfig,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the full NIDS pipeline on a test set.

    Parameters
    ----------
    threshold : float
        Decision boundary for binary classification (default: 0.5).
        Can be tuned post-hoc using the ROC curve to optimise for
        recall (important for security applications).

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, auc, confusion_matrix
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MODEL PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def save_model(model: NIDSModel, cfg: NIDSConfig, path: str):
    """
    Save model weights + config to a single .pt checkpoint file.

    The saved checkpoint contains enough information to fully reconstruct
    the model without needing to pass the original NIDSConfig separately.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "architecture": "AG-SDAE-v1",
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved to: {path}")


def load_model(path: str) -> Tuple[NIDSModel, NIDSConfig]:
    """
    Load a saved AG-SDAE checkpoint.

    Returns
    -------
    model : NIDSModel (ready for inference)
    cfg   : NIDSConfig (restored configuration)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint["config"]
    cfg = NIDSConfig(**cfg_dict)
    model = NIDSModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_mode("finetune")
    model.eval()
    logger.info(f"Model loaded from: {path}")
    return model, cfg


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — GA CHROMOSOME INTERFACE (STUB)
# ═════════════════════════════════════════════════════════════════════════════

class GAChromosome:
    """
    Stub demonstrating how the AG-SDAE plugs into a Genetic Algorithm.

    A chromosome encodes the key hyperparameters of the NIDS pipeline.
    The GA evolves a population of chromosomes, evaluating each by:
      1. Building a NIDSConfig from the chromosome genes.
      2. Pre-training the AG-SDAE on the training set.
      3. Extracting Z = extract_latent_features(model, X_val).
      4. Training a lightweight XGBoost on Z.
      5. Returning F1 on the validation set as the fitness score.

    This separates GA exploration (hyperparameter space) from DL training
    (fixed architecture + loss functions), making the fitness evaluation
    efficient and parallelisable.
    """

    # Gene bounds — extend as needed for your GA implementation
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
        """
        Convert a flat chromosome list to a NIDSConfig.

        Expected chromosome order (length 10):
            [enc_dim_0, enc_dim_1, enc_dim_2, latent_dim, dropout,
             sparsity_weight, sparsity_target, noise_std, lr, batch_size]
        """
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
        """
        Evaluate fitness of a single chromosome.

        Returns
        -------
        float : Validation F1 score (the GA maximises this).

        NOTE: In practice, wrap this in a try-except and return 0.0 on
        failure to prevent the GA from crashing on degenerate chromosomes.
        """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FULL PIPELINE DEMO (Entry Point)
# ═════════════════════════════════════════════════════════════════════════════

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
    """
    Run the full two-phase AG-SDAE training pipeline.

    This is the top-level entry point that ties together all sections.

    Parameters
    ----------
    X_train, y_train : pre-processed training data (post-SMOTE)
    X_val, y_val     : validation split (from original data, no SMOTE)
    X_test, y_test   : held-out test set
    cfg              : NIDSConfig (auto-built if None)
    save_path        : path to save the trained model checkpoint

    Returns
    -------
    model    : trained NIDSModel
    results  : evaluation metrics on the test set
    Z_test   : latent feature matrix for the test set (for GA / ML reuse)

    Example usage
    -------------
    >>> import numpy as np
    >>> # Simulate preprocessed data matching the PDF description
    >>> X_train = np.random.randn(350_000, 50).astype(np.float32)  # after SMOTE
    >>> y_train = np.random.randint(0, 2, 350_000).astype(np.float32)
    >>> X_val   = np.random.randn(20_000, 50).astype(np.float32)
    >>> y_val   = np.random.randint(0, 2, 20_000).astype(np.float32)
    >>> X_test  = np.random.randn(100_000, 50).astype(np.float32)
    >>> y_test  = np.random.randint(0, 2, 100_000).astype(np.float32)
    >>>
    >>> model, results, Z_test = run_full_pipeline(
    ...     X_train, y_train, X_val, y_val, X_test, y_test
    ... )
    """
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

    # ── Build dataloaders ─────────────────────────────────────────────────
    train_loader = build_dataloader(X_train, y_train, cfg, shuffle=True, use_weighted_sampler=False)
    val_loader   = build_dataloader(X_val,   y_val,   cfg, shuffle=False)
    test_loader  = build_dataloader(X_test,  y_test,  cfg, shuffle=False)

    # ── Instantiate model ─────────────────────────────────────────────────
    model = NIDSModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {n_params:,}")

    # ── Phase 1: Pre-training ─────────────────────────────────────────────
    trainer = NIDSTrainer(model, cfg, y_train)
    trainer.pretrain(train_loader, val_loader)

    # ── Phase 2: Fine-tuning (unfreeze after 5 warm-up epochs) ───────────
    trainer.finetune(train_loader, val_loader, freeze_extractor_epochs=5)

    # ── Evaluate on test set ──────────────────────────────────────────────
    results = evaluate_model(model, X_test, y_test, cfg)

    # ── Extract latent features for GA / external ML use ─────────────────
    Z_test, A_test = extract_latent_features(model, X_test, cfg, return_attention=True)
    logger.info(f"  Latent Z_test shape  : {Z_test.shape}")
    logger.info(f"  Attention A_test shape: {A_test.shape}")

    # ── Save model checkpoint ─────────────────────────────────────────────
    save_model(model, cfg, save_path)

    return model, results, Z_test


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick smoke-test with synthetic data matching the hybrid dataset dimensions
    from the preprocessing PDF (Section 3.7):
        - ~350k SMOTE-augmented training samples
        - ~100k test samples
        - ~50 features after feature selection
    """
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
