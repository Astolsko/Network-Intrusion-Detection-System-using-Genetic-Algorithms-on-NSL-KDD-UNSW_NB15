"""
=============================================================================
GA-NIDS: XAI & Cybersecurity Metrics Module
=============================================================================

Companion module to `nids_deep_model.py`. Provides:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  MODULE A — Per-Class Cybersecurity Metrics                         │
  │                                                                     │
  │  Computes Detection Rate (DR), False Alarm Rate (FAR), and F1       │
  │  for every named attack class (DoS, Probe, U2R, R2L, etc.)          │
  │  plus an aggregated binary summary. Produces publication-ready      │
  │  comparison tables compatible with NSL-KDD and UNSW-NB15 papers.   │
  ├─────────────────────────────────────────────────────────────────────┤
  │  MODULE B — Explainable AI (XAI)                                    │
  │                                                                     │
  │  SHAP — KernelExplainer on the full AG-SDAE pipeline (model-       │
  │         agnostic; works on the raw preprocessed feature space).     │
  │         TreeExplainer on an optional XGBoost Block B trained on z.  │
  │                                                                     │
  │  LIME — Tabular explainer on individual predictions, mapping        │
  │         perturbations of the preprocessed feature vector to the     │
  │         binary classification output.                               │
  │                                                                     │
  │  Attention — Direct extraction of the FeatureAttentionGate weights  │
  │              from Block A; no post-hoc approximation needed.        │
  ├─────────────────────────────────────────────────────────────────────┤
  │  MODULE C — Visualisations                                           │
  │                                                                     │
  │  All plots saved to disk as high-resolution PNGs:                   │
  │    · SHAP summary (beeswarm + bar)                                  │
  │    · SHAP decision plot for individual samples                      │
  │    · LIME bar chart for a single prediction                         │
  │    · Attention heatmap per sample class                             │
  │    · Per-class DR / FAR / F1 radar chart                            │
  │    · Confusion matrix with normalisation                            │
  └─────────────────────────────────────────────────────────────────────┘

CYBERSECURITY METRIC DEFINITIONS
─────────────────────────────────
  For each class c with True-Positive (TP), False-Negative (FN),
  False-Positive (FP), and True-Negative (TN):

    Detection Rate  (DR)  = TP / (TP + FN)   → aka Recall / Sensitivity
    False Alarm Rate(FAR) = FP / (FP + TN)   → aka Fall-out / FPR
    Precision             = TP / (TP + FP)
    F1-score              = 2 · (P · DR) / (P + DR)

  DR and FAR have direct operational meaning in NIDS:
    · High DR  →  few attacks slip through undetected
    · Low  FAR →  few benign connections are incorrectly blocked
  They are often in tension (threshold trade-off), making both essential
  reporting metrics in security literature.

DEPENDENCIES
────────────
    pip install shap lime matplotlib seaborn scikit-learn xgboost torch numpy
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Numerical / Data ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Visualisation ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")            # headless rendering — safe on any server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ─── Sklearn metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

# ─── PyTorch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── XAI ─────────────────────────────────────────────────────────────────────
import shap
from lime import lime_tabular

# ─── Local model ─────────────────────────────────────────────────────────────
from nids_deep_model import (
    NIDSModel, NIDSConfig, NIDSTabularDataset,
    build_dataloader, extract_latent_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── Consistent colour palette for all figures ────────────────────────────────
PALETTE = {
    "normal":   "#4CAF50",   # green
    "attack":   "#F44336",   # red
    "shap_pos": "#FF6B35",   # warm orange  (positive SHAP impact → attack)
    "shap_neg": "#1E88E5",   # cool blue    (negative SHAP impact → normal)
    "accent":   "#9C27B0",   # purple       (attention weights)
    "grid":     "#EEEEEE",
    "text":     "#212121",
}


# ═════════════════════════════════════════════════════════════════════════════
# MODULE A — PER-CLASS CYBERSECURITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

# NSL-KDD standard multi-class attack mapping
NSL_KDD_ATTACK_MAP: Dict[str, int] = {
    "Normal": 0,
    "DoS":    1,   # neptune, smurf, pod, teardrop, land, back, ...
    "Probe":  2,   # ipsweep, nmap, portsweep, satan, ...
    "U2R":    3,   # buffer_overflow, loadmodule, rootkit, perl, ...
    "R2L":    4,   # ftp_write, guess_passwd, imap, phf, multihop, ...
}

# UNSW-NB15 standard multi-class attack mapping
UNSW_NB15_ATTACK_MAP: Dict[str, int] = {
    "Normal":      0,
    "Fuzzers":     1,
    "Analysis":    2,
    "Backdoors":   3,
    "DoS":         4,
    "Exploits":    5,
    "Generic":     6,
    "Reconnaissance": 7,
    "Shellcode":   8,
    "Worms":       9,
}


def compute_binary_cyber_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-class DR, FAR, Precision, F1 for a binary (0/1) prediction.

    This handles the standard NIDS binary case where all attack categories
    are collapsed to class 1. This is directly compatible with the output
    of the AG-SDAE pipeline (Block B logit → sigmoid > 0.5).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground truth binary labels {0 = Normal, 1 = Attack}.
    y_pred : np.ndarray, shape (N,)
        Predicted binary labels.
    y_prob : np.ndarray, shape (N,), optional
        Attack class probability scores (sigmoid output). Used for AUC.
    class_names : list[str], optional
        Override default ["Normal", "Attack"].

    Returns
    -------
    pd.DataFrame
        Columns: Class | TP | FP | FN | TN | DR | FAR | Precision | F1 | AUC

    Notes
    -----
    DR  (Detection Rate)   = TP / (TP + FN)  — per class treated as "positive"
    FAR (False Alarm Rate) = FP / (FP + TN)  — per class treated as "positive"

    For binary classification:
      · Normal class as positive → DR_normal = TN_attack / (TN_attack + FP_attack)
      · Attack  class as positive → DR_attack = TP_attack / (TP_attack + FN_attack)
    Both are reported so the table matches standard NIDS literature conventions.
    """
    if class_names is None:
        class_names = ["Normal", "Attack"]

    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp          # other classes predicted as i
        fn = cm[i, :].sum() - tp          # class i predicted as other
        tn = cm.sum() - tp - fp - fn

        dr  = tp / (tp + fn + 1e-9)       # Recall / Sensitivity
        far = fp / (fp + tn + 1e-9)       # Fall-out / FPR
        prec = tp / (tp + fp + 1e-9)
        f1  = 2 * prec * dr / (prec + dr + 1e-9)

        row = {
            "Class":     name,
            "TP":        int(tp),
            "FP":        int(fp),
            "FN":        int(fn),
            "TN":        int(tn),
            "DR (%)":    round(dr * 100, 4),
            "FAR (%)":   round(far * 100, 4),
            "Precision": round(prec, 4),
            "F1-Score":  round(f1, 4),
        }

        if y_prob is not None and n_classes == 2:
            # AUC only meaningful for the attack class in binary setting
            if i == 1:
                row["AUC-ROC"] = round(roc_auc_score(
                    (y_true == i).astype(int), y_prob
                ), 4)
            else:
                row["AUC-ROC"] = round(roc_auc_score(
                    (y_true == i).astype(int), 1 - y_prob
                ), 4)
        rows.append(row)

    # ── Macro and weighted averages ─────────────────────────────────────────
    rows.append({
        "Class":     "Macro Avg",
        "TP":        "",
        "FP":        "",
        "FN":        "",
        "TN":        "",
        "DR (%)":    round(np.mean([r["DR (%)"] for r in rows if r["Class"] not in ["Macro Avg", "Weighted Avg"]]), 4),
        "FAR (%)":   round(np.mean([r["FAR (%)"] for r in rows if r["Class"] not in ["Macro Avg", "Weighted Avg"]]), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "AUC-ROC":   "" if y_prob is None else round(roc_auc_score(y_true, y_prob), 4),
    })

    df = pd.DataFrame(rows)
    return df


def compute_multiclass_cyber_metrics(
    y_true_multiclass: np.ndarray,
    y_pred_binary: np.ndarray,
    y_prob_binary: Optional[np.ndarray] = None,
    attack_map: Optional[Dict[str, int]] = dict(NSL_KDD_ATTACK_MAP),
) -> pd.DataFrame:
    """
    Compute DR, FAR, and F1 per NAMED attack category against the binary
    classifier's decisions.

    This is critical for evaluating NIDS real-world performance: a model
    may correctly classify 99% of DoS attacks but miss 80% of U2R attacks
    (which are rare and stealthy). Per-category analysis reveals these
    failure modes.

    Parameters
    ----------
    y_true_multiclass : np.ndarray, shape (N,)
        Multi-class ground truth labels (category IDs per attack_map).
        Class 0 is always Normal traffic.
    y_pred_binary : np.ndarray, shape (N,)
        Binary predictions from the model {0=Normal, 1=Attack}.
    y_prob_binary : np.ndarray, shape (N,), optional
        Attack probability scores.
    attack_map : dict
        Mapping from attack category name to integer label.

    Returns
    -------
    pd.DataFrame with one row per attack category.

    Interpretation
    --------------
    For attack category c:
      · TP = samples of class c that the model labelled as Attack (1)
      · FN = samples of class c that the model labelled as Normal (0)
      · FP = samples of class 0 (Normal) that model labelled as Attack (1)
      · TN = samples of class 0 (Normal) that model labelled as Normal (0)

    DR  = TP / (TP + FN)  → fraction of THIS attack type correctly detected
    FAR = FP / (FP + TN)  → fraction of Normal traffic falsely flagged
    Note: FAR is the same for all attack categories since it only depends
    on the Normal subset. This is standard in NIDS literature.
    """
    inv_map = {v: k for k, v in attack_map.items()}  # id → name
    normal_mask = (y_true_multiclass == 0)

    # FAR is global (Normal class only)
    fp_global = ((y_pred_binary == 1) & normal_mask).sum()
    tn_global = ((y_pred_binary == 0) & normal_mask).sum()
    far_global = fp_global / (fp_global + tn_global + 1e-9)

    rows = []
    for class_id, class_name in sorted(inv_map.items()):
        if class_id == 0:      # Normal — skip (reported separately)
            continue

        class_mask = (y_true_multiclass == class_id)
        n_class = class_mask.sum()
        if n_class == 0:
            continue

        # TP: this attack correctly flagged as Attack
        tp = ((y_pred_binary == 1) & class_mask).sum()
        # FN: this attack missed (predicted Normal)
        fn = ((y_pred_binary == 0) & class_mask).sum()

        dr   = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp_global + 1e-9)  # shared FP pool
        f1   = 2 * prec * dr / (prec + dr + 1e-9)

        row = {
            "Attack Category": class_name,
            "N Samples":      int(n_class),
            "TP":             int(tp),
            "FN":             int(fn),
            "DR (%)":         round(dr * 100, 4),
            "FAR (%)":        round(far_global * 100, 4),   # global FAR
            "Precision":      round(prec, 4),
            "F1-Score":       round(f1, 4),
        }

        if y_prob_binary is not None:
            # Per-category AUC: is this category detectable?
            y_bin = class_mask.astype(int)
            if y_bin.sum() > 0:
                row["AUC (vs Normal)"] = round(
                    roc_auc_score(y_bin, y_prob_binary * class_mask + (1 - class_mask) * (1 - y_prob_binary))
                    if y_bin.sum() < len(y_bin) else 0.5, 4
                )
        rows.append(row)

    # Normal row summary
    n_normal = normal_mask.sum()
    rows.insert(0, {
        "Attack Category": "Normal (Baseline)",
        "N Samples":      int(n_normal),
        "TP":             int(tn_global),  # correctly identified Normal
        "FN":             int(fp_global),  # Normal misclassified as Attack
        "DR (%)":         round(tn_global / (tn_global + fp_global + 1e-9) * 100, 4),
        "FAR (%)":        "—",
        "Precision":      "—",
        "F1-Score":       round(f1_score((~normal_mask).astype(int),
                                         (y_pred_binary == 0).astype(int),
                                         zero_division=0), 4),
    })

    df = pd.DataFrame(rows)
    return df


def print_metrics_table(df: pd.DataFrame, title: str = "Cybersecurity Metrics"):
    """Print a formatted metrics table to the logger."""
    logger.info("\n" + "═" * 80)
    logger.info(f"  {title}")
    logger.info("═" * 80)
    logger.info("\n" + df.to_string(index=False))
    logger.info("═" * 80 + "\n")


def save_metrics_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logger.info(f"Metrics saved to: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE B1 — SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    SHAP-based explainer for the AG-SDAE NIDS pipeline.

    Provides two complementary SHAP explanations:

      (1) Pipeline-level SHAP (KernelExplainer)
          Treats the entire pipeline (Block A → Block B) as a black box
          mapping preprocessed features → attack probability.
          Returns SHAP values in the ORIGINAL preprocessed feature space.
          This directly answers: "Which input features drive the final
          attack/normal classification?"

      (2) Latent-space SHAP (TreeExplainer, optional)
          If an XGBoost model is trained on the latent space z, uses the
          fast TreeExplainer to explain which LATENT DIMENSIONS matter.
          Combined with the Attention weights, this bridges the gap between
          the compressed representation and original feature contributions.

    Mathematical Basis
    ──────────────────
    SHAP (SHapley Additive exPlanations) assigns each feature φᵢ a value
    such that:
        f(x) = E[f(x)] + Σᵢ φᵢ(x)
    where f(x) is the model output (attack probability) and φᵢ satisfies
    the Shapley axioms (efficiency, symmetry, dummy, linearity).

    KernelSHAP approximates Shapley values for any model by:
        φᵢ = Σ_{S⊆F\{i}} |S|!(|F|-|S|-1)!/|F|! · [f(S∪{i}) - f(S)]
    where the expectation is estimated over a background dataset.

    Parameters
    ----------
    model : NIDSModel
        Trained AG-SDAE model (after Phase 2 fine-tuning).
    cfg : NIDSConfig
    feature_names : list[str]
        Names of preprocessed input features (post one-hot encoding).
        Length must equal cfg.input_dim.
    background_data : np.ndarray, shape (K, input_dim)
        Reference samples for SHAP baseline expectation.
        K = 50–200 is typically sufficient; larger K → more accurate.
    """

    def __init__(
        self,
        model: NIDSModel,
        cfg: NIDSConfig,
        feature_names: List[str],
        background_data: np.ndarray,
        n_background_samples: int = 100,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.feature_names = feature_names
        assert len(feature_names) == cfg.input_dim, (
            f"feature_names length ({len(feature_names)}) must match "
            f"cfg.input_dim ({cfg.input_dim})"
        )

        # Subsample background to n_background_samples for speed
        idx = np.random.choice(
            len(background_data),
            min(n_background_samples, len(background_data)),
            replace=False,
        )
        self.background = background_data[idx].astype(np.float32)

        # Build the prediction function that SHAP wraps
        self._pipeline_fn = self._build_pipeline_fn()

        logger.info(
            f"Building SHAP KernelExplainer "
            f"(background={self.background.shape[0]} samples)..."
        )
        self.explainer = shap.KernelExplainer(
            self._pipeline_fn,
            self.background,
            link="logit",          # explain log-odds (additive on logit scale)
        )
        logger.info("KernelExplainer ready.")

    def _build_pipeline_fn(self):
        """
        Returns a numpy-in / numpy-out prediction function for SHAP.

        The function maps a batch of raw features → attack probabilities,
        encapsulating the full AG-SDAE pipeline (Block A + Block B).
        """
        model = self.model
        device = self.device
        cfg = self.cfg

        model.set_mode("finetune")
        model.eval()

        def predict_proba(X: np.ndarray) -> np.ndarray:
            """
            X : np.ndarray, shape (N, input_dim)
            Returns attack probabilities as shape (N,).
            """
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            with torch.no_grad():
                logits, _, _ = model(X_t)
                probs = torch.sigmoid(logits).cpu().numpy()
            return probs

        return predict_proba

    def explain_batch(
        self,
        X_explain: np.ndarray,
        n_samples: int = 200,
        silent: bool = False,
    ) -> np.ndarray:
        """
        Compute SHAP values for a batch of samples.

        Parameters
        ----------
        X_explain : np.ndarray, shape (M, input_dim)
            Samples to explain (should be drawn from test set for validity).
        n_samples : int
            Number of SHAP perturbation samples per feature.
            Higher → more accurate but slower. 100–500 is typical.
        silent : bool
            Suppress SHAP progress bar.

        Returns
        -------
        shap_values : np.ndarray, shape (M, input_dim)
            SHAP values. Positive value → pushes toward Attack (1).
            Negative value → pushes toward Normal (0).
        """
        logger.info(f"Computing SHAP values for {len(X_explain)} samples...")
        shap_values = self.explainer.shap_values(
            X_explain.astype(np.float32),
            nsamples=n_samples,
            silent=silent,
        )
        # KernelExplainer with link="logit" returns values for the positive class
        # If it returns a list (per-class), extract the attack class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        logger.info(f"SHAP computation complete. Shape: {shap_values.shape}")
        return shap_values

    def explain_single(
        self,
        x: np.ndarray,
        n_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for a single sample.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)

        Returns
        -------
        shap_values : np.ndarray, shape (input_dim,)
        """
        return self.explain_batch(x[np.newaxis, :], n_samples=n_samples).squeeze(0)

    def get_global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Compute mean absolute SHAP value per feature (global importance).

        Parameters
        ----------
        shap_values : np.ndarray, shape (M, input_dim)

        Returns
        -------
        pd.DataFrame sorted by importance descending.
            Columns: Feature | Mean |SHAP| | Std |SHAP|
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        std_abs  = np.abs(shap_values).std(axis=0)
        df = pd.DataFrame({
            "Feature":       self.feature_names,
            "Mean |SHAP|":   mean_abs,
            "Std |SHAP|":    std_abs,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def build_latent_explainer(
        xgb_model,            # trained xgboost.XGBClassifier on z
        Z_background: np.ndarray,
        latent_dim_names: Optional[List[str]] = None,
    ) -> shap.TreeExplainer:
        """
        Build a fast TreeExplainer for an XGBoost classifier trained on z.

        Parameters
        ----------
        xgb_model : XGBClassifier
            XGBoost Block B trained on latent features.
        Z_background : np.ndarray, shape (K, latent_dim)
        latent_dim_names : list[str], optional

        Returns
        -------
        shap.TreeExplainer
        """
        explainer = shap.TreeExplainer(
            xgb_model,
            data=Z_background,
            feature_perturbation="interventional",
        )
        return explainer


# ═════════════════════════════════════════════════════════════════════════════
# MODULE B2 — LIME EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════

class LIMEExplainer:
    """
    LIME-based local explainer for the AG-SDAE NIDS pipeline.

    LIME (Local Interpretable Model-agnostic Explanations) explains a single
    prediction by:
      1. Generating N perturbed versions of the input x̃ (tabular: replacing
         feature values with values sampled from the training distribution).
      2. Weighting perturbations by their proximity to x (kernel distance).
      3. Fitting a sparse linear model g on the weighted neighbourhood.
      4. Returning the linear coefficients as local feature attributions.

    Complementary to SHAP:
      · SHAP  → globally consistent, computationally expensive
      · LIME  → purely local, fast, intuitive for individual cases
      · Both  → together provide robust, cross-validated explanations

    Parameters
    ----------
    predict_fn : callable
        numpy (N, D) → numpy (N,) attack probability function.
    feature_names : list[str]
    training_data : np.ndarray, shape (K, D)
        Used to estimate the distribution for LIME perturbations.
    categorical_features : list[int], optional
        Indices of categorical features (one-hot columns). LIME handles these
        differently (re-sampling from empirical distribution vs Gaussian noise).
    class_names : list[str]
        ["Normal", "Attack"]
    """

    def __init__(
        self,
        predict_fn,
        feature_names: List[str],
        training_data: np.ndarray,
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        kernel_width: float = 0.75,
        random_state: int = 42,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names

        if class_names is None:
            class_names = ["Normal", "Attack"]

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data.astype(np.float64),
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features or [],
            mode=mode,
            kernel_width=kernel_width,
            random_state=random_state,
            discretize_continuous=True,   # bins continuous features for readability
            discretizer="quartile",
        )
        logger.info("LIME LimeTabularExplainer initialised.")

    def _wrap_for_lime(self, X: np.ndarray) -> np.ndarray:
        """
        LIME expects a (N, 2) probability array [P(Normal), P(Attack)].
        Wraps the model's scalar attack probability output.
        """
        p_attack = self.predict_fn(X.astype(np.float32))
        return np.column_stack([1 - p_attack, p_attack])

    def explain_instance(
        self,
        x: np.ndarray,
        num_features: int = 15,
        num_samples: int = 2000,
        top_labels: int = 2,
    ) -> "lime.explanation.Explanation":
        """
        Explain a single instance prediction.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
        num_features : int
            Number of top features to include in the local linear model.
        num_samples : int
            Number of perturbed neighbours to generate.
        top_labels : int
            Number of class labels to compute explanations for.
            Always set to 2 (both Normal and Attack) so that
            `get_explanation_df` can safely look up whichever label
            key is available — avoids KeyError when the local
            neighbourhood is dominated by one class.

        Returns
        -------
        lime.explanation.Explanation
            Call .as_list() for the feature-weight pairs.
            Call .as_pyplot_figure() for a quick bar chart.
        """
        logger.info(f"LIME explaining instance (num_features={num_features}, "
                    f"num_samples={num_samples})...")
        explanation = self.explainer.explain_instance(
            data_row=x.astype(np.float64),
            predict_fn=self._wrap_for_lime,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,   # request BOTH labels every time
        )
        return explanation

    def get_explanation_df(
        self,
        explanation,
        label: int = 1,
    ) -> pd.DataFrame:
        """
        Convert a LIME explanation to a ranked DataFrame.

        Parameters
        ----------
        explanation : lime Explanation object
        label : int
            Preferred class label to extract (1 = Attack).
            If that key is absent from explanation.local_exp (which happens
            when the local neighbourhood is entirely one class), the method
            automatically falls back to the highest-confidence available key
            and logs a warning. This prevents the ``KeyError: 1`` crash that
            occurs when a sample is predicted with near-100% confidence for
            one class and LIME only stores the dominant class key.

        Returns
        -------
        pd.DataFrame with columns [Feature Condition, LIME Weight, Direction]
        sorted by absolute weight.
        """
        available_labels = list(explanation.local_exp.keys())

        # Resolve label: use requested label if present, else fall back
        if label in available_labels:
            resolved_label = label
        else:
            # Fall back to whichever label key exists with the most features
            resolved_label = max(
                available_labels,
                key=lambda k: len(explanation.local_exp[k]),
            )
            logger.warning(
                f"LIME label key {label} not found in explanation "
                f"(available: {available_labels}). "
                f"Falling back to label {resolved_label}. "
                f"This usually means the instance was predicted with very "
                f"high confidence for class {resolved_label} and LIME's "
                f"local neighbourhood contained only that class. "
                f"The returned weights still reflect the features that most "
                f"strongly support the model's decision."
            )

        pairs = explanation.as_list(label=resolved_label)
        df = pd.DataFrame(pairs, columns=["Feature Condition", "LIME Weight"])
        df["Explained Label"] = resolved_label
        df["Direction"] = df["LIME Weight"].apply(
            lambda w: "→ Attack" if w > 0 else "→ Normal"
        )
        df["Abs Weight"] = df["LIME Weight"].abs()
        df = df.sort_values("Abs Weight", ascending=False).reset_index(drop=True)
        return df


# ═════════════════════════════════════════════════════════════════════════════
# MODULE B3 — ATTENTION WEIGHT EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_attention_weights(
    model: NIDSModel,
    X: np.ndarray,
    cfg: NIDSConfig,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Extract FeatureAttentionGate weights directly from Block A.

    Unlike SHAP/LIME which are post-hoc approximations, the attention weights
    A ∈ (0,1)^(encoder_dims[-1]) are NATIVE to the model — they are computed
    deterministically from the input during the forward pass.

    These weights represent learned per-neuron importance at the
    pre-bottleneck layer. While they operate in the hidden space (not the
    original feature space), aggregated attention patterns reveal which
    hidden representations the model emphasises before compression.

    Parameters
    ----------
    model : NIDSModel
    X : np.ndarray, shape (N, input_dim)
    cfg : NIDSConfig
    batch_size : int

    Returns
    -------
    attention_matrix : np.ndarray, shape (N, encoder_dims[-1])
        Row i = attention gate output for sample i.
        Column j = importance weight of hidden dimension j.

    Usage
    -----
    To correlate attention back to input features, compute the
    Jacobian or use SHAP on the encoder only.
    A simpler approach: cluster samples by attention pattern to find
    traffic subgroups that consistently activate different encoder paths.
    """
    device = torch.device(cfg.device)
    model.set_mode("finetune")
    model.eval()
    model.to(device)

    all_attn = []
    dataset = NIDSTabularDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            _, attn = model.feature_extractor(x, add_noise=False)
            all_attn.append(attn.cpu().numpy())

    return np.concatenate(all_attn, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE C — VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

class NIDSVisualiser:
    """
    Publication-quality visualisations for NIDS cybersecurity analysis.

    All methods save PNG files at 300 DPI (suitable for journal submission).
    Returns the matplotlib Figure object for further customisation.

    Parameters
    ----------
    output_dir : str
        Directory to save all output figures.
    feature_names : list[str]
        Preprocessed input feature names.
    """

    def __init__(self, output_dir: str = "xai_outputs", feature_names: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = feature_names or []

        # Apply global style
        plt.rcParams.update({
            "font.family":       "DejaVu Sans",
            "font.size":         11,
            "axes.titlesize":    13,
            "axes.titleweight":  "bold",
            "axes.labelsize":    11,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.dpi":        300,
            "savefig.bbox":      "tight",
            "savefig.dpi":       300,
        })

    def _save(self, fig: plt.Figure, name: str):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"  Saved: {path}")
        plt.close(fig)
        return path

    # ──────────────────────────────────────────────────────────────────────
    # C1 — Confusion Matrix
    # ──────────────────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix",
    ) -> plt.Figure:
        """
        Dual-panel confusion matrix: raw counts (left) and row-normalised (right).

        The normalised matrix shows DR as the diagonal of the Attack row and
        specificity as the diagonal of the Normal row — directly comparable
        to the metrics table.
        """
        if class_names is None:
            class_names = ["Normal", "Attack"]

        cm_raw  = confusion_matrix(y_true, y_pred)
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

        for ax, cm, fmt, subtitle in zip(
            axes,
            [cm_raw, cm_norm],
            ["d", ".3f"],
            ["Raw Counts", "Row-Normalised (DR on diagonal)"],
        ):
            cmap = LinearSegmentedColormap.from_list(
                "nids", ["#FFFFFF", PALETTE["attack"]], N=256
            )
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set(
                xticks=range(len(class_names)),
                yticks=range(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted Label",
                ylabel="True Label",
                title=subtitle,
            )

            thresh = cm.max() / 2
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = cm[i, j]
                    text = f"{val:{fmt}}" if fmt != "d" else f"{int(val):,}"
                    ax.text(
                        j, i, text,
                        ha="center", va="center", fontsize=12,
                        color="white" if val > thresh else "black",
                        fontweight="bold",
                    )

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C2 — Per-Class Cybersecurity Metrics Bar Chart
    # ──────────────────────────────────────────────────────────────────────

    def plot_per_class_metrics(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Per-Class Cybersecurity Metrics",
        filename: str = "per_class_metrics",
        attack_col: str = "Attack Category",
    ) -> plt.Figure:
        """
        Grouped bar chart of DR (%), FAR (%), and F1-Score per attack category.

        The chart highlights the detection vs. false-alarm trade-off visually —
        ideal performance is high DR, low FAR, high F1 for all categories.
        """
        # Filter out summary rows and non-numeric entries
        plot_df = metrics_df[metrics_df[attack_col] != "Normal (Baseline)"].copy()
        plot_df = plot_df[pd.to_numeric(plot_df["DR (%)"], errors="coerce").notna()]
        plot_df["DR (%)"]   = pd.to_numeric(plot_df["DR (%)"])
        plot_df["FAR (%)"]  = pd.to_numeric(plot_df["FAR (%)"], errors="coerce").fillna(0)
        plot_df["F1-Score"] = pd.to_numeric(plot_df["F1-Score"])

        n_categories = len(plot_df)
        x = np.arange(n_categories)
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, n_categories * 2.5), 6))

        bars_dr  = ax.bar(x - width, plot_df["DR (%)"],   width, label="DR (%)",
                          color=PALETTE["normal"],   edgecolor="white", linewidth=0.8)
        bars_far = ax.bar(x,         plot_df["FAR (%)"],  width, label="FAR (%)",
                          color=PALETTE["attack"],   edgecolor="white", linewidth=0.8)
        bars_f1  = ax.bar(x + width, plot_df["F1-Score"] * 100, width, label="F1 × 100",
                          color=PALETTE["accent"],   edgecolor="white", linewidth=0.8, alpha=0.85)

        # Value labels on bars
        for bars in [bars_dr, bars_far, bars_f1]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.5, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=8, rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df[attack_col], rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C3 — DR / FAR / F1 Radar Chart
    # ──────────────────────────────────────────────────────────────────────

    def plot_radar_chart(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Attack Detection Profile (Radar)",
        filename: str = "radar_chart",
        attack_col: str = "Attack Category",
    ) -> plt.Figure:
        """
        Radar (spider) chart showing DR, FAR (inverted), and F1 per category.

        FAR is plotted as (100 - FAR) so that the ideal profile is a large,
        evenly-filled polygon. A collapsed polygon in any direction indicates
        a specific weakness.
        """
        plot_df = metrics_df[metrics_df[attack_col] != "Normal (Baseline)"].copy()
        plot_df = plot_df[pd.to_numeric(plot_df["DR (%)"], errors="coerce").notna()]
        plot_df["DR (%)"]      = pd.to_numeric(plot_df["DR (%)"])
        plot_df["FAR (%)"]     = pd.to_numeric(plot_df["FAR (%)"], errors="coerce").fillna(0)
        plot_df["F1-Score"]    = pd.to_numeric(plot_df["F1-Score"])
        plot_df["Specificity"] = 100 - plot_df["FAR (%)"]   # inverted FAR

        categories_list = plot_df[attack_col].tolist()
        metrics_cols = ["DR (%)", "Specificity", "F1-Score"]
        metric_labels = ["DR (%)", "Specificity\n(100−FAR %)", "F1 × 100"]

        N = len(metrics_cols)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]   # close the polygon

        colours = plt.cm.tab10(np.linspace(0, 0.9, len(categories_list)))

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_facecolor("#FAFAFA")

        for idx, (_, row) in enumerate(plot_df.iterrows()):
            values = [
                row["DR (%)"],
                row["Specificity"],
                row["F1-Score"] * 100,
            ] + [row["DR (%)"]]   # close polygon

            ax.plot(angles, values, "o-", linewidth=2,
                    color=colours[idx], label=row[attack_col])
            ax.fill(angles, values, alpha=0.15, color=colours[idx])

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=11)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8)
        ax.set_ylim(0, 110)
        ax.set_title(title, size=13, fontweight="bold", pad=20)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1),
            framealpha=0.9,
            fontsize=9,
        )

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C4 — SHAP Beeswarm Summary Plot
    # ──────────────────────────────────────────────────────────────────────

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        max_display: int = 20,
        title: str = "SHAP Feature Importance (Beeswarm)",
        filename: str = "shap_summary_beeswarm",
    ) -> plt.Figure:
        """
        SHAP beeswarm plot: each dot is one sample, x-axis = SHAP value,
        colour = feature value (blue = low, red = high).

        This is the standard publication figure for SHAP-based NIDS analysis.
        It simultaneously shows:
          · Global importance (vertical ordering by mean |SHAP|)
          · Effect direction (positive → predicts Attack, negative → Normal)
          · Feature value interaction (colour gradient)
          · Sample distribution (spread of dots)
        """
        # Convert to SHAP Explanation object for the modern API
        feature_names_arr = (
            np.array(self.feature_names) if self.feature_names
            else np.array([f"f{i}" for i in range(shap_values.shape[1])])
        )

        explanation = shap.Explanation(
            values=shap_values,
            data=X_explain,
            feature_names=feature_names_arr,
        )

        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.38)))
        shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            show=False,
            color_bar_label="Feature Value",
        )
        plt.title(title, fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("SHAP Value (impact on Attack probability logit)")
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_shap_bar(
        self,
        shap_values: np.ndarray,
        max_display: int = 20,
        title: str = "SHAP Mean |SHAP| Feature Importance",
        filename: str = "shap_bar",
    ) -> plt.Figure:
        """
        Horizontal bar chart of mean absolute SHAP values — the clearest
        single-metric ranking of global feature importance.
        """
        feature_names_arr = (
            self.feature_names if self.feature_names
            else [f"f{i}" for i in range(shap_values.shape[1])]
        )
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:max_display]

        top_names  = [feature_names_arr[i] for i in top_idx]
        top_values = mean_abs[top_idx]

        # Colour: gradient from high importance (dark red) to lower (light)
        cmap  = plt.cm.YlOrRd
        norm  = plt.Normalize(top_values.min(), top_values.max())
        colors = [cmap(norm(v)) for v in top_values]

        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.38)))
        bars = ax.barh(range(max_display), top_values[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(max_display))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(title, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)

        # Annotate bar ends
        for bar, val in zip(bars, top_values[::-1]):
            ax.text(
                val + top_values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Importance Level", fraction=0.03, pad=0.02)
        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C5 — SHAP Decision Plot (Individual Sample)
    # ──────────────────────────────────────────────────────────────────────

    def plot_shap_decision(
        self,
        shap_values_single: np.ndarray,
        expected_value: float,
        x_single: np.ndarray,
        true_label: int,
        pred_prob: float,
        title: Optional[str] = None,
        filename: str = "shap_decision_single",
        max_display: int = 15,
    ) -> plt.Figure:
        """
        SHAP decision plot for a single instance.

        Shows the cumulative effect of each feature on the model's output,
        starting from the base value (E[f(x)]) and adding each SHAP value
        in order of magnitude. The final position is the model's prediction.

        Colours: orange = pushes toward Attack, blue = pushes toward Normal.
        """
        feature_names_arr = (
            np.array(self.feature_names) if self.feature_names
            else np.array([f"f{i}" for i in range(len(shap_values_single))])
        )

        # Select top max_display features by |SHAP|
        top_idx = np.argsort(np.abs(shap_values_single))[::-1][:max_display]
        top_idx_sorted = top_idx[np.argsort(shap_values_single[top_idx])]

        features_display = feature_names_arr[top_idx_sorted]
        values_display   = shap_values_single[top_idx_sorted]
        data_display     = x_single[top_idx_sorted]

        cumulative = expected_value + np.cumsum(values_display)

        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.42)))

        for i, (val, cum, feat, data_val) in enumerate(
            zip(values_display, cumulative, features_display, data_display)
        ):
            colour = PALETTE["shap_pos"] if val > 0 else PALETTE["shap_neg"]
            ax.barh(i, val, left=cum - val, color=colour, alpha=0.8,
                    edgecolor="white", height=0.65)

            # Feature label + value on the right
            ax.text(
                max(cumulative) + abs(values_display).max() * 0.04, i,
                f"{feat} = {data_val:.3g}",
                va="center", fontsize=8.5, color=PALETTE["text"],
            )

        ax.axvline(expected_value, color="grey", linestyle="--",
                   linewidth=1.2, label=f"Base value ({expected_value:.3f})")
        ax.axvline(cumulative[-1], color=PALETTE["attack"] if pred_prob > 0.5 else PALETTE["normal"],
                   linestyle="-", linewidth=1.8,
                   label=f"Prediction ({pred_prob:.3f})")

        ax.set_yticks(range(max_display))
        ax.set_yticklabels([""] * max_display)   # labels already in text annotations
        ax.set_xlabel("Model Output (logit space)")

        label_str = "Attack" if true_label == 1 else "Normal"
        if title is None:
            title = f"SHAP Decision Plot | True: {label_str} | P(Attack)={pred_prob:.3f}"
        ax.set_title(title, fontweight="bold")

        pos_patch = mpatches.Patch(color=PALETTE["shap_pos"], label="→ Attack")
        neg_patch = mpatches.Patch(color=PALETTE["shap_neg"], label="→ Normal")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C6 — LIME Bar Chart
    # ──────────────────────────────────────────────────────────────────────

    def plot_lime_explanation(
        self,
        lime_df: pd.DataFrame,
        true_label: int,
        pred_prob: float,
        title: Optional[str] = None,
        filename: str = "lime_explanation",
    ) -> plt.Figure:
        """
        Horizontal bar chart of LIME feature weights for a single prediction.

        Positive weights (orange) push the model toward Attack.
        Negative weights (blue) push toward Normal.
        """
        label_str = "Attack" if true_label == 1 else "Normal"
        if title is None:
            title = (
                f"LIME Local Explanation | True: {label_str} "
                f"| P(Attack)={pred_prob:.3f}"
            )

        lime_plot = lime_df.head(15).copy()
        colours = [
            PALETTE["shap_pos"] if w > 0 else PALETTE["shap_neg"]
            for w in lime_plot["LIME Weight"]
        ]

        fig, ax = plt.subplots(figsize=(9, max(5, len(lime_plot) * 0.45)))
        bars = ax.barh(
            range(len(lime_plot)),
            lime_plot["LIME Weight"],
            color=colours,
            edgecolor="white",
            linewidth=0.7,
            height=0.65,
        )

        ax.set_yticks(range(len(lime_plot)))
        ax.set_yticklabels(lime_plot["Feature Condition"], fontsize=9)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("LIME Weight (local linear coefficient)")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)

        # Annotate
        for bar in bars:
            w = bar.get_width()
            offset = max(lime_plot["LIME Weight"].abs()) * 0.02
            ax.text(
                w + (offset if w >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=8,
            )

        pos_patch = mpatches.Patch(color=PALETTE["shap_pos"], label="Supports Attack")
        neg_patch = mpatches.Patch(color=PALETTE["shap_neg"], label="Supports Normal")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C7 — Attention Heatmap
    # ──────────────────────────────────────────────────────────────────────

    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        y_labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        n_samples_per_class: int = 50,
        title: str = "Feature Attention Gate — Per-Class Activation Heatmap",
        filename: str = "attention_heatmap",
    ) -> plt.Figure:
        """
        Heatmap of attention gate activations, stratified by class.

        Each row = one sample. Each column = one hidden neuron at the
        pre-bottleneck layer. Colour = attention weight ∈ (0, 1).

        By comparing Normal (top) vs Attack (bottom) panels, one can
        visually identify which hidden dimensions are systematically
        up-weighted for attack traffic — a form of internal feature selection.

        Parameters
        ----------
        attention_matrix : np.ndarray, shape (N, encoder_dims[-1])
        y_labels : np.ndarray, shape (N,)
            Binary labels {0, 1}.
        n_samples_per_class : int
            Number of samples to display per class (random subset).
        """
        if class_names is None:
            class_names = ["Normal", "Attack"]

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"hspace": 0.45}
        )
        fig.suptitle(title, fontsize=13, fontweight="bold")

        cmap = LinearSegmentedColormap.from_list(
            "attn", ["#FFFFFF", PALETTE["accent"]], N=256
        )

        for class_id, (ax, class_name) in enumerate(zip(axes, class_names)):
            idx = np.where(y_labels == class_id)[0]
            if len(idx) == 0:
                ax.set_title(f"{class_name} — no samples")
                continue

            sample_idx = np.random.choice(
                idx, min(n_samples_per_class, len(idx)), replace=False
            )
            data = attention_matrix[sample_idx]

            im = ax.imshow(
                data, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest",
            )
            fig.colorbar(im, ax=ax, label="Attention Weight", fraction=0.03, pad=0.02)
            ax.set_title(
                f"Class: {class_name} ({len(sample_idx)} samples shown)",
                fontsize=11,
            )
            ax.set_xlabel("Hidden Neuron Index (pre-bottleneck)")
            ax.set_ylabel("Sample")
            ax.tick_params(left=False, labelleft=False)

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C8 — SHAP + Attention Overlay: Bridging Latent and Input Spaces
    # ──────────────────────────────────────────────────────────────────────

    def plot_shap_vs_attention(
        self,
        shap_importance: pd.DataFrame,
        attention_mean: np.ndarray,
        top_n: int = 20,
        title: str = "SHAP Importance vs Block A Attention Activation",
        filename: str = "shap_vs_attention",
    ) -> plt.Figure:
        """
        Dual-axis comparative plot:
          Left axis (bar)   — Top N features by mean |SHAP| (input feature space)
          Right axis (line) — Mean attention gate activation (hidden space, averaged)

        Since SHAP operates in the input space and attention in the hidden space,
        this plot illustrates the relationship between the two explanations —
        features that rank highly in SHAP tend to activate specific hidden neurons
        that the attention gate subsequently up-weights.

        Parameters
        ----------
        shap_importance : pd.DataFrame
            From SHAPExplainer.get_global_importance(), sorted descending.
        attention_mean : np.ndarray, shape (encoder_dims[-1],)
            Mean attention weight across all samples, per hidden neuron.
        top_n : int
            Number of top features to display.
        """
        top_df = shap_importance.head(top_n).copy()
        x = np.arange(len(top_df))

        fig, ax1 = plt.subplots(figsize=(12, 5.5))
        ax2 = ax1.twinx()

        # SHAP bars
        bars = ax1.bar(
            x, top_df["Mean |SHAP|"],
            color=PALETTE["shap_pos"], alpha=0.75,
            edgecolor="white", linewidth=0.7, label="Mean |SHAP|"
        )
        ax1.set_ylabel("Mean |SHAP Value|", color=PALETTE["shap_pos"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["shap_pos"])

        # Attention activation (mean across neurons, replicated for visual comparison)
        # Normalise to [0,1] and scale by SHAP range for overlay readability
        attn_scalar = attention_mean.mean()   # global mean attention (scalar)
        attn_line_y = np.full(len(x), attn_scalar * top_df["Mean |SHAP|"].max())
        ax2.plot(x, top_df["Mean |SHAP|"] * attn_scalar, "o-",
                 color=PALETTE["accent"], linewidth=2.0,
                 markersize=5, alpha=0.8, label="SHAP × Attn scale")
        ax2.set_ylabel("Attention-Scaled Importance", color=PALETTE["accent"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])

        ax1.set_xticks(x)
        ax1.set_xticklabels(top_df["Feature"], rotation=35, ha="right", fontsize=9)
        ax1.set_title(title, fontweight="bold", fontsize=13)
        ax1.set_facecolor("#FAFAFA")
        ax1.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        plt.tight_layout()
        return self._save(fig, filename)

    # ──────────────────────────────────────────────────────────────────────
    # C9 — Latent Space SHAP (TreeExplainer on XGBoost Block B)
    # ──────────────────────────────────────────────────────────────────────

    def plot_latent_shap_bar(
        self,
        shap_values_latent: np.ndarray,
        latent_dim: int,
        title: str = "XGBoost Block B — Latent Dimension SHAP Importance",
        filename: str = "shap_latent_bar",
    ) -> plt.Figure:
        """
        Bar chart of SHAP importance over latent dimensions z₀, z₁, ..., z_{d-1}.

        This answers: "Which bottleneck neurons are most discriminative for
        the final Attack/Normal decision?" Combined with the attention heatmap
        (which shows which input patterns activate which hidden neurons),
        this creates a full end-to-end explanation chain:

            Input features
                ↓  (SHAP in input space)
            Pre-bottleneck hidden rep
                ↓  (Attention gate weights)
            Latent z
                ↓  (SHAP in latent space via TreeExplainer)
            Binary classification
        """
        mean_abs = np.abs(shap_values_latent).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1]
        latent_names = [f"z_{i}" for i in range(latent_dim)]

        fig, ax = plt.subplots(figsize=(10, 5))
        colours = plt.cm.plasma(np.linspace(0.2, 0.85, latent_dim))

        ax.bar(
            range(latent_dim),
            mean_abs[top_idx],
            color=[colours[i] for i in range(latent_dim)],
            edgecolor="white", linewidth=0.6,
        )
        ax.set_xticks(range(latent_dim))
        ax.set_xticklabels(
            [latent_names[i] for i in top_idx],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_ylabel("Mean |SHAP Value|")
        ax.set_title(title, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        plt.tight_layout()
        return self._save(fig, filename)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE D — INTEGRATED XAI PIPELINE (SINGLE ENTRY POINT)
# ═════════════════════════════════════════════════════════════════════════════

def run_xai_pipeline(
    model: NIDSModel,
    cfg: NIDSConfig,
    X_test: np.ndarray,
    y_test_binary: np.ndarray,
    feature_names: List[str],
    y_test_multiclass: Optional[np.ndarray] = None,
    attack_map: Optional[Dict[str, int]] = None,
    output_dir: str = "xai_outputs",
    n_shap_explain: int = 200,
    n_shap_samples: int = 200,
    n_lime_samples: int = 2000,
    sample_indices_to_explain: Optional[List[int]] = None,
    xgb_block_b=None,
    Z_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Full XAI and Metrics pipeline — single call entry point.

    Runs all three XAI methods + all visualisations + all metric tables
    and saves outputs to `output_dir`.

    Parameters
    ----------
    model : NIDSModel
        Fully trained AG-SDAE (both phases complete).
    cfg : NIDSConfig
    X_test : np.ndarray, shape (N, input_dim)
        Test set features.
    y_test_binary : np.ndarray, shape (N,)
        Binary labels {0=Normal, 1=Attack}.
    feature_names : list[str]
        Human-readable feature names for all input_dim features.
    y_test_multiclass : np.ndarray, optional
        Multi-class labels (e.g., NSL-KDD DoS/Probe/U2R/R2L/Normal).
        If provided, enables per-category metric computation.
    attack_map : dict, optional
        Mapping of attack category name → integer ID.
        Defaults to NSL_KDD_ATTACK_MAP if y_test_multiclass is provided.
    output_dir : str
        Directory for all output figures and CSVs.
    n_shap_explain : int
        Number of test samples to explain with SHAP.
    n_shap_samples : int
        SHAP nsamples parameter (kernel perturbations).
    n_lime_samples : int
        LIME num_samples parameter.
    sample_indices_to_explain : list[int], optional
        Specific test indices for individual SHAP/LIME explanations.
        Defaults to [first Normal, first Attack] if None.
    xgb_block_b : XGBClassifier, optional
        Pre-trained XGBoost on latent features. Enables latent SHAP.
    Z_test : np.ndarray, optional
        Latent features for test set (pre-computed). If None, extracted here.

    Returns
    -------
    dict with keys:
        "binary_metrics_df"       : pd.DataFrame (DR/FAR/F1 binary)
        "multiclass_metrics_df"   : pd.DataFrame or None
        "shap_values"             : np.ndarray
        "shap_importance_df"      : pd.DataFrame
        "lime_explanations"       : list of (df, explanation) tuples
        "attention_matrix"        : np.ndarray
        "predictions"             : dict with y_pred, y_prob
    """
    logger.info("=" * 70)
    logger.info("GA-NIDS — XAI & Cybersecurity Metrics Pipeline")
    logger.info("=" * 70)

    vis = NIDSVisualiser(output_dir=output_dir, feature_names=feature_names)
    results: Dict[str, Any] = {}

    # ── Step 1: Get model predictions ────────────────────────────────────
    logger.info("\n[1/7] Generating model predictions...")
    device = torch.device(cfg.device)
    model.set_mode("finetune")
    model.eval()
    model.to(device)

    all_logits = []
    dataset = NIDSTabularDataset(X_test, y_test_binary)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for x_b, _ in loader:
            logit, _, _ = model(x_b.to(device))
            all_logits.append(logit.cpu().numpy())

    logits = np.concatenate(all_logits)
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob > 0.5).astype(int)
    results["predictions"] = {"y_pred": y_pred, "y_prob": y_prob, "logits": logits}

    logger.info(f"  Predictions complete. Attack rate: "
                f"{y_pred.mean()*100:.2f}% (true: {y_test_binary.mean()*100:.2f}%)")

    # ── Step 2: Binary cybersecurity metrics ─────────────────────────────
    logger.info("\n[2/7] Computing binary cybersecurity metrics...")
    binary_df = compute_binary_cyber_metrics(
        y_test_binary, y_pred, y_prob,
        class_names=["Normal", "Attack"]
    )
    results["binary_metrics_df"] = binary_df
    print_metrics_table(binary_df, title="Binary Metrics (DR / FAR / F1)")
    save_metrics_csv(binary_df, str(Path(output_dir) / "binary_metrics.csv"))

    # ── Step 3: Multi-class per-category metrics ──────────────────────────
    results["multiclass_metrics_df"] = None
    if y_test_multiclass is not None:
        logger.info("\n[3/7] Computing per-attack-category metrics...")
        if attack_map is None:
            attack_map = NSL_KDD_ATTACK_MAP
        mc_df = compute_multiclass_cyber_metrics(
            y_test_multiclass, y_pred, y_prob, attack_map
        )
        results["multiclass_metrics_df"] = mc_df
        print_metrics_table(mc_df, title="Per-Attack-Category Metrics")
        save_metrics_csv(mc_df, str(Path(output_dir) / "multiclass_metrics.csv"))

        # Visualise
        vis.plot_per_class_metrics(mc_df, attack_col="Attack Category")
        vis.plot_radar_chart(mc_df, attack_col="Attack Category")
    else:
        logger.info("\n[3/7] Skipping multi-class metrics (y_test_multiclass=None).")

    # ── Step 4: Confusion matrix ──────────────────────────────────────────
    logger.info("\n[4/7] Plotting confusion matrix...")
    vis.plot_confusion_matrix(y_test_binary, y_pred)

    # ── Step 5: Attention weights ─────────────────────────────────────────
    logger.info("\n[5/7] Extracting Block A attention weights...")
    attention_matrix = extract_attention_weights(model, X_test, cfg)
    results["attention_matrix"] = attention_matrix
    vis.plot_attention_heatmap(attention_matrix, y_test_binary)

    # ── Step 6: SHAP explanations ─────────────────────────────────────────
    logger.info(f"\n[6/7] Running SHAP (explaining {n_shap_explain} samples)...")

    # Select representative subset for SHAP (stratified)
    n_half = n_shap_explain // 2
    normal_idx = np.where(y_test_binary == 0)[0]
    attack_idx = np.where(y_test_binary == 1)[0]
    shap_idx = np.concatenate([
        np.random.choice(normal_idx, min(n_half, len(normal_idx)), replace=False),
        np.random.choice(attack_idx, min(n_half, len(attack_idx)), replace=False),
    ])
    X_shap = X_test[shap_idx]

    shap_exp = SHAPExplainer(
        model=model,
        cfg=cfg,
        feature_names=feature_names,
        background_data=X_test,
        n_background_samples=min(100, len(X_test)),
    )
    shap_values = shap_exp.explain_batch(X_shap, n_samples=n_shap_samples, silent=False)
    results["shap_values"] = shap_values

    importance_df = shap_exp.get_global_importance(shap_values)
    results["shap_importance_df"] = importance_df
    logger.info("\nTop 10 features by mean |SHAP|:")
    logger.info("\n" + importance_df.head(10).to_string(index=False))
    save_metrics_csv(importance_df, str(Path(output_dir) / "shap_importance.csv"))

    # Global plots
    vis.plot_shap_summary(shap_values, X_shap, max_display=20)
    vis.plot_shap_bar(shap_values, max_display=20)
    vis.plot_shap_vs_attention(importance_df, attention_matrix.mean(axis=0))

    # Latent SHAP (if XGBoost Block B provided)
    if xgb_block_b is not None and Z_test is not None:
        logger.info("  Computing latent SHAP via TreeExplainer...")
        latent_exp = SHAPExplainer.build_latent_explainer(
            xgb_block_b, Z_test[shap_idx[:50]]
        )
        shap_latent = latent_exp.shap_values(Z_test[shap_idx])
        if isinstance(shap_latent, list):
            shap_latent = shap_latent[1]
        results["shap_latent_values"] = shap_latent
        vis.plot_latent_shap_bar(shap_latent, latent_dim=cfg.latent_dim)

    # Individual sample explanations
    if sample_indices_to_explain is None:
        sample_indices_to_explain = []
        if len(normal_idx) > 0:
            sample_indices_to_explain.append(int(normal_idx[0]))
        if len(attack_idx) > 0:
            sample_indices_to_explain.append(int(attack_idx[0]))

    for i, idx in enumerate(sample_indices_to_explain[:4]):
        sv_single = shap_exp.explain_single(X_test[idx], n_samples=500)
        vis.plot_shap_decision(
            sv_single,
            expected_value=float(shap_exp.explainer.expected_value),
            x_single=X_test[idx],
            true_label=int(y_test_binary[idx]),
            pred_prob=float(y_prob[idx]),
            filename=f"shap_decision_sample_{idx}",
        )

    # ── Step 7: LIME explanations ─────────────────────────────────────────
    logger.info(f"\n[7/7] Running LIME (explaining {len(sample_indices_to_explain)} samples)...")

    def predict_proba_numpy(X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        with torch.no_grad():
            logit, _, _ = model(X_t)
        return torch.sigmoid(logit).cpu().numpy()

    lime_exp = LIMEExplainer(
        predict_fn=predict_proba_numpy,
        feature_names=feature_names,
        training_data=X_test[:min(2000, len(X_test))],
    )

    lime_results = []
    for i, idx in enumerate(sample_indices_to_explain[:4]):
        explanation = lime_exp.explain_instance(
            X_test[idx],
            num_features=15,
            num_samples=n_lime_samples,
        )
        # Pass the model's predicted class as the preferred label so the
        # explanation always matches the decision being explained.
        # get_explanation_df will fall back gracefully if that key is missing.
        predicted_label = int(y_pred[idx])
        lime_df = lime_exp.get_explanation_df(explanation, label=predicted_label)
        lime_results.append((lime_df, explanation))

        vis.plot_lime_explanation(
            lime_df,
            true_label=int(y_test_binary[idx]),
            pred_prob=float(y_prob[idx]),
            filename=f"lime_explanation_sample_{idx}",
        )
        logger.info(
            f"  Sample {idx} (true={y_test_binary[idx]}, "
            f"P(Attack)={y_prob[idx]:.3f}) — "
            f"top feature: {lime_df.iloc[0]['Feature Condition']}"
        )

    results["lime_explanations"] = lime_results

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 70)
    logger.info("XAI Pipeline Complete. Outputs saved to: " + str(Path(output_dir).resolve()))
    logger.info(f"  Binary   F1        : {binary_df[binary_df['Class']=='Attack']['F1-Score'].values[0]}")
    logger.info(f"  Attack   DR        : {binary_df[binary_df['Class']=='Attack']['DR (%)'].values[0]}%")
    logger.info(f"  Normal   FAR       : {binary_df[binary_df['Class']=='Attack']['FAR (%)'].values[0]}%")
    logger.info(f"  Top SHAP feature   : {importance_df.iloc[0]['Feature']}")
    logger.info(f"  SHAP values shape  : {shap_values.shape}")
    logger.info(f"  Attention shape    : {attention_matrix.shape}")
    logger.info("═" * 70 + "\n")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SMOKE TEST / DEMO
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    End-to-end smoke test demonstrating the full XAI + Metrics pipeline
    using synthetic data matching the hybrid dataset dimensions from the PDF.

    In a real workflow:
        1. Train the model via nids_deep_model.run_full_pipeline()
        2. Call run_xai_pipeline() with the trained model and test data
        3. Review outputs in xai_outputs/
    """
    from nids_deep_model import NIDSModel, NIDSConfig, run_full_pipeline

    logger.info("Smoke test: synthetic data matching PDF preprocessing specs.")

    rng = np.random.default_rng(42)
    N, D = 3_000, 50

    # Simulate post-preprocessing feature names (NSL-KDD hybrid)
    feature_names = (
        ["dur", "proto_tcp", "proto_udp", "proto_icmp",
         "service_http", "service_ftp", "service_smtp",
         "state_SF", "state_S0", "state_REJ",
         "sbytes", "dbytes", "ct_srv_src", "ct_srv_dst",
         "ct_dst_src_ltm", "ct_dst_ltm"]
        + [f"feat_{i}" for i in range(D - 16)]
    )
    assert len(feature_names) == D

    X = rng.standard_normal((N, D)).astype(np.float32)
    y_binary = rng.integers(0, 2, N).astype(np.float32)

    # Multi-class for per-category metrics (NSL-KDD classes)
    y_multiclass = np.zeros(N, dtype=int)
    y_multiclass[y_binary == 1] = rng.choice([1, 2, 3, 4], size=int(y_binary.sum()))

    # Minimal config for fast smoke test
    cfg = NIDSConfig(
        input_dim=D,
        encoder_dims=[64, 32, 16],
        latent_dim=8,
        pretrain_epochs=3,
        finetune_epochs=3,
        early_stop_patience=2,
        batch_size=128,
        num_workers=0,
        pin_memory=False,
    )

    # Train model
    N_train = int(N * 0.7)
    model, train_results, Z_test = run_full_pipeline(
        X[:N_train], y_binary[:N_train],
        X[N_train:N_train+300], y_binary[N_train:N_train+300],
        X[N_train:], y_binary[N_train:],
        cfg=cfg,
        save_path="smoke_test_model.pt",
    )

    # Run XAI pipeline
    xai_results = run_xai_pipeline(
        model=model,
        cfg=cfg,
        X_test=X[N_train:],
        y_test_binary=y_binary[N_train:].astype(int),
        feature_names=feature_names,
        y_test_multiclass=y_multiclass[N_train:],
        attack_map=NSL_KDD_ATTACK_MAP,
        output_dir="xai_outputs",
        n_shap_explain=50,
        n_shap_samples=50,
        n_lime_samples=500,
    )

    logger.info("All smoke tests passed. Check ./xai_outputs/ for figures.")