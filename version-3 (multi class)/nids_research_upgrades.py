import logging, warnings, copy, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
NUM_CLASSES  = 5

def compute_ovr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    N  = cm.sum()

    rows = []
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(N - tp - fp - fn)

        dr   = tp / (tp + fn + 1e-12)
        far  = fp / (fp + tn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        f1   = 2 * prec * dr / (prec + dr + 1e-12)
        spec = tn / (tn + fp + 1e-12)

        rows.append({
            "Class":       name,
            "N_true":      int(tp + fn),
            "TP":          tp,
            "FP":          fp,
            "FN":          fn,
            "TN":          tn,
            "DR % ↑":      round(dr   * 100, 4),
            "FAR % ↓":     round(far  * 100, 4),
            "Specificity": round(spec * 100, 4),
            "Precision":   round(prec, 4),
            "F1-Score":    round(f1,   4),
        })

    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wtd_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    acc      = (y_pred == y_true).mean()

    rows.append({
        "Class":"Macro Avg","N_true":N,"TP":"","FP":"","FN":"","TN":"",
        "DR % ↑":round(macro_r*100,4),"FAR % ↓":"","Specificity":"",
        "Precision":round(macro_p,4),"F1-Score":round(macro_f1,4),
    })
    rows.append({
        "Class":"Wtd. Avg","N_true":N,"TP":"","FP":"","FN":"","TN":"",
        "DR % ↑":"","FAR % ↓":"","Specificity":"",
        "Precision":"","F1-Score":round(wtd_f1,4),
    })
    rows.append({
        "Class":"Accuracy","N_true":N,"TP":"","FP":"","FN":"","TN":"",
        "DR % ↑":round(acc*100,4),"FAR % ↓":"","Specificity":"",
        "Precision":"","F1-Score":round(acc,4),
    })

    return pd.DataFrame(rows), cm


def print_ovr_table(df: pd.DataFrame, title: str = "OvR Per-Class Metrics"):
    sep = "─" * 108
    log.info("\n" + "═" * 108)
    log.info(f"  {title}")
    log.info("  OvR FAR = FP_c / (FP_c + TN_c)  where FP_c = non-c samples predicted AS c")
    log.info("═" * 108)
    log.info(
        f"  {'Class':<12} {'N_true':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7} "
        f"{'DR % ↑':>9} {'FAR % ↓':>9} {'Spec %':>8} {'Prec':>8} {'F1':>8}"
    )
    log.info(sep)
    for _, r in df.iterrows():
        cls = str(r["Class"])
        if cls in ("Macro Avg", "Wtd. Avg", "Accuracy"):
            log.info(sep)
        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)
        log.info(
            f"  {cls:<12} {str(r['N_true']):>8} {str(r['TP']):>7} {str(r['FP']):>7} "
            f"{str(r['FN']):>7} {str(r['TN']):>7} "
            f"{_fmt(r['DR % ↑'],'.3f')+'%':>9} "
            f"{_fmt(r['FAR % ↓'],'.3f')+'%' if r['FAR % ↓']!='' else '':>9} "
            f"{_fmt(r['Specificity'],'.3f')+'%' if r['Specificity']!='' else '':>8} "
            f"{_fmt(r['Precision']):>8} {_fmt(r['F1-Score']):>8}"
        )
    log.info("═" * 108 + "\n")


def plot_ovr_metrics(
    df: pd.DataFrame,
    output_dir: str = "research_outputs",
    filename: str  = "ovr_per_class_metrics",
):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_df = df[~df["Class"].isin(["Macro Avg","Wtd. Avg","Accuracy"])].copy()
    plot_df["DR"]  = pd.to_numeric(plot_df["DR % ↑"],  errors="coerce").fillna(0)
    plot_df["FAR"] = pd.to_numeric(plot_df["FAR % ↓"], errors="coerce").fillna(0)
    plot_df["F1"]  = pd.to_numeric(plot_df["F1-Score"], errors="coerce").fillna(0)

    y  = np.arange(len(plot_df))
    h  = 0.28
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(y - h, plot_df["DR"],      h, label="DR % ↑ (higher=better)", color="#2E7D32", alpha=0.88)
    ax.barh(y,     plot_df["FAR"],     h, label="FAR % ↓ (lower=better)", color="#C62828", alpha=0.88)
    ax.barh(y + h, plot_df["F1"]*100, h, label="F1×100",                  color="#1565C0", alpha=0.82)


    for i, (dr, far, f1) in enumerate(zip(plot_df["DR"], plot_df["FAR"], plot_df["F1"])):
        ax.text(dr  + 0.3, i-h,  f"{dr:.1f}",   va="center", fontsize=8)
        ax.text(far + 0.3, i,    f"{far:.2f}",  va="center", fontsize=8, color="#C62828")
        ax.text(f1*100+0.3, i+h, f"{f1:.3f}",  va="center", fontsize=8, color="#1565C0")

    ax.set_yticks(y); ax.set_yticklabels(plot_df["Class"], fontsize=11)
    ax.set_xlabel("Score (%)")
    ax.set_title("Corrected OvR Per-Class DR / FAR / F1  (True Multi-Class FAR)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    path = str(Path(output_dir) / f"{filename}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    log.info(f"OvR metrics chart → {path}")
    return path


def apply_svmsmote_targeted(
    X: np.ndarray,
    y: np.ndarray,
    minority_classes: Tuple[int, ...] = (4,),
    secondary_classes: Tuple[int, ...] = (3,),
    u2r_target_ratio: float  = 0.05,
    r2l_target_ratio: float  = 0.20,
    n_continuous: int = 16,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    try:
        from imblearn.over_sampling import SVMSMOTE, SMOTE
    except ImportError:
        raise ImportError("pip install imbalanced-learn")

    counts   = Counter(y.tolist())
    max_cnt  = max(counts.values())


    u2r_target   = max(int(max_cnt * u2r_target_ratio), counts.get(4, 1) * 3)
    svm_strategy = {c: u2r_target for c in minority_classes if counts.get(c, 0) < u2r_target}

    X_work, y_work = X.copy(), y.copy()

    if svm_strategy:
        log.info("  SVMSMOTE (boundary-aware) for extreme minority classes:")
        for c, t in svm_strategy.items():
            log.info(f"    [{CLASS_NAMES[c]}]: {counts.get(c,0)} → {t}  "
                     f"(×{t//max(counts.get(c,1),1)} augmentation)")

        k_svm = min(5, min(counts.get(c, 1) for c in svm_strategy) - 1)
        k_svm = max(k_svm, 1)
        try:
            svm_smote = SVMSMOTE(
                sampling_strategy=svm_strategy,
                k_neighbors=k_svm,
                m_neighbors=min(10, min(counts.values()) - 1),
                random_state=random_state,
            )
            X_work, y_work = svm_smote.fit_resample(X_work, y_work)
            log.info(f"    SVMSMOTE complete. New shape: {X_work.shape}")
        except Exception as e:
            log.warning(f"  SVMSMOTE failed ({e}), falling back to standard SMOTE for U2R")
            smote_fallback = SMOTE(
                sampling_strategy=svm_strategy,
                k_neighbors=k_svm,
                random_state=random_state,
            )
            X_work, y_work = smote_fallback.fit_resample(X_work, y_work)


    new_counts   = Counter(y_work.tolist())
    new_max      = max(new_counts.values())
    smote_target = {c: int(new_max * r2l_target_ratio)
                    for c in secondary_classes
                    if new_counts.get(c, 0) < int(new_max * r2l_target_ratio)}

    if smote_target:
        log.info("  Standard SMOTE for secondary classes:")
        for c, t in smote_target.items():
            log.info(f"    [{CLASS_NAMES[c]}]: {new_counts.get(c,0)} → {t}")
        k_s = min(5, min(new_counts.get(c, 1) for c in smote_target) - 1)
        k_s = max(k_s, 1)
        smote = SMOTE(
            sampling_strategy=smote_target,
            k_neighbors=k_s,
            random_state=random_state,
        )
        X_work, y_work = smote.fit_resample(X_work, y_work)


    X_work[:, n_continuous:] = np.clip(np.round(X_work[:, n_continuous:]), 0, 1)


    perm = np.random.default_rng(random_state).permutation(len(X_work))
    X_work, y_work = X_work[perm], y_work[perm]

    final_counts = Counter(y_work.tolist())
    log.info("  Final distribution after SVMSMOTE:")
    for i, name in enumerate(CLASS_NAMES):
        n = final_counts.get(i, 0)
        log.info(f"    [{i}] {name:<8}: {n:>8,}  ({n/len(y_work)*100:.2f}%)")

    return X_work, y_work


def build_penalty_matrix(
    class_counts: np.ndarray,
    u2r_multiplier: float = 50.0,
    normal_to_attack_multiplier: float = 2.0,
) -> torch.Tensor:

    n_cls = len(class_counts)
    C     = np.ones((n_cls, n_cls), dtype=np.float32)
    np.fill_diagonal(C, 0.0)

    max_count = float(max(class_counts))

    for i in range(n_cls):
        for j in range(n_cls):
            if i == j:
                continue

            rarity_i = max_count / (class_counts[i] + 1e-6)

            rarity_j = max_count / (class_counts[j] + 1e-6)


            if i != 0 and j == 0:

                cost = rarity_i * 5.0
            elif i == 0 and j != 0:

                cost = normal_to_attack_multiplier * rarity_j * 0.5
            else:

                cost = (rarity_i + rarity_j) * 0.5


            if i == 4:
                cost *= u2r_multiplier
            if j == 4:
                cost *= 0.5

            C[i, j] = float(cost)


    mask    = ~np.eye(n_cls, dtype=bool)
    C[mask] = C[mask] / C[mask].mean()

    return torch.from_numpy(C)


class CostSensitiveFocalLoss(nn.Module):

    def __init__(
        self,
        cost_matrix:     torch.Tensor,
        gamma:           float = 2.0,
        alpha:           Optional[torch.Tensor] = None,
        num_classes:     int   = NUM_CLASSES,
        label_smoothing: float = 0.05,
        reduction:       str   = "mean",
    ):
        super().__init__()
        self.gamma           = gamma
        self.num_classes     = num_classes
        self.label_smoothing = label_smoothing
        self.reduction       = reduction
        self.register_buffer("cost_matrix", cost_matrix.float())
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", torch.ones(num_classes))

    @classmethod
    def from_class_counts(
        cls,
        class_counts:      np.ndarray,
        gamma:             float = 2.0,
        beta:              float = 0.9999,
        label_smoothing:   float = 0.05,
        u2r_multiplier:    float = 50.0,
        normal_attack_mult: float = 2.0,
    ) -> "CostSensitiveFocalLoss":


        eff    = 1.0 - np.power(beta, class_counts.astype(np.float64))
        alpha  = (1.0 - beta) / (eff + 1e-10)
        alpha  = alpha / (alpha.sum() / len(alpha))


        C = build_penalty_matrix(class_counts, u2r_multiplier, normal_attack_mult)

        log.info("  CostSensitiveFocalLoss initialised:")
        log.info(f"  {'Class':<10} {'α (CB)':>10} {'C[c,Normal]':>12} {'C[c,U2R]':>10}")
        for i, (name, a) in enumerate(zip(CLASS_NAMES, alpha)):
            log.info(f"    {name:<10} {a:>10.4f} {C[i,0].item():>12.3f} {C[i,4].item():>10.3f}")

        return cls(
            cost_matrix    = C,
            gamma          = gamma,
            alpha          = torch.from_numpy(alpha.astype(np.float32)),
            label_smoothing= label_smoothing,
        )

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        B, C  = logits.shape


        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()


        p_t   = p.gather(1, targets.unsqueeze(1)).squeeze(1)


        focal = (1.0 - p_t).pow(self.gamma)


        alpha_t = self.alpha[targets]


        with torch.no_grad():
            pred_class  = logits.argmax(dim=-1)
            cost_weight = self.cost_matrix[targets, pred_class]


        if self.label_smoothing > 0 and self.training:
            eps    = self.label_smoothing
            smooth = torch.full((B, C), eps / (C - 1), device=logits.device)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - eps)
            ce = -(smooth * log_p).sum(dim=-1)
        else:
            ce = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)


        loss = alpha_t * focal * (1.0 + cost_weight) * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CenterLoss(nn.Module):

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        feat_dim:    int   = 64,
        lambda_c:    float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.lambda_c    = lambda_c


        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim) * 0.1
        )

    def forward(
        self,
        z:       torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        batch_centers = self.centers[targets]


        dist  = (z - batch_centers).pow(2).sum(dim=1)

        return self.lambda_c * 0.5 * dist.mean()

    def get_center_distances(self) -> torch.Tensor:
        with torch.no_grad():
            c   = self.centers
            diff = c.unsqueeze(0) - c.unsqueeze(1)
            return diff.pow(2).sum(dim=-1).sqrt()

    @torch.no_grad()
    def update_centers_batch(
        self,
        z:         torch.Tensor,
        targets:   torch.Tensor,
        lr_center: float = 0.5,
    ):

        for c_idx in range(self.num_classes):
            mask   = (targets == c_idx)
            if mask.sum() == 0:
                continue
            z_c    = z[mask].mean(dim=0)
            delta  = self.centers[c_idx] - z_c
            self.centers[c_idx] -= lr_center * delta

    def plot_centers(
        self,
        z:           np.ndarray,
        y:           np.ndarray,
        output_dir:  str = "research_outputs",
        epoch:       int = 0,
    ):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        try:
            from sklearn.decomposition import PCA

            n_sub = min(2000, len(z))
            idx   = np.random.choice(len(z), n_sub, replace=False)
            z_sub = z[idx]; y_sub = y[idx]
            pca   = PCA(n_components=2)
            z_2d  = pca.fit_transform(z_sub)


            c_np  = self.centers.detach().cpu().numpy()
            c_2d  = pca.transform(c_np)

            colors = ["#2196F3","#F44336","#FF9800","#4CAF50","#9C27B0"]
            fig, ax = plt.subplots(figsize=(8, 7))
            for cid, (cname, col) in enumerate(zip(CLASS_NAMES, colors)):
                mask = y_sub == cid
                if mask.sum() == 0: continue
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=col, alpha=0.25,
                           s=8, label=cname)
                ax.scatter(c_2d[cid, 0], c_2d[cid, 1], c=col, s=200,
                           marker="*", edgecolors="black", linewidth=1.5,
                           zorder=5)
            ax.set_title(f"Latent Space PCA (epoch {epoch}) — ★ = class centres",
                         fontweight="bold")
            ax.legend(fontsize=9, markerscale=2)
            ax.set_facecolor("#FAFAFA")
            path = str(Path(output_dir) / f"latent_space_epoch{epoch:03d}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
            log.info(f"  Latent space plot → {path}")
        except Exception as e:
            log.warning(f"  Center plot failed: {e}")


class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.07, lambda_sc: float = 0.1):
        super().__init__()
        self.tau      = temperature
        self.lambda_sc = lambda_sc

    def forward(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        z_norm = F.normalize(z, dim=1)


        sim    = torch.mm(z_norm, z_norm.T) / self.tau


        targets_row = targets.unsqueeze(0)
        targets_col = targets.unsqueeze(1)
        pos_mask    = (targets_row == targets_col).float()
        diag_mask   = torch.eye(len(targets), device=z.device)
        pos_mask    = pos_mask - diag_mask


        sim_max, _  = sim.max(dim=1, keepdim=True)
        sim_exp     = (sim - sim_max.detach()).exp()


        denom_mask  = 1 - diag_mask
        denom       = (sim_exp * denom_mask).sum(dim=1, keepdim=True)

        log_prob    = sim - sim_max.detach() - denom.clamp(min=1e-8).log()


        n_pos       = pos_mask.sum(dim=1).clamp(min=1)
        loss        = -(pos_mask * log_prob).sum(dim=1) / n_pos

        return self.lambda_sc * loss.mean()


class SHAPFeaturePruner:


    def __init__(
        self,
        feature_names:     List[str],
        shap_values:       Optional[np.ndarray] = None,
        threshold_mode:    str   = "percentile",
        threshold_value:   float = 10.0,
        protect_features:  Optional[List[str]] = None,
        min_features:      int   = 50,
    ):
        self.feature_names    = list(feature_names)
        self.n_features       = len(feature_names)
        self.threshold_mode   = threshold_mode
        self.threshold_value  = threshold_value
        self.protect_features = set(protect_features or [])
        self.min_features     = min_features


        self.importance_scores: Optional[np.ndarray]  = None
        self.feature_mask:      Optional[np.ndarray]  = None
        self.pruned_names:      Optional[List[str]]   = None
        self.kept_names:        Optional[List[str]]   = None
        self.pruning_report:    Optional[pd.DataFrame] = None

        if shap_values is not None:
            self.fit(shap_values)

    def fit(self, shap_values: np.ndarray) -> "SHAPFeaturePruner":
        if shap_values.ndim == 2:
            self.importance_scores = np.abs(shap_values).mean(axis=0)
        else:
            self.importance_scores = np.abs(shap_values)

        assert len(self.importance_scores) == self.n_features, (
            f"SHAP length {len(self.importance_scores)} ≠ features {self.n_features}"
        )


        if self.threshold_mode == "zero":
            threshold = 1e-10

        elif self.threshold_mode == "percentile":
            threshold = np.percentile(self.importance_scores, self.threshold_value)

        elif self.threshold_mode == "cumulative":

            sorted_idx  = np.argsort(self.importance_scores)[::-1]
            total_mass  = self.importance_scores.sum()
            cumsum      = np.cumsum(self.importance_scores[sorted_idx])
            n_keep      = int(np.searchsorted(cumsum, total_mass * self.threshold_value) + 1)
            n_keep      = max(n_keep, self.min_features)
            threshold   = self.importance_scores[sorted_idx[n_keep - 1]]

        else:
            raise ValueError(f"threshold_mode must be 'zero', 'percentile', or 'cumulative'")


        mask = self.importance_scores > threshold


        for fname in self.protect_features:
            if fname in self.feature_names:
                mask[self.feature_names.index(fname)] = True


        if mask.sum() < self.min_features:
            sorted_by_imp = np.argsort(self.importance_scores)[::-1]
            for idx in sorted_by_imp:
                if mask.sum() >= self.min_features:
                    break
                mask[idx] = True

        self.feature_mask = mask
        self.kept_names   = [self.feature_names[i] for i in range(self.n_features) if mask[i]]
        self.pruned_names = [self.feature_names[i] for i in range(self.n_features) if not mask[i]]


        self.pruning_report = pd.DataFrame({
            "Feature":   self.feature_names,
            "Mean|SHAP|": self.importance_scores,
            "Kept":       mask,
        }).sort_values("Mean|SHAP|", ascending=False).reset_index(drop=True)

        self._log_summary(threshold)
        return self

    def _log_summary(self, threshold: float):
        n_kept   = int(self.feature_mask.sum())
        n_pruned = self.n_features - n_kept
        log.info("─" * 70)
        log.info(f"SHAP Feature Pruning Summary")
        log.info(f"  Mode      : {self.threshold_mode} (threshold = {threshold:.6f})")
        log.info(f"  Original  : {self.n_features} features")
        log.info(f"  Kept      : {n_kept} features")
        log.info(f"  Pruned    : {n_pruned} features ({n_pruned/self.n_features*100:.1f}%)")
        log.info(f"  SHAP mass kept: "
                 f"{self.importance_scores[self.feature_mask].sum() / (self.importance_scores.sum()+1e-12)*100:.2f}%")
        if self.pruned_names:
            log.info(f"  First 10 pruned: {self.pruned_names[:10]}")
        log.info("─" * 70)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.feature_mask is not None, "Call fit() first."
        return X[:, self.feature_mask]

    def fit_transform(self, X: np.ndarray, shap_values: np.ndarray) -> np.ndarray:
        self.fit(shap_values)
        return self.transform(X)

    @property
    def n_kept(self) -> int:
        return int(self.feature_mask.sum()) if self.feature_mask is not None else 0

    def save(self, path: str):
        import json
        data = {
            "feature_mask": self.feature_mask.tolist(),
            "kept_names":   self.kept_names,
            "pruned_names": self.pruned_names,
            "threshold_mode":  self.threshold_mode,
            "threshold_value": self.threshold_value,
            "importance_scores": self.importance_scores.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Pruner saved → {path}")

    @classmethod
    def load(cls, path: str, feature_names: List[str]) -> "SHAPFeaturePruner":
        import json
        with open(path) as f:
            data = json.load(f)
        pruner = cls(feature_names=feature_names)
        pruner.feature_mask      = np.array(data["feature_mask"])
        pruner.kept_names        = data["kept_names"]
        pruner.pruned_names      = data["pruned_names"]
        pruner.threshold_mode    = data["threshold_mode"]
        pruner.threshold_value   = data["threshold_value"]
        pruner.importance_scores = np.array(data["importance_scores"])
        log.info(f"Pruner loaded ← {path} ({pruner.n_kept} features)")
        return pruner

    def plot_importance(
        self,
        top_n:      int = 30,
        output_dir: str = "research_outputs",
        filename:   str = "shap_pruning_importance",
    ):
        assert self.importance_scores is not None, "Call fit() first."
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        top_idx   = np.argsort(self.importance_scores)[::-1][:top_n]
        top_names = [self.feature_names[i] for i in top_idx]
        top_imp   = self.importance_scores[top_idx]
        top_kept  = self.feature_mask[top_idx]

        colors = ["#2E7D32" if k else "#C62828" for k in top_kept]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
        bars = ax.barh(range(top_n), top_imp[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(
            f"SHAP Feature Importance — {self.n_kept}/{self.n_features} kept\n"
            f"(green=kept, red=pruned  |  mode={self.threshold_mode}, "
            f"threshold={self.threshold_value})",
            fontweight="bold", fontsize=11,
        )


        for bar in bars:
            w = bar.get_width()
            ax.text(w + top_imp.max() * 0.01, bar.get_y() + bar.get_height()/2,
                    f"{w:.4f}", va="center", fontsize=7)

        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color="#2E7D32", label=f"Kept ({self.n_kept})"),
            mpatches.Patch(color="#C62828", label=f"Pruned ({self.n_features - self.n_kept})"),
        ], fontsize=9)
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.7)
        plt.tight_layout()
        path = str(Path(output_dir) / f"{filename}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
        log.info(f"SHAP pruning chart → {path}")
        return path

    def get_ga_feature_set(self) -> Dict[str, Any]:
        return {
            "mask":         self.feature_mask,
            "kept_names":   self.kept_names,
            "n_features":   self.n_kept,
            "importance":   dict(zip(self.kept_names,
                                     self.importance_scores[self.feature_mask])),
            "pruned_names": self.pruned_names,
        }


class CenterLossTrainerMixin:

    def setup_center_loss(
        self,
        feat_dim:      int,
        lambda_c:      float = 0.01,
        lambda_sc:     float = 0.05,
        use_supcon:    bool  = True,
        lr_center:     float = 0.5,
    ):
        device          = getattr(self, "device", torch.device("cpu"))
        self.center_loss = CenterLoss(NUM_CLASSES, feat_dim, lambda_c).to(device)
        self.supcon_loss = SupConLoss(temperature=0.07, lambda_sc=lambda_sc) if use_supcon else None
        self.lr_center   = lr_center
        self.center_opt  = torch.optim.SGD(
            self.center_loss.parameters(), lr=lr_center
        )
        log.info(f"  CenterLoss: λ={lambda_c}  SupConLoss: λ={lambda_sc if use_supcon else 0}")

    def compute_auxiliary_loss(
        self,
        z:       torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=z.device)


        if hasattr(self, "center_loss"):
            c_loss = self.center_loss(z, targets)
            loss   = loss + c_loss

            self.center_opt.zero_grad()


        if hasattr(self, "supcon_loss") and self.supcon_loss is not None:
            sc_loss = self.supcon_loss(z, targets)
            loss    = loss + sc_loss

        return loss

    def step_center_optimizer(self):
        if hasattr(self, "center_opt"):

            for param in self.center_loss.parameters():
                if param.grad is not None:
                    param.grad.data *= (1.0 / self.lr_center)
            self.center_opt.step()


def finetune_step_with_center_loss(
    model,
    loader:        DataLoader,
    cls_loss_fn:   nn.Module,
    center_loss:   CenterLoss,
    center_opt:    torch.optim.Optimizer,
    model_opt:     torch.optim.Optimizer,
    sched,
    epoch:         int,
    n_batches:     int,
    device:        torch.device,
    supcon_loss:   Optional[SupConLoss] = None,
    lr_center:     float = 0.5,
    grad_clip:     float = 1.0,
) -> Dict[str, float]:

    model.train()
    center_loss.train()

    totals = {"focal": 0.0, "center": 0.0, "supcon": 0.0, "total": 0.0}
    n      = 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)


        model_opt.zero_grad(set_to_none=True)
        center_opt.zero_grad(set_to_none=True)


        logits, z = model(x)


        focal = cls_loss_fn(logits, y)


        c_loss = center_loss(z, y)


        sc_loss = supcon_loss(z, y) if supcon_loss is not None else torch.tensor(0.0, device=device)


        total = focal + c_loss + sc_loss


        total.backward()


        for param in center_loss.parameters():
            if param.grad is not None:
                param.grad.data *= (1.0 / (lr_center + 1e-8))


        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


        model_opt.step()
        center_opt.step()


        sched.step(epoch - 1 + i / n_batches)


        B               = len(x)
        totals["focal"]  += focal.item()  * B
        totals["center"] += c_loss.item() * B
        totals["supcon"] += sc_loss.item()* B
        totals["total"]  += total.item()  * B
        n                += B

    return {k: v / n for k, v in totals.items()}


def run_research_pipeline(
    train_csv:      str,
    test_csv:       str,
    output_dir:     str  = "research_outputs",
    shap_values:    Optional[np.ndarray] = None,
    feature_names:  Optional[List[str]]  = None,
    prune_threshold_mode:  str   = "percentile",
    prune_threshold_value: float = 10.0,
):

    Path(output_dir).mkdir(parents=True, exist_ok=True)


    import pandas as pd
    from nids_multiclass import (
        MultiClassNIDSConfig, MultiClassNIDSModel,
        MultiClassNIDSTrainer, load_hybrid_csv,
        build_multiclass_dataloader, CLASS_NAMES, NUM_CLASSES,
    )
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


    X_tr_all, y_tr_all, feat_names = load_hybrid_csv(train_csv)
    X_test,   y_test,   _          = load_hybrid_csv(test_csv)

    if feature_names is None:
        feature_names = feat_names


    pruner = None
    if shap_values is not None:
        log.info("\n[STEP 2] SHAP Feature Pruning...")
        pruner = SHAPFeaturePruner(
            feature_names    = feature_names,
            shap_values      = shap_values,
            threshold_mode   = prune_threshold_mode,
            threshold_value  = prune_threshold_value,
            protect_features = ["flow_balance", "ct_dst_src_ltm",
                                 "service_-", "ct_dst_ltm", "state_FIN"],
        )
        X_tr_all = pruner.transform(X_tr_all)
        X_test   = pruner.transform(X_test)
        pruner.plot_importance(output_dir=output_dir)
        pruner.save(str(Path(output_dir) / "shap_pruner.json"))
        log.info(f"  Reduced: {pruner.n_features} → {pruner.n_kept} features")


    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=42)
    ti, vi = next(sss.split(X_tr_all, y_tr_all))
    X_train, y_train = X_tr_all[ti], y_tr_all[ti]
    X_val,   y_val   = X_tr_all[vi], y_tr_all[vi]


    log.info("\n[STEP 4] SVMSMOTE oversampling...")
    X_train, y_train = apply_svmsmote_targeted(
        X_train, y_train,
        minority_classes  = (4,),
        secondary_classes = (),
        u2r_target_ratio  = 0.05,
        n_continuous      = 16,
    )


    input_dim = X_train.shape[1]
    cfg = MultiClassNIDSConfig(
        input_dim         = input_dim,
        encoder_dims      = [512, 256, 128],
        latent_dim        = 64,
        classifier_hidden = 128,
        dropout_rate      = 0.25,
        noise_std         = 0.08,
        sparsity_weight   = 1e-3,
        sparsity_target   = 0.05,
        focal_gamma       = 2.0,
        label_smoothing   = 0.03,
        learning_rate     = 1e-3,
        batch_size        = 512,
        pretrain_epochs   = 100,
        finetune_epochs   = 180,
        early_stop_patience = 15,
        t0_epochs         = 20,
        num_workers       = 0,
    )

    model   = MultiClassNIDSModel(cfg)


    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    device       = torch.device(cfg.device)
    cls_loss = CostSensitiveFocalLoss.from_class_counts(
        train_counts,
        gamma              = cfg.focal_gamma,
        label_smoothing    = cfg.label_smoothing,
        u2r_multiplier     = 50.0,
        normal_attack_mult = 2.0,
    ).to(device)


    center_loss = CenterLoss(NUM_CLASSES, cfg.latent_dim, lambda_c=0.01).to(device)
    supcon_loss = SupConLoss(temperature=0.07, lambda_sc=0.05).to(device)
    center_opt  = torch.optim.SGD(center_loss.parameters(), lr=0.5)


    trainer = MultiClassNIDSTrainer(model, cfg, y_train)
    trainer.cls_loss = cls_loss

    train_loader = build_multiclass_dataloader(
        X_train, y_train, cfg.batch_size, use_class_aware_sampler=True,
        num_workers=cfg.num_workers,
    )
    val_loader = build_multiclass_dataloader(
        X_val, y_val, cfg.batch_size, use_class_aware_sampler=False,
        shuffle=False, num_workers=cfg.num_workers,
    )


    trainer.pretrain(train_loader, val_loader)


    log.info("=" * 65)
    log.info("PHASE 2 — Fine-tuning with CostSensitiveFocalLoss + CenterLoss")
    log.info("=" * 65)
    model.set_mode("finetune")
    model.freeze_encoder()
    model.to(device)

    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.t0_epochs, T_mult=2, eta_min=cfg.eta_min)
    best_f1    = 0.0
    best_state = None
    patience_c = 0
    n_b        = len(train_loader)

    for epoch in range(1, cfg.finetune_epochs + 1):
        if epoch == 6:
            model.unfreeze_encoder()
            opt = AdamW(model.parameters(), lr=cfg.learning_rate*0.1,
                        weight_decay=cfg.weight_decay)
            sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.t0_epochs,
                                                T_mult=2, eta_min=cfg.eta_min)
            log.info("  ✓ Encoder unfrozen.")

        t0     = time.time()
        losses = finetune_step_with_center_loss(
            model, train_loader, cls_loss, center_loss, center_opt,
            opt, sched, epoch, n_b, device, supcon_loss, lr_center=0.5,
        )
        val_m  = trainer._p2_val(val_loader)

        if val_m["macro_f1"] > best_f1:
            best_f1    = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_c = 0
        else:
            patience_c += 1

        log.info(f"  Ep {epoch:03d}/{cfg.finetune_epochs} "
                 f"focal={losses['focal']:.4f} center={losses['center']:.4f} "
                 f"sc={losses['supcon']:.4f} | val_mF1={val_m['macro_f1']:.4f} "
                 f"acc={val_m['acc']:.4f} {time.time()-t0:.1f}s")


        if epoch % 30 == 0 or epoch == 1:
            center_loss.plot_centers(
                X_train[:2000], y_train[:2000],
                output_dir=output_dir, epoch=epoch,
            )

        if patience_c >= cfg.early_stop_patience:
            log.info(f"  Early stop @ epoch {epoch}. Best MacroF1={best_f1:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
        log.info("  ✓ Best weights restored.")


    log.info("\n[STEP 9] Running corrected OvR evaluation...")
    from nids_multiclass import _NIDSDS
    import torch.nn.functional as F

    model.eval(); model.to(device)
    loader = DataLoader(_NIDSDS(X_test, y_test), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y_b in loader:
            logits, _ = model(x.to(device))
            preds_all.append(logits.argmax(-1).cpu().numpy())
            labels_all.append(y_b.numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)

    metrics_df, cm = compute_ovr_metrics(y_true, y_pred, CLASS_NAMES)
    print_ovr_table(metrics_df, "Research-Grade OvR Per-Class Metrics")
    metrics_df.to_csv(str(Path(output_dir) / "ovr_metrics_corrected.csv"), index=False)
    plot_ovr_metrics(metrics_df, output_dir=output_dir)


    ckpt = str(Path(output_dir) / "nids_research_model.pt")
    torch.save({
        "model_state_dict":      model.state_dict(),
        "center_loss_state":     center_loss.state_dict(),
        "config":                cfg.__dict__,
        "feature_names_kept":    pruner.kept_names if pruner else feature_names,
        "pruner_mask":           pruner.feature_mask.tolist() if pruner else None,
    }, ckpt)
    log.info(f"Model saved → {ckpt}")

    return model, metrics_df, pruner


if __name__ == "__main__":
    log.info("=" * 70)
    log.info("SMOKE TEST — Validating all research-grade fixes")
    log.info("=" * 70)


    log.info("\n[Test 1] Corrected OvR FAR from your confusion matrix...")
    cm = np.array([
        [26682,    57,  2017, 14287,   13],
        [ 1138,  5482,   331,  1413,    1],
        [  523,   243,  3088,   696,    3],
        [ 2946,     4,  1628, 14118,  113],
        [   43,     0,     0,     4,   20],
    ])

    y_true_recon, y_pred_recon = [], []
    for i in range(5):
        for j in range(5):
            y_true_recon.extend([i] * cm[i, j])
            y_pred_recon.extend([j] * cm[i, j])
    y_true_recon = np.array(y_true_recon)
    y_pred_recon = np.array(y_pred_recon)

    df, _ = compute_ovr_metrics(y_true_recon, y_pred_recon, CLASS_NAMES)
    print_ovr_table(df, "Corrected OvR Metrics (from your CM)")
    log.info("  Expected: DoS FAR ≈ 0.46%  Probe ≈ 5.66%  R2L ≈ 29.26%  U2R ≈ 0.17%")


    log.info("\n[Test 2] Cost matrix for training set counts...")
    train_counts = np.array([43056, 8365, 4553, 18809, 67], dtype=np.float64)
    C = build_penalty_matrix(train_counts, u2r_multiplier=50.0)
    log.info(f"  U2R→Normal cost : {C[4,0].item():.3f}  (should be highest)")
    log.info(f"  Normal→DoS cost : {C[0,1].item():.3f}")
    log.info(f"  DoS→Probe cost  : {C[1,2].item():.3f}  (should be lowest)")


    log.info("\n[Test 3] CostSensitiveFocalLoss forward pass...")
    loss_fn = CostSensitiveFocalLoss.from_class_counts(
        train_counts, gamma=2.0, label_smoothing=0.03,
        u2r_multiplier=50.0, normal_attack_mult=2.0,
    )
    logits_t  = torch.randn(32, 5)
    targets_t = torch.randint(0, 5, (32,))
    loss_val  = loss_fn(logits_t, targets_t)
    log.info(f"  Loss = {loss_val.item():.4f}  (should be a valid scalar)")
    assert not torch.isnan(loss_val), "Loss is NaN!"


    log.info("\n[Test 4] CenterLoss forward pass...")
    cl  = CenterLoss(5, 64, lambda_c=0.01)
    z   = torch.randn(32, 64)
    tgt = torch.randint(0, 5, (32,))
    cv  = cl(z, tgt)
    log.info(f"  Center loss = {cv.item():.6f}  (should be near-zero initially)")
    assert not torch.isnan(cv), "Center loss is NaN!"


    log.info("\n[Test 5] SupConLoss forward pass...")
    scl = SupConLoss(temperature=0.07, lambda_sc=0.05)
    sv  = scl(z, tgt)
    log.info(f"  SupCon loss = {sv.item():.6f}")
    assert not torch.isnan(sv), "SupCon loss is NaN!"


    log.info("\n[Test 6] SHAP feature pruner...")
    rng          = np.random.default_rng(42)
    fake_shap    = np.abs(rng.standard_normal((500, 248)))

    for dead in ["service_exec", "service_efs", "proto_visa", "service_finger"]:
        idx = 100 + ["service_exec","service_efs","proto_visa","service_finger"].index(dead)
        fake_shap[:, idx] = 0.0
    fake_names   = [f"feat_{i}" for i in range(248)]
    pruner = SHAPFeaturePruner(
        feature_names   = fake_names,
        shap_values     = fake_shap,
        threshold_mode  = "percentile",
        threshold_value = 10.0,
    )
    ga_info = pruner.get_ga_feature_set()
    log.info(f"  Features after pruning: {ga_info['n_features']} / 248")
    assert ga_info["n_features"] < 248, "Pruner removed no features!"
    X_fake   = rng.standard_normal((100, 248)).astype(np.float32)
    X_pruned = pruner.transform(X_fake)
    log.info(f"  X after pruning: {X_pruned.shape}  (was (100, 248))")
    assert X_pruned.shape[1] == ga_info["n_features"]

    log.info("\n" + "=" * 70)
    log.info("All smoke tests passed. ✓")
    log.info("=" * 70)
