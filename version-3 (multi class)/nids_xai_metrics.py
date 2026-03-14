import os, warnings, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import shap
from lime import lime_tabular

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

PALETTE = {
    "normal":   "#4CAF50", "attack": "#F44336",
    "shap_pos": "#FF6B35", "shap_neg": "#1E88E5",
    "accent":   "#9C27B0", "grid": "#EEEEEE", "text": "#212121",
}

CLASS_NAMES_5 = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def compute_binary_cyber_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    if class_names is None:
        class_names = ["Normal", "Attack"]
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        dr = tp / (tp + fn + 1e-9)
        far = fp / (fp + tn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1 = 2 * prec * dr / (prec + dr + 1e-9)
        row = {
            "Class": name,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "DR (%)": round(dr * 100, 4),
            "FAR (%)": round(far * 100, 4),
            "Precision": round(prec, 4),
            "F1-Score": round(f1, 4),
        }
        if y_prob is not None and n == 2:
            prob_c = y_prob if i == 1 else 1 - y_prob
            row["AUC-ROC"] = round(roc_auc_score((y_true == i).astype(int), prob_c), 4)
        rows.append(row)
    rows.append({
        "Class": "Macro Avg",
        "TP": "",
        "FP": "",
        "FN": "",
        "TN": "",
        "DR (%)": round(np.mean([r["DR (%)"] for r in rows]), 4),
        "FAR (%)": round(np.mean([r["FAR (%)"] for r in rows]), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "AUC-ROC": "" if y_prob is None else round(roc_auc_score(y_true, y_prob), 4),
    })
    return pd.DataFrame(rows)


def compute_multiclass_cyber_metrics(
    y_true_mc: np.ndarray,
    y_pred_bin: np.ndarray,
    y_prob_bin: Optional[np.ndarray] = None,
    attack_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    if attack_map is None:
        attack_map = {n: i for i, n in enumerate(CLASS_NAMES_5)}
    inv = {v: k for k, v in attack_map.items()}
    normal_mask = y_true_mc == 0
    fp_g = ((y_pred_bin == 1) & normal_mask).sum()
    tn_g = ((y_pred_bin == 0) & normal_mask).sum()
    far_g = fp_g / (fp_g + tn_g + 1e-9)
    rows = []
    for cid, cname in sorted(inv.items()):
        if cid == 0:
            continue
        mask = y_true_mc == cid
        if not mask.any():
            continue
        tp = ((y_pred_bin == 1) & mask).sum()
        fn = ((y_pred_bin == 0) & mask).sum()
        dr = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp_g + 1e-9)
        f1 = 2 * prec * dr / (prec + dr + 1e-9)
        rows.append({
            "Attack Category": cname,
            "N Samples": int(mask.sum()),
            "TP": int(tp),
            "FN": int(fn),
            "DR (%)": round(dr * 100, 4),
            "FAR (%)": round(far_g * 100, 4),
            "Precision": round(prec, 4),
            "F1-Score": round(f1, 4),
        })
    n_norm = normal_mask.sum()
    rows.insert(0, {
        "Attack Category": "Normal (Baseline)",
        "N Samples": int(n_norm),
        "TP": int(tn_g),
        "FN": int(fp_g),
        "DR (%)": round(tn_g / (tn_g + fp_g + 1e-9) * 100, 4),
        "FAR (%)": "—",
        "Precision": "—",
        "F1-Score": round(
            f1_score((~normal_mask).astype(int), (y_pred_bin == 0).astype(int), zero_division=0), 4
        ),
    })
    return pd.DataFrame(rows)


def print_metrics_table(df: pd.DataFrame, title: str = "Metrics"):
    logger.info("\n" + "═" * 80)
    logger.info(f"  {title}")
    logger.info("═" * 80)
    logger.info("\n" + df.to_string(index=False))
    logger.info("═" * 80 + "\n")


def save_metrics_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logger.info(f"Metrics → {path}")


class SHAPExplainer:
    def __init__(self, predict_fn, feature_names: List[str], background_data: np.ndarray, n_background: int = 100):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        idx = np.random.choice(len(background_data), min(n_background, len(background_data)), replace=False)
        self.background = background_data[idx].astype(np.float32)
        logger.info(f"Building SHAP KernelExplainer (background={len(self.background)})...")
        self.explainer = shap.KernelExplainer(predict_fn, self.background, link="logit")
        logger.info("KernelExplainer ready.")

    @classmethod
    def for_binary_model(cls, model, cfg, feature_names, background_data, n_background=100):
        device = torch.device(cfg.device)
        model.set_mode("finetune")
        model.eval()
        model.to(device)

        def fn(X):
            with torch.no_grad():
                logit, _, _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
            return torch.sigmoid(logit).cpu().numpy()

        return cls(fn, feature_names, background_data, n_background)

    @classmethod
    def for_multiclass_model(cls, model, cfg, feature_names, background_data, n_background=100, target_class=1):
        device = torch.device(cfg.device)
        model.set_mode("finetune")
        model.eval()
        model.to(device)

        def fn(X):
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
            return F.softmax(logits, -1)[:, target_class].cpu().numpy()

        return cls(fn, feature_names, background_data, n_background)

    def explain_batch(self, X: np.ndarray, n_samples=200, silent=False) -> np.ndarray:
        logger.info(f"SHAP: explaining {len(X)} samples (nsamples={n_samples})...")
        sv = self.explainer.shap_values(X.astype(np.float32), nsamples=n_samples, silent=silent)
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def explain_single(self, x: np.ndarray, n_samples=500) -> np.ndarray:
        return self.explain_batch(x[np.newaxis], n_samples).squeeze(0)

    def get_global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "Feature": self.feature_names,
            "Mean |SHAP|": np.abs(shap_values).mean(0),
            "Std |SHAP|": np.abs(shap_values).std(0),
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    @staticmethod
    def build_tree_explainer(xgb_model, Z_background):
        return shap.TreeExplainer(xgb_model, data=Z_background, feature_perturbation="interventional")


class LIMEExplainer:
    def __init__(
        self,
        predict_fn,
        feature_names,
        training_data,
        class_names=None,
        categorical_features=None,
        kernel_width=0.75,
        random_state=42,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        if class_names is None:
            class_names = ["Normal", "Attack"]
        self.n_classes = len(class_names)
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data.astype(np.float64),
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features or [],
            mode="classification",
            kernel_width=kernel_width,
            random_state=random_state,
            discretize_continuous=True,
            discretizer="quartile",
        )
        logger.info(f"LIME LimeTabularExplainer initialised ({len(class_names)} classes).")

    def _wrap(self, X):
        out = self.predict_fn(X.astype(np.float32))
        if out.ndim == 1:
            return np.column_stack([1 - out, out])
        return out

    def explain_instance(self, x, num_features=15, num_samples=2000):
        logger.info(f"LIME explaining instance (features={num_features}, samples={num_samples})...")
        return self.explainer.explain_instance(
            data_row=x.astype(np.float64),
            predict_fn=self._wrap,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=self.n_classes,
        )

    def get_explanation_df(self, explanation, label=1) -> pd.DataFrame:
        available = list(explanation.local_exp.keys())
        if label in available:
            resolved = label
        else:
            resolved = max(available, key=lambda k: len(explanation.local_exp[k]))
            logger.warning(
                f"LIME label {label} not in explanation (available={available}). "
                f"Falling back to label {resolved}."
            )
        pairs = explanation.as_list(label=resolved)
        df = pd.DataFrame(pairs, columns=["Feature Condition", "LIME Weight"])
        df["Explained Label"] = resolved
        df["Direction"] = df["LIME Weight"].apply(lambda w: "→ Attack" if w > 0 else "→ Normal")
        df["Abs Weight"] = df["LIME Weight"].abs()
        return df.sort_values("Abs Weight", ascending=False).reset_index(drop=True)


def extract_attention_weights_binary(model, X, cfg, batch_size=512):
    device = torch.device(cfg.device)
    model.set_mode("finetune")
    model.eval()
    model.to(device)
    from nids_deep_model import NIDSTabularDataset
    loader = DataLoader(NIDSTabularDataset(X), batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for batch in loader:
            _, attn = model.feature_extractor(batch.to(device), add_noise=False)
            out.append(attn.cpu().numpy())
    return np.concatenate(out)


def extract_latent_multiclass(model, X, cfg, batch_size=512):
    device = torch.device(cfg.device)
    return model.extract_latent(X, device, batch_size)


class NIDSVisualiser:
    def __init__(self, output_dir="xai_outputs", feature_names=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = feature_names or []
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        })

    def _save(self, fig, name):
        p = self.output_dir / f"{name}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"  Saved: {p}")
        plt.close(fig)
        return str(p)

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title="Confusion Matrix", filename="confusion_matrix"):
        if class_names is None:
            class_names = ["Normal", "Attack"]
        cm_raw = confusion_matrix(y_true, y_pred)
        cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-9)
        n = len(class_names)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(title, fontsize=15, fontweight="bold")
        for ax, (data, fmt, sub, cmap) in zip(axes, [
            (cm_raw, "d", "Raw Counts", "Blues"),
            (cm_norm, ".3f", "Row-Normalised (DR on diagonal)", "RdYlGn"),
        ]):
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=(1.0 if fmt == ".3f" else None))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set(
                xticks=range(n),
                yticks=range(n),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted",
                ylabel="True",
                title=sub,
            )
            ax.set_xticklabels(class_names, rotation=30, ha="right")
            t = data.max() / 2
            for i in range(n):
                for j in range(n):
                    v = data[i, j]
                    ax.text(
                        j,
                        i,
                        f"{int(v):,}" if fmt == "d" else f"{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white" if v > t else "black",
                    )
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_per_class_metrics(self, metrics_df, title="Per-Class Metrics", filename="per_class_metrics", attack_col="Attack Category"):
        pf = metrics_df[metrics_df[attack_col] != "Normal (Baseline)"].copy()
        pf = pf[pd.to_numeric(pf["DR (%)"], errors="coerce").notna()]
        pf["DR"] = pd.to_numeric(pf["DR (%)"])
        pf["FAR"] = pd.to_numeric(pf["FAR (%)"], errors="coerce").fillna(0)
        pf["F1"] = pd.to_numeric(pf["F1-Score"])
        x = np.arange(len(pf))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(pf) * 2.5), 6))
        ax.bar(x - w, pf["DR"], w, label="DR %", color=PALETTE["normal"], edgecolor="white")
        ax.bar(x, pf["FAR"], w, label="FAR %", color=PALETTE["attack"], edgecolor="white")
        ax.bar(x + w, pf["F1"] * 100, w, label="F1×100", color=PALETTE["accent"], edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(pf[attack_col], rotation=20, ha="right")
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.legend()
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", color=PALETTE["grid"])
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_radar_chart(self, metrics_df, title="Attack Detection Radar", filename="radar_chart", attack_col="Attack Category"):
        pf = metrics_df[metrics_df[attack_col] != "Normal (Baseline)"].copy()
        pf = pf[pd.to_numeric(pf["DR (%)"], errors="coerce").notna()]
        pf["DR"] = pd.to_numeric(pf["DR (%)"])
        pf["FAR"] = pd.to_numeric(pf["FAR (%)"], errors="coerce").fillna(0)
        pf["F1"] = pd.to_numeric(pf["F1-Score"])
        pf["Spec"] = 100 - pf["FAR"]
        cats = pf[attack_col].tolist()
        N = 3
        angles = (np.linspace(0, 2 * np.pi, N, endpoint=False) + [0]).tolist()
        angles += angles[:1]
        colours = plt.cm.tab10(np.linspace(0, 0.9, len(cats)))
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_facecolor("#FAFAFA")
        for idx, (_, row) in enumerate(pf.iterrows()):
            vals = [row["DR"], row["Spec"], row["F1"] * 100] + [row["DR"]]
            ax.plot(angles, vals, "o-", linewidth=2, color=colours[idx], label=row[attack_col])
            ax.fill(angles, vals, alpha=0.15, color=colours[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["DR %", "Specificity\n(100−FAR)", "F1×100"], size=11)
        ax.set_ylim(0, 110)
        ax.set_title(title, size=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_shap_summary(self, shap_values, X_explain, max_display=20, title="SHAP Beeswarm", filename="shap_summary_beeswarm"):
        fn = np.array(self.feature_names) if self.feature_names else np.array([f"f{i}" for i in range(shap_values.shape[1])])
        exp = shap.Explanation(values=shap_values, data=X_explain, feature_names=fn)
        fig, _ = plt.subplots(figsize=(10, max(6, max_display * 0.38)))
        shap.plots.beeswarm(exp, max_display=max_display, show=False)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("SHAP Value")
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_shap_bar(self, shap_values, max_display=20, title="SHAP Global Importance", filename="shap_bar"):
        fn = self.feature_names if self.feature_names else [f"f{i}" for i in range(shap_values.shape[1])]
        ma = np.abs(shap_values).mean(0)
        ti = np.argsort(ma)[::-1][:max_display]
        tn = [fn[i] for i in ti]
        tv = ma[ti]
        norm = plt.Normalize(tv.min(), tv.max())
        colors = [plt.cm.YlOrRd(norm(v)) for v in tv]
        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.38)))
        ax.barh(range(max_display), tv[::-1], color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(max_display))
        ax.set_yticklabels(tn[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(title, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"])
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_shap_decision(self, sv_single, expected_value, x_single, true_label, pred_prob, title=None, filename="shap_decision", max_display=15):
        fn = np.array(self.feature_names) if self.feature_names else np.array([f"f{i}" for i in range(len(sv_single))])
        ti = np.argsort(np.abs(sv_single))[::-1][:max_display]
        ts = ti[np.argsort(sv_single[ti])]
        fnd = fn[ts]
        vd = sv_single[ts]
        dd = x_single[ts]
        cum = expected_value + np.cumsum(vd)
        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.42)))
        for i, (val, c, feat, dv) in enumerate(zip(vd, cum, fnd, dd)):
            colour = PALETTE["shap_pos"] if val > 0 else PALETTE["shap_neg"]
            ax.barh(i, val, left=c - val, color=colour, alpha=0.8, edgecolor="white", height=0.65)
            ax.text(max(cum) + abs(vd).max() * 0.04, i, f"{feat} = {dv:.3g}", va="center", fontsize=8.5)
        ax.axvline(expected_value, color="grey", linestyle="--", linewidth=1.2, label=f"Base ({expected_value:.3f})")
        c_pred = PALETTE["attack"] if pred_prob > 0.5 else PALETTE["normal"]
        ax.axvline(cum[-1], color=c_pred, linewidth=1.8, label=f"Pred ({pred_prob:.3f})")
        ax.set_yticks(range(max_display))
        ax.set_yticklabels([""] * max_display)
        ax.set_xlabel("Model Output (logit)")
        lbl = "Attack" if true_label == 1 else "Normal"
        ax.set_title(title or f"SHAP Decision | True={lbl} | P(Attack)={pred_prob:.3f}", fontweight="bold")
        ax.legend(
            handles=[
                mpatches.Patch(color=PALETTE["shap_pos"], label="→Attack"),
                mpatches.Patch(color=PALETTE["shap_neg"], label="→Normal"),
            ],
            loc="lower right",
            fontsize=9,
        )
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"])
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_lime_explanation(self, lime_df, true_label, pred_prob, title=None, filename="lime_explanation"):
        lbl = "Attack" if true_label == 1 else "Normal"
        if title is None:
            title = f"LIME | True={lbl} | P(Attack)={pred_prob:.3f}"
        top = lime_df.head(15).copy()
        colours = [PALETTE["shap_pos"] if w > 0 else PALETTE["shap_neg"] for w in top["LIME Weight"]]
        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.45)))
        bars = ax.barh(range(len(top)), top["LIME Weight"], color=colours, edgecolor="white", height=0.65)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["Feature Condition"], fontsize=9)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("LIME Weight")
        ax.set_title(title, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="x", color=PALETTE["grid"])
        for bar in bars:
            w = bar.get_width()
            off = top["LIME Weight"].abs().max() * 0.02
            ax.text(
                w + (off if w >= 0 else -off),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}",
                va="center",
                ha="left" if w >= 0 else "right",
                fontsize=8,
            )
        ax.legend(
            handles=[
                mpatches.Patch(color=PALETTE["shap_pos"], label="→Attack"),
                mpatches.Patch(color=PALETTE["shap_neg"], label="→Normal"),
            ],
            fontsize=9,
        )
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_attention_heatmap(self, attn_matrix, y_labels, class_names=None, n_samples=50, title="Attention Heatmap", filename="attention_heatmap"):
        if class_names is None:
            class_names = ["Normal", "Attack"]
        n_cls = len(class_names)
        fig, axes = plt.subplots(n_cls, 1, figsize=(14, 4 * n_cls), gridspec_kw={"hspace": 0.45})
        if n_cls == 1:
            axes = [axes]
        cmap = LinearSegmentedColormap.from_list("attn", ["#FFFFFF", PALETTE["accent"]], N=256)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        for cid, (ax, cname) in enumerate(zip(axes, class_names)):
            idx = np.where(y_labels == cid)[0]
            if not len(idx):
                ax.set_title(f"{cname} — no samples")
                continue
            si = np.random.choice(idx, min(n_samples, len(idx)), replace=False)
            im = ax.imshow(attn_matrix[si], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            plt.colorbar(im, ax=ax, label="Attn Weight", fraction=0.03, pad=0.02)
            ax.set_title(f"{cname} ({len(si)} samples)", fontsize=11)
            ax.set_xlabel("Hidden Neuron Index")
            ax.set_ylabel("Sample")
            ax.tick_params(left=False, labelleft=False)
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_shap_vs_attention(self, importance_df, attention_mean, top_n=20, filename="shap_vs_attention"):
        top = importance_df.head(top_n).copy()
        x = np.arange(len(top))
        fig, ax1 = plt.subplots(figsize=(12, 5.5))
        ax2 = ax1.twinx()
        ax1.bar(x, top["Mean |SHAP|"], color=PALETTE["shap_pos"], alpha=0.75, edgecolor="white", label="Mean |SHAP|")
        ax1.set_ylabel("Mean |SHAP|", color=PALETTE["shap_pos"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["shap_pos"])
        attn_s = attention_mean.mean()
        ax2.plot(x, top["Mean |SHAP|"] * attn_s, "o-", color=PALETTE["accent"], linewidth=2, markersize=5, label="SHAP×Attn")
        ax2.set_ylabel("Attention-Scaled", color=PALETTE["accent"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(top["Feature"], rotation=35, ha="right", fontsize=9)
        ax1.set_title("SHAP Importance vs Block A Attention", fontweight="bold")
        ax1.set_facecolor("#FAFAFA")
        ax1.grid(axis="y", color=PALETTE["grid"])
        lines1, l1 = ax1.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, l1 + l2, loc="upper right", fontsize=9)
        plt.tight_layout()
        return self._save(fig, filename)


def run_xai_pipeline(
    model,
    cfg,
    X_test: np.ndarray,
    y_test_binary: np.ndarray,
    feature_names: List[str],
    model_type: str = "binary",
    y_test_multiclass: Optional[np.ndarray] = None,
    attack_map: Optional[Dict] = None,
    output_dir: str = "xai_outputs",
    n_shap_explain: int = 200,
    n_shap_samples: int = 200,
    n_lime_samples: int = 2000,
    sample_indices: Optional[List[int]] = None,
    xgb_block_b=None,
    Z_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info(f"GA-NIDS XAI Pipeline  [{model_type.upper()}]")
    logger.info("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vis = NIDSVisualiser(output_dir=output_dir, feature_names=feature_names)
    results: Dict[str, Any] = {}
    device = torch.device(cfg.device)

    logger.info("\n[1/7] Generating predictions...")
    model.eval()
    model.to(device)

    from torch.utils.data import TensorDataset
    X_t = torch.from_numpy(X_test.astype(np.float32))
    ds = TensorDataset(X_t, torch.zeros(len(X_t)))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            if model_type == "binary":
                model.set_mode("finetune")
                out, _, _ = model(xb)
                all_logits.append(out.cpu().numpy())
            else:
                model.set_mode("finetune")
                out, _ = model(xb)
                all_logits.append(out.cpu().numpy())

    if model_type == "binary":
        logits = np.concatenate(all_logits)
        y_prob = 1 / (1 + np.exp(-logits))
        y_pred = (y_prob > 0.5).astype(int)
    else:
        logits = np.concatenate(all_logits)
        y_prob = np.exp(logits - logits.max(1, keepdims=True))
        y_prob = y_prob / y_prob.sum(1, keepdims=True)
        y_pred = y_prob.argmax(1)
        y_prob_scalar = y_prob[:, 1]

    results["predictions"] = {"y_pred": y_pred, "y_prob": y_prob}

    logger.info("\n[2/7] Binary metrics...")
    prob_for_binary = y_prob if model_type == "binary" else y_prob_scalar
    binary_df = compute_binary_cyber_metrics(
        y_test_binary,
        y_pred if model_type == "binary" else (y_pred > 0).astype(int),
        prob_for_binary,
    )
    results["binary_metrics_df"] = binary_df
    print_metrics_table(binary_df, "Binary Metrics (DR / FAR / F1)")
    save_metrics_csv(binary_df, str(Path(output_dir) / "binary_metrics.csv"))

    results["multiclass_metrics_df"] = None
    if y_test_multiclass is not None:
        logger.info("\n[3/7] Per-attack-category metrics...")
        mc_df = compute_multiclass_cyber_metrics(
            y_test_multiclass,
            y_pred if model_type == "binary" else (y_pred > 0).astype(int),
            prob_for_binary,
            attack_map,
        )
        results["multiclass_metrics_df"] = mc_df
        print_metrics_table(mc_df, "Per-Attack-Category Metrics")
        save_metrics_csv(mc_df, str(Path(output_dir) / "multiclass_metrics.csv"))
        vis.plot_per_class_metrics(mc_df, attack_col="Attack Category")
        vis.plot_radar_chart(mc_df, attack_col="Attack Category")
    else:
        logger.info("\n[3/7] Skipping per-attack metrics (y_test_multiclass=None).")

    logger.info("\n[4/7] Confusion matrix...")
    cm_class = CLASS_NAMES_5 if model_type == "multiclass" else None
    vis.plot_confusion_matrix(
        y_test_binary if model_type == "binary" else y_pred,
        y_pred if model_type == "binary" else y_pred,
        class_names=cm_class,
    )

    attention_matrix = None
    if model_type == "binary":
        logger.info("\n[5/7] Extracting attention weights...")
        attention_matrix = extract_attention_weights_binary(model, X_test, cfg)
        results["attention_matrix"] = attention_matrix
        vis.plot_attention_heatmap(attention_matrix, y_test_binary)
    else:
        logger.info("\n[5/7] Skipping attention heatmap (multiclass model).")

    logger.info(f"\n[6/7] SHAP ({n_shap_explain} samples)...")
    normal_idx = np.where(y_test_binary == 0)[0]
    attack_idx = np.where(y_test_binary == 1)[0]
    n_h = n_shap_explain // 2
    shap_idx = np.concatenate([
        np.random.choice(normal_idx, min(n_h, len(normal_idx)), replace=False),
        np.random.choice(attack_idx, min(n_h, len(attack_idx)), replace=False),
    ])
    X_shap = X_test[shap_idx]

    if model_type == "binary":
        shap_exp = SHAPExplainer.for_binary_model(
            model,
            cfg,
            feature_names,
            X_test,
            n_background=min(100, len(X_test)),
        )
    else:
        shap_exp = SHAPExplainer.for_multiclass_model(
            model,
            cfg,
            feature_names,
            X_test,
            n_background=min(100, len(X_test)),
            target_class=1,
        )

    shap_values = shap_exp.explain_batch(X_shap, n_samples=n_shap_samples, silent=False)
    results["shap_values"] = shap_values
    imp_df = shap_exp.get_global_importance(shap_values)
    results["shap_importance_df"] = imp_df
    save_metrics_csv(imp_df, str(Path(output_dir) / "shap_importance.csv"))
    logger.info("Top 10 features:\n" + imp_df.head(10).to_string(index=False))

    vis.plot_shap_summary(shap_values, X_shap, max_display=20)
    vis.plot_shap_bar(shap_values, max_display=20)
    if attention_matrix is not None:
        vis.plot_shap_vs_attention(imp_df, attention_matrix.mean(0))

    if sample_indices is None:
        sample_indices = []
        if len(normal_idx):
            sample_indices.append(int(normal_idx[0]))
        if len(attack_idx):
            sample_indices.append(int(attack_idx[0]))

    for idx in sample_indices[:4]:
        sv = shap_exp.explain_single(X_test[idx], n_samples=500)
        prob_val = float(y_prob[idx]) if model_type == "binary" else float(y_prob_scalar[idx])
        vis.plot_shap_decision(
            sv,
            float(shap_exp.explainer.expected_value),
            X_test[idx],
            int(y_test_binary[idx]),
            prob_val,
            filename=f"shap_decision_sample_{idx}",
        )

    logger.info(f"\n[7/7] LIME ({len(sample_indices[:4])} samples)...")

    if model_type == "binary":
        model.set_mode("finetune")

        def lime_fn(X):
            with torch.no_grad():
                lg, _, _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
            return torch.sigmoid(lg).cpu().numpy()

        class_names_lime = ["Normal", "Attack"]
    else:
        model.set_mode("finetune")

        def lime_fn(X):
            with torch.no_grad():
                lg, _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
            return F.softmax(lg, -1).cpu().numpy()

        class_names_lime = CLASS_NAMES_5

    lime_exp = LIMEExplainer(
        lime_fn,
        feature_names,
        X_test[: min(2000, len(X_test))],
        class_names=class_names_lime,
    )

    lime_results = []
    for idx in sample_indices[:4]:
        exp = lime_exp.explain_instance(X_test[idx], num_features=15, num_samples=n_lime_samples)
        pred_l = int(y_pred[idx])
        ldf = lime_exp.get_explanation_df(exp, label=pred_l)
        lime_results.append((ldf, exp))
        prob_val = float(y_prob[idx]) if model_type == "binary" else float(y_prob_scalar[idx])
        vis.plot_lime_explanation(ldf, int(y_test_binary[idx]), prob_val, filename=f"lime_explanation_sample_{idx}")
        logger.info(
            f"  Sample {idx}: true={y_test_binary[idx]} "
            f"P(attack)={prob_val:.3f} top={ldf.iloc[0]['Feature Condition']}"
        )

    results["lime_explanations"] = lime_results

    logger.info("\n" + "═" * 70)
    logger.info(f"XAI complete → {Path(output_dir).resolve()}")
    logger.info("═" * 70 + "\n")
    return results