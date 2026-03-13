r"""
=============================================================================
GA-NIDS — Main Integration Script
=============================================================================

CONFIGURED FOR YOUR DATASET (from inspect_dataset.py output):
  ✓ 246 columns total  →  245 feature cols + 1 "label" col
  ✓ All features are float64, already Z-score scaled — NO re-scaling needed
  ✓ No categorical columns — NO one-hot encoding needed
  ✓ Binary label column: "label"  (0 = Normal, 1 = Attack)
  ✓ No multiclass attack category column present
  ✓ ~120 constant/zero-variance columns detected → auto-dropped before training
  ✓ Test CSV uses Python CSV engine to avoid C-parser crash on Windows

HOW TO RUN:
    cd "C:\Users\abul4\OneDrive\Desktop\IDS Project"
    python run_nids.py

OUTPUT:
    outputs/
    ├── nids_model.pt              ← saved model weights
    ├── binary_metrics.csv         ← DR / FAR / F1 table
    ├── shap_importance.csv        ← ranked feature importance
    ├── confusion_matrix.png
    ├── shap_summary_beeswarm.png
    ├── shap_bar.png
    ├── shap_decision_sample_N.png
    ├── lime_explanation_sample_N.png
    └── attention_heatmap.png
=============================================================================
"""

import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nids_deep_model import NIDSConfig, run_full_pipeline
from nids_xai_metrics import run_xai_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (verified against your inspect_dataset.py output)
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_CSV    = Path("dataset/Hybrid_IDS_train.csv")
TEST_CSV     = Path("dataset/Hybrid_IDS_test.csv")
LABEL_COLUMN = "label"      # only label col; values: {0, 1}
OUTPUT_DIR   = Path("outputs")
VAL_SPLIT    = 0.15         # 15% of train → validation

# ── Model hyperparameters ─────────────────────────────────────────────────────
# input_dim is set automatically after constant-column removal.
# Estimated active features after dropping ~120 constants: ~120–130.
MODEL_CONFIG = dict(
    encoder_dims      = [256, 128, 64],
    latent_dim        = 32,
    dropout_rate      = 0.3,
    noise_std         = 0.05,
    sparsity_weight   = 1e-3,
    sparsity_target   = 0.05,
    classifier_hidden = 64,
    learning_rate     = 1e-3,
    weight_decay      = 1e-4,
    batch_size        = 512,
    pretrain_epochs   = 30,
    finetune_epochs   = 50,
    early_stop_patience = 8,
    num_workers       = 0,      # keep 0 on Windows — avoids multiprocessing issues
    pin_memory        = False,
)

# ── XAI configuration ─────────────────────────────────────────────────────────
XAI_CONFIG = dict(
    n_shap_explain = 200,   # test samples to explain (more = slower but richer)
    n_shap_samples = 200,   # SHAP kernel perturbations per sample
    n_lime_samples = 2000,  # LIME neighbourhood size
    output_dir     = str(OUTPUT_DIR),
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CSV  (Python engine avoids the C-parser crash on Windows)
# ═════════════════════════════════════════════════════════════════════════════

def load_csv_safe(path: Path, label: str) -> pd.DataFrame:
    """
    Load a CSV using Python's built-in engine, which is slower than pandas'
    C engine but tolerates malformed rows and Windows Unicode edge-cases
    that cause the C engine to hang or crash (as seen in the test CSV).
    """
    if not path.exists():
        logger.error(
            f"File not found: {path.resolve()}\n"
            f"  Run from the IDS Project folder:\n"
            f"      cd \"C:\\Users\\abul4\\OneDrive\\Desktop\\IDS Project\"\n"
            f"      python run_nids.py"
        )
        sys.exit(1)

    logger.info(f"Loading {label}: {path}  (Python CSV engine)...")
    df = pd.read_csv(
        path,
        engine        = "python",          # tolerant parser — won't hang
        encoding      = "utf-8",
        encoding_errors = "replace",        # replace undecodable bytes instead of crashing
        on_bad_lines  = "skip",             # skip malformed rows silently
    )
    logger.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPARE ARRAYS
# ═════════════════════════════════════════════════════════════════════════════

def prepare_arrays(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
) -> tuple:
    """
    Convert the already-preprocessed CSVs into clean numpy arrays.

    What this does  (and why):
    ──────────────────────────
    1. Separate the "label" column from features.
       The data is ALREADY Z-score scaled — no StandardScaler applied.

    2. Drop zero-variance (constant) columns from TRAINING data.
       The inspector showed ~120 columns that take a single constant value
       across all rows (one-hot dummies for categories absent from this split).
       Constant features add nothing mathematically:
           BatchNorm1d will divide by σ=0 → NaN gradients
           SHAP values will always be 0 → meaningless explainability output
       We fit the "live" column list on train and apply the same mask to test.

    3. Align train/test column sets.
       After removing constants from train, any column that was constant in
       train but variable in test is also dropped from test so both arrays
       have identical column counts.

    4. Convert to float32 (PyTorch default dtype).

    5. Stratified train / validation split (preserves class ratio).

    Returns
    ───────
    X_train, y_train  : training features + labels
    X_val,   y_val    : validation features + labels  (from train CSV)
    X_test,  y_test   : test features + labels        (from test CSV)
    feature_names     : list[str] of surviving column names
    """
    logger.info("\n" + "─" * 60)
    logger.info("PREPARING ARRAYS")
    logger.info("─" * 60)

    # ── 1. Separate labels ────────────────────────────────────────────────
    if LABEL_COLUMN not in df_train.columns:
        logger.error(
            f"Label column '{LABEL_COLUMN}' not found.\n"
            f"  Columns present: {df_train.columns.tolist()}"
        )
        sys.exit(1)

    y_train_full = df_train[LABEL_COLUMN].values.astype(np.float32)
    y_test       = df_test[LABEL_COLUMN].values.astype(np.float32)

    X_train_df = df_train.drop(columns=[LABEL_COLUMN])
    X_test_df  = df_test.drop(columns=[LABEL_COLUMN])

    logger.info(f"  Raw feature count (train): {X_train_df.shape[1]}")
    logger.info(f"  Class distribution — Train:  Normal={(y_train_full==0).sum():,}  "
                f"Attack={(y_train_full==1).sum():,}")
    logger.info(f"  Class distribution — Test:   Normal={(y_test==0).sum():,}  "
                f"Attack={(y_test==1).sum():,}")

    # ── 2. Drop zero-variance columns (fit on FULL train set) ─────────────
    # A column is "constant" if its standard deviation across ALL training
    # rows is below 1e-8. We check on the full data, not just the 50k sample,
    # because some columns are near-constant but do vary in the full dataset.
    logger.info("  Scanning for zero-variance columns (full train data)...")

    # Convert to float first so std() works correctly
    X_train_numeric = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test_numeric  = X_test_df.apply(pd.to_numeric,  errors="coerce").fillna(0)

    col_std = X_train_numeric.std(axis=0)
    live_cols = col_std[col_std > 1e-8].index.tolist()
    dropped_const = X_train_df.shape[1] - len(live_cols)

    logger.info(f"  Dropped {dropped_const} zero-variance columns.")
    logger.info(f"  Remaining informative features: {len(live_cols)}")

    X_train_numeric = X_train_numeric[live_cols]

    # ── 3. Align test to same column set ──────────────────────────────────
    # Add any missing columns to test as zeros; drop any extras.
    missing_in_test = [c for c in live_cols if c not in X_test_numeric.columns]
    for c in missing_in_test:
        X_test_numeric[c] = 0.0
    X_test_numeric = X_test_numeric[live_cols]   # enforce same column order

    if missing_in_test:
        logger.info(f"  Added {len(missing_in_test)} zero-filled cols to test "
                    f"(absent from test CSV).")

    feature_names = live_cols

    # ── 4. Convert to float32 numpy arrays ───────────────────────────────
    X_train_full = X_train_numeric.values.astype(np.float32)
    X_test       = X_test_numeric.values.astype(np.float32)

    # ── 5. Stratified train / validation split ────────────────────────────
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_full)),
        test_size  = VAL_SPLIT,
        random_state = 42,
        stratify   = y_train_full.astype(int),
    )

    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    logger.info(
        f"\n  Final array shapes:\n"
        f"    X_train : {X_train.shape}   y_train : {y_train.shape}\n"
        f"    X_val   : {X_val.shape}   y_val   : {y_val.shape}\n"
        f"    X_test  : {X_test.shape}   y_test  : {y_test.shape}\n"
        f"    Features: {len(feature_names)}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("\n" + "=" * 60)
    logger.info("  GA-NIDS — Full Training + XAI Pipeline")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    df_train = load_csv_safe(TRAIN_CSV, "TRAIN")
    df_test  = load_csv_safe(TEST_CSV,  "TEST")

    # ── Prepare arrays ────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        prepare_arrays(df_train, df_test)

    # ── Build config with correct input_dim ───────────────────────────────
    cfg = NIDSConfig(
        input_dim = X_train.shape[1],   # set automatically from surviving features
        **MODEL_CONFIG,
    )
    logger.info(f"\n  NIDSConfig.input_dim  = {cfg.input_dim}")
    logger.info(f"  NIDSConfig.latent_dim = {cfg.latent_dim}")
    logger.info(f"  NIDSConfig.device     = {cfg.device}")

    # ── Phase 1 + Phase 2 training ────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("  TRAINING")
    logger.info("─" * 60)
    model, train_results, Z_test = run_full_pipeline(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        cfg       = cfg,
        save_path = str(OUTPUT_DIR / "nids_model.pt"),
    )

    logger.info(
        f"\n  Training complete.\n"
        f"    Test Accuracy : {train_results['accuracy']:.4f}\n"
        f"    Test F1       : {train_results['f1']:.4f}\n"
        f"    Test Precision: {train_results['precision']:.4f}\n"
        f"    Test Recall   : {train_results['recall']:.4f}\n"
        f"    Test AUC-ROC  : {train_results['auc_roc']:.4f}"
    )

    # ── XAI + per-class cybersecurity metrics ─────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("  XAI + METRICS")
    logger.info("─" * 60)
    xai_results = run_xai_pipeline(
        model             = model,
        cfg               = cfg,
        X_test            = X_test,
        y_test_binary     = y_test.astype(int),
        feature_names     = feature_names,
        y_test_multiclass = None,   # no multiclass column in your dataset
        attack_map        = None,
        Z_test            = Z_test,
        **XAI_CONFIG,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ALL DONE")
    logger.info("=" * 60)
    logger.info(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    logger.info("  Files generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        logger.info(f"    {f.name:<45} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()