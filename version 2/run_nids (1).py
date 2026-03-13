
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

TRAIN_CSV    = Path("dataset/Hybrid_IDS_train.csv")
TEST_CSV     = Path("dataset/Hybrid_IDS_test.csv")
LABEL_COLUMN = "label"      
OUTPUT_DIR   = Path("outputs")
VAL_SPLIT    = 0.15         


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
    num_workers       = 0,      
    pin_memory        = False,
)
XAI_CONFIG = dict(
    n_shap_explain = 200,   # test samples to explain (more = slower but richer)
    n_shap_samples = 200,   # SHAP kernel perturbations per sample
    n_lime_samples = 2000,  # LIME neighbourhood size
    output_dir     = str(OUTPUT_DIR),
)


def load_csv_safe(path: Path, label: str) -> pd.DataFrame:
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
        engine        = "python",          
        encoding      = "utf-8",
        encoding_errors = "replace",       
        on_bad_lines  = "skip",             
    )
    logger.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def prepare_arrays(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
) -> tuple:

    logger.info("\n" + "─" * 60)
    logger.info("PREPARING ARRAYS")
    logger.info("─" * 60)

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

    logger.info("  Scanning for zero-variance columns (full train data)...")

    X_train_numeric = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test_numeric  = X_test_df.apply(pd.to_numeric,  errors="coerce").fillna(0)

    col_std = X_train_numeric.std(axis=0)
    live_cols = col_std[col_std > 1e-8].index.tolist()
    dropped_const = X_train_df.shape[1] - len(live_cols)

    logger.info(f"  Dropped {dropped_const} zero-variance columns.")
    logger.info(f"  Remaining informative features: {len(live_cols)}")

    X_train_numeric = X_train_numeric[live_cols]


    missing_in_test = [c for c in live_cols if c not in X_test_numeric.columns]
    for c in missing_in_test:
        X_test_numeric[c] = 0.0
    X_test_numeric = X_test_numeric[live_cols]  

    if missing_in_test:
        logger.info(f"  Added {len(missing_in_test)} zero-filled cols to test "
                    f"(absent from test CSV).")

    feature_names = live_cols

    X_train_full = X_train_numeric.values.astype(np.float32)
    X_test       = X_test_numeric.values.astype(np.float32)

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



def main():
    logger.info("\n" + "=" * 60)
    logger.info("  GA-NIDS — Full Training + XAI Pipeline")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_csv_safe(TRAIN_CSV, "TRAIN")
    df_test  = load_csv_safe(TEST_CSV,  "TEST")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        prepare_arrays(df_train, df_test)

    cfg = NIDSConfig(
        input_dim = X_train.shape[1], 
        **MODEL_CONFIG,
    )
    logger.info(f"\n  NIDSConfig.input_dim  = {cfg.input_dim}")
    logger.info(f"  NIDSConfig.latent_dim = {cfg.latent_dim}")
    logger.info(f"  NIDSConfig.device     = {cfg.device}")


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

    logger.info("\n" + "─" * 60)
    logger.info("  XAI + METRICS")
    logger.info("─" * 60)
    xai_results = run_xai_pipeline(
        model             = model,
        cfg               = cfg,
        X_test            = X_test,
        y_test_binary     = y_test.astype(int),
        feature_names     = feature_names,
        y_test_multiclass = None,   
        attack_map        = None,
        Z_test            = Z_test,
        **XAI_CONFIG,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  ALL DONE")
    logger.info("=" * 60)
    logger.info(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    logger.info("  Files generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        logger.info(f"    {f.name:<45} {size_kb:>8.1f} KB")

main()