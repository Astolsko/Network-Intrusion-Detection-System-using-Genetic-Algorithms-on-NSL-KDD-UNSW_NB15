TRAIN_CSV  = r"C:\Users\abul4\OneDrive\Desktop\IDS Project\dataset\Hybrid_IDS_train_multiclass.csv"
TEST_CSV   = r"C:\Users\abul4\OneDrive\Desktop\IDS Project\dataset\Hybrid_IDS_test_multiclass.csv"
OUTPUT_DIR = r"C:\Users\abul4\OneDrive\Desktop\IDS Project\nids_outputs"

MODE = "multiclass_research"

SAVED_MODEL_PATH = r"C:\Users\abul4\OneDrive\Desktop\IDS Project\nids_outputs\research_outputs\nids_research_model.pt"

U2R_TARGET_RATIO      = 0.05
U2R_COST_MULTIPLIER   = 50.0
CENTER_LOSS_LAMBDA    = 0.01
SUPCON_LOSS_LAMBDA    = 0.05
SHAP_PRUNE_PERCENTILE = 10.0

import os, sys, logging, copy, time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))


def _check_files():
    missing = [p for p in [TRAIN_CSV, TEST_CSV] if not Path(p).exists()]
    if missing:
        for m in missing:
            log.error(f"File not found: {m}")
        sys.exit(1)


def _try_xai(model, cfg, out_dir, model_type,
             X_test=None, y_test=None, feat_names=None):
    try:
        import shap, lime
    except ImportError:
        log.warning("shap/lime not installed — XAI skipped. pip install shap lime")
        return
    import numpy as np
    from nids_xai_metrics import run_xai_pipeline
    if "multiclass" in model_type:
        from nids_multiclass import load_hybrid_csv
        X_test, y_test, feat_names = load_hybrid_csv(TEST_CSV, verbose=False)
        y_bin = (y_test > 0).astype(int)
    else:
        y_bin = y_test.astype(int)
    run_xai_pipeline(
        model=model, cfg=cfg,
        X_test=X_test, y_test_binary=y_bin,
        feature_names=feat_names or [f"f{i}" for i in range(X_test.shape[1])],
        model_type="multiclass" if "multiclass" in model_type else "binary",
        y_test_multiclass=y_test,
        output_dir=str(Path(out_dir) / "xai_outputs"),
        n_shap_explain=200, n_shap_samples=200, n_lime_samples=2000,
    )


def run_multiclass_research():
    _check_files()
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.utils.data import DataLoader

    from nids_multiclass import (
        MultiClassNIDSConfig, MultiClassNIDSModel, MultiClassNIDSTrainer,
        load_hybrid_csv, build_multiclass_dataloader,
        CLASS_NAMES, NUM_CLASSES, _NIDSDS,
    )
    from nids_research_upgrades import (
        apply_svmsmote_targeted, CostSensitiveFocalLoss,
        CenterLoss, SupConLoss, finetune_step_with_center_loss,
        compute_ovr_metrics, print_ovr_table, plot_ovr_metrics,
        SHAPFeaturePruner,
    )

    out_dir = str(Path(OUTPUT_DIR) / "research_outputs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("━" * 70)
    log.info("GA-NIDS v3 — Research-Grade 5-Class Pipeline")
    log.info(f"  Train  : {TRAIN_CSV}")
    log.info(f"  Test   : {TEST_CSV}")
    log.info(f"  Output : {out_dir}")
    log.info("━" * 70)

    log.info("\n[1/8] Loading data...")
    X_tr_all, y_tr_all, feat_names = load_hybrid_csv(TRAIN_CSV)
    X_test,   y_test,   _          = load_hybrid_csv(TEST_CSV)

    pruner = None
    if SHAP_PRUNE_PERCENTILE > 0:
        pruner_path = str(Path(out_dir) / "shap_pruner.json")
        if Path(pruner_path).exists():
            log.info("\n[2/8] Loading saved SHAP pruner...")
            pruner = SHAPFeaturePruner.load(pruner_path, feat_names)
            X_tr_all   = pruner.transform(X_tr_all)
            X_test     = pruner.transform(X_test)
            feat_names = pruner.kept_names
            log.info(f"  Features: 248 -> {pruner.n_kept}")
        else:
            log.info("\n[2/8] No saved pruner found — using all 248 features.")
            log.info("  Run XAI first to generate shap_pruner.json")
    else:
        log.info("\n[2/8] SHAP pruning disabled.")

    log.info("\n[3/8] Splitting data...")
    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=42)
    ti, vi = next(sss.split(X_tr_all, y_tr_all))
    X_train, y_train = X_tr_all[ti], y_tr_all[ti]
    X_val,   y_val   = X_tr_all[vi], y_tr_all[vi]
    log.info(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    log.info("\n[4/8] SVMSMOTE for U2R...")
    X_train, y_train = apply_svmsmote_targeted(
        X_train, y_train,
        minority_classes=(4,), secondary_classes=(),
        u2r_target_ratio=U2R_TARGET_RATIO, n_continuous=16,
    )

    log.info("\n[5/8] Building model...")
    input_dim = X_train.shape[1]
    cfg = MultiClassNIDSConfig(
        input_dim=input_dim, encoder_dims=[512, 256, 128], latent_dim=64,
        classifier_hidden=128, dropout_rate=0.25, noise_std=0.08,
        sparsity_weight=1e-3, sparsity_target=0.05,
        focal_gamma=2.0, label_smoothing=0.03,
        learning_rate=1e-3, weight_decay=1e-4,
        batch_size=512, pretrain_epochs=100, finetune_epochs=180,
        early_stop_patience=15, t0_epochs=20,
        attn_num_heads=4, skip_proj_dim=16, num_workers=0, seed=42,
    )
    model  = MultiClassNIDSModel(cfg)
    device = torch.device(cfg.device)

    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)

    cls_loss = CostSensitiveFocalLoss.from_class_counts(
        train_counts, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing,
        u2r_multiplier=U2R_COST_MULTIPLIER, normal_attack_mult=2.0,
    ).to(device)

    center_loss = CenterLoss(NUM_CLASSES, cfg.latent_dim, CENTER_LOSS_LAMBDA).to(device)
    supcon_loss = SupConLoss(0.07, SUPCON_LOSS_LAMBDA).to(device)
    center_opt  = torch.optim.SGD(center_loss.parameters(), lr=0.5)

    log.info("\n[6/8] Building DataLoaders...")
    train_loader = build_multiclass_dataloader(
        X_train, y_train, cfg.batch_size,
        use_class_aware_sampler=True, num_workers=0,
    )
    val_loader = build_multiclass_dataloader(
        X_val, y_val, cfg.batch_size,
        use_class_aware_sampler=False, shuffle=False, num_workers=0,
    )

    log.info("\n[7/8] Training...")
    trainer = MultiClassNIDSTrainer(model, cfg, y_train)
    trainer.cls_loss = cls_loss
    trainer.pretrain(train_loader, val_loader)

    log.info("=" * 65)
    log.info("PHASE 2 — CostSensitiveFocal + CenterLoss + SupCon")
    log.info("=" * 65)
    model.set_mode("finetune")
    model.freeze_encoder()
    model.to(device)

    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.t0_epochs, T_mult=2, eta_min=cfg.eta_min)
    best_f1, best_state, patience_c = 0.0, None, 0
    n_b = len(train_loader)

    for epoch in range(1, cfg.finetune_epochs + 1):
        if epoch == 6:
            model.unfreeze_encoder()
            opt   = AdamW(model.parameters(), lr=cfg.learning_rate * 0.1,
                          weight_decay=cfg.weight_decay)
            sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.t0_epochs, T_mult=2, eta_min=cfg.eta_min)
            log.info("  Encoder unfrozen.")

        t0     = time.time()
        losses = finetune_step_with_center_loss(
            model, train_loader, cls_loss, center_loss, center_opt,
            opt, sched, epoch, n_b, device, supcon_loss, 0.5,
        )
        val_m  = trainer._p2_val(val_loader)

        if val_m["macro_f1"] > best_f1:
            best_f1, best_state, patience_c = val_m["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            patience_c += 1

        log.info(
            f"  Ep {epoch:03d}/{cfg.finetune_epochs} "
            f"focal={losses['focal']:.4f} center={losses['center']:.5f} "
            f"sc={losses['supcon']:.5f} | "
            f"val_mF1={val_m['macro_f1']:.4f} acc={val_m['acc']:.4f} "
            f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.1f}s"
        )

        if epoch % 30 == 0 or epoch == 1:
            try:
                center_loss.plot_centers(X_train[:2000], y_train[:2000],
                                          out_dir, epoch)
            except Exception:
                pass

        if patience_c >= cfg.early_stop_patience:
            log.info(f"  Early stop @ epoch {epoch}. Best={best_f1:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
        log.info("  Best weights restored.")

    log.info("\n[8/8] Corrected OvR evaluation...")
    model.eval(); model.to(device)
    loader = DataLoader(_NIDSDS(X_test, y_test),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    pa, la = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb.to(device))
            pa.append(logits.argmax(-1).cpu().numpy())
            la.append(yb.numpy())
    y_pred = np.concatenate(pa)
    y_true = np.concatenate(la)

    metrics_df, _ = compute_ovr_metrics(y_true, y_pred, CLASS_NAMES)
    print_ovr_table(metrics_df, "RESEARCH-GRADE OvR Metrics (Corrected FAR)")
    metrics_df.to_csv(str(Path(out_dir) / "ovr_metrics_corrected.csv"), index=False)
    plot_ovr_metrics(metrics_df, out_dir)

    torch.save({
        "model_state_dict":  model.state_dict(),
        "center_loss_state": center_loss.state_dict(),
        "config":            cfg.__dict__,
        "feature_names_kept": pruner.kept_names if pruner else feat_names,
        "pruner_mask":        pruner.feature_mask.tolist() if pruner else None,
    }, str(Path(out_dir) / "nids_research_model.pt"))

    log.info(f"\nAll outputs saved to: {out_dir}")
    log.info("  ovr_metrics_corrected.csv")
    log.info("  ovr_per_class_metrics.png")
    log.info("  latent_space_epoch*.png")
    log.info("  nids_research_model.pt")

    _try_xai(model, cfg, out_dir, "multiclass",
             X_test=X_test, y_test=y_test, feat_names=feat_names)
    return model, metrics_df


def run_multiclass():
    _check_files()
    from nids_multiclass import MultiClassNIDSConfig, run_multiclass_pipeline
    cfg = MultiClassNIDSConfig(
        input_dim=248, encoder_dims=[512,256,128], latent_dim=64,
        classifier_hidden=128, dropout_rate=0.30, noise_std=0.10,
        sparsity_weight=1e-3, sparsity_target=0.05, focal_gamma=2.0,
        label_smoothing=0.05, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=512, pretrain_epochs=100, finetune_epochs=180,
        early_stop_patience=15, t0_epochs=20, attn_num_heads=4,
        skip_proj_dim=16, num_workers=0, seed=42,
    )
    mc_out = str(Path(OUTPUT_DIR) / "multiclass_outputs")
    model, results = run_multiclass_pipeline(
        train_csv=TRAIN_CSV, test_csv=TEST_CSV,
        cfg=cfg, output_dir=mc_out, apply_smote=True, val_split=0.15,
    )
    log.info(f"Accuracy={results['overall_accuracy']*100:.3f}%  MacroF1={results['macro_f1']:.4f}")
    _try_xai(model, cfg, mc_out, "multiclass")


def run_binary():
    _check_files()
    import numpy as np, pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    from nids_deep_model import NIDSConfig, run_full_pipeline

    train_df  = pd.read_csv(TRAIN_CSV)
    test_df   = pd.read_csv(TEST_CSV)
    feat_cols = [c for c in train_df.columns if c != "label"]
    X_all     = train_df[feat_cols].values.astype(np.float32)
    y_all     = (train_df["label"].values > 0).astype(np.float32)
    X_test    = test_df[feat_cols].values.astype(np.float32)
    y_test    = (test_df["label"].values > 0).astype(np.float32)

    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=42)
    ti, vi = next(sss.split(X_all, y_all))
    X_tr, y_tr = X_all[ti], y_all[ti]
    X_va, y_va = X_all[vi], y_all[vi]

    cfg = NIDSConfig(
        input_dim=X_tr.shape[1], encoder_dims=[512,256,128], latent_dim=64,
        classifier_hidden=128, dropout_rate=0.30, noise_std=0.05,
        sparsity_weight=1e-3, sparsity_target=0.05,
        learning_rate=1e-3, weight_decay=1e-4,
        batch_size=512, pretrain_epochs=100, finetune_epochs=180,
        early_stop_patience=15, num_workers=0, seed=42,
    )
    bin_out = str(Path(OUTPUT_DIR) / "binary_outputs")
    Path(bin_out).mkdir(parents=True, exist_ok=True)

    model, results, Z_test = run_full_pipeline(
        X_tr, y_tr, X_va, y_va, X_test, y_test,
        cfg=cfg, save_path=str(Path(bin_out) / "nids_binary_model.pt"),
    )
    _try_xai(model, cfg, bin_out, "binary",
             X_test=X_test, y_test=y_test.astype(int), feat_names=feat_cols)


def run_xai_only():
    _check_files()
    import torch, numpy as np
    from nids_multiclass import MultiClassNIDSModel, MultiClassNIDSConfig, load_hybrid_csv
    from nids_xai_metrics import run_xai_pipeline

    if not Path(SAVED_MODEL_PATH).exists():
        log.error(f"Model not found: {SAVED_MODEL_PATH}")
        sys.exit(1)

    ckpt = torch.load(SAVED_MODEL_PATH, map_location="cpu", weights_only=False)
    cfg  = MultiClassNIDSConfig(**ckpt["config"])
    model = MultiClassNIDSModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    feat_names = ckpt.get("feature_names_kept", [f"f{i}" for i in range(cfg.input_dim)])
    log.info(f"Loaded: {SAVED_MODEL_PATH}  ({cfg.input_dim} features, latent={cfg.latent_dim})")

    X_test, y_test, _ = load_hybrid_csv(TEST_CSV)
    mask = ckpt.get("pruner_mask")
    if mask:
        X_test = X_test[:, np.array(mask)]
        log.info(f"Feature mask applied: {X_test.shape[1]} features")

    run_xai_pipeline(
        model=model, cfg=cfg,
        X_test=X_test, y_test_binary=(y_test > 0).astype(int),
        feature_names=feat_names, model_type="multiclass",
        y_test_multiclass=y_test,
        output_dir=str(Path(OUTPUT_DIR) / "xai_outputs"),
        n_shap_explain=200, n_shap_samples=200, n_lime_samples=2000,
    )


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log.info(f"MODE = {MODE}")

    if   MODE == "multiclass_research": run_multiclass_research()
    elif MODE == "multiclass":          run_multiclass()
    elif MODE == "binary":              run_binary()
    elif MODE == "xai_only":            run_xai_only()
    else:
        log.error(f"Unknown MODE '{MODE}'. Choose: multiclass_research | multiclass | binary | xai_only")
        sys.exit(1)
