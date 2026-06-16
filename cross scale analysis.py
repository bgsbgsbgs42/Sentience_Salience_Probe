import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
from transformer_lens import HookedTransformer
import torch
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")

warnings.filterwarnings('ignore')

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [
        logging.FileHandler("cross_scale.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Everything that varies across model sizes."""
    name:          str         # TransformerLens model name
    short_name:    str         # display label
    n_layers:      int
    n_heads:       int
    d_model:       int
    hdf5_path:     str
    color:         str         # plot colour
    linestyle:     str
    marker:        str
    params_b:      float       # approximate parameter count in billions

MODEL_CONFIGS = [
    ModelConfig(
        name       = "pythia-160m",
        short_name = "160M",
        n_layers   = 12,
        n_heads    = 12,
        d_model    = 768,
        hdf5_path  = "activations_160m.h5",
        color      = "#74c476",
        linestyle  = ":",
        marker     = "^",
        params_b   = 0.16,
    ),
    ModelConfig(
        name       = "EleutherAI/pythia-410m",
        short_name = "410M",
        n_layers   = 24,
        n_heads    = 16,
        d_model    = 1024,
        hdf5_path  = "activations_410m.h5",
        color      = "#fd8d3c",
        linestyle  = "--",
        marker     = "s",
        params_b   = 0.41,
    ),
    ModelConfig(
        name       = "pythia-1.4b",
        short_name = "1.4B",
        n_layers   = 24,
        n_heads    = 16,
        d_model    = 2048,
        hdf5_path  = "activations_1.4b.h5",
        color      = "#08306b",
        linestyle  = "-",
        marker     = "o",
        params_b   = 1.4,
    ),
]

TIER_ORDER  = ["T1","T2","T3","T4","T5","T6"]
TIER_CONFIG = {
    "T1": {"label": "Humans",              "color": "#08306b"},
    "T2": {"label": "Mammals",             "color": "#2171b5"},
    "T3": {"label": "Vertebrates",         "color": "#6baed6"},
    "T4": {"label": "Invertebrates (w+)",  "color": "#fd8d3c"},
    "T5": {"label": "Invertebrates (min)", "color": "#d7301f"},
    "T6": {"label": "Non-sentient",        "color": "#bdbdbd"},
}

WELFARE_TYPES = ["pain_suffering","fear","moral_consideration","cross_framing"]
NEUTRAL_TYPES = ["neutral_control"]

N_FOLDS      = 5
RANDOM_SEED  = 42
N_SHUFFLE    = 20

LOOKUP_PATH  = "GitHub/Sentience_Salience_Probe (Futurekind_Fellowship)/entity_position_lookup.json"

log.info("Cross-Scale Sentience Gradient Analysis")
log.info("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACTION PIPELINE (self-contained, model-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def extract_corpus_for_model(cfg:        ModelConfig,
                              lookup:     dict,
                              device:     str = "cpu") -> None:
    """
    Run full corpus extraction for one model size and save to HDF5.
    Skips if HDF5 already exists and contains the expected number of groups.
    """
    hdf5_path = Path(cfg.hdf5_path)
    n_jobs    = sum(len(data["entries"]) for data in lookup.values())

    # Check if already complete
    if hdf5_path.exists():
        try:
            with h5py.File(hdf5_path, "r") as f:
                n_done = len(f.keys())
                if n_done >= n_jobs:
                    log.info(f"  {cfg.short_name}: HDF5 already complete "
                            f"({n_done} groups) — skipping extraction")
                return
        except OSError:
            print(
                f"Corrupted HDF5 detected: {hdf5_path}"
            )

            hdf5_path.unlink()

            n_done = 0
        
    """if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as f:
            n_done = len(f.keys())"""
            
    
    log.info(f"  {cfg.short_name}: resuming extraction")

    log.info(f"  Loading {cfg.name}...")
    t0    = time.time()
    model = HookedTransformer.from_pretrained(
        cfg.name,
        dtype = torch.float16,
    )
    model.eval()
    model = model.to(device)
    log.info(f"  Loaded in {time.time()-t0:.1f}s")

    # Build job list
    jobs = []
    for entity, data in lookup.items():
        for tid, entry in data["entries"].items():
            jobs.append({
                "sentence_id": f"{tid}_{entity.replace(' ','_')}",
                "entity":      entity,
                "tier":        data["tier"],
                "template_id": tid,
                "template_type": entry["template_type"],
                "sentence":    entry["sentence"],
                "positions":   entry["positions"],
            })
    jobs.sort(key=lambda j: (j["tier"], j["entity"], j["template_id"]))

    # Load completed set
    completed = set()
    if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as f:
            completed = set(f.keys())

    remaining = [j for j in jobs if j["sentence_id"] not in completed]
    log.info(f"  {cfg.short_name}: {len(remaining)} sentences to extract")

    np_dtype   = np.float16
    hdf5_mode  = "a" if hdf5_path.exists() else "w"
    n_errors   = 0

    with h5py.File(hdf5_path, hdf5_mode) as f:
        f.attrs.update({
            "model_name": cfg.name,
            "n_layers":   cfg.n_layers,
            "n_heads":    cfg.n_heads,
            "d_model":    cfg.d_model,
        })

        for idx, job in enumerate(remaining):
            try:
                tokens  = model.to_tokens(job["sentence"])
                seq_len = tokens.shape[1]
                positions = job["positions"]

                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens)

                resid_list = []
                mlp_list   = []
                attn_to_list = []
                attn_from_list = []

                for layer in range(cfg.n_layers):
                    resid = cache[f"blocks.{layer}.hook_resid_post"]
                    resid_list.append(
                        resid[0, positions, :].mean(0)
                        .cpu().float().numpy().astype(np_dtype)
                    )

                    mlp = cache[f"blocks.{layer}.hook_mlp_out"]
                    mlp_list.append(
                        mlp[0, positions, :].mean(0)
                        .cpu().float().numpy().astype(np_dtype)
                    )

                    attn = cache[f"blocks.{layer}.attn.hook_pattern"]
                    attn_to_list.append(
                        attn[0, :, :, positions].mean(-1)
                        .cpu().float().numpy().astype(np_dtype)
                    )
                    attn_from_list.append(
                        attn[0, :, positions, :].mean(1)
                        .cpu().float().numpy().astype(np_dtype)
                    )

                grp = f.create_group(job["sentence_id"])
                grp.attrs.update({
                    "entity":        job["entity"],
                    "tier":          job["tier"],
                    "template_id":   job["template_id"],
                    "template_type": job["template_type"],
                    "sentence":      job["sentence"],
                    "positions":     job["positions"],
                    "n_tokens":      seq_len,
                })

                opts = dict(compression="gzip", compression_opts=4)
                grp.create_dataset("resid_post",
                    data=np.stack(resid_list),   **opts)
                grp.create_dataset("mlp_out",
                    data=np.stack(mlp_list),     **opts)
                grp.create_dataset("attn_to",
                    data=np.stack(attn_to_list), **opts)
                grp.create_dataset("attn_from",
                    data=np.stack(attn_from_list),**opts)

                del cache
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                n_errors += 1
                log.error(f"  Error on {job['sentence_id']}: {e}")

            if (idx+1) % 200 == 0:
                elapsed = time.time() - t0
                rate    = (idx+1) / elapsed
                eta     = (len(remaining) - idx - 1) / (rate + 1e-9)
                log.info(f"  {cfg.short_name}: {idx+1}/{len(remaining)} "
                         f"({rate:.1f}/s  ETA {eta/60:.1f}min  "
                         f"errors={n_errors})")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    log.info(f"  {cfg.short_name}: extraction complete "
             f"({n_errors} errors)  "
             f"file={hdf5_path.stat().st_size/1e9:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROBE PIPELINE (model-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def load_resid_matrix(hdf5_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load residual stream matrix and metadata from HDF5."""
    matrices  = []
    meta_rows = []

    with h5py.File(hdf5_path, "r") as f:
        for sid in sorted(f.keys()):
            grp = f[sid]
            matrices.append(grp["resid_post"][:].astype(np.float32))
            meta_rows.append({
                "sentence_id":   sid,
                "entity":        grp.attrs["entity"],
                "tier":          grp.attrs["tier"],
                "template_type": grp.attrs["template_type"],
            })

    meta_df = pd.DataFrame(meta_rows)
    meta_df["is_welfare"] = meta_df["template_type"].isin(WELFARE_TYPES)
    meta_df["is_neutral"]  = meta_df["template_type"].isin(NEUTRAL_TYPES)
    meta_df["binary_label"] = (meta_df["tier"] != "T6").astype(int)
    meta_df["tier_rank"]    = meta_df["tier"].map(
        {"T1":1,"T2":2,"T3":3,"T4":4,"T5":5,"T6":6}
    )

    return np.stack(matrices), meta_df


def make_lr_pipeline(n_classes: int) -> Pipeline:
    params = dict(max_iter=1000, solver="lbfgs", C=1.0,
                  random_state=RANDOM_SEED,
                  multi_class="ovr" if n_classes==2 else "multinomial")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(**params)),
    ])


def run_probe_layer(X: np.ndarray,
                    y: np.ndarray,
                    n_folds: int = N_FOLDS) -> dict:
    """5-fold CV probe for one layer."""
    n_classes = len(np.unique(y))
    pipe = make_lr_pipeline(n_classes)
    cv   = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=RANDOM_SEED)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy","f1_macro"], n_jobs=-1
    )
    return {
        "mean_acc":     scores["test_accuracy"].mean(),
        "std_acc":      scores["test_accuracy"].std(),
        "fold_accs":    scores["test_accuracy"].tolist(),
        "mean_f1":      scores["test_f1_macro"].mean(),
        "chance":       1.0 / n_classes,
    }


def run_generalisation_probe(matrix:       np.ndarray,
                              y:            np.ndarray,
                              train_mask:   np.ndarray,
                              test_mask:    np.ndarray,
                              n_layers:     int) -> np.ndarray:
    """Train on train_mask, evaluate on test_mask at every layer."""
    accs = []
    for layer in range(n_layers):
        pipe = make_lr_pipeline(len(np.unique(y[train_mask])))
        pipe.fit(matrix[train_mask, layer, :], y[train_mask])
        accs.append(pipe.score(matrix[test_mask, layer, :], y[test_mask]))
    return np.array(accs)


def run_shuffle_null(matrix:   np.ndarray,
                     y:         np.ndarray,
                     n_layers:  int,
                     n_runs:    int = N_SHUFFLE) -> np.ndarray:
    """Build shuffled-label null distribution. Returns (n_runs, n_layers)."""
    rng  = np.random.default_rng(RANDOM_SEED)
    accs = np.zeros((n_runs, n_layers))
    for run in range(n_runs):
        y_shuf = rng.permutation(y)
        for layer in range(n_layers):
            res = run_probe_layer(matrix[:, layer, :], y_shuf, n_folds=3)
            accs[run, layer] = res["mean_acc"]
    return accs


def full_probe_suite(cfg:     ModelConfig,
                     matrix:  np.ndarray,
                     meta_df: pd.DataFrame) -> dict:
    """
    Run the complete probe suite for one model.
    Returns dict of result arrays.
    """
    n_layers  = cfg.n_layers
    y_6class  = meta_df["tier"].values
    y_binary  = meta_df["binary_label"].values
    welfare   = meta_df["is_welfare"].values
    neutral   = meta_df["is_neutral"].values

    log.info(f"  [{cfg.short_name}] Running 6-class probe...")
    probe_6    = []
    probe_bin  = []
    probe_f1   = []

    for layer in range(n_layers):
        X = matrix[:, layer, :]
        r6  = run_probe_layer(X, y_6class)
        rb  = run_probe_layer(X, y_binary)
        probe_6.append(r6["mean_acc"])
        probe_bin.append(rb["mean_acc"])
        probe_f1.append(r6["mean_f1"])
        if layer % (n_layers // 4) == 0:
            log.info(f"    Layer {layer:2d}/{n_layers-1}  "
                     f"6cls={r6['mean_acc']:.3f}  "
                     f"bin={rb['mean_acc']:.3f}")

    log.info(f"  [{cfg.short_name}] Running generalisation probe...")
    gen_w2n = run_generalisation_probe(
        matrix, y_6class, welfare, neutral, n_layers
    )

    log.info(f"  [{cfg.short_name}] Running shuffle null...")
    shuffle_null = run_shuffle_null(matrix, y_6class, n_layers, N_SHUFFLE)

    # Spearman r: tier rank vs PC1 proxy
    # Use layer-wise mean representational structure
    tier_rank    = meta_df["tier_rank"].values
    spearman_r   = []
    for layer in range(n_layers):
        X = matrix[:, layer, :]
        # Use first principal component as scalar proxy
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=1, random_state=RANDOM_SEED)
        pc1  = pca.fit_transform(
            StandardScaler().fit_transform(X)
        )[:, 0]
        r, p = stats.spearmanr(tier_rank, pc1)
        spearman_r.append(abs(r))

    # Per-tier accuracy at peak layer
    peak_layer = int(np.argmax(probe_6))
    X_peak     = matrix[:, peak_layer, :]
    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=RANDOM_SEED)
    pipe_peak  = make_lr_pipeline(6)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in cv.split(X_peak, y_6class):
        pipe_peak.fit(X_peak[train_idx], y_6class[train_idx])
        y_pred_all.extend(pipe_peak.predict(X_peak[test_idx]))
        y_true_all.extend(y_6class[test_idx])
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    tier_accs_peak = {
        t: (y_pred_all[y_true_all==t]==t).mean()
        for t in TIER_ORDER if (y_true_all==t).sum() > 0
    }

    return {
        "probe_6":        np.array(probe_6),
        "probe_binary":   np.array(probe_bin),
        "probe_f1":       np.array(probe_f1),
        "gen_w2n":        gen_w2n,
        "shuffle_null":   shuffle_null,
        "shuffle_mean":   shuffle_null.mean(axis=0),
        "shuffle_95":     np.percentile(shuffle_null, 95, axis=0),
        "spearman_r":     np.array(spearman_r),
        "peak_layer":     peak_layer,
        "tier_accs_peak": tier_accs_peak,
        "n_layers":       n_layers,
        "chance_6":       1/6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LAYER NORMALISATION (fractional depth)
# ══════════════════════════════════════════════════════════════════════════════

def normalise_layers(values:   np.ndarray,
                     n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert layer indices to fractional depth [0, 1].
    Essential for comparing models with different layer counts.
    Returns (fractional_depths, values).
    """
    depths = np.linspace(0, 1, n_layers)
    return depths, values


def interpolate_to_common_grid(values:   np.ndarray,
                                n_layers: int,
                                n_grid:   int = 100) -> np.ndarray:
    """Interpolate layer-wise values to a common fractional grid."""
    depths = np.linspace(0, 1, n_layers)
    grid   = np.linspace(0, 1, n_grid)
    return np.interp(grid, depths, values)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCALE EFFECT METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale_metrics(all_results: dict) -> pd.DataFrame:
    """
    Compute metrics that quantify how the sentience gradient changes with scale.
    """
    rows = []
    for short_name, (cfg, results) in all_results.items():
        n     = results["n_layers"]
        probe = results["probe_6"]
        null  = results["shuffle_mean"]
        r     = results["spearman_r"]

        # Peak accuracy above chance
        peak_above_chance = probe.max() - results["chance_6"]

        # Area under the probe curve (above chance)
        auc = np.trapz(np.maximum(probe - results["chance_6"], 0)) / n

        # Layer at which probe first exceeds null 95th pct
        null_95 = results["shuffle_95"]
        exceeds = np.where(probe > null_95)[0]
        first_exceed_frac = (exceeds[0] / n) if len(exceeds) > 0 else np.nan

        # Fraction of layers significantly above null
        frac_sig = (probe > null_95).mean()

        # Peak Spearman r
        peak_r = r.max()

        # Generalisation retention: gen_w2n peak / in-dist peak
        gen_retention = results["gen_w2n"].max() / (probe.max() + 1e-10)

        rows.append({
            "model":              short_name,
            "params_b":           cfg.params_b,
            "n_layers":           n,
            "peak_acc_6class":    probe.max(),
            "peak_above_chance":  peak_above_chance,
            "peak_layer":         results["peak_layer"],
            "peak_layer_frac":    results["peak_layer"] / n,
            "auc_above_chance":   auc,
            "first_exceed_frac":  first_exceed_frac,
            "frac_sig_layers":    frac_sig,
            "peak_spearman_r":    peak_r,
            "peak_binary":        results["probe_binary"].max(),
            "gen_w2n_peak":       results["gen_w2n"].max(),
            "gen_retention":      gen_retention,
        })

    return pd.DataFrame(rows).sort_values("params_b")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RUN FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

log.info("\nLoading corpus lookup...")
with open(LOOKUP_PATH) as f:
    lookup = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {device}")

# ── Step 1: Extract activations for each model ─────────────────────────────
log.info("\n── Step 1: Extraction ──")
for cfg in MODEL_CONFIGS:
    log.info(f"\nExtracting {cfg.short_name}...")
    extract_corpus_for_model(cfg, lookup, device)

# ── Step 2: Load matrices and run probes ───────────────────────────────────
log.info("\n── Step 2: Probe analysis ──")
all_results = {}   # short_name → (cfg, results_dict)

for cfg in MODEL_CONFIGS:
    log.info(f"\n[{cfg.short_name}] Loading matrix...")
    matrix, meta_df = load_resid_matrix(cfg.hdf5_path)
    log.info(f"  Matrix shape: {matrix.shape}")
    log.info(f"  Running probe suite...")
    results = full_probe_suite(cfg, matrix, meta_df)
    all_results[cfg.short_name] = (cfg, results)
    log.info(f"  Peak 6-class accuracy: {results['probe_6'].max():.3f} "
             f"at layer {results['peak_layer']}/{cfg.n_layers-1}")

# ── Step 3: Scale metrics ──────────────────────────────────────────────────
scale_df = compute_scale_metrics(all_results)

log.info("\n── Scale metrics ──")
log.info(scale_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

log.info("\nBuilding figures...")

GRID_N  = 100
x_grid  = np.linspace(0, 1, GRID_N)   # common fractional depth axis

# Interpolate all curves to common grid
interp_curves = {}
for short_name, (cfg, results) in all_results.items():
    interp_curves[short_name] = {
        "probe_6":      interpolate_to_common_grid(
                            results["probe_6"], cfg.n_layers),
        "probe_binary": interpolate_to_common_grid(
                            results["probe_binary"], cfg.n_layers),
        "probe_f1":     interpolate_to_common_grid(
                            results["probe_f1"], cfg.n_layers),
        "gen_w2n":      interpolate_to_common_grid(
                            results["gen_w2n"], cfg.n_layers),
        "spearman_r":   interpolate_to_common_grid(
                            results["spearman_r"], cfg.n_layers),
        "shuffle_mean": interpolate_to_common_grid(
                            results["shuffle_mean"], cfg.n_layers),
        "shuffle_95":   interpolate_to_common_grid(
                            results["shuffle_95"], cfg.n_layers),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main cross-scale comparison
# ══════════════════════════════════════════════════════════════════════════════

fig1 = plt.figure(figsize=(22, 26))
gs1  = gridspec.GridSpec(4, 2, figure=fig1,
                          hspace=0.45, wspace=0.32)
fig1.suptitle(
    "Cross-Scale Sentience Gradient Analysis\n"
    "Pythia 160M  →  410M  →  1.4B\n"
    "Key question: does the sentience gradient intensify with model scale?",
    fontsize=14, fontweight="bold", y=0.99
)


def draw_model_curves(ax, metric_key, ylabel, title,
                       show_null=True, chance=1/6,
                       raw_layers=False):
    """
    Draw probe accuracy curves for all models on one axis.
    raw_layers=True: use raw layer index (for per-model panels).
    raw_layers=False: use fractional depth (for cross-model comparison).
    """
    for short_name, (cfg, results) in all_results.items():
        if raw_layers:
            x     = np.arange(cfg.n_layers)
            y     = results[metric_key]
            y_std = results.get(metric_key + "_std",
                                np.zeros(cfg.n_layers))
        else:
            x = x_grid
            y = interp_curves[short_name][metric_key]

        ax.plot(x, y,
                color     = cfg.color,
                linestyle = cfg.linestyle,
                linewidth = 2.2,
                marker    = cfg.marker,
                markersize= 3.5 if raw_layers else 0,
                markevery = max(1, cfg.n_layers // 8),
                label     = f"Pythia-{short_name}  "
                            f"(peak={results[metric_key].max():.3f})",
                alpha     = 0.9,
                zorder    = 5)

        # Peak marker
        peak_y   = results[metric_key].max()
        peak_idx = int(np.argmax(results[metric_key]))
        if raw_layers:
            peak_x = peak_idx
        else:
            peak_x = peak_idx / cfg.n_layers
        ax.plot(peak_x, peak_y,
                marker    = "*",
                color     = cfg.color,
                markersize= 12,
                zorder    = 6)

    # Null distribution from largest model
    if show_null:
        lg_name   = "1.4B"
        lg_cfg, lg_res = all_results[lg_name]
        if raw_layers:
            x_null   = np.arange(lg_cfg.n_layers)
            null_m   = lg_res["shuffle_mean"]
            null_95  = lg_res["shuffle_95"]
        else:
            x_null  = x_grid
            null_m  = interp_curves[lg_name]["shuffle_mean"]
            null_95 = interp_curves[lg_name]["shuffle_95"]

        ax.fill_between(x_null, null_m - 0.01, null_95,
                        alpha=0.12, color="grey",
                        label="Null band (1.4B, shuffled labels)")
        ax.plot(x_null, null_m,
                color="grey", linewidth=1.2,
                linestyle="--", alpha=0.6, label="Null mean")

    if chance is not None:
        ax.axhline(chance, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5,
                   label=f"Chance ({chance*100:.0f}%)")

    ax.set_xlabel(
        "Layer (raw)" if raw_layers else "Fractional depth [0 = first, 1 = final]",
        fontsize=10
    )
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    )


# ── Panel A: 6-class probe (fractional depth) ─────────────────────────────
ax_a = fig1.add_subplot(gs1[0, :])
draw_model_curves(
    ax_a,
    metric_key = "probe_6",
    ylabel     = "6-class accuracy",
    title      = "Panel A — 6-class Tier Probe: All Model Sizes\n"
                 "(x-axis = fractional depth; enables direct layer comparison across sizes)",
    show_null  = True,
    chance     = 1/6,
    raw_layers = False,
)

# Add vertical annotation: where does each model's peak fall?
for short_name, (cfg, results) in all_results.items():
    peak_frac = results["peak_layer"] / cfg.n_layers
    ax_a.axvline(peak_frac, color=cfg.color,
                 linewidth=1, linestyle=":", alpha=0.5)
    ax_a.text(peak_frac, 0.02, f"L{results['peak_layer']}\n({short_name})",
              ha="center", va="bottom", fontsize=7,
              color=cfg.color, fontweight="bold")


# ── Panel B: Binary probe ─────────────────────────────────────────────────
ax_b = fig1.add_subplot(gs1[1, 0])
draw_model_curves(
    ax_b,
    metric_key = "probe_binary",
    ylabel     = "Binary accuracy (sentient vs non-sentient)",
    title      = "Panel B — Binary Probe\nSentient (T1–T5) vs Non-sentient (T6)",
    show_null  = False,
    chance     = 0.5,
    raw_layers = False,
)


# ── Panel C: Spearman r (tier rank vs PC1) ────────────────────────────────
ax_c = fig1.add_subplot(gs1[1, 1])
draw_model_curves(
    ax_c,
    metric_key = "spearman_r",
    ylabel     = "|Spearman r| (tier rank vs PC1)",
    title      = "Panel C — PC1 Tier-Separation Strength\n"
                 "|Spearman r| between tier rank and first principal component",
    show_null  = False,
    chance     = None,
    raw_layers = False,
)
ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))


# ── Panel D: Generalisation welfare→neutral ───────────────────────────────
ax_d = fig1.add_subplot(gs1[2, 0])
draw_model_curves(
    ax_d,
    metric_key = "gen_w2n",
    ylabel     = "Test accuracy (neutral sentences)",
    title      = "Panel D — Generalisation: Welfare→Neutral\n"
                 "Does the gradient persist in entity-level encoding?",
    show_null  = False,
    chance     = 1/6,
    raw_layers = False,
)


# ── Panel E: Scale effect — peak accuracy vs log(params) ──────────────────
ax_e = fig1.add_subplot(gs1[2, 1])

for _, row in scale_df.iterrows():
    cfg_match = next(c for c in MODEL_CONFIGS
                     if c.short_name == row["model"])
    ax_e.scatter(
        row["params_b"], row["peak_acc_6class"],
        color     = cfg_match.color,
        s         = 120,
        zorder    = 5,
        marker     = cfg_match.marker,
        edgecolors = "white",
        linewidths = 1,
    )
    ax_e.annotate(
        f"Pythia-{row['model']}\n"
        f"peak={row['peak_acc_6class']*100:.1f}%\n"
        f"L{int(row['peak_layer'])}",
        (row["params_b"], row["peak_acc_6class"]),
        xytext    = (5, 8), textcoords="offset points",
        fontsize  = 8, color=cfg_match.color,
        fontweight= "bold",
    )

# Fit log-linear trend if 3+ points
if len(scale_df) >= 2:
    log_params = np.log(scale_df["params_b"].values)
    peak_accs  = scale_df["peak_acc_6class"].values
    slope, intercept, r, p, se = stats.linregress(log_params, peak_accs)
    x_trend = np.linspace(scale_df["params_b"].min() * 0.7,
                           scale_df["params_b"].max() * 1.3, 100)
    y_trend = intercept + slope * np.log(x_trend)
    ax_e.plot(x_trend, y_trend, color="grey",
              linestyle="--", linewidth=1.5, alpha=0.7,
              label=f"Log-linear fit (r={r:.2f})")
    ax_e.legend(fontsize=8)

ax_e.axhline(1/6, color="black", linewidth=0.8,
             linestyle=":", alpha=0.5, label="Chance")
ax_e.set_xscale("log")
ax_e.set_xlabel("Model size (billion parameters)", fontsize=10)
ax_e.set_ylabel("Peak 6-class probe accuracy", fontsize=10)
ax_e.set_title(
    "Panel E — Scale Effect\nDoes the gradient intensify with model size?",
    fontsize=10, fontweight="bold"
)
ax_e.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
)


# ── Panel F: Per-tier accuracy at peak layer, by model ────────────────────
ax_f = fig1.add_subplot(gs1[3, :])

n_models = len(MODEL_CONFIGS)
n_tiers  = len(TIER_ORDER)
x_f      = np.arange(n_tiers)
width_f  = 0.25
offsets  = np.linspace(-(n_models-1)*width_f/2,
                        (n_models-1)*width_f/2, n_models)

for m_idx, (short_name, (cfg, results)) in enumerate(all_results.items()):
    tier_accs = [
        results["tier_accs_peak"].get(t, 0) for t in TIER_ORDER
    ]
    ax_f.bar(
        x_f + offsets[m_idx],
        tier_accs,
        width   = width_f,
        color   = cfg.color,
        alpha   = 0.85,
        label   = f"Pythia-{short_name} (peak L{results['peak_layer']})",
        edgecolor = "white",
        linewidth = 0.5,
    )

ax_f.axhline(1/6, color="black", linewidth=1,
             linestyle="--", alpha=0.6, label="Chance")
ax_f.set_xticks(x_f)
ax_f.set_xticklabels(
    [f"{t}\n{TIER_CONFIG[t]['label']}" for t in TIER_ORDER],
    fontsize=9
)
for tick, tier in zip(ax_f.get_xticklabels(), TIER_ORDER):
    tick.set_color(TIER_CONFIG[tier]["color"])
ax_f.set_ylabel("Per-tier accuracy at peak layer", fontsize=10)
ax_f.set_title(
    "Panel F — Per-Tier Accuracy at Peak Layer, by Model Size\n"
    "Which tiers become more discriminable as scale increases?",
    fontsize=10, fontweight="bold"
)
ax_f.legend(fontsize=9, loc="upper right")
ax_f.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
)
ax_f.set_ylim(0, 1)

plt.savefig("cross_scale_main.png", dpi=150, bbox_inches="tight")
plt.show()
log.info("Saved: cross_scale_main.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Raw-layer probe curves (one panel per model)
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig2.suptitle(
    "Probe Accuracy vs Raw Layer Depth — Per Model\n"
    "Note different x-axis ranges (12 vs 24 layers)",
    fontsize=13, fontweight="bold"
)

for ax, (short_name, (cfg, results)) in zip(axes2, all_results.items()):
    layers = np.arange(cfg.n_layers)

    # Null band
    ax.fill_between(layers,
                    results["shuffle_mean"],
                    results["shuffle_95"],
                    alpha=0.15, color="grey",
                    label="Null band (shuffled)")
    ax.plot(layers, results["shuffle_mean"],
            color="grey", linewidth=1, linestyle="--",
            alpha=0.6)

    # 6-class probe
    ax.fill_between(layers,
                    results["probe_6"] - 0.01,
                    results["probe_6"] + 0.01,
                    alpha=0.15, color=cfg.color)
    ax.plot(layers, results["probe_6"],
            color=cfg.color, linewidth=2.5,
            marker=cfg.marker, markersize=5,
            label=f"6-class (peak={results['probe_6'].max():.3f})")

    # Binary probe
    ax.plot(layers, results["probe_binary"],
            color=cfg.color, linewidth=1.8,
            linestyle="--", marker=cfg.marker,
            markersize=4, alpha=0.7,
            label=f"Binary (peak={results['probe_binary'].max():.3f})")

    # Generalisation
    ax.plot(layers, results["gen_w2n"],
            color=cfg.color, linewidth=1.5,
            linestyle=":", marker=cfg.marker,
            markersize=3, alpha=0.6,
            label=f"Gen w→n (peak={results['gen_w2n'].max():.3f})")

    ax.axhline(1/6, color="black", linewidth=0.8,
               linestyle=":", alpha=0.5)
    ax.axvline(results["peak_layer"], color=cfg.color,
               linewidth=1.2, linestyle=":", alpha=0.5)

    ax.set_xlabel("Layer (raw)", fontsize=10)
    ax.set_ylabel("Accuracy" if ax == axes2[0] else "", fontsize=10)
    ax.set_title(
        f"Pythia-{short_name}\n"
        f"{cfg.n_layers} layers  |  d_model={cfg.d_model}  |  "
        f"{cfg.params_b}B params",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xticks(range(0, cfg.n_layers, max(1, cfg.n_layers//6)))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    )
    ax.set_ylim(0.1, 1.0)

plt.tight_layout()
plt.savefig("cross_scale_per_model.png", dpi=150, bbox_inches="tight")
plt.show()
log.info("Saved: cross_scale_per_model.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Scale effect deep dive
# ══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle(
    "Scale Effect Deep Dive\n"
    "Quantifying how the sentience gradient changes across model sizes",
    fontsize=13, fontweight="bold"
)

metrics_to_plot = [
    ("peak_acc_6class",   "Peak 6-class accuracy",          "Panel A"),
    ("auc_above_chance",  "AUC above chance (6-class)",      "Panel B"),
    ("peak_spearman_r",   "Peak |Spearman r| (tier vs PC1)", "Panel C"),
    ("gen_retention",     "Generalisation retention",         "Panel D"),
    ("peak_layer_frac",   "Peak layer (fractional depth)",   "Panel E"),
    ("frac_sig_layers",   "Fraction of significant layers",  "Panel F"),
]

for ax, (metric, ylabel, panel) in zip(axes3.flatten(), metrics_to_plot):
    for _, row in scale_df.iterrows():
        cfg_m = next(c for c in MODEL_CONFIGS if c.short_name == row["model"])
        ax.scatter(
            row["params_b"], row[metric],
            color=cfg_m.color, s=130, zorder=5,
            marker=cfg_m.marker,
            edgecolors="white", linewidths=1.2,
        )
        ax.annotate(
            row["model"],
            (row["params_b"], row[metric]),
            xytext=(4, 5), textcoords="offset points",
            fontsize=8.5, color=cfg_m.color, fontweight="bold"
        )

    if len(scale_df) >= 2:
        log_p = np.log(scale_df["params_b"].values)
        vals  = scale_df[metric].values
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            sl, ic, rv, pv, _ = stats.linregress(log_p[valid], vals[valid])
            x_t = np.linspace(scale_df["params_b"].min()*0.7,
                               scale_df["params_b"].max()*1.3, 50)
            ax.plot(x_t, ic + sl*np.log(x_t),
                    color="grey", linestyle="--", linewidth=1.5,
                    alpha=0.6, label=f"r={rv:.2f}  p={pv:.3f}")
            ax.legend(fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Params (B)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(panel, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("cross_scale_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
log.info("Saved: cross_scale_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUALITATIVE LAYER COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  QUALITATIVE LAYER COMPARISON")
print("═"*70)

print(f"\n  {'Region':<18}", end="")
for short_name in all_results:
    print(f"  {'Pythia-'+short_name:<18}", end="")
print()
print("  " + "-"*18 + ("  " + "-"*18) * len(all_results))

regions = [
    ("Early (0–25%)",     0.00, 0.25),
    ("Early-mid (25–50%)",0.25, 0.50),
    ("Late-mid (50–75%)", 0.50, 0.75),
    ("Late (75–100%)",    0.75, 1.00),
]

for region_label, frac_lo, frac_hi in regions:
    print(f"\n  {region_label:<18}", end="")
    for short_name, (cfg, results) in all_results.items():
        depths = np.linspace(0, 1, cfg.n_layers)
        mask   = (depths >= frac_lo) & (depths < frac_hi)
        if mask.sum() > 0:
            mean_acc = results["probe_6"][mask].mean()
            peak_acc = results["probe_6"][mask].max()
            print(f"  mean={mean_acc:.3f} max={peak_acc:.3f}  ", end="")
        else:
            print(f"  —{'':16}", end="")
    print()

print(f"\n  Peak layer (fractional depth):")
for short_name, (cfg, results) in all_results.items():
    peak_frac = results["peak_layer"] / cfg.n_layers
    print(f"    Pythia-{short_name:<6}: "
          f"layer {results['peak_layer']:>2}/{cfg.n_layers-1}  "
          f"= {peak_frac:.2f} fractional depth")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAFETY INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("  SAFETY-RELEVANT INTERPRETATION")
print("═"*70)

smallest = scale_df.iloc[0]
largest  = scale_df.iloc[-1]

acc_increase = largest["peak_acc_6class"] - smallest["peak_acc_6class"]
r_increase   = largest["peak_spearman_r"] - smallest["peak_spearman_r"]

emergent = acc_increase > 0.05

print(f"""
  Scale effect on sentience gradient:
    Smallest model ({smallest['model']}): peak acc = {smallest['peak_acc_6class']*100:.1f}%
    Largest model  ({largest['model']}):  peak acc = {largest['peak_acc_6class']*100:.1f}%
    Increase across scale:               {acc_increase*100:+.1f} pp

    Spearman r (tier vs PC1):
    Smallest: {smallest['peak_spearman_r']:.3f}   Largest: {largest['peak_spearman_r']:.3f}
    Increase: {r_increase:+.3f}

  Verdict:
    {'  EMERGENT: Sentience gradient strengthens with scale.'
     if emergent else
     '  STABLE: Sentience gradient does not scale with model size.'}

    {'This is the safety-relevant finding: models may inherit increasingly'
     if emergent else
     'The gradient is present even in small models, suggesting it is not'} 
    {'structured implicit moral hierarchies as they scale toward frontier'
     if emergent else
     'an emergent property of scale but rather a stable artifact of'}
    {'capability levels. Mitigation should happen now, not at AGI.'
     if emergent else
     'training data composition at all scales examined.'}

  Peak layer shift across scale:
""")
for short_name, (cfg, results) in all_results.items():
    frac = results["peak_layer"] / cfg.n_layers
    region = ("early" if frac < 0.33 else
              "middle" if frac < 0.67 else "late")
    print(f"    Pythia-{short_name}: {region} layers "
          f"({frac:.2f} fractional depth)")

print(f"""
  If peak layer migrates toward earlier layers as scale increases:
    → Sentience encoding becomes more fundamental, computed earlier
    → Harder to remove via fine-tuning (entangled with early representations)

  If peak layer stays constant in fractional depth:
    → Encoding depth scales proportionally with model depth
    → Suggests systematic rather than incidental encoding
""")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

scale_df.to_csv("cross_scale_metrics.csv", index=False)

for short_name, (cfg, results) in all_results.items():
    pd.DataFrame({
        "layer":        range(cfg.n_layers),
        "frac_depth":   np.linspace(0, 1, cfg.n_layers),
        "probe_6":      results["probe_6"],
        "probe_binary": results["probe_binary"],
        "probe_f1":     results["probe_f1"],
        "gen_w2n":      results["gen_w2n"],
        "spearman_r":   results["spearman_r"],
        "shuffle_mean": results["shuffle_mean"],
        "shuffle_95":   results["shuffle_95"],
    }).to_csv(f"probe_results_{short_name.lower()}.csv", index=False)

log.info("\n── Output files ──")
for fname in [
    "cross_scale_main.png",
    "cross_scale_per_model.png",
    "cross_scale_metrics.png",
    "cross_scale_metrics.csv",
    *[f"probe_results_{cfg.short_name.lower()}.csv"
      for cfg in MODEL_CONFIGS],
    "cross_scale.log",
]:
    size = Path(fname).stat().st_size/1e3 if Path(fname).exists() else 0
    log.info(f"  {fname:<45} {size:.0f} KB")

log.info("\n Cross-scale analysis complete")