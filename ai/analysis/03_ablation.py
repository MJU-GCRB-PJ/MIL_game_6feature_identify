"""Run a weight-scaling ablation for one cross-validation fold."""

import argparse
import ast
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────
# Configuration.
# ──────────────────────────────────────────────────────────────
# Model setup.
SCALE_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ──────────────────────────────────────────────────────────────
# Load input.
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DIR = PROJECT_ROOT / "ai" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from cv_config import CV_OUTPUT_DIR, N_FOLDS, fold_data_csv, fold_dir  # noqa: E402

ENSEMBLE_PY = TRAINING_DIR / "08_ensemble.py"

if not ENSEMBLE_PY.exists():
    raise FileNotFoundError(f"08_ensemble.py not found: {ENSEMBLE_PY}")

spec = importlib.util.spec_from_file_location("ensemble_mod", ENSEMBLE_PY)
ensemble_mod = importlib.util.module_from_spec(spec)
sys.modules["ensemble_mod"] = ensemble_mod
spec.loader.exec_module(ensemble_mod)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--ensemble-row", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=CV_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    target_row_number = args.ensemble_row
    print(f"=== Ablation Study: FOLD `{args.fold}`, ENSEMBLE ROW No `{target_row_number}` ===")
    
    # Read input.
    output_dir = fold_dir(args.fold, args.output_root)
    ensemble_excel = output_dir / "ensemble" / "ensemble_results.xlsx"
    
    if not ensemble_excel.exists():
        raise FileNotFoundError(f"Ensemble result workbook not found: {ensemble_excel}")
        
    df_results = pd.read_excel(ensemble_excel, sheet_name="All_Results")
    

    row_mask = df_results['No'] == target_row_number
    if not row_mask.any():
        raise ValueError(f"All_Results has no row where No == {target_row_number}.")
        
    target_row = df_results[row_mask].iloc[0]
    combo_str = str(target_row['Model_Combination'])
    combo_labels = [label.strip() for label in combo_str.split('+')]
    weights_str = str(target_row['Model_Weights']).strip()
    

    label_to_key = {reg['label']: reg['key'] for reg in ensemble_mod.MODEL_REGISTRY}
    try:
        target_keys = [label_to_key[lbl] for lbl in combo_labels]
    except KeyError as e:
        raise KeyError(f"Model label not found in MODEL_REGISTRY: {e}\n(ensemble string: {combo_str})")
        

    base_weights_dict = {}
    if weights_str in ["N/A", "equal", "nan", ""]:
        base_w = 1.0 / len(target_keys)
        base_weights_dict = {k: base_w for k in target_keys}
    else:
        try:
            w_parsed = ast.literal_eval(weights_str)
            if isinstance(w_parsed, dict):

                for k_or_l, w_val in w_parsed.items():
                    if k_or_l in label_to_key:
                        base_weights_dict[label_to_key[k_or_l]] = float(w_val)
                    elif k_or_l in target_keys:
                        base_weights_dict[k_or_l] = float(w_val)
            else:
                base_weights_dict = {k: 1.0 / len(target_keys) for k in target_keys}
        except Exception as e:
            print(f"Warning: failed to parse weights ({weights_str}); using equal weights. Error: {e}")
            base_weights_dict = {k: 1.0 / len(target_keys) for k in target_keys}
            

    for k in target_keys:
        if k not in base_weights_dict:
            base_weights_dict[k] = 1.0 / len(target_keys)
            
    print(f"\n[Target model combination]: {combo_str}")
    print(f"[Extracted baseline weights]: {base_weights_dict}\n")
    
    # Model setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")
    
    ensemble_mod.ACTIVE_FOLD = args.fold
    ensemble_mod.OUTPUT_BASE = output_dir
    ensemble_mod.DATA_CSV = fold_data_csv(args.fold, args.output_root)
    ensemble_mod.ENSEMBLE_DIR = output_dir / "ensemble"
    
    data_csv_path = ensemble_mod.DATA_CSV
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {data_csv_path}")
        
    df_data = ensemble_mod._read_csv_df(data_csv_path)
    

    print("Loading and collecting model prediction probabilities...")
    predictions, labels, splits, all_file_names = ensemble_mod.collect_predictions(df_data, device)
    
    # Model setup.
    y_true, prob_tensor, avail_mask, split_flags, valid_fns = ensemble_mod._get_available_data(
        target_keys, predictions, labels, splits, all_file_names
    )
    
    # Convert data.
    base_w_tensor = torch.tensor([base_weights_dict[k] for k in target_keys], dtype=torch.float32, device=device)
    prob_tensor_gpu = torch.tensor(prob_tensor, dtype=torch.float32, device=device) # (M, N, C)
    avail_mask_gpu = torch.tensor(avail_mask, dtype=torch.float32, device=device)   # (M, N)
    

    scale_combinations = list(itertools.product(SCALE_STEPS, repeat=len(target_keys)))
    print(f"\nTotal ablation combinations in the scale-factor grid: {len(scale_combinations)}")
    print("Starting GPU-accelerated ablation...")
    
    rows = []
    
    for scale_tuple in tqdm(scale_combinations, desc="Weight-scaling ablation"):
        s_tensor = torch.tensor(scale_tuple, dtype=torch.float32, device=device)
        

        w_tensor = base_w_tensor * s_tensor
        sum_w = w_tensor.sum()
        
        # Model setup.
        if sum_w == 0.0:
            continue
            
        # Ensemble evaluation.
        w_norm = w_tensor / sum_w
        

        w_norm_cpu = w_norm.cpu().numpy()
        w_dict = {key: round(float(w), 4) for key, w in zip(target_keys, w_norm_cpu)}
        
        # ==================== Parallel processing ====================
        # w_norm: (M), prob_tensor_gpu: (M, N, C), avail_mask_gpu: (M, N)
        w_3d = w_norm.view(-1, 1, 1)
        # (M, N, C)
        weighted_prob = w_3d * prob_tensor_gpu * avail_mask_gpu.unsqueeze(-1)
        weighted_prob_sum = torch.sum(weighted_prob, dim=0) # (N, C)
        

        w_sum_avail = torch.sum(w_norm.view(-1, 1) * avail_mask_gpu, dim=0).unsqueeze(-1)
        w_sum_avail = torch.clamp(w_sum_avail, min=1e-12)
        
        y_prob_all_gpu = weighted_prob_sum / w_sum_avail  # (N, C)
        y_prob_all = y_prob_all_gpu.cpu().numpy()
        
        # ==================== Compute values ====================
        metrics = ensemble_mod._metrics_for_splits(y_true, y_prob_all, split_flags)
        

        row = ensemble_mod._flatten_metrics_to_row(
            no=len(rows)+1, 
            method=target_row['Ensemble_Method'] + "_Ablation", 
            model_combo=combo_str, 
            weights=str(w_dict), 
            metrics_dict=metrics
        )
        
        # Model setup.
        for k in target_keys:
            row[f"Weight_{k}"] = w_dict.get(k, 0.0)
            
        rows.append(row)
        
    # Save results.
    if not rows:
        print("No valid combinations were available.")
        return
        
    ablation_df = pd.DataFrame(rows)
    ablation_df = ablation_df.sort_values("Val_Macro_AUC", ascending=False).reset_index(drop=True)
    ablation_df['No'] = range(1, len(ablation_df)+1)
    
    # Batch processing.
    original_columns = df_results.columns.tolist()
    target_columns = [col for col in original_columns if col in ablation_df.columns]
    

    try:
        mw_idx = target_columns.index("Model_Weights")

        added_weight_cols = [col for col in ablation_df.columns if col.startswith("Weight_")]
        

        final_columns = target_columns[:mw_idx+1] + added_weight_cols + target_columns[mw_idx+1:]
    except ValueError:

        added_weight_cols = [col for col in ablation_df.columns if col.startswith("Weight_")]
        final_columns = target_columns + added_weight_cols
        
    ablation_df = ablation_df[final_columns]
    
    ablation_out_dir = output_dir / "ablation"
    ablation_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = ablation_out_dir / f"ablation_row_{target_row_number}_results.xlsx"
    
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        ablation_df.to_excel(writer, sheet_name="All_Results", index=False)
        
    print("\nAblation against the baseline weights completed successfully.")
    print(f"Results saved: {out_file}")

    # =========================================================================
    # Visualization.
    # =========================================================================
    print("\nGenerating plots...")
    

    x_indices = range(len(ablation_df))
    
    # Visualization.
    fig, ax1 = plt.subplots(figsize=(20, 11.25), dpi=100)
    

    # Sort values.
    desired_weight_order = [
        "Weight_vision", 
        "Weight_original_audio", 
        "Weight_vocal_audio", 
        "Weight_non_vocal_audio", 
        "Weight_ocr", 
        "Weight_stt"
    ]
    
    # Create required output.
    weight_cols = [col for col in desired_weight_order if col in ablation_df.columns]
    
    w_data = ablation_df[weight_cols].values
    
    # Graph generation.
    w_data_T = w_data.T
    stack_labels = [c.replace("Weight_", "") for c in weight_cols]
    

    ax1.set_zorder(1)
    ax1.stackplot(x_indices, w_data_T, labels=stack_labels, alpha=0.75)
    
    # Configuration.
    ax1.set_xlabel("Ranking (Tested Combinations, Sorted by Val_Macro_AUC)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Ensemble Weights (0~100%)", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    

    import matplotlib.ticker as mtick
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.margins(x=0)
    
    # Graph generation.
    # Create required output.
    ax2 = ax1.twinx()
    # Graph generation.
    ax2.set_zorder(2)
    # Configuration.
    ax2.patch.set_visible(False)
    
    # Metric handling.
    total_auc = ablation_df["Total_Macro_AUC"].values
    train_auc = ablation_df["Train_Macro_AUC"].values
    val_auc   = ablation_df["Val_Macro_AUC"].values
    

    min_auc = min(total_auc.min(), train_auc.min(), val_auc.min())
    max_auc = max(total_auc.max(), train_auc.max(), val_auc.max())
    padding = (max_auc - min_auc) * 0.1 if (max_auc - min_auc) > 0 else 0.05
    auc_ylim_bottom = float(max(0.0, min_auc - padding))
    auc_ylim_top    = float(min(1.0, max_auc + padding))
    
    # Metric handling.
    l1 = ax2.plot(x_indices, total_auc, label="Total AUC", color='black', linewidth=3, linestyle='-')
    l2 = ax2.plot(x_indices, train_auc, label="Train AUC", color='blue', linewidth=3, linestyle='--')
    l3 = ax2.plot(x_indices, val_auc,   label="Val AUC",   color='red', linewidth=3, linestyle='-.')
    
    # Configuration.
    ax2.set_ylabel("Macro AUC Score", fontsize=14, fontweight='bold')
    ax2.set_ylim(auc_ylim_bottom, auc_ylim_top)
    
    ax2.grid(True, linestyle=':', alpha=0.6)
    

    # Graph generation.
    
    handles1, labels1 = ax1.get_legend_handles_labels()

    leg1 = ax1.legend(handles1, labels1, loc="upper right", bbox_to_anchor=(-0.05, 1.0),
                      title="Weights Stack (Background)", fontsize=12)
    leg1.get_title().set_fontsize('12')
    
    handles2, labels2 = ax2.get_legend_handles_labels()

    leg2 = ax2.legend(handles2, labels2, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                      title="AUC Metrics (Foreground)", fontsize=12)
    leg2.get_title().set_fontsize('12')
    
    plt.title(f"Ablation Results for Ensemble Row {target_row_number}\n({combo_str})", fontsize=18, fontweight='bold')
    

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
    
    # Save output.
    plot_file = ablation_out_dir / f"ablation_row_{target_row_number}_trend.png"
    plt.savefig(plot_file, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_file}")

if __name__ == "__main__":
    main()
