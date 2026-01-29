#!/usr/bin/env python3
"""
Experiment Comparison Script
Runs evaluation on both baseline and multiscale models and compares results

Usage:
    python run_comparison.py
"""

import os
import sys
import json
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import get_data_loaders
from model import SwinUNetpp
from evaluate import evaluate_metrics

import numpy as np

def to_python(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj



def run_evaluation(config, experiment_name, checkpoint_path):
    """
    Run evaluation for a specific experiment
    
    Args:
        config: Configuration object
        experiment_name: Name of the experiment (e.g., "baseline", "multiscale")
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING: {experiment_name.upper()}")
    print("=" * 80)
    
    # Load validation data
    print("\nLoading validation data...")
    try:
        _, val_loader = get_data_loaders(config)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    if len(val_loader.dataset) == 0:
        print("✗ No validation data found!")
        return None
    
    print(f"✓ Loaded {len(val_loader.dataset)} validation images")
    
    # Create model
    print("\nCreating model...")
    model = SwinUNetpp(config)
    print(f"✓ Model created")
    
    # Evaluate
    try:
        metrics = evaluate_metrics(model, val_loader, config, checkpoint_path)
        
        # Add metadata
        metrics['experiment_name'] = experiment_name
        metrics['checkpoint'] = checkpoint_path
        metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics['config'] = {
            'L1_WEIGHT': config.L1_WEIGHT,
            'PERCEPTUAL_WEIGHT': config.PERCEPTUAL_WEIGHT,
            'MULTISCALE_WEIGHT': getattr(config, 'MULTISCALE_WEIGHT', 0.0),
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
        }
        
        return metrics
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_metrics(baseline_metrics, multiscale_metrics, output_dir):
    """
    Compare metrics between baseline and multiscale experiments
    
    Args:
        baseline_metrics: Dictionary of baseline metrics
        multiscale_metrics: Dictionary of multiscale metrics
        output_dir: Directory to save comparison results
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison dictionary
    comparison = {
        'baseline': baseline_metrics,
        'multiscale': multiscale_metrics,
        'improvements': {}
    }
    
    # Calculate improvements
    metrics_to_compare = ['mae', 'psnr', 'ssim', 'pixel_accuracy']
    
    print("\n{:<20} {:<15} {:<15} {:<15} {:<10}".format(
        "Metric", "Baseline", "Multiscale", "Difference", "Better?"
    ))
    print("-" * 80)
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics[metric]
        multiscale_val = multiscale_metrics[metric]
        
        # For MAE, lower is better; for others, higher is better
        if metric == 'mae':
            diff = baseline_val - multiscale_val
            improvement = (diff / baseline_val) * 100 if baseline_val != 0 else 0
            better = "✓ Yes" if diff > 0 else "✗ No"
        else:
            diff = multiscale_val - baseline_val
            improvement = (diff / baseline_val) * 100 if baseline_val != 0 else 0
            better = "✓ Yes" if diff > 0 else "✗ No"
        
        comparison['improvements'][metric] = {
            'absolute_difference': float(diff),
            'percentage_improvement': float(improvement),
            'is_better': diff > 0 if metric != 'mae' else diff < 0
        }
        
        # Format values based on metric type
        if metric == 'mae':
            print("{:<20} {:<15.6f} {:<15.6f} {:<15.6f} {:<10}".format(
                metric.upper(), baseline_val, multiscale_val, diff, better
            ))
        elif metric == 'psnr':
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10}".format(
                metric.upper(), baseline_val, multiscale_val, diff, better
            ))
        elif metric == 'ssim':
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<10}".format(
                metric.upper(), baseline_val, multiscale_val, diff, better
            ))
        else:  # pixel_accuracy
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10}".format(
                metric.upper(), baseline_val, multiscale_val, diff, better
            ))
    
    print("-" * 80)
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    improvements = comparison['improvements']
    better_count = sum(1 for m in improvements.values() if m['is_better'])
    total_count = len(improvements)
    
    print(f"\nMetrics improved: {better_count}/{total_count}")
    
    if better_count >= 3:
        verdict = "✓ MULTISCALE LOSS SIGNIFICANTLY IMPROVES RESULTS"
    elif better_count >= 2:
        verdict = "~ MULTISCALE LOSS SHOWS MODERATE IMPROVEMENT"
    else:
        verdict = "✗ MULTISCALE LOSS DOES NOT IMPROVE RESULTS"
    
    print(f"\n{verdict}")
    comparison['verdict'] = verdict
    
    # Save comparison to JSON
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'comparison_results.json')
    
    with open(comparison_file, 'w') as f:
        comparison = to_python(comparison)
        json.dump(comparison, f, indent=4)

    
    print(f"\n✓ Comparison results saved to: {comparison_file}")
    
    # Save human-readable report
    report_file = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SWIN-UNET: BASELINE vs MULTISCALE LOSS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BASELINE CONFIGURATION:\n")
        f.write(f"  - Checkpoint: {baseline_metrics['checkpoint']}\n")
        f.write(f"  - L1 Weight: {baseline_metrics['config']['L1_WEIGHT']}\n")
        f.write(f"  - Perceptual Weight: {baseline_metrics['config']['PERCEPTUAL_WEIGHT']}\n")
        f.write(f"  - Multiscale Weight: {baseline_metrics['config']['MULTISCALE_WEIGHT']}\n\n")
        
        f.write("MULTISCALE CONFIGURATION:\n")
        f.write(f"  - Checkpoint: {multiscale_metrics['checkpoint']}\n")
        f.write(f"  - L1 Weight: {multiscale_metrics['config']['L1_WEIGHT']}\n")
        f.write(f"  - Perceptual Weight: {multiscale_metrics['config']['PERCEPTUAL_WEIGHT']}\n")
        f.write(f"  - Multiscale Weight: {multiscale_metrics['config']['MULTISCALE_WEIGHT']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRICS COMPARISON:\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("{:<20} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Metric", "Baseline", "Multiscale", "Difference", "Improvement %"
        ))
        f.write("-" * 80 + "\n")
        
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            multiscale_val = multiscale_metrics[metric]
            diff = improvements[metric]['absolute_difference']
            imp_pct = improvements[metric]['percentage_improvement']
            
            if metric == 'mae':
                f.write("{:<20} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, multiscale_val, diff, imp_pct
                ))
            elif metric == 'psnr':
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, multiscale_val, diff, imp_pct
                ))
            elif metric == 'ssim':
                f.write("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, multiscale_val, diff, imp_pct
                ))
            else:
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, multiscale_val, diff, imp_pct
                ))
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Metrics improved: {better_count}/{total_count}\n")
        f.write(f"Verdict: {verdict}\n")
    
    print(f"✓ Comparison report saved to: {report_file}")
    
    return comparison


def main():
    """Main comparison workflow"""
    
    print("\n" + "=" * 80)
    print("SWIN-UNET EXPERIMENT COMPARISON")
    print("Baseline vs Multiscale Loss")
    print("=" * 80)
    
    # Create output directory for comparison results
    comparison_dir = os.path.join('outputs', 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Evaluate Baseline Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: EVALUATING BASELINE MODEL")
    print("=" * 80)
    
    # Create baseline config
    class BaselineConfig(Config):
        CHECKPOINT_DIR = 'checkpoints/baseline/'
        MULTISCALE_WEIGHT = 0.0  # No multiscale loss
    
    baseline_checkpoint = os.path.join(BaselineConfig.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(baseline_checkpoint):
        print(f"\n✗ Baseline checkpoint not found: {baseline_checkpoint}")
        print("Please train the baseline model first by:")
        print("  1. Setting MULTISCALE_WEIGHT = 0.0 in config.py")
        print("  2. Setting CHECKPOINT_DIR = 'checkpoints/baseline/'")
        print("  3. Running: python train.py")
        return
    
    baseline_metrics = run_evaluation(BaselineConfig, "baseline", baseline_checkpoint)
    baseline_metrics = to_python(baseline_metrics)

    if baseline_metrics is None:
        print("\n✗ Baseline evaluation failed!")
        return
    
    # Save baseline metrics
    baseline_file = os.path.join(comparison_dir, 'baseline_metrics.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"\n✓ Baseline metrics saved to: {baseline_file}")
    
    # ========================================================================
    # STEP 2: Evaluate Multiscale Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING MULTISCALE MODEL")
    print("=" * 80)
    
    # Create multiscale config
    class MultiscaleConfig(Config):
        CHECKPOINT_DIR = 'checkpoints/exp_multiscale/'
        MULTISCALE_WEIGHT = 0.5
    
    multiscale_checkpoint = os.path.join(MultiscaleConfig.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(multiscale_checkpoint):
        print(f"\n✗ Multiscale checkpoint not found: {multiscale_checkpoint}")
        print("Please train the multiscale model first by:")
        print("  1. Setting MULTISCALE_WEIGHT = 0.5 in config.py")
        print("  2. Setting CHECKPOINT_DIR = 'checkpoints/exp_multiscale/'")
        print("  3. Running: python train.py")
        return
    
    multiscale_metrics = run_evaluation(MultiscaleConfig, "multiscale", multiscale_checkpoint)
    multiscale_metrics = to_python(multiscale_metrics)

    if multiscale_metrics is None:
        print("\n✗ Multiscale evaluation failed!")
        return
    
    # Save multiscale metrics
    multiscale_file = os.path.join(comparison_dir, 'multiscale_metrics.json')
    with open(multiscale_file, 'w') as f:
        json.dump(multiscale_metrics, f, indent=4)
    print(f"\n✓ Multiscale metrics saved to: {multiscale_file}")
    
    # ========================================================================
    # STEP 3: Compare Results
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: COMPARING RESULTS")
    print("=" * 80)
    
    comparison = compare_metrics(baseline_metrics, multiscale_metrics, comparison_dir)
    
    # ========================================================================
    # STEP 4: Recommendations
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    improvements = comparison['improvements']
    better_count = sum(1 for m in improvements.values() if m['is_better'])
    
    if better_count >= 3:
        print("\n✓ Multiscale loss shows significant improvement!")
        print("\nRecommended next steps:")
        print("  1. Keep multiscale loss in your final model")
        print("  2. Consider experimenting with:")
        print("     - Different multiscale weights (try 0.3, 0.7, 1.0)")
        print("     - Different scale combinations in MultiscaleLoss")
        print("     - Adding edge-aware loss")
        print("     - Adding adversarial loss (GAN)")
        print("  3. Run generate.py to visualize the improvements")
    elif better_count >= 2:
        print("\n~ Multiscale loss shows moderate improvement")
        print("\nRecommended next steps:")
        print("  1. Try tuning the multiscale weight")
        print("  2. Experiment with different scales")
        print("  3. Consider combining with other loss functions")
    else:
        print("\n✗ Multiscale loss does not improve results")
        print("\nRecommended next steps:")
        print("  1. Stick with baseline configuration")
        print("  2. Try other improvements:")
        print("     - Edge-aware loss")
        print("     - Adversarial loss (GAN)")
        print("     - Different learning rate schedules")
        print("     - Data augmentation")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved in: {comparison_dir}")
    print("\nGenerated files:")
    print(f"  - comparison_results.json  (structured data)")
    print(f"  - comparison_report.txt    (human-readable report)")
    print(f"  - baseline_metrics.json    (baseline metrics)")
    print(f"  - multiscale_metrics.json  (multiscale metrics)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()