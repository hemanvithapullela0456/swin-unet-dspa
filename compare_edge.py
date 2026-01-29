#!/usr/bin/env python3
"""
Compare Baseline vs Edge-Aware Loss
Evaluate and compare baseline Swin-UNET with edge-aware variant

Usage:
    python compare_edge.py
"""

import os
import sys
import json
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import Config as BaselineConfig
from config_edge import Config as EdgeConfig
from dataset import get_data_loaders
from model import SwinUNetpp
from evaluate import evaluate_metrics


def main():
    print("\n" + "=" * 80)
    print("SWIN-UNET: BASELINE vs EDGE-AWARE LOSS COMPARISON")
    print("=" * 80)
    
    comparison_dir = os.path.join('outputs', 'comparison_edge')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Evaluate Baseline
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: EVALUATING BASELINE MODEL")
    print("=" * 80)
    
    baseline_checkpoint = os.path.join('checkpoints/baseline/', 'best_model.pth')
    
    if not os.path.exists(baseline_checkpoint):
        print(f"\n✗ Baseline checkpoint not found: {baseline_checkpoint}")
        print("Please train the baseline model first")
        return
    
    # Load validation data
    _, val_loader = get_data_loaders(BaselineConfig)
    
    if len(val_loader.dataset) == 0:
        print("✗ No validation data found!")
        return
    
    print(f"✓ Loaded {len(val_loader.dataset)} validation images")
    
    # Create and evaluate baseline model
    baseline_model = SwinUNetpp(BaselineConfig)
    baseline_metrics = evaluate_metrics(
        baseline_model, val_loader, BaselineConfig, baseline_checkpoint
    )
    
    baseline_metrics['experiment_name'] = 'baseline'
    baseline_metrics['checkpoint'] = baseline_checkpoint
    
    # Save baseline metrics
    baseline_file = os.path.join(comparison_dir, 'baseline_metrics.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"\n✓ Baseline metrics saved to: {baseline_file}")
    
    # ========================================================================
    # STEP 2: Evaluate Edge-Aware
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING EDGE-AWARE MODEL")
    print("=" * 80)
    
    edge_checkpoint = os.path.join('checkpoints/exp_edge_aware/', 'best_model.pth')
    
    if not os.path.exists(edge_checkpoint):
        print(f"\n✗ Edge-aware checkpoint not found: {edge_checkpoint}")
        print("Please train the edge-aware model first:")
        print("  python train_edge.py --epochs 50")
        return
    
    # Load validation data
    _, val_loader = get_data_loaders(EdgeConfig)
    
    # Create and evaluate edge-aware model
    edge_model = SwinUNetpp(EdgeConfig)
    edge_metrics = evaluate_metrics(
        edge_model, val_loader, EdgeConfig, edge_checkpoint
    )
    
    edge_metrics['experiment_name'] = 'edge_aware'
    edge_metrics['checkpoint'] = edge_checkpoint
    
    # Save edge-aware metrics
    edge_file = os.path.join(comparison_dir, 'edge_aware_metrics.json')
    with open(edge_file, 'w') as f:
        json.dump(edge_metrics, f, indent=4)
    print(f"\n✓ Edge-aware metrics saved to: {edge_file}")
    
    # ========================================================================
    # STEP 3: Compare Results
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    metrics_to_compare = ['mae', 'psnr', 'ssim', 'pixel_accuracy']
    
    print("\n{:<20} {:<15} {:<15} {:<15} {:<10}".format(
        "Metric", "Baseline", "Edge-Aware", "Difference", "Better?"
    ))
    print("-" * 80)
    
    improvements = {}
    better_count = 0
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics[metric]
        edge_val = edge_metrics[metric]
        
        # For MAE, lower is better; for others, higher is better
        if metric == 'mae':
            diff = baseline_val - edge_val
            improvement = (diff / baseline_val) * 100 if baseline_val != 0 else 0
            is_better = diff > 0
        else:
            diff = edge_val - baseline_val
            improvement = (diff / baseline_val) * 100 if baseline_val != 0 else 0
            is_better = diff > 0
        
        better = "✓ Yes" if is_better else "✗ No"
        if is_better:
            better_count += 1
        
        improvements[metric] = {
            'baseline': float(baseline_val),
            'edge_aware': float(edge_val),
            'absolute_difference': float(diff),
            'percentage_improvement': float(improvement),
            'is_better': is_better
        }
        
        # Format output
        if metric == 'mae':
            print("{:<20} {:<15.6f} {:<15.6f} {:<15.6f} {:<10}".format(
                metric.upper(), baseline_val, edge_val, diff, better
            ))
        elif metric == 'psnr':
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10}".format(
                metric.upper(), baseline_val, edge_val, diff, better
            ))
        elif metric == 'ssim':
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<10}".format(
                metric.upper(), baseline_val, edge_val, diff, better
            ))
        else:
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10}".format(
                metric.upper(), baseline_val, edge_val, diff, better
            ))
    
    print("-" * 80)
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    print(f"\nMetrics improved: {better_count}/{len(metrics_to_compare)}")
    
    if better_count >= 3:
        verdict = "✓ EDGE-AWARE LOSS SIGNIFICANTLY IMPROVES RESULTS"
        color = "green"
    elif better_count >= 2:
        verdict = "~ EDGE-AWARE LOSS SHOWS MODERATE IMPROVEMENT"
        color = "yellow"
    else:
        verdict = "✗ EDGE-AWARE LOSS DOES NOT IMPROVE RESULTS"
        color = "red"
    
    print(f"\n{verdict}")
    
    # Save comparison
    comparison = {
        'baseline': baseline_metrics,
        'edge_aware': edge_metrics,
        'improvements': improvements,
        'verdict': verdict,
        'better_count': better_count,
        'total_metrics': len(metrics_to_compare)
    }
    
    comparison_file = os.path.join(comparison_dir, 'comparison_results.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Generate report
    report_file = os.path.join(comparison_dir, 'comparison_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SWIN-UNET: BASELINE vs EDGE-AWARE LOSS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATIONS:\n")
        f.write(f"  Baseline checkpoint: {baseline_checkpoint}\n")
        f.write(f"  Edge-aware checkpoint: {edge_checkpoint}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRICS COMPARISON:\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("{:<20} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Metric", "Baseline", "Edge-Aware", "Difference", "Improvement %"
        ))
        f.write("-" * 80 + "\n")
        
        for metric, data in improvements.items():
            baseline_val = data['baseline']
            edge_val = data['edge_aware']
            diff = data['absolute_difference']
            imp_pct = data['percentage_improvement']
            
            if metric == 'mae':
                f.write("{:<20} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, edge_val, diff, imp_pct
                ))
            elif metric == 'psnr':
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, edge_val, diff, imp_pct
                ))
            elif metric == 'ssim':
                f.write("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, edge_val, diff, imp_pct
                ))
            else:
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    metric.upper(), baseline_val, edge_val, diff, imp_pct
                ))
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Metrics improved: {better_count}/{len(metrics_to_compare)}\n")
        f.write(f"Verdict: {verdict}\n")
    
    print(f"\n✓ Comparison saved to: {comparison_file}")
    print(f"✓ Report saved to: {report_file}")
    
    # ========================================================================
    # Recommendations
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if better_count >= 3:
        print("\n✓ Edge-aware loss shows significant improvement!")
        print("\nNext steps:")
        print("  1. Use edge-aware loss in final model")
        print("  2. Visualize improvements:")
        print("     python generate.py --checkpoint checkpoints/exp_edge_aware/best_model.pth")
        print("  3. Try tuning edge weight (0.3, 0.7, 1.0)")
        print("  4. Consider adding more loss components:")
        print("     - Total variation loss (reduce noise)")
        print("     - GAN loss (more realistic outputs)")
    elif better_count >= 2:
        print("\n~ Edge-aware loss shows moderate improvement")
        print("\nNext steps:")
        print("  1. Try different edge weights:")
        print("     python train_edge.py --edge-weight 0.7 --epochs 50")
        print("  2. Combine with other losses")
        print("  3. Check visual comparisons to see where it helps")
    else:
        print("\n✗ Edge-aware loss does not improve results")
        print("\nNext steps:")
        print("  1. Try GAN/adversarial loss instead")
        print("  2. Experiment with different architectures")
        print("  3. Add data augmentation")
        print("  4. Try feature matching loss")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)
    print(f"\nResults saved in: {comparison_dir}/")


if __name__ == "__main__":
    main()