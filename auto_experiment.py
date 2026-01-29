#!/usr/bin/env python3
"""
Automated Edge-Aware Loss Experiment Runner
Runs multiple experiments with different edge weights, compares all results,
and generates comprehensive reports.

This script will:
1. Train edge-aware models with weights: 0.3, 0.5, 0.7, 1.0
2. Evaluate each model on validation set
3. Compare all results with baseline
4. Generate visual comparisons
5. Create detailed reports
6. Recommend best configuration

Expected runtime: 2-3 hours (depends on your hardware)

Usage:
    python auto_experiment.py
"""

import os
import sys
import json
import time
import torch
import subprocess
from datetime import datetime
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


sys.path.insert(0, os.path.dirname(__file__))


def log(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also save to log file
    with open('experiment_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def run_command(command, description, log_file=None):
    """Run a command and log output"""
    log(f"Starting: {description}")
    log(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log(f"✓ Completed: {description} ({elapsed/60:.1f} minutes)")
            return True, elapsed
        else:
            log(f"✗ Failed: {description}")
            if result.stderr:
                log(f"Error: {result.stderr[:500]}")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"✗ Exception in {description}: {str(e)}")
        return False, elapsed


def update_config_edge_weight(weight):
    """Update config_edge.py with new edge weight"""
    config_path = 'config_edge.py'
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'EDGE_WEIGHT' in line and '=' in line and 'NEW:' in line:
            new_lines.append(f"    EDGE_WEIGHT = {weight}  # NEW: Edge-aware loss weight\n")
        elif 'CHECKPOINT_DIR' in line and 'exp_edge_aware' in line:
            new_lines.append(f"    CHECKPOINT_DIR = 'checkpoints/exp_edge_{weight}/'  # Edge weight {weight}\n")
        else:
            new_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    log(f"✓ Updated config: EDGE_WEIGHT = {weight}")


def evaluate_model(checkpoint_path, experiment_name):
    """Evaluate a trained model"""
    log(f"Evaluating {experiment_name}...")
    
    # Import here to avoid circular imports
    from config_edge import Config
    from dataset import get_data_loaders
    from model import SwinUNetpp
    from evaluate import evaluate_metrics
    
    try:
        # Load validation data
        _, val_loader = get_data_loaders(Config)
        
        if len(val_loader.dataset) == 0:
            log("✗ No validation data found!")
            return None
        
        # Create and evaluate model
        model = SwinUNetpp(Config)
        metrics = evaluate_metrics(model, val_loader, Config, checkpoint_path)
        
        metrics['experiment_name'] = experiment_name
        metrics['checkpoint'] = checkpoint_path
        metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log(f"✓ Evaluation complete: MAE={metrics['mae']:.2f}, PSNR={metrics['psnr']:.2f}")
        
        return metrics
        
    except Exception as e:
        log(f"✗ Evaluation failed: {str(e)}")
        return None


def compare_all_results(baseline_metrics, experiment_results, output_dir):
    """Compare all experiments and find best configuration"""
    
    log("=" * 80)
    log("COMPARING ALL EXPERIMENTS")
    log("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare comparison table
    experiments = [('baseline', baseline_metrics)] + experiment_results
    
    metrics_names = ['mae', 'psnr', 'ssim', 'pixel_accuracy']
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPLETE RESULTS COMPARISON")
    print("=" * 100)
    print(f"{'Experiment':<25} {'MAE':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Pixel Acc (%)':<15}")
    print("-" * 100)
    
    for exp_name, metrics in experiments:
        if metrics:
            print(f"{exp_name:<25} {metrics['mae']:<15.2f} {metrics['psnr']:<15.2f} "
                  f"{metrics['ssim']:<15.4f} {metrics['pixel_accuracy']:<15.2f}")
    
    print("-" * 100)
    
    # Find best for each metric
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS PER METRIC")
    print("=" * 80)
    
    best_results = {}
    
    for metric in metrics_names:
        if metric == 'mae':
            # Lower is better
            best_exp = min(
                [(name, m) for name, m in experiments if m],
                key=lambda x: x[1][metric]
            )
        else:
            # Higher is better
            best_exp = max(
                [(name, m) for name, m in experiments if m],
                key=lambda x: x[1][metric]
            )
        
        best_results[metric] = best_exp
        print(f"{metric.upper():<20} Best: {best_exp[0]:<25} Value: {best_exp[1][metric]:.4f}")
    
    # Calculate improvements over baseline
    print("\n" + "=" * 80)
    print("IMPROVEMENTS OVER BASELINE")
    print("=" * 80)
    print(f"{'Experiment':<25} {'MAE Δ':<15} {'PSNR Δ':<15} {'SSIM Δ':<15} {'Pixel Acc Δ':<15} {'Score':<10}")
    print("-" * 100)
    
    improvement_scores = []
    
    for exp_name, metrics in experiment_results:
        if not metrics:
            continue
        
        mae_diff = baseline_metrics['mae'] - metrics['mae']  # Positive = better
        psnr_diff = metrics['psnr'] - baseline_metrics['psnr']  # Positive = better
        ssim_diff = metrics['ssim'] - baseline_metrics['ssim']  # Positive = better
        pixel_diff = metrics['pixel_accuracy'] - baseline_metrics['pixel_accuracy']  # Positive = better
        
        # Calculate improvement score (number of improved metrics)
        score = sum([
            mae_diff > 0,
            psnr_diff > 0,
            ssim_diff > 0,
            pixel_diff > 0
        ])
        
        improvement_scores.append((exp_name, score, metrics))
        
        print(f"{exp_name:<25} {mae_diff:<15.2f} {psnr_diff:<15.2f} "
              f"{ssim_diff:<15.4f} {pixel_diff:<15.2f} {score}/4")
    
    print("-" * 100)
    
    # Find overall best
    if improvement_scores:
        best_overall = max(improvement_scores, key=lambda x: (x[1], x[2]['psnr']))
        
        print("\n" + "=" * 80)
        print("OVERALL BEST CONFIGURATION")
        print("=" * 80)
        print(f"Winner: {best_overall[0]}")
        print(f"Improved metrics: {best_overall[1]}/4")
        print(f"MAE: {best_overall[2]['mae']:.2f}")
        print(f"PSNR: {best_overall[2]['psnr']:.2f} dB")
        print(f"SSIM: {best_overall[2]['ssim']:.4f}")
        print(f"Pixel Accuracy: {best_overall[2]['pixel_accuracy']:.2f}%")
    
    # Save comprehensive report
    report_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'baseline': baseline_metrics,
        'experiments': {name: metrics for name, metrics in experiment_results if metrics},
        'best_per_metric': {
            metric: {'experiment': name, 'value': float(m[metric])}
            for metric, (name, m) in best_results.items()
        },
        'best_overall': {
            'experiment': best_overall[0],
            'score': best_overall[1],
            'metrics': best_overall[2]
        } if improvement_scores else None
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, 'complete_comparison.json')
    with open(json_path, 'w') as f:
        report_data = to_python(report_data)
        json.dump(report_data, f, indent=4)
    log(f"✓ Saved comparison data: {json_path}")
    
    # Save text report
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EDGE-AWARE LOSS: COMPLETE EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENTS RUN:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Baseline (L1 + Perceptual, no edge loss)\n")
        for i, (exp_name, _) in enumerate(experiment_results, 2):
            f.write(f"{i}. {exp_name}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("COMPLETE RESULTS\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"{'Experiment':<25} {'MAE':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Pixel Acc (%)':<15}\n")
        f.write("-" * 100 + "\n")
        
        for exp_name, metrics in experiments:
            if metrics:
                f.write(f"{exp_name:<25} {metrics['mae']:<15.2f} {metrics['psnr']:<15.2f} "
                       f"{metrics['ssim']:<15.4f} {metrics['pixel_accuracy']:<15.2f}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("IMPROVEMENTS OVER BASELINE\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"{'Experiment':<25} {'MAE Δ':<15} {'PSNR Δ':<15} {'SSIM Δ':<15} {'Pixel Acc Δ':<15} {'Score':<10}\n")
        f.write("-" * 100 + "\n")
        
        for exp_name, metrics in experiment_results:
            if not metrics:
                continue
            
            mae_diff = baseline_metrics['mae'] - metrics['mae']
            psnr_diff = metrics['psnr'] - baseline_metrics['psnr']
            ssim_diff = metrics['ssim'] - baseline_metrics['ssim']
            pixel_diff = metrics['pixel_accuracy'] - baseline_metrics['pixel_accuracy']
            
            score = sum([mae_diff > 0, psnr_diff > 0, ssim_diff > 0, pixel_diff > 0])
            
            f.write(f"{exp_name:<25} {mae_diff:<15.2f} {psnr_diff:<15.2f} "
                   f"{ssim_diff:<15.4f} {pixel_diff:<15.2f} {score}/4\n")
        f.write("\n")
        
        if improvement_scores:
            f.write("-" * 80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Best Configuration: {best_overall[0]}\n")
            f.write(f"Metrics Improved: {best_overall[1]}/4\n\n")
            
            if best_overall[1] >= 3:
                f.write("✓ EDGE-AWARE LOSS SIGNIFICANTLY IMPROVES RESULTS\n\n")
                f.write("Recommendations:\n")
                f.write("1. Use this configuration in your final model\n")
                f.write("2. Include edge-aware loss in your paper as a contribution\n")
                f.write("3. Generate visual comparisons to show improvements\n")
                f.write("4. Consider adding additional losses (GAN, Total Variation)\n")
            elif best_overall[1] >= 2:
                f.write("~ EDGE-AWARE LOSS SHOWS MODERATE IMPROVEMENT\n\n")
                f.write("Recommendations:\n")
                f.write("1. Check visual quality - may be better than metrics suggest\n")
                f.write("2. Try fine-tuning with other edge weights\n")
                f.write("3. Consider combining with other loss functions\n")
            else:
                f.write("✗ EDGE-AWARE LOSS DOES NOT IMPROVE RESULTS\n\n")
                f.write("Recommendations:\n")
                f.write("1. Stick with baseline configuration\n")
                f.write("2. Try GAN/adversarial loss instead\n")
                f.write("3. Experiment with attention mechanisms\n")
                f.write("4. Consider different architectures\n")
    
    log(f"✓ Saved text report: {report_path}")
    
    return best_overall if improvement_scores else None


def main():
    """Main experiment runner"""
    
    start_time = time.time()
    
    # Clear previous log
    if os.path.exists('experiment_log.txt'):
        os.remove('experiment_log.txt')
    
    log("=" * 80)
    log("AUTOMATED EDGE-AWARE LOSS EXPERIMENTS")
    log("=" * 80)
    log("This will take 2-3 hours. You can leave and come back!")
    log("")
    log("Experiments to run:")
    log("  1. Edge weight = 0.3 (conservative)")
    log("  2. Edge weight = 0.5 (balanced)")
    log("  3. Edge weight = 0.7 (aggressive)")
    log("  4. Edge weight = 1.0 (very aggressive)")
    log("")
    log("Starting in 5 seconds... (Ctrl+C to cancel)")
    
    time.sleep(5)
    
    # Configuration
    edge_weights = [0.3, 0.5, 0.7, 1.0]
    epochs = 50
    results_dir = 'outputs/auto_experiment'
    os.makedirs(results_dir, exist_ok=True)
    
    # Track all results
    experiment_results = []
    training_times = {}
    
    # ========================================================================
    # STEP 0: Load or evaluate baseline
    # ========================================================================
    
    log("")
    log("=" * 80)
    log("STEP 0: LOADING BASELINE RESULTS")
    log("=" * 80)
    
    baseline_checkpoint = 'checkpoints/baseline/best_model.pth'
    
    if not os.path.exists(baseline_checkpoint):
        log("✗ Baseline checkpoint not found!")
        log("Please train baseline first:")
        log("  python switch_config.py baseline")
        log("  python train.py --epochs 50")
        return
    
    # Evaluate baseline
    baseline_metrics = evaluate_model(baseline_checkpoint, 'baseline')
    baseline_metrics = to_python(baseline_metrics)

    if not baseline_metrics:
        log("✗ Failed to evaluate baseline")
        return
    
    # Save baseline metrics
    baseline_file = os.path.join(results_dir, 'baseline_metrics.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    
    log(f"✓ Baseline metrics saved: {baseline_file}")
    log(f"  MAE: {baseline_metrics['mae']:.2f}")
    log(f"  PSNR: {baseline_metrics['psnr']:.2f} dB")
    log(f"  SSIM: {baseline_metrics['ssim']:.4f}")
    log(f"  Pixel Accuracy: {baseline_metrics['pixel_accuracy']:.2f}%")
    
    # ========================================================================
    # MAIN EXPERIMENT LOOP
    # ========================================================================
    
    for i, weight in enumerate(edge_weights, 1):
        log("")
        log("=" * 80)
        log(f"EXPERIMENT {i}/{len(edge_weights)}: EDGE WEIGHT = {weight}")
        log("=" * 80)
        
        exp_name = f"edge_{weight}"
        checkpoint_dir = f"checkpoints/exp_{exp_name}"
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        
        # Check if already trained
        if os.path.exists(checkpoint_path):
            log(f"✓ Checkpoint already exists: {checkpoint_path}")
            log("  Skipping training, will evaluate existing model")
            skip_training = True
        else:
            skip_training = False
        
        if not skip_training:
            # Update config
            update_config_edge_weight(weight)
            
            # Train model
            log(f"Training with edge weight {weight} for {epochs} epochs...")
            log("This will take ~30-45 minutes...")
            
            train_cmd = f"python train_edge.py --epochs {epochs} --edge-weight {weight}"
            train_log = os.path.join(results_dir, f'training_{exp_name}.log')
            
            success, train_time = run_command(
                train_cmd,
                f"Training with edge weight {weight}",
                train_log
            )
            
            training_times[exp_name] = train_time
            
            if not success:
                log(f"✗ Training failed for edge weight {weight}")
                log(f"  Check log: {train_log}")
                experiment_results.append((exp_name, None))
                continue
            
            log(f"✓ Training completed in {train_time/60:.1f} minutes")
        
        # Evaluate model
        if os.path.exists(checkpoint_path):
            metrics = evaluate_model(checkpoint_path, exp_name)
            
            if metrics:
                # Save individual metrics
                metrics = to_python(metrics)
                metrics_file = os.path.join(results_dir, f'{exp_name}_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                log(f"✓ Metrics saved: {metrics_file}")
                experiment_results.append((exp_name, metrics))
            else:
                log(f"✗ Evaluation failed for {exp_name}")
                experiment_results.append((exp_name, None))
        else:
            log(f"✗ Checkpoint not found: {checkpoint_path}")
            experiment_results.append((exp_name, None))
        
        # Print current progress
        completed = i
        remaining = len(edge_weights) - i
        elapsed_hours = (time.time() - start_time) / 3600
        
        log("")
        log(f"Progress: {completed}/{len(edge_weights)} experiments completed")
        log(f"Elapsed time: {elapsed_hours:.1f} hours")
        if completed > 0 and remaining > 0:
            avg_time_per_exp = elapsed_hours / completed
            estimated_remaining = avg_time_per_exp * remaining
            log(f"Estimated time remaining: {estimated_remaining:.1f} hours")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    
    log("")
    log("=" * 80)
    log("ALL EXPERIMENTS COMPLETED!")
    log("=" * 80)
    
    total_time = time.time() - start_time
    log(f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    # Compare all results
    best_config = compare_all_results(baseline_metrics, experiment_results, results_dir)
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    
    log("")
    log("=" * 80)
    log("GENERATING VISUALIZATIONS")
    log("=" * 80)
    
    if best_config:
        best_exp_name = best_config[0]
        best_checkpoint = f"checkpoints/exp_{best_exp_name}/best_model.pth"
        
        if os.path.exists(best_checkpoint):
            log(f"Generating samples from best configuration: {best_exp_name}")
            
            gen_cmd = f"python generate.py --checkpoint {best_checkpoint} --num-samples 10 --output-dir {results_dir}/predictions_{best_exp_name}"
            success, _ = run_command(
                gen_cmd,
                f"Generating predictions for {best_exp_name}",
                os.path.join(results_dir, f'generation_{best_exp_name}.log')
            )
            
            if success:
                log(f"✓ Predictions saved to: {results_dir}/predictions_{best_exp_name}/")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    log("")
    log("=" * 80)
    log("EXPERIMENT SUMMARY")
    log("=" * 80)
    log("")
    log("Generated files:")
    log(f"  - {results_dir}/complete_comparison.json")
    log(f"  - {results_dir}/experiment_report.txt")
    log(f"  - {results_dir}/baseline_metrics.json")
    for exp_name, metrics in experiment_results:
        if metrics:
            log(f"  - {results_dir}/{exp_name}_metrics.json")
    log(f"  - experiment_log.txt (this log)")
    log("")
    
    if best_config:
        log("=" * 80)
        log("RECOMMENDATION")
        log("=" * 80)
        log(f"Best configuration: {best_config[0]}")
        log(f"Improved metrics: {best_config[1]}/4")
        log("")
        
        if best_config[1] >= 3:
            log("✓ Edge-aware loss significantly improves results!")
            log("")
            log("Next steps:")
            log("  1. Review visual samples in predictions folder")
            log("  2. Use this configuration for your paper")
            log("  3. Consider adding more losses (GAN, TV)")
        elif best_config[1] >= 2:
            log("~ Edge-aware loss shows moderate improvement")
            log("")
            log("Next steps:")
            log("  1. Check visual quality carefully")
            log("  2. Try combining with other losses")
            log("  3. May still be worth including in paper")
        else:
            log("✗ Edge-aware loss does not improve results")
            log("")
            log("Next steps:")
            log("  1. Stick with baseline")
            log("  2. Try GAN/adversarial loss")
            log("  3. Experiment with attention mechanisms")
    
    log("")
    log("=" * 80)
    log("ALL DONE! Check the reports in outputs/auto_experiment/")
    log("=" * 80)
    log("")
    log(f"Total runtime: {total_time/3600:.2f} hours")
    log(f"Main report: {results_dir}/experiment_report.txt")
    log(f"Detailed log: experiment_log.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("")
        log("=" * 80)
        log("INTERRUPTED BY USER")
        log("=" * 80)
        log("Partial results may be available in outputs/auto_experiment/")
    except Exception as e:
        log("")
        log("=" * 80)
        log("ERROR OCCURRED")
        log("=" * 80)
        log(f"Error: {str(e)}")
        import traceback
        log(traceback.format_exc())