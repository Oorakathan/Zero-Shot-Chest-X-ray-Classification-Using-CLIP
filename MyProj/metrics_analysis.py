"""
Common Metrics Analysis Module for CLIP Classification

Simple metrics calculator using confusion matrix.
Can be imported and used by both classify_5class.py and biomedclip_5class.py

Calculates:
  - Accuracy
  - Precision (per-class and average)
  - Recall (per-class and average)
  - F1-Score (per-class and average)
  - Confusion Matrix

Usage as module:
  from metrics_analysis import calculate_metrics, print_metrics
  metrics = calculate_metrics(y_true, y_pred, class_names)
  print_metrics(metrics)

Usage as script:
  python metrics_analysis.py --true labels.txt --pred predictions.txt

Requirements:
  numpy, scikit-learn
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support  # type: ignore

# Configuration
SELECTED_CLASSES = ["Pneumonia", "Effusion", "Emphysema", "Fibrosis", "Hernia"]


def calculate_metrics(y_true: List[str], y_pred: List[str], class_names: List[str] = None) -> Dict:
    """
    Calculate classification metrics from true and predicted labels.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names (default: SELECTED_CLASSES)
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix
    """
    if class_names is None:
        class_names = SELECTED_CLASSES
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate per-class precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )
    
    # Calculate average metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average='macro', zero_division=0
    )
    
    # Build metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class': {},
        'averages': {
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1_score': float(avg_f1)
        },
        'total_samples': len(y_true)
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return metrics


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """
    Print metrics in a readable format.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics()
        model_name: Name of the model (for display)
    """
    print("\n" + "=" * 80)
    print(f"📊 EVALUATION METRICS - {model_name}")
    print("=" * 80)
    
    # Overall accuracy
    print(f"\n✓ Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Total Samples: {metrics['total_samples']}")
    
    # Per-class metrics
    print(f"\n📈 Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<15} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1_score']:<12.4f} "
              f"{class_metrics['support']:<10}")
    
    print("-" * 80)
    print(f"{'Average':<15} "
          f"{metrics['averages']['precision']:<12.4f} "
          f"{metrics['averages']['recall']:<12.4f} "
          f"{metrics['averages']['f1_score']:<12.4f}")
    
    # Confusion Matrix
    print(f"\n🔢 Confusion Matrix:")
    print("-" * 80)
    cm = np.array(metrics['confusion_matrix'])
    class_names = list(metrics['per_class'].keys())
    
    # Print header
    print(f"{'True \\ Pred':<15}", end="")
    for name in class_names:
        print(f"{name[:10]:<12}", end="")
    print()
    print("-" * 80)
    
    # Print matrix rows
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:<12}", end="")
        print()
    
    print("=" * 80 + "\n")


def save_metrics(metrics: Dict, output_path: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics()
        output_path: Path to save JSON file
    """
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {output_path}")


# Standalone usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate metrics from prediction files')
    parser.add_argument('--true', required=True, help='File with true labels (one per line)')
    parser.add_argument('--pred', required=True, help='File with predicted labels (one per line)')
    parser.add_argument('--output', help='Output JSON file for metrics')
    parser.add_argument('--model', default='Model', help='Model name for display')
    
    args = parser.parse_args()
    
    # Load labels from files
    with open(args.true, 'r', encoding='utf-8') as f:
        y_true = [line.strip() for line in f if line.strip()]
    
    with open(args.pred, 'r', encoding='utf-8') as f:
        y_pred = [line.strip() for line in f if line.strip()]
    
    if len(y_true) != len(y_pred):
        print(f"❌ Error: Number of true labels ({len(y_true)}) != predicted labels ({len(y_pred)})")
        exit(1)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, args.model)
    
    # Save metrics if output specified
    if args.output:
        save_metrics(metrics, args.output)
