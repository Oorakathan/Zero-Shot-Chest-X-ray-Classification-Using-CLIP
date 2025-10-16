"""
Calculate metrics from confusion matrices
Based on the actual confusion matrix results from the images
"""

import numpy as np

# Confusion Matrix - BiomedCLIP
# Rows: True Class, Columns: Predicted Class
# Order: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia
biomedclip_cm = np.array([
    [22, 7, 57, 30, 34],   # Pneumonia
    [5, 81, 22, 8, 34],    # Effusion
    [17, 10, 80, 8, 35],   # Emphysema
    [4, 4, 91, 19, 32],    # Fibrosis
    [7, 5, 68, 2, 68]      # Hernia
])

# Confusion Matrix - Standard CLIP
standard_clip_cm = np.array([
    [141, 2, 0, 0, 7],     # Pneumonia
    [141, 5, 0, 0, 4],     # Effusion
    [139, 2, 0, 0, 9],     # Emphysema
    [144, 0, 0, 0, 6],     # Fibrosis
    [144, 2, 0, 0, 4]      # Hernia
])

classes = ['Pneumonia', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia']

def calculate_metrics(cm, model_name):
    """Calculate accuracy, precision, recall, and F1-score from confusion matrix"""
    print(f"\n{'='*70}")
    print(f"Metrics for {model_name}")
    print(f"{'='*70}\n")
    
    # Overall Accuracy
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total * 100
    
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {correct}/{total}\n")
    
    # Per-class metrics
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for i, class_name in enumerate(classes):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of column minus TP
        fp = np.sum(cm[:, i]) - tp
        
        # False Negatives: sum of row minus TP
        fn = np.sum(cm[i, :]) - tp
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-Score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"{class_name:<15} {precision*100:>10.2f}%  {recall*100:>10.2f}%  {f1*100:>10.2f}%")
    
    # Average metrics
    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_f1 = np.mean(f1_scores) * 100
    
    print("-" * 70)
    print(f"{'Average':<15} {avg_precision:>10.2f}%  {avg_recall:>10.2f}%  {avg_f1:>10.2f}%")
    
    # Detailed breakdown
    print(f"\n{'Class Performance Details:'}")
    print("-" * 70)
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        total_actual = np.sum(cm[i, :])
        total_predicted = np.sum(cm[:, i])
        
        print(f"\n{class_name}:")
        print(f"  True Positives: {tp}")
        print(f"  Total Actual: {total_actual}")
        print(f"  Total Predicted: {total_predicted}")
        print(f"  Precision: {precisions[i]*100:.2f}% (of predicted {class_name}, {precisions[i]*100:.1f}% were correct)")
        print(f"  Recall: {recalls[i]*100:.2f}% (of actual {class_name}, {recalls[i]*100:.1f}% were found)")
        print(f"  F1-Score: {f1_scores[i]*100:.2f}%")

# Calculate metrics for both models
calculate_metrics(biomedclip_cm, "BiomedCLIP")
calculate_metrics(standard_clip_cm, "Standard CLIP")

# Summary comparison
print(f"\n{'='*70}")
print("SUMMARY COMPARISON")
print(f"{'='*70}\n")

biomedclip_acc = np.trace(biomedclip_cm) / np.sum(biomedclip_cm) * 100
standard_acc = np.trace(standard_clip_cm) / np.sum(standard_clip_cm) * 100

print(f"Model Comparison:")
print(f"  BiomedCLIP Accuracy:     {biomedclip_acc:.2f}%")
print(f"  Standard CLIP Accuracy:  {standard_acc:.2f}%")
print(f"  Difference:              {biomedclip_acc - standard_acc:+.2f}%")
print(f"\nKey Observation:")
if biomedclip_acc > standard_acc:
    print(f"  BiomedCLIP performs {biomedclip_acc - standard_acc:.1f}% better than Standard CLIP")
    print(f"  This confirms that medical-specialized training improves performance")
else:
    print(f"  Standard CLIP performs better (unusual - may indicate dataset issues)")

print(f"\nStandard CLIP Issue:")
print(f"  Standard CLIP predicts almost everything as Pneumonia")
print(f"  This is a clear sign of model bias on medical images")
print(f"  The model cannot distinguish between different chest X-ray conditions")
