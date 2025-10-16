# Chest X-ray Classification with CLIP

## Project Overview

This project implements zero-shot medical image classification using CLIP (Contrastive Language-Image Pre-training). The system classifies chest X-rays into five disease categories without training on the target dataset.

**Dataset**: NIH Chest X-ray Dataset (750 images, 150 per class)  
**Classes**: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia  
**Technique**: Zero-shot learning with vision-language models  
**Models**: Standard CLIP (local code) and BiomedCLIP (medical-specialized)

---

## Project Files

### Core Scripts

- **batch_infer_5class.py** - Main inference script
  - Supports single image and batch inference
  - Works with both Standard CLIP and BiomedCLIP
  - Generates confusion matrix and prediction visualizations
  
- **metrics_analysis.py** - Evaluation metrics module
  - Calculates accuracy, precision, recall, F1-score
  - Can be imported or run standalone

### Reference Scripts

- **classify_5class.py** - Standard CLIP single-image classifier
- **biomedclip_5class.py** - BiomedCLIP single-image classifier

### Supporting Files

- **Data_Entry_2017.csv** - NIH dataset metadata
- **requirements.txt** - Python dependencies
- **README.md** - Project documentation
- **steps.md** - This file

### Directories

- **finalData/** - 750 chest X-ray images and ground truth labels
- **results/** - Generated visualizations (confusion matrices, plots)

---

## Model Comparison

| Feature | Standard CLIP | BiomedCLIP |
|---------|---------------|------------|
| **Source** | Local repository | HuggingFace Hub |
| **Installation** | None needed | pip install open_clip_torch |
| **Training Data** | 400M general images | 15M medical images |
| **Domain** | General-purpose | Medical-specialized |
| **Architecture** | ViT-B/32 | ViT-base + PubMedBERT |
| **Best For** | Baseline, learning | Production, accuracy |

---

## Setup Instructions

### Install Dependencies

The project uses local CLIP code from the parent directory, so no CLIP installation is needed.

```bash
# Core dependencies
pip install torch torchvision pillow numpy scikit-learn matplotlib seaborn

# Optional: For BiomedCLIP support
pip install open_clip_torch huggingface-hub
```

---

## Usage

### Single Image Inference

```bash
# Standard CLIP
python batch_infer_5class.py --image finalData/00000001_001.png

# BiomedCLIP
python batch_infer_5class.py --image finalData/00000001_001.png --model biomed
```

### Batch Inference

```bash
# Standard CLIP
python batch_infer_5class.py --batch --data finalData

# BiomedCLIP
python batch_infer_5class.py --batch --data finalData --model biomed
```

This will process all 750 images, calculate metrics, and generate visualizations.

---

## How It Works

### Zero-Shot Classification

Unlike traditional machine learning that requires training, zero-shot classification works by:

1. Converting the X-ray image to a vector representation
2. Converting disease descriptions to vector representations
3. Finding which description best matches the image
4. Predicting the class with highest similarity

### Text Prompts

The system uses these descriptions for classification:

```python
TEXT_PROMPTS = [
    "a chest X-ray showing pneumonia",
    "a chest X-ray showing pleural effusion", 
    "a chest X-ray showing emphysema",
    "a chest X-ray showing pulmonary fibrosis",
    "a chest X-ray showing diaphragmatic hernia"
]
```

### Using Local CLIP

Standard CLIP is imported from the parent directory:

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clip  # Imports from ../clip/ directory
```

This avoids redundant installation since the CLIP code is already available.

---

## Evaluation Metrics

The system calculates:

- **Overall Accuracy**: Percentage of correct predictions
- **Per-Class Precision**: How many predicted cases were actually that disease
- **Per-Class Recall**: How many actual cases were correctly identified  
- **Per-Class F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which classes get confused

---

## Visualizations

### Confusion Matrix

Shows prediction patterns as a heatmap. Diagonal cells represent correct predictions, off-diagonal cells show misclassifications.

### Prediction Comparison Plot

Two subplots:

1. **Scatter Plot**: Ground truth (green dots) vs predictions (red X) with connecting lines
   - Green lines: Correct predictions
   - Red lines: Incorrect predictions

2. **Confidence Distribution**: Histogram showing confidence scores
   - Green bars: Correct predictions
   - Red bars: Incorrect predictions

Results are saved to the **results/** directory.

---

## Project Structure

```
MyProj/
├── batch_infer_5class.py       # Main script
├── metrics_analysis.py         # Metrics module
├── classify_5class.py          # Reference script
├── biomedclip_5class.py        # Reference script  
├── Data_Entry_2017.csv         # Dataset metadata
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── steps.md                    # This file
├── finalData/                  # Dataset
│   ├── *.png                   # 750 X-ray images
│   ├── image_labels.txt        # Ground truth
│   └── dataset_summary.txt     # Statistics
└── results/                    # Visualizations
    ├── confusion_matrix_*.png
    └── predictions_analysis_*.png
```

---

## Technical Notes

### Why Zero-Shot Learning?

- Medical data is expensive to label
- Can leverage knowledge from large general datasets
- Works on new categories without retraining
- Reduces need for specialized training data

### Advantages of CLIP

- Pre-trained on internet-scale data
- Flexible text-based classification
- No training required
- Human-readable descriptions

### Limitations

- Lower accuracy than specialized trained models
- Results depend on text prompt quality
- General CLIP not optimized for medical images (use BiomedCLIP instead)

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'clip'"**

Make sure you're running from the MyProj directory. The code adds the parent directory automatically.

**"ModuleNotFoundError: No module named 'open_clip'"**

BiomedCLIP requires installation:
```bash
pip install open_clip_torch huggingface-hub
```

**Low Accuracy**

Try BiomedCLIP instead of Standard CLIP for medical images, or experiment with different text prompts.

---

**Last Updated**: October 16, 2025  
**Status**: Production-ready
