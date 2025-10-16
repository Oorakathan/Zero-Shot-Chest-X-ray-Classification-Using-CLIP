# Chest X-ray Classification with CLIP

## Overview

This project uses CLIP (Contrastive Language-Image Pre-training) to classify chest X-rays into five disease categories. The interesting part is that it works without any training - we just describe the diseases in text, and CLIP figures out which description matches each X-ray image.

**What we're classifying**: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia  
**How many images**: 750 chest X-rays (150 per disease)  
**Models used**: Standard CLIP (general-purpose) and BiomedCLIP (medical-focused)

---

## Project Structure

```
MyProj/
├── batch_infer_5class.py       # Main script (single/batch inference)
├── metrics_analysis.py         # Evaluation metrics module
├── classify_5class.py          # Reference: Standard CLIP
├── biomedclip_5class.py        # Reference: BiomedCLIP
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── finalData/                  # Dataset (750 images)
│   ├── *.png                   # Chest X-ray images
│   └── image_labels.txt        # Ground truth labels
└── results/                    # Generated visualizations
    ├── confusion_matrix_*.png
    └── predictions_analysis_*.png
```

## Setup

### System Requirements

- Python 3.8 or higher
- 2-4GB free disk space for models and dependencies
- GPU with CUDA support (optional, but recommended for faster inference)
- 8GB+ RAM recommended for batch processing

### Installation

#### Option 1: Full Install (Recommended - includes BiomedCLIP)

```bash
pip install -r requirements.txt
```

#### Option 2: Minimal Install (Standard CLIP only)

If you only want to use Standard CLIP and not BiomedCLIP:

```bash
pip install torch torchvision Pillow numpy scikit-learn matplotlib seaborn ftfy regex
```

#### Option 3: GPU Support (Recommended for faster inference)

Install PyTorch with CUDA support **before** installing other dependencies:

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Option 4: Install CLIP from GitHub (Alternative to local code)

If you want to install CLIP from GitHub instead of using the local code:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Additional Dependencies

**For data analysis script** (data_analysis.py):
```bash
pip install pandas
```

**For Jupyter notebooks** (if needed):
```bash
pip install jupyter ipykernel
```

### Why Two Models?

- **Standard CLIP**: Trained on general images (cats, cars, etc.). Good baseline but not specialized for medicine.
- **BiomedCLIP**: Trained specifically on medical images. Better accuracy for X-rays.

### Using Local CLIP Code

This project uses the CLIP code that's already in the parent directory (`../clip/`), so you don't need to install CLIP separately. The scripts automatically add the parent directory to the Python path:

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clip  # This gets CLIP from ../clip/ folder
```

---

## How to Use

### Quick Start

**Classify a single X-ray image:**
```bash
python batch_infer_5class.py --image finalData/00000001_001.png
```

**Process all images in a folder:**
```bash
python batch_infer_5class.py --batch --data finalData
```

### All Available Scripts

#### 1. Main Inference Script (batch_infer_5class.py)

**Single image with Standard CLIP:**
```bash
python batch_infer_5class.py --image finalData/00000001_001.png
```

**Single image with BiomedCLIP (better for medical images):**
```bash
python batch_infer_5class.py --image finalData/00000001_001.png --model biomed
```

**Batch inference with Standard CLIP:**
```bash
python batch_infer_5class.py --batch --data finalData
```

**Batch inference with BiomedCLIP:**
```bash
python batch_infer_5class.py --batch --data finalData --model biomed
```

**Custom labels file:**
```bash
python batch_infer_5class.py --batch --data finalData --labels finalData/image_labels.txt
```

#### 2. Simple Classifiers

**Standard CLIP classifier:**
```bash
python classify_5class.py path/to/xray.png
```

**BiomedCLIP classifier:**
```bash
python biomedclip_5class.py path/to/xray.png
```

#### 3. Data Analysis & Metrics

**Analyze the NIH dataset CSV:**
```bash
python data_analysis.py
# Note: Requires pandas (pip install pandas)
```

**Calculate metrics from confusion matrices:**
```bash
python confusion_matrix_analysis.py
```

**Calculate metrics from prediction files:**
```bash
python metrics_analysis.py --true ground_truth.txt --pred predictions.txt
```

#### 4. Learning Exercises

**CLIP embeddings tutorial:**
```bash
python learning/exercise_1_embeddings.py
```

### Classify a Single Image

```bash
# Using Standard CLIP
python batch_infer_5class.py --image finalData/00000001_001.png

# Using BiomedCLIP (better for medical images)
python batch_infer_5class.py --image finalData/00000001_001.png --model biomed
```

### Process All Images

```bash
# Standard CLIP
python batch_infer_5class.py --batch --data finalData

# BiomedCLIP
python batch_infer_5class.py --batch --data finalData --model biomed
```

This will:
- Classify all 750 images
- Calculate accuracy and other metrics
- Create visualizations showing where the model got things right/wrong
- Save results to the `results/` folder

### What Each Script Does

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `batch_infer_5class.py` | Main inference script (single/batch) | torch, numpy, PIL, sklearn, matplotlib, seaborn |
| `classify_5class.py` | Simple Standard CLIP classifier | torch, PIL, clip (local) |
| `biomedclip_5class.py` | Simple BiomedCLIP classifier | torch, PIL, open_clip |
| `data_analysis.py` | CSV dataset analysis with plots | pandas, numpy, matplotlib, seaborn |
| `confusion_matrix_analysis.py` | Calculate metrics from pre-computed confusion matrices | numpy |
| `metrics_analysis.py` | Reusable metrics calculation module | numpy, sklearn |
| `learning/exercise_1_embeddings.py` | Tutorial on CLIP embeddings | torch, PIL, clip |

---

## How It Works

### The Basic Idea

Instead of training a model from scratch, CLIP uses a clever trick:

1. **Image Understanding**: Converts the X-ray to a bunch of numbers (vector)
2. **Text Understanding**: Converts disease descriptions to numbers too
3. **Matching**: Compares which description's numbers are closest to the image's numbers
4. **Prediction**: The closest match wins

### The Text Descriptions We Use

```python
TEXT_PROMPTS = [
    "a chest X-ray showing pneumonia",
    "a chest X-ray showing pleural effusion", 
    "a chest X-ray showing emphysema",
    "a chest X-ray showing pulmonary fibrosis",
    "a chest X-ray showing diaphragmatic hernia"
]
```

CLIP figures out which description best matches each X-ray image.

### Using Local CLIP Code

The project imports CLIP from the parent directory instead of installing it:

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clip  # This gets CLIP from ../clip/ folder
```

This is more efficient since the code is already there.

---

## What You Get

### Console Output

When you run batch inference, you'll see:
- Overall accuracy (how many it got right)
- Per-class metrics (how well it did for each disease)
- Confusion matrix (which diseases it mixes up)

### Visualizations

Saved in the `results/` folder:

1. **confusion_matrix_[model].png** - Heatmap showing predictions
   - Diagonal = correct predictions
   - Off-diagonal = where it got confused

2. **predictions_analysis_[model].png** - Two charts:
   - Scatter plot with lines showing correct (green) vs wrong (red) predictions
   - Histogram showing how confident the model was

---

## Expected Results

| Model | Typical Accuracy |
|-------|-----------------|
| Standard CLIP | 36% |
| BiomedCLIP | 20% |

BiomedCLIP does better because it was trained on medical images specifically.

---

## Common Issues & Troubleshooting

### Installation Issues

**"Can't find clip module"**  
- Make sure you're running the script from inside the MyProj folder
- The script automatically looks for CLIP in the parent directory (`../clip/`)

**"Can't find open_clip"**  
- You need to install it for BiomedCLIP:
  ```bash
  pip install open_clip_torch huggingface-hub
  ```

**"torch not found" or CUDA errors**
- Install PyTorch with the correct CUDA version first (see Installation section)
- For CPU-only systems, use the CPU-only installation option

**"ImportError: No module named sklearn"**
- Install scikit-learn:
  ```bash
  pip install scikit-learn
  ```

### Performance Issues

**Low accuracy**  
- Try using BiomedCLIP instead of Standard CLIP with the `--model biomed` flag
- BiomedCLIP is specifically trained on medical images

**Slow inference**
- Install PyTorch with CUDA support to use GPU acceleration
- Use batch mode instead of processing images one by one

**Out of memory errors**
- Use CPU instead of GPU if you have limited GPU memory
- Process images in smaller batches

### Usage Issues

**"No images found in directory"**
- Check that your image directory contains .png, .jpg, or .jpeg files
- Verify the path is correct: `--data path/to/images`

**Missing visualizations**
- Visualizations are only generated in batch mode
- Check the `results/` folder after batch inference
- Requires matplotlib and seaborn to be installed

---

## Technical Details

### Why Zero-Shot Learning?

- Medical datasets are expensive to create and label
- Zero-shot learning lets us use models trained on huge general datasets
- We can add new disease categories just by changing the text descriptions
- No need to retrain the whole model

### Advantages

- Works immediately without training
- Can classify new categories by just changing text
- Uses human-readable descriptions
- Pre-trained on massive datasets

### Limitations

- Not as accurate as models specifically trained on this exact task
- Results depend on how you write the text descriptions
- Standard CLIP wasn't trained on medical images (hence BiomedCLIP performs better)

---
