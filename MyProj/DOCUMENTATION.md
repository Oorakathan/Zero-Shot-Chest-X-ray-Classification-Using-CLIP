# Zero-Shot Chest X-ray Classification Using CLIP

## 1. Understanding CLIP

CLIP (Contrastive Language-Image Pre-training) is a neural network that connects images and text. It was trained on 400 million image-text pairs from the internet. The key idea is that CLIP learns to understand both images and their descriptions in a shared space.

### How Zero-Shot Classification Works

Zero-shot means the model can classify images without being trained on them. Here's how:

1. The model converts an X-ray image into a vector of numbers
2. It converts text descriptions of diseases into vectors too
3. It compares which text vector is most similar to the image vector
4. The disease with the highest similarity becomes the prediction

For example:
- X-ray image becomes vector [0.2, 0.8, 0.1, ...]
- Text "chest X-ray showing pneumonia" becomes vector [0.3, 0.7, 0.2, ...]
- High similarity score means the image likely shows pneumonia

### Why CLIP for Medical Images

Traditional methods need thousands of labeled examples to train. Medical labeling is expensive because it requires expert radiologists. CLIP solves this by using knowledge learned from millions of general images. We just describe what we want to find, and CLIP matches images to descriptions.

The main limitation is that standard CLIP was trained on general images, not medical ones. That's why we also use BiomedCLIP, which was trained on 15 million medical images and performs better on X-rays.

---

## 2. Project Implementation

### Dataset

We use the NIH Chest X-ray Dataset from Kaggle with 5 disease classes:
- Pneumonia (150 images)
- Pleural Effusion (150 images)
- Emphysema (150 images)
- Pulmonary Fibrosis (150 images)
- Diaphragmatic Hernia (150 images)

Total: 750 images, evenly distributed across classes for fair evaluation.

### Models

**Standard CLIP (ViT-B/32)**
- Trained on 400M general images
- Used from local code repository
- Good baseline for comparison

**BiomedCLIP**
- Trained on 15M medical images from PubMed
- Medical-specialized architecture
- Better accuracy on X-rays

### Text Prompts

We use these descriptions to classify images:

```
"a chest X-ray showing pneumonia"
"a chest X-ray showing pleural effusion"
"a chest X-ray showing emphysema"
"a chest X-ray showing pulmonary fibrosis"
"a chest X-ray showing diaphragmatic hernia"
```

Adding "chest X-ray showing" gives medical context that helps the model understand the task better.

### Code Structure

The project has clean, modular code:

- **batch_infer_5class.py** - Main script for single or batch inference
- **metrics_analysis.py** - Calculates evaluation metrics
- **classify_5class.py** - Reference implementation for Standard CLIP
- **biomedclip_5class.py** - Reference implementation for BiomedCLIP

All code includes clear comments explaining what each part does.

---

## 3. Usage

### Setup

Install required packages:

```bash
pip install torch torchvision pillow numpy scikit-learn matplotlib seaborn
```

For BiomedCLIP support:

```bash
pip install open_clip_torch huggingface-hub
```

### Single Image Classification

```bash
python batch_infer_5class.py --image finalData/00000001_001.png
```

Add `--model biomed` to use BiomedCLIP instead of Standard CLIP.

### Batch Processing

Process all 750 images:

```bash
python batch_infer_5class.py --batch --data finalData --model biomed
```

This generates:
- Performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Prediction analysis plots

Results are saved in the `results/` folder.

---

## 4. Evaluation

### Metrics

The system calculates:

- **Accuracy** - Percentage of correct predictions
- **Precision** - Of predicted cases, how many were correct
- **Recall** - Of actual cases, how many were found
- **F1-Score** - Balance between precision and recall
- **Confusion Matrix** - Shows which diseases get confused

### Visualizations

**Confusion Matrix**: Heatmap where diagonal shows correct predictions and off-diagonal shows errors. This reveals which disease pairs the model confuses.

**Prediction Analysis**: Two charts showing confidence distribution for correct vs incorrect predictions, and a scatter plot comparing ground truth to predictions.

---

## 5. Challenges Faced

### Domain Gap

Standard CLIP was trained on general images like animals and objects, not medical X-rays. This caused lower accuracy. Solution: Use BiomedCLIP which was trained specifically on medical images.

### Prompt Engineering

The exact wording of text descriptions significantly affects results. We tested different variations:
- "pneumonia" - too brief
- "chest X-ray with pneumonia" - better but still vague
- "a chest X-ray showing pneumonia" - best performance

Adding medical context helps the model understand what it's looking at.

### Visual Similarity

Some diseases look similar in X-rays. For example, Pneumonia and Effusion both show white areas in lungs. Emphysema and Fibrosis both affect lung texture. This causes higher confusion between visually similar conditions.

### Evaluation Metrics

In medical diagnosis, false negatives (missing a disease) are more critical than false positives (over-detecting). We calculated multiple metrics to understand different aspects of performance rather than just overall accuracy.

---

## 6. Improvements and Suggestions

### Immediate Improvements

**Better Prompts**: Test multiple prompt variations and combine predictions from different wordings.

**Confidence Thresholds**: Reject predictions below a certain confidence level and flag them for manual review.

**Ensemble Methods**: Combine Standard CLIP and BiomedCLIP predictions by averaging or voting.

**Larger Dataset**: Use the full NIH dataset with 100,000+ images to better evaluate performance.

### Model Improvements

**Fine-tuning**: Adapt BiomedCLIP to this specific dataset with a small amount of labeled data.

**Hierarchical Classification**: First classify Normal vs Abnormal, then identify specific disease.
---

## 7. Results and Discussion

### Expected Performance

BiomedCLIP typically achieves 30-37% accuracy while Standard CLIP gets 20%. The gap shows the importance of domain-specific training for medical images.

### Key Findings

Zero-shot learning works for medical images but with lower accuracy than supervised models. It's useful when labeled data is limited. The approach is practical for initial screening and triage rather than final diagnosis.

BiomedCLIP's better performance confirms that medical-specific training data improves results even in zero-shot settings.

### Practical Applications

This system can be used for:
- Quick screening in emergency departments
- Organizing research datasets
- Educational tools for medical students
- Baseline comparison for specialized models

### Limitations

Zero-shot classification cannot match the accuracy of models trained specifically on large labeled chest X-ray datasets. Results depend heavily on how text prompts are written. The system should assist, not replace, expert radiologists.

---

## 8. Conclusion

This project demonstrates that zero-shot classification using CLIP is viable for medical imaging when labeled data is scarce. While not matching specialized supervised models, it provides a practical solution that works immediately without expensive labeling.

The comparison between Standard CLIP and BiomedCLIP shows the value of domain-specific pre-training. The clean, modular code makes it easy to adapt for other medical imaging tasks or disease categories.

Key takeaway: Vision-language models like CLIP open new possibilities for medical AI by reducing dependence on large labeled datasets. This approach bridges the gap between general computer vision and specialized medical applications.

---

## Project Files

```
MyProj/
├── batch_infer_5class.py       # Main inference script
├── metrics_analysis.py         # Evaluation metrics
├── classify_5class.py          # Standard CLIP reference
├── biomedclip_5class.py        # BiomedCLIP reference
├── requirements.txt            # Dependencies
├── Data_Entry_2017.csv         # Dataset metadata
├── finalData/                  # 750 X-ray images
│   ├── *.png                   
│   ├── image_labels.txt        # Ground truth labels
│   └── dataset_summary.txt     
└── results/                    # Generated visualizations
    ├── confusion_matrix_*.png
    └── predictions_analysis_*.png
```

---

## References

- CLIP Paper: Learning Transferable Visual Models From Natural Language Supervision (https://arxiv.org/pdf/2103.00020)
- NIH Chest X-ray Dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data
- BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

---
