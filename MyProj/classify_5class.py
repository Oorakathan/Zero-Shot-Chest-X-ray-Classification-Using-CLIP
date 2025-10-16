"""
Standard CLIP Chest X-ray Classifier
Classifies chest X-rays into five disease categories using zero-shot learning.

Classes: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia

Usage:
  python classify_5class.py <path_to_xray_image>
  python classify_5class.py  # Uses sample from finalData if available
"""

import os
import sys
import torch
from PIL import Image
from typing import List, Tuple

# Add parent directory to path to use local CLIP code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import clip

# Disease classes we're working with
SELECTED_CLASSES = ["Pneumonia", "Effusion", "Emphysema", "Fibrosis", "Hernia"]

# Text descriptions for zero-shot classification
# Adding medical context helps CLIP understand what to look for
TEXT_PROMPTS = [
    "a chest X-ray showing pneumonia",
    "a chest X-ray showing pleural effusion", 
    "a chest X-ray showing emphysema",
    "a chest X-ray showing pulmonary fibrosis",
    "a chest X-ray showing diaphragmatic hernia"
]

def load_clip_model(device: str = None):
    """
    Load Standard CLIP model from local repository.
    Uses the CLIP code in the parent directory instead of an installed package.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Loaded Standard CLIP model on {device}")
    return model, preprocess, clip.tokenize, device

@torch.no_grad()
def predict_xray(image_path: str, model, preprocess, tokenizer, device: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Predict disease class for a chest X-ray using CLIP.
    
    How it works:
    1. Load and preprocess the X-ray image
    2. Convert text descriptions to embeddings
    3. Compare image similarity to each text description
    4. Return the most similar diseases with confidence scores
    
    Args:
        image_path: Path to the X-ray image file
        model: CLIP model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
        device: Computing device (cuda or cpu)
        top_k: Number of top predictions to return
    
    Returns:
        List of (disease_name, confidence) tuples, sorted by confidence
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize the text descriptions
    text_inputs = tokenizer(TEXT_PROMPTS).to(device)
    
    # Get embeddings from CLIP
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    # Normalize features (important for cosine similarity)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores and convert to probabilities
    logits = (100.0 * image_features @ text_features.T)
    probs = logits.softmax(dim=-1).cpu().numpy()[0]
    
    # Get top k predictions
    top_indices = probs.argsort()[::-1][:top_k]
    results = [(SELECTED_CLASSES[i], float(probs[i])) for i in top_indices]
    
    return results

def format_results(results: List[Tuple[str, float]], image_path: str) -> str:
    """Format prediction results for clean display"""
    lines = []
    lines.append("=" * 70)
    lines.append("PREDICTIONS")
    lines.append("=" * 70)
    
    for rank, (class_name, prob) in enumerate(results, 1):
        # Create a simple progress bar
        bar = "█" * int(prob * 40)
        lines.append(f"{rank}. {class_name:15s}: {prob*100:6.2f}% {bar}")
    
    # Show top prediction
    top_class, top_prob = results[0]
    lines.append("\n" + "=" * 70)
    lines.append(f"Diagnosis: {top_class}")
    lines.append(f"Confidence: {top_prob*100:.2f}%")
    lines.append(f"Model: Standard CLIP (ViT-B/32)")
    
    # Confidence level assessment
    if top_prob > 0.6:
        lines.append("Confidence Level: High")
    elif top_prob > 0.4:
        lines.append("Confidence Level: Medium")
    else:
        lines.append("Confidence Level: Low")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    """Main function to handle command line usage"""
    print("\nStandard CLIP - Chest X-ray Classifier")
    print("=" * 70)
    print("Classes: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia")
    print("Method: Zero-shot classification with medical prompts")
    print("=" * 70)
    
    # Get image path from command line or use sample
    if len(sys.argv) < 2:
        print("\nUsage: python classify_5class.py <path_to_xray_image>")
        print("\nExample:")
        print("  python classify_5class.py finalData/00000001_001.png")
        
        # Try to find a sample image in finalData
        sample_dir = "finalData"
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                image_path = os.path.join(sample_dir, images[0])
                print(f"\nUsing sample image: {image_path}")
            else:
                print("\nNo images found in finalData folder")
                return
        else:
            print("\nfinalData folder not found")
            return
    else:
        image_path = sys.argv[1]
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        return
    
    # Load the CLIP model
    try:
        model, preprocess, tokenizer, device = load_clip_model()
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        return
    
    # Run prediction
    print(f"\nAnalyzing: {os.path.basename(image_path)}")
    print(f"Device: {device}")
    print("Running inference...")
    
    try:
        results = predict_xray(image_path, model, preprocess, tokenizer, device, top_k=5)
        
        # Display results
        print("\n" + format_results(results, image_path))
        
        # Additional info
        print("\nModel Information:")
        print("  - Standard CLIP (ViT-B/32)")
        print("  - Trained on 400M image-text pairs")
        print("  - Zero-shot classification (no training on X-rays)")
        print("  - Uses medical context in text prompts")
        
        print("\nInference complete!\n")
        
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        return

if __name__ == "__main__":
    main()