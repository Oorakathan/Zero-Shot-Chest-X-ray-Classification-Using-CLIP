"""
BiomedCLIP Chest X-ray Classifier
Uses Microsoft's medical-specialized CLIP model trained on 15M medical images.

Classes: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia

Usage:
  python biomedclip_5class.py <path_to_xray_image>
  python biomedclip_5class.py  # Uses sample from finalData if available
"""

import torch
from PIL import Image
import os
import sys

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
except ImportError:
    print("Error: open_clip_torch not installed")
    print("\nPlease install it first:")
    print("  pip install open_clip_torch")
    print("\nOr use the standard CLIP model in classify_5class.py")
    sys.exit(1)

# Disease classes we're working with
SELECTED_CLASSES = ["Pneumonia", "Effusion", "Emphysema", "Fibrosis", "Hernia"]

# Text descriptions for zero-shot classification
# BiomedCLIP was trained on medical images, so simpler prompts work well
text_descriptions = [
    "chest X-ray showing pneumonia",
    "chest X-ray showing pleural effusion",
    "chest X-ray showing emphysema",
    "chest X-ray showing pulmonary fibrosis",
    "chest X-ray showing diaphragmatic hernia"
]

def load_biomedclip_model():
    """
    Load BiomedCLIP model from Hugging Face.
    This is a specialized version of CLIP trained on medical images.
    
    Note: First run will download the model (may take a few minutes)
    """
    print("\nLoading BiomedCLIP model from Microsoft...")
    print("This may take a moment on first run (downloading model)...")
    
    try:
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        print("BiomedCLIP model loaded successfully!")
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Error loading BiomedCLIP: {e}")
        print("\nTrying alternative loading method...")
        raise

def predict_xray(image_path, model, preprocess, tokenizer, top_k=5):
    """
    Predict disease class for a chest X-ray using BiomedCLIP.
    
    How it works:
    1. Load and preprocess the X-ray image
    2. Convert text descriptions to embeddings using medical tokenizer
    3. Compare image similarity to each text description
    4. Return the most similar diseases with confidence scores
    
    Args:
        image_path: Path to the X-ray image file
        model: BiomedCLIP model
        preprocess: Image preprocessing function
        tokenizer: Medical text tokenizer
        top_k: Number of top predictions to return
    
    Returns:
        List of (disease_name, confidence) tuples, sorted by confidence
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize the text descriptions
    # BiomedCLIP uses longer context length (256 vs 77 for standard CLIP)
    context_length = 256
    text_inputs = tokenizer(text_descriptions, context_length=context_length).to(device)
    
    # Get embeddings and calculate similarities
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_input, text_inputs)
        logits = (logit_scale * image_features @ text_features.t()).detach()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    
    # Get top k predictions
    top_indices = probs.argsort()[::-1][:top_k]
    results = [(SELECTED_CLASSES[i], probs[i]) for i in top_indices]
    
    return results

def main():
    """Main function to handle command line usage"""
    print("\nBiomedCLIP - Chest X-ray Classifier")
    print("=" * 70)
    print("Classes: Pneumonia, Effusion, Emphysema, Fibrosis, Hernia")
    print("Method: Zero-shot classification with medical-specialized model")
    print("=" * 70)
    
    # Get image path from command line or use sample
    if len(sys.argv) < 2:
        print("\nUsage: python biomedclip_5class.py <path_to_xray_image>")
        print("\nExample:")
        print("  python biomedclip_5class.py finalData/00000001_001.png")
        
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
            print("Run extract_sample_images.py first")
            return
    else:
        image_path = sys.argv[1]
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        return
    
    # Load the BiomedCLIP model
    try:
        model, preprocess, tokenizer = load_biomedclip_model()
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("\nPlease ensure you have installed open_clip_torch:")
        print("  pip install open_clip_torch")
        return
    
    # Run prediction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nAnalyzing: {os.path.basename(image_path)}")
    print(f"Device: {device}")
    print("Running inference...")
    
    results = predict_xray(image_path, model, preprocess, tokenizer, top_k=5)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    
    for rank, (class_name, prob) in enumerate(results, 1):
        # Create a simple progress bar
        bar = "█" * int(prob * 40)
        print(f"{rank}. {class_name:15s}: {prob*100:6.2f}% {bar}")
    
    # Show top prediction
    top_class, top_prob = results[0]
    print("\n" + "=" * 70)
    print(f"Diagnosis: {top_class}")
    print(f"Confidence: {top_prob*100:.2f}%")
    print(f"Model: BiomedCLIP (Medical-specialized)")
    
    # Confidence level assessment
    if top_prob > 0.6:
        print("Confidence Level: High")
    elif top_prob > 0.4:
        print("Confidence Level: Medium")
    else:
        print("Confidence Level: Low")
    
    print("=" * 70)
    
    # Additional information about the model
    print("\nModel Information:")
    print("  - BiomedCLIP by Microsoft Research")
    print("  - Trained on 15M medical image-text pairs from PubMed")
    print("  - Published in NEJM AI (2024)")
    print("  - Specialized for medical imaging tasks")
    print("  - Better performance on medical images than standard CLIP")
    
    print("\nInference complete!\n")

if __name__ == "__main__":
    main()
