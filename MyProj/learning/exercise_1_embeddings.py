"""
EXERCISE 1: Extract and Understand Image Embeddings
===================================================

GOAL: Load a single X-ray image, pass it through CLIP, and understand what comes out.

LEARNING OBJECTIVES:
- Load a pretrained CLIP model
- Preprocess an image correctly
- Extract an image embedding
- Understand tensor shapes (what each dimension means)
- Print and inspect the embedding values

TIME: 20-30 minutes

BEFORE YOU START:
1. Read CLIP_basics.md (at least the "Embeddings" section)
2. Have one X-ray image path ready (e.g., finalData/00000001_001.png)
3. Open a new terminal

YOUR TASK: Fill in the TODOs below and run this script
"""

import os
import torch
from PIL import Image

# TODO 1: Import CLIP library
# Hint: You'll need either 'import clip' (OpenAI) or 'import open_clip'
# Try this first: import clip
# ??? import ???
import clip
# the above is simple
import open_clip
#this is advanced


# TODO 2: Set device (CPU or CUDA)
# Hint: Check if CUDA available, otherwise use CPU
# device = ???
device = "cuda" if torch.cuda.is_available() else "cpu"

# Listing the available models in the clip library
print("this is from clip lib",clip.available_models())
#print("this is from open_clip lib", open_clip.list_pretrained())

# TODO 3: Load CLIP model
# Hint: For OpenAI CLIP, use: model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = ???
model, preprocess = clip.load("ViT-B/32", device = device)
print(type(model))
# TODO 4: Load an X-ray image
# Hint: Use PIL Image.open(), convert to RGB
# image_path = "finalData/00000001_001.png"  # Change this to an actual file
# image = ???
image_path = "D:\\Self Projects\\CLIP-main\\CLIP-main\\MyProj\\finalData\\00000003_005.png"

# TODO 5: Preprocess the image
# Hint: The preprocess function you loaded earlier does this
# This converts PIL Image → PyTorch tensor with correct shape/normalization
# image_tensor = ???
image = Image.open(image_path).convert("RGB")

# TODO 6: Add batch dimension
# Hint: CLIP expects shape [batch_size, channels, height, width]
# Use .unsqueeze(0) to add batch dimension
# image_tensor = image_tensor.???
image_tensor = preprocess(image).unsqueeze(0)

# TODO 7: Move tensor to device
# Hint: Use .to(device)
# image_tensor = ???
image_tensor = image_tensor.to(device)

# TODO 8: Extract image embedding (with no gradient computation)
# Hint: Wrap in torch.no_grad() for efficiency
# Use model.encode_image(image_tensor)
# with torch.no_grad():
#     image_embedding = ???

with torch.no_grad():
    image_embedding = model.encode_image(image_tensor)


# TODO 9: Print information about the embedding
print("=" * 60)
print("IMAGE EMBEDDING ANALYSIS")
print("=" * 60)

# TODO 10: Print the shape
# What does each dimension represent?
# print(f"Embedding shape: {???}")
print("Shape: ", image_embedding.shape)

# TODO 11: Print the data type
# print(f"Data type: {???}")
print("Data type: ", image_embedding.dtype)


# TODO 12: Calculate and print the norm (length of the vector)
# Hint: Use torch.norm(image_embedding)
# norm = ???
# print(f"Embedding norm: {???}")
print("Norm: ", torch.norm(image_embedding))

# TODO 13: Print first 10 values
# This shows you what the actual numbers look like
# print(f"First 10 values: {???}")
print("First 10 numbers in vector: ", image_embedding.flatten()[:10].cpu().numpy())

# TODO 14: Print min and max values
# print(f"Min value: {???}")
# print(f"Max value: {???}")
print("MIN: ", torch.min(image_embedding))
print("MAX: ", torch.max(image_embedding))

# TODO 15: Calculate mean and standard deviation
# print(f"Mean: {???}")
# print(f"Std: {???}")
print("Mean: ", torch.mean(image_embedding))
print("Standard Deviation: ",torch.std(image_embedding))
print("=" * 60)

# REFLECTION QUESTIONS (Answer these in your learning journal):
# 1. What is the shape of the embedding? What does each dimension represent?
# 2. What's the norm of the embedding? Why is this value significant?
# 3. Are the embedding values mostly positive, negative, or mixed?
# 4. If you run this on a different image, does the embedding change?
# 5. What do you think determines these specific numbers?

# EXTENSION CHALLENGES (Optional):
# - Extract embeddings for 3 different X-rays and compare their norms
# - Compute the cosine similarity between two image embeddings manually
# - Try normalizing the embedding to unit length yourself

print("\n✅ Exercise 1 complete! Write your observations in learning_journal.md")
