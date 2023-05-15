###############################################
### Test simple model loading and inference
###############################################
import os
import re
import pandas as pd
import requests
from collections.abc import Iterable
import imghdr
import tarfile
from collections import Counter

import random
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np
from skimage import io, transform


IMAGE_SIZE = (512, 512)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



def preprocess_image(feature_extractor, image):
  preprocessed_image = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
  preprocessed_image = feature_extractor(preprocessed_image, return_tensors='pt')['pixel_values'].to(device)
  return preprocessed_image


def apply_model_to_image(model, feature_extractor, image_path):
  image = io.imread(image_path)
  logits = model(preprocess_image(feature_extractor, image)).logits
  pred_proba = torch.nn.functional.softmax(logits).cpu().detach().numpy()
  return pred_proba


# Load model from save file (not state dict)
model = torch.load('models/pi_vit.pt', map_location=device)
model.eval()

# Load pretrained feature extractor
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

# Test model inference
IMAGE_URL = "https://media-photos.depop.com/b0/32370405/1198876747_3e9d3b87feee4066843003a272062a56/P0.jpg"
print(apply_model_to_image(model, feature_extractor, IMAGE_URL))