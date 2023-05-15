##############################################################################################
### Convert serialized model to state dict format
##############################################################################################
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
CLASSES = ["Nudity", "Alcohol", "Drugs", "Hate Symbols", "Violence", "Credit Cards", "Misc"]
pretrained_model_name_or_path = 'google/vit-base-patch16-224-in21k'
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

# Save state dict
torch.save(model.state_dict(), 'models/pi_vit.pth')

# Load state dict format model
state_dict_model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path,
    num_labels=len(CLASSES),
    id2label={str(i): c for i, c in enumerate(CLASSES)},
    label2id={c: str(i) for i, c in enumerate(CLASSES)}
)
state_dict_model.load_state_dict(torch.load("models/pi_vit.pth"))
state_dict_model.to(device)

# Load pretrained feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

# Test state dict model inference
print("APPLYING MODEL LOADED FROM STATE DICT...")
IMAGE_URL = "https://media-photos.depop.com/b0/32370405/1198876747_3e9d3b87feee4066843003a272062a56/P0.jpg"
print(apply_model_to_image(state_dict_model, feature_extractor, IMAGE_URL))

# Test original serialized model inference
print("APPLYING MODEL .PT SERIALIZED MODEL...")
IMAGE_URL = "https://media-photos.depop.com/b0/32370405/1198876747_3e9d3b87feee4066843003a272062a56/P0.jpg"
print(apply_model_to_image(model, feature_extractor, IMAGE_URL))