import torch
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import skimage
import io
from PIL import Image
import numpy as np


# Create model object
CLASSES = ["Nudity", "Alcohol", "Drugs", "Hate Symbols", "Violence", "Credit Cards", "Misc"]
IMAGE_SIZE = (512, 512)
pretrained_model_name_or_path = 'google/vit-base-patch16-224-in21k'
model = None
feature_extractor = None

def preprocess(input_data):
    image = input_data[0].get("data") or input_data[0].get("body")

    if isinstance(image, (bytearray, bytes)):
        image = np.array(Image.open(io.BytesIO(image)))

    preprocessed_image = skimage.transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
    preprocessed_image = feature_extractor(preprocessed_image, return_tensors='pt')['pixel_values']

    return preprocessed_image

def postprocess(pred_proba):
    probabilities = pred_proba[0]
    return dict(zip(CLASSES, probabilities.tolist()))



def apply_fn(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model, feature_extractor

    if not data:
        manifest = context.manifest
        print(manifest)

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.load(model_pt_path, map_location=device)
        feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    else:
        #infer and return result
        # return model(data)
        processed_input = preprocess(data)
        logits = model(processed_input).logits
        pred_proba = torch.nn.functional.softmax(logits).cpu().detach().numpy()
        res = postprocess(pred_proba)
        print(res)
        return [res]
    