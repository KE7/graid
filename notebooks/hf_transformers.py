import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs[0]

# We have to force return_dict=False for tracing
model.config.return_dict = False

with torch.no_grad():
    traced_model = torch.jit.trace(model, [inputs.pixel_values])
    traced_outputs = traced_model(inputs.pixel_values)

print((last_hidden_states - traced_outputs[0]).abs().max())