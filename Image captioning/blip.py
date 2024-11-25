# Install the transformers library
# !pip install transformers Pillow torch torchvision torchaudio
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image = Image.open("path_to_your_image.jpg")

# Prepare the image
inputs = processor(image, return_tensors="pt")

# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)

# Video Question Answering
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Image URL 
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Specify the question you want to ask about the image
question = "What is in the image?"

# Use the processor to prepare inputs for VQA (image + question)
inputs = processor(raw_image, question, return_tensors="pt")

# Generate the answer from the model
out = model.generate(**inputs)

# Decode and print the answer to the question
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")
