import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

raw_image = Image.open("C:\Users\DELL\OneDrive\Pictures\Camera Roll\Landscape photo of mine.jpeg").convert('RGB')

inputs = processor(raw_image, return_tensors="pt")
print("Generating caption...")
out = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Resulting Caption: {caption}")