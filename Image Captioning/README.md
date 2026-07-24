# Image Captioning with BLIP Model

This Python script utilizes Salesforce's **BLIP (Bootstrapping Language-Image Pre-training)** model via the Hugging Face `transformers` library to automatically generate a descriptive text caption for a local image.

## Features

- **Hugging Face Transformers**: Uses pre-trained `Salesforce/blip-image-captioning-base`.
- **PIL (Pillow)**: Loads and converts local images to the required `RGB` format.
- **Automated Captioning**: Processes the input image tensor and decodes generated output into human-readable text.

## Prerequisites & Installation

Make sure you have Python installed, then install the required dependencies:

```bash
pip install requests pillow transformers torch

Setup & Usage
Open the script (caption_generator.py).

Update the image file path to match your local file location:

*(Note: Use a raw string `r"..."` or forward slashes `/` in Windows paths to prevent backslash escape sequence issues).*

3. Run the script:
   ```bash
   python caption_generator.py
Workflow Explanation
Load Processor & Model: Fetches pre-trained tokenizer/processor and weights for Salesforce/blip-image-captioning-base.

Load Image: Opens the target image and converts it into standard RGB mode.

Pre-processing: Transforms the image into tensors suitable for PyTorch model input.

Inference: Generates caption token IDs with a maximum limit of 50 new tokens.

Decoding: Converts output token IDs back to a readable text string and prints it.
