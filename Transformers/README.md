# PyTorch Transformer Model from Scratch 🚀

An end-to-end implementation of the original Transformer architecture (*Attention Is All You Need*) built entirely from scratch in **PyTorch** for English-to-Italian Machine Translation.

---

## 🛠️ Key Technical Features
- **Pure PyTorch Architecture:** Implemented Multi-Head Attention, Scaled Dot-Product Attention, Positional Encoding, and Layer Normalization without using built-in `nn.Transformer` modules.
- **Advanced Decoding:** Utilizes **Beam Search Decoding** for optimized sequence generation over standard greedy search.
- **Custom Pipeline:** Tokenization, dataset preparation with causal/padding masks, custom training loop, and checkpointing.
- **Interactive UI & API:** Packaged with **Gradio** for real-time translation testing and **FastAPI** for API serving.

---

## 📂 Project Structure
```text
├── configuration.py          # Training and model hyperparameter settings
├── dataset.py                # Dataset parsing & padding/causal masking
├── transformer_from_scratch.py # Raw PyTorch Transformer Architecture
├── train.py                  # Custom training loop & TensorBoard tracking
├── inference.py              # Beam Search inference pipeline
├── demo.py                   # Interactive Gradio Web UI
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Local Interactive Web UI
```bash
python demo.py
```
