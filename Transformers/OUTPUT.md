# Model Evaluation & Project Output Details

## 📌 Project Architecture
- **Framework:** PyTorch (from scratch, avoiding `nn.Transformer`)[cite: 8, 13]
- **Task:** English-to-Italian Seq2Seq Neural Machine Translation[cite: 5, 13]
- **Key Modules:** 
  - Tokenizers with standard special tokens (`[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`)[cite: 14]
  - Custom dataset preparation with causal and padding masks (`dataset.py`)[cite: 7, 13]
  - Multi-Head Attention, Positional Encoding, and Layer Normalization[cite: 13]
  - Beam Search Decoder (`inference.py`)[cite: 10, 13]
- **Deployment Interfaces:**
  - REST API via FastAPI (`app.py`)[cite: 5]
  - Interactive Web UI via Gradio (`demo.py`)[cite: 8]

---

## ⚙️ Hyperparameter Configuration
* **Batch Size:** 8[cite: 6, 11]
* **Max Length:** 350[cite: 6, 11]
* **Embedding Dimension ($d_{model}$):** 512[cite: 6, 11]
* **Learning Rate:** $10^{-3}$[cite: 6, 11]
* **Epochs:** 25[cite: 6, 11]
* **Beam Search Size:** 3[cite: 10]

---

## 📊 Evaluation Results
- **Dataset Evaluated:** `Helsinki-NLP/opus_books` (`en-it`) validation subset[cite: 9]
- **Metric:** SacreBLEU Corpus BLEU Score[cite: 9]
- **Evaluation Script:** `compute_bleu_score()` in `bleu.py`[cite: 9]

---

## 🏃 Quick Execution Commands

### 1. Launch FastAPI Server
```bash
uvicorn main:app --reload