import torch
from torchtext.data.metrics import bleu_score
import sacrebleu
from tqdm import tqdm
from dataset import casual_mask
from datasets import load_dataset
from inference import Translator

def compute_bleu_score():
    print("Loading test dataset and translator...")
    translator = Translator(weights_path="weights/tmodel_02.pt") # Path to your best checkpoint
    
    # Load validation split of the same dataset used during training
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-it", split="train[-30:]")
    
    predictions = []
    references = []

    print("Generating translations for evaluation...")
    for item in tqdm(dataset):
        src_text = item["translation"]["en"]
        tgt_text = item["translation"]["it"]

        # Run translation
        pred_text = translator.translate(src_text)

        predictions.append(pred_text)
        references.append(tgt_text)

    # Compute SacreBLEU score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print("\n" + "="*40)
    print(f"Final BLEU Score: {bleu.score:.2f}")
    print("="*40)

if __name__ == "__main__":
    compute_bleu_score()