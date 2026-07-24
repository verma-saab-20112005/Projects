# English to Hindi Translator (NLLB-200)

An interactive command-line application that translates English text into Hindi using Meta's **NLLB-200** (No Language Left Behind) model via Hugging Face Transformers.

## Features

- **Pre-trained Translation Model**: Employs `facebook/nllb-200-distilled-600M` for sequence-to-sequence machine translation.
- **Language Codes**: Explicitly configured for English input (`eng_Latn`) and Hindi output (`hin_Deva`).
- **Interactive CLI**: Continuous command-line loop allowing real-time translation queries.
- **Clean Output**: Ignores non-critical runtime warnings for a clean user interface.

## Prerequisites & Installation

Ensure Python is installed on your system, then install the required dependencies:

```bash
pip install transformers torch

How to RunSave the script (e.g., translator.py).  Execute the file:
Type any English sentence and press Enter to see the Hindi translation.  Type quit, exit, or q (or press Ctrl+C) to terminate the session.  Code Flow BreakdownModel & Tokenizer Initialization: Loads tokenizer with src_lang="eng_Latn" and the seq2seq model.  Translation Function:Tokenizes the English input string[cite: 2].Passes input through model.generate(), forcing the target output language token to Hindi (hin_Deva)[cite: 2].Decodes and returns the generated text[cite: 2].User Loop: Continuously prompts the user for text input until an exit command or signal is received[cite: 2].