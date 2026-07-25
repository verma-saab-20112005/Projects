import torch
from pathlib import Path
from tokenizers import Tokenizer
from configuration import get_config
from dataset import casual_mask
from transformer_from_scratch import build_transformer

class Translator:
    def __init__(self, weights_path: str = None, beam_size: int = 3):
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size

        # Load Tokenizers
        self.tokenizer_src = Tokenizer.from_file(self.config["tokenizer_file"].format(self.config["source_language"]))
        self.tokenizer_tgt = Tokenizer.from_file(self.config["tokenizer_file"].format(self.config["target_language"]))

        # Build Model
        self.model = build_transformer(
            self.tokenizer_src.get_vocab_size(),
            self.tokenizer_tgt.get_vocab_size(),
            self.config["max_len"],
            self.config["max_len"],
            self.config["d_embedding"]
        ).to(self.device)

        # Load Trained Weights
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Loaded weights from {weights_path}")
        else:
            print("Warning: Running inference with uninitialized/random weights!")

        self.model.eval()

    def translate(self, text: str) -> str:
        """Translates input source string using Beam Search."""
        sos_idx = self.tokenizer_src.token_to_id("[SOS]")
        eos_idx = self.tokenizer_src.token_to_id("[EOS]")
        pad_idx = self.tokenizer_src.token_to_id("[PAD]")

        # Tokenize and format input
        tokens = self.tokenizer_src.encode(text).ids
        encoder_input = torch.cat([
            torch.tensor([sos_idx], dtype=torch.int64),
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor([eos_idx], dtype=torch.int64),
            torch.tensor([pad_idx] * (self.config["max_len"] - len(tokens) - 2), dtype=torch.int64)
        ]).unsqueeze(0).to(self.device)

        encoder_mask = (encoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int().to(self.device)

        with torch.no_grad():
            encoder_output = self.model.encode(encoder_input, encoder_mask)
            
            # Beam Search Initialization
            # Beams store tuples of: (sequence_tensor, cumulative_score)
            beams = [(torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(self.device), 0.0)]

            for _ in range(self.config["max_len"]):
                new_beams = []
                all_done = True

                for seq, score in beams:
                    if seq[0, -1].item() == eos_idx:
                        new_beams.append((seq, score))
                        continue

                    all_done = False
                    decoder_mask = casual_mask(seq.size(1)).type_as(encoder_mask).to(self.device)
                    out = self.model.decode(encoder_output, encoder_mask, seq, decoder_mask)

                    log_probs = self.model.project(out[:, -1])

                    topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size, dim=-1)

                    for i in range(self.beam_size):
                        next_token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                        token_score = topk_log_probs[0, i].item()
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_beams.append((new_seq, score + token_score))

                # Keep top K best sequences
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]
                if all_done:
                    break

            best_seq = beams[0][0].squeeze(0).cpu().numpy()
            return self.tokenizer_tgt.decode(best_seq)

if __name__ == "__main__":
    # Point directly to your local trained weights file:
    translator = Translator(weights_path="weights/tmodel_02.pt") # Or whatever your epoch number is
    
    sample = "Books are full of knowledge."
    print(f"Input: {sample}")
    print(f"Output: {translator.translate(sample)}")