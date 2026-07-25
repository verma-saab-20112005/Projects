import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_language, target_language, max_len)->None:
        super().__init__()

        self.dataset= dataset
        self.tokenizer_source= tokenizer_source
        self.tokenizer_target= tokenizer_target
        self.source_language= source_language
        self.target_language= target_language
        self.max_len= max_len

        self.sos_token= torch.tensor([tokenizer_source.token_to_id("[SOS]")], dtype= torch.int64)
        self.eos_token= torch.tensor([tokenizer_source.token_to_id("[EOS]")], dtype= torch.int64)
        self.pad_token= torch.tensor([tokenizer_source.token_to_id("[PAD]")], dtype= torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: any)->any:
        source_target_pair= self.dataset[index]
        source_text= source_target_pair["translation"][self.source_language]
        target_text= source_target_pair["translation"][self.target_language]

        enc_input_tokens= self.tokenizer_source.encode(source_text).ids
        dec_input_tokens= self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens= self.max_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens= self.max_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError("Sentence is too long !")
        
        # Add SOS and EOS to source text
        encoder_input= torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype= torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype= torch.int64)
        ])

        # Add SOS to the decoder input
        decoder_input= torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype= torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype= torch.int64)
        ])

        # Add EOS to the label (what we expect as output from the decoder)
        label= torch.cat([
            torch.tensor(dec_input_tokens, dtype= torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype= torch.int64)
        ])

        assert encoder_input.size(0)== self.max_len
        assert decoder_input.size(0)== self.max_len
        assert label.size(0)== self.max_len

        return {
            "encoder_input": encoder_input, # (max_len)
            "decoder_input": decoder_input, # (max_len)
            "encoder_mask": (encoder_input!= self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,max_len)
            "decoder_mask": (decoder_input!= self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, max_len) & (1, max_len, max_len)
            "label": label, # (max_len)
            "source_text": source_text,
            "target_text": target_text
        }
    
def casual_mask(size):
    mask= torch.triu(torch.ones(1, size, size), diagonal= 1).type(torch.int64)
    return mask==0
