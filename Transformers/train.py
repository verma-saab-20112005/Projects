import torch
from torch import nn
import math
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset, casual_mask
from transformer_from_scratch import build_transformer
from configuration import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

def greedy_decode(model, source, source_mask, tokenizer_source, tokenizer_target, max_len, device):
    sos_idx= tokenizer_target.token_to_id("[SOS]")
    eos_idx= tokenizer_target.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output= model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input= torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1)== max_len:
            break
        # Build mask for the taregt (decoder input)
        decoder_mask= casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out= model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob= model.project(out[:,-1])
        # Select the token with the max probability (because it is a greedy search)
        _, next_word= torch.max(prob, dim=1)
        decoder_input= torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word== eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_dataset, tokenizer_source, tokenizer_target, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count= 0
    # source_texts= []
    # expected= []
    # predicted= []

    # Size of the control window (just use as default value)
    console_width= 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input= batch["encoder_input"].to(device)
            encoder_mask= batch["encoder_mask"].to(device)

            assert encoder_input.size(0)== 1,"Batch size must be 1 for validation"
            
            model_out= greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)

            source_text= batch["source_text"][0]
            target_text= batch["target_text"][0]
            model_out_text= tokenizer_target.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print to the console
            print_msg("-"*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break
            
def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path= Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer= Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer= Whitespace()
        trainer= WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer= trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer= Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
def get_dataset(config):

    dataset_raw= load_dataset("Helsinki-NLP/opus_books", f"{config['source_language']}-{config['target_language']}", split= "train")

    # Build Tokenizers
    tokenizer_source= get_or_build_tokenizer(config, dataset_raw, config["source_language"])
    tokenizer_target= get_or_build_tokenizer(config, dataset_raw, config["target_language"])

    # Keep 80% for training and 20% for validation
    train_dataset_size= int(0.8*len(dataset_raw))
    val_dataset_size= len(dataset_raw)- train_dataset_size
    train_dataset_raw, val_dataset_raw= random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    train_dataset= BilingualDataset(train_dataset_raw, tokenizer_source, tokenizer_target, config["source_language"], config["target_language"], config["max_len"])
    val_dataset= BilingualDataset(val_dataset_raw, tokenizer_source, tokenizer_target, config["source_language"], config["target_language"], config["max_len"])

    max_len_source= 0
    max_len_target= 0

    for item in dataset_raw:
        source_ids= tokenizer_source.encode(item["translation"][config["source_language"]]).ids
        target_ids= tokenizer_target.encode(item["translation"][config["target_language"]]).ids # Here instead of tokenizer_target, tokenizer_source was written

        max_len_source= max(max_len_source, len(source_ids))
        max_len_target= max(max_len_target, len(target_ids))
        
    print(f"Max length of source sentence: {max_len_source}")
    print(f"Max length of target sentence: {max_len_target}")

    train_dataloader= DataLoader(train_dataset, batch_size= config["batch_size"], shuffle= True)
    val_dataloader= DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocab_source_len, vocab_target_len):
    model= build_transformer(vocab_source_len, vocab_target_len, config["max_len"], config["max_len"], config["d_embedding"])
    return model

def train_model(config):
    # Define the device
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_source, tokenizer_target= get_dataset(config)
    model= get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)
    # Tensorboard
    writer= SummaryWriter(config["experiment_name"])
    optimizer= torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch= 0
    global_step= 0
    if config["preload"]:
        model_filename= get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state= torch.load(model_filename)
        initial_epoch= state["epoch"]+1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step= state["global_step"]
    
    # loss_fn= nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_target.token_to_id("[PAD]"), 
    label_smoothing=0.1
    ).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        
        batch_iterator= tqdm(train_dataloader, desc= f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            encoder_input= batch["encoder_input"].to(device) # (32, max_len)
            deocder_input= batch["decoder_input"].to(device) # (32, max_len)
            encoder_mask= batch["encoder_mask"].to(device) # (32, 1, 1, max_len)
            decoder_mask= batch["decoder_mask"].to(device) # (32, 1, max_len, max_len)

            # Run the tensors through the transformers
            encoder_output= model.encode(encoder_input, encoder_mask) # (32, max_len, d_embedding)
            decoder_output= model.decode(encoder_output, encoder_mask, deocder_input, decoder_mask) # (32, max_len, d_embedding)
            proj_output= model.project(decoder_output) # (32, max_len, target_vocab_size)

            label= batch["label"].to(device) # (32, max_len)

            # (32, max_len, target_vocab_size) --> (32, max_len, target_vocab_size)
            loss= loss_fn(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_source, tokenizer_target, config["max_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model at the end of every epoch
        model_filename= get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gloabal_step": global_step
        }, model_filename)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    config= get_config()
    train_model(config)
