import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import math

# ====================================================================
# Encoder Part
class InputEmbedding(nn.Module):

    def __init__(self, d_embedding: int, vocab_size: int):
        super().__init__()
        self.d_embedding= d_embedding
        self.vocab_size= vocab_size
        self.embedding= nn.Embedding(vocab_size, d_embedding)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_embedding)

class PositionalEncoding(nn.Module):

    def __init__(self, d_embedding: int, max_len: int, dropout: float)->None:
        super().__init__()
        self.d_embedding= d_embedding
        self.max_len= max_len
        self.dropout= nn.Dropout(dropout)

        # Create a matrix of shape (max_len, d_embedding)
        pe= torch.zeros(max_len, d_embedding)
        #Create a vector of shape (max_len, 1)
        position= torch.arange(0, max_len, dtype= torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_embedding, 2).float()*(-math.log(10000.0)/d_embedding))
        # Apply the sin to the even positions and cos to the odd positions
        pe[:, 0::2]= torch.sin(position * div_term)
        pe[:, 1::2]= torch.cos(position * div_term)

        pe= pe.unsqueeze(0) # (1, max_len, d_embedding)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x= x+ (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    # Here eps stands for epsilon
    def __init__(self, eps: float= 10**-6)->None:
        super().__init__()
        self.eps= eps
        self.alpha= nn.Parameter(torch.ones(1)) # Multiplied
        self.beta= nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean= x.mean(dim= -1, keepdim= True)
        std= x.std(dim= -1, keepdim= True)
        return self.alpha*(x-mean)/(std*self.eps)+self.beta
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_embedding: int, d_ff: int, dropout: float)->None:
        super().__init__()
        self.linear_1= nn.Linear(d_embedding, d_ff) # W1 and B1
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(d_ff, d_embedding) #W2 and B2

    def forward(self, x):
        # (Batch, max_len, d_embedding) --> (Batch, max_len, d_ff) --> (Batch, max_len, d_embedding)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_embedding: int, h: int, dropout: float)->None: # Here h is the number of heads and also remember that d_embedding should be divisible by h and d_embedding/h= d_k or d_v
        super().__init__()
        self.d_embedding= d_embedding
        self.h= h
        assert d_embedding % h==0, "d_embedding is not divisible by h"
        self.d_k= d_embedding//h
        self.w_q= nn.Linear(d_embedding, d_embedding) # Wq
        self.w_k= nn.Linear(d_embedding, d_embedding) # Wk
        self.w_v= nn.Linear(d_embedding, d_embedding) # Wv
        
        self.w_o= nn.Linear(d_embedding, d_embedding) # Wo
        self.dropout= nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k= query.shape[-1]
        # (Batch, h, max_len, d_k) --> (Batch, h, max_len, max_len)
        attention_scores= (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -math.inf)
        attention_scores= attention_scores.softmax(dim= -1) # (Batch, h, max_len, max_len)
        if dropout is not None:
            attention_scores= dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q,k,v,mask):
        query= self.w_q(q) # (Batch, max_len, d_embedding) --> (Batch, max_len, d_embedding)
        key= self.w_k(k) # (Batch, max_len, d_embedding) --> (Batch, max_len, d_embedding)
        value= self.w_v(v) # (Batch, max_len, d_embedding) --> (Batch, max_len, d_embedding)

        # (Batch, max_len, d_embedding) --> (Batch, max_len, h, d_k) --> (Batch, h, max_len, d_k)
        query= query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key= key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value= value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores= MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, max_len, d_k) --> (Batch, max_len, h, d_k) --> (Batch, max_len, d_embedding)
        x= x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, max_len, d_embedding) --> (Batch, max_len, d_embedding)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float)->None:
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm= LayerNormalization()

    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)->None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.feed_forward_block= feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, source_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,source_mask))
        x= self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.norm(x)
# ================================================================================

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)->None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block= feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x= self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))
        x= self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x= layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_embedding: int, vocab_size: int)->None:
        super().__init__()
        self.proj= nn.Linear(d_embedding, vocab_size)
    
    def forward(self, x):
        # (Batch, max_len, d_embedding) --> (Batch, max_len, vocab_size)
        # return torch.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding, source_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer)->None:
        super().__init__()
        self.encoder= encoder
        self.decoder= decoder
        self.source_embedding= source_embedding
        self.target_embedding= target_embedding
        self.source_pos= source_pos
        self.target_pos= target_pos
        self.projection_layer= projection_layer

    def encode(self, source, source_mask):
        source= self.source_embedding(source)
        source= self.source_pos(source)
        return self.encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask, target, target_mask):
        target= self.target_embedding(target)
        target= self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(source_vocab_size:int, target_vocab_size:int, source_max_len:int, target_max_len:int, d_embedding:int=512, N:int=6, h:int=8, dropout:float=0.1, d_ff:int=2048)->Transformer:
    # Create the embedding layers
    source_embedding= InputEmbedding(d_embedding, source_vocab_size)
    target_embedding= InputEmbedding(d_embedding, target_vocab_size)

    # Create the positional encoding layers
    source_pos= PositionalEncoding(d_embedding, source_max_len, dropout)
    target_pos= PositionalEncoding(d_embedding, target_max_len, dropout)

    # Create the encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block= MultiHeadAttentionBlock(d_embedding, h, dropout)
        feed_forward_block= FeedForwardBlock(d_embedding, d_ff, dropout)
        encoder_block= EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Creating decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block= MultiHeadAttentionBlock(d_embedding, h, dropout)
        decoder_cross_attention_block= MultiHeadAttentionBlock(d_embedding, h, dropout)
        feed_forward_block= FeedForwardBlock(d_embedding, d_ff, dropout)
        decoder_block= DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the encoder and the decoder
    encoder= Encoder(nn.ModuleList(encoder_blocks))
    decoder= Decoder(nn.ModuleList(decoder_blocks))

    # Create the Projection Layer
    projection_layer= ProjectionLayer(d_embedding, target_vocab_size)

    # Create the transformer
    transformer= Transformer(encoder, decoder, source_embedding, target_embedding, source_pos, target_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer

