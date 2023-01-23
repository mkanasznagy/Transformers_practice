import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(self.head_dim*self.heads == self.embed_size), "Embed size needs to be divisible by number of heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim*self.heads, self.embed_size)
    
    def forward(self, values, keys, query, mask):
        # How many queries we are sending in at the same time
        N = query.shape[0] 
        
        # Corresponds to the source sentence length (used in encoder) or the target sentence (used in decoder)
        value_len = values.shape[1] 
        key_len = keys.shape[1]
        query_len = query.shape[1]
        
        # Split embedding into self.heads pieces along dim 2 -> dim 2, 3
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim) # are we sure this is key_len and not query_len??
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, value_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("1e-20"))
            
        attention = torch.softmax(energy / (self.embed_size)**(1/2), dim=3) # dim=3 means that we are normalizing across the key length
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim), where value_len==key_len
        # out shape: (N, query_len, heads, head_dim), and then flatten the last two dimensions
        
        out = self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropuot, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        
        # Similar to batch normalization, but does the normalization per example (and not for the entire batch)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        
        self.droupout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query)) # Skip connection 
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # Skip connection
        
        return out    



class EncoderModule(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length # maximal length of input sentences
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    dorward_expansion=forward_expansion
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # input is N sequences of length seq_length
        N, seq_length = x.shape 
        
        # positions is N identical sequences going from 0 to seq_length-1, in the numerical format of device
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # learns how words are structured through the position embeddings
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            # In the encoder, value, key and query are all the same input: 'out'
            out = layer(out, out, out, mask) 
        
        return out



class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )



