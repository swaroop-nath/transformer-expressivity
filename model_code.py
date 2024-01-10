import torch
import torch.nn as nn
import math
# from torch.nn import Transformer # Changed to our implementation of Transformer
from transformer import Transformer

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_length=5000):
        super().__init__()
        self.dropout = dropout
        pe = torch.empty(max_length, d_model)
        nn.init.xavier_uniform_(pe, gain=nn.init.calculate_gain('relu'))
        self.pe = nn.Parameter(pe, requires_grad=True).unsqueeze(dim=0) # (1, max_length, d_model)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device) # (bsz, seq_len, d_model)

# Code for Positional Encoding, based on Attention Is All You Need, from https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_length=5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    super().__init__()     

    self.dropout = nn.Dropout(p=dropout)      

    pe = torch.zeros(max_length, d_model)    
    k = torch.arange(0, max_length).unsqueeze(1)  
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(k * div_term)    
    pe[:, 1::2] = torch.cos(k * div_term)  
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    x = x + self.pe[:, :x.size(1)].requires_grad_(False) 

    return self.dropout(x)

class TransformerCheck(nn.Module):
    def __init__(self, mode, num_classes=None, padding_value=-100, **transformer_kwargs):
        super().__init__()
        self._emb_dim = transformer_kwargs['emb-dim']
        self._dec_start_token_emb = None
        self._model = Transformer(
                    d_model=transformer_kwargs['emb-dim'],
                    nhead=transformer_kwargs['num-heads'],
                    num_encoder_layers=transformer_kwargs['num-encoder-layers'],
                    num_decoder_layers=transformer_kwargs['num-decoder-layers'],
                    dim_feedforward=transformer_kwargs['pffn-dim'],
                    dropout=transformer_kwargs['dropout'],
                    activation=transformer_kwargs['activation'],
                    batch_first=True,
                    norm_first=transformer_kwargs['pre-ln']).train()
        
        assert transformer_kwargs['pe-type'] in ['sinusoid', 'learned'], f"{transformer_kwargs['pe-type']} not supported"
        if transformer_kwargs['pe-type'] == 'sinusoid':
            self._pe = SinusoidalPositionalEncoding(transformer_kwargs['emb-dim'], dropout=transformer_kwargs['dropout'], max_length=10)
        elif transformer_kwargs['pe-type'] == 'learned':
            self._pe = LearnedPositionalEncoding(transformer_kwargs['emb-dim'], dropout=transformer_kwargs['dropout'], max_length=10)
        
        assert mode in ['classification', 'regression'], f"Specified mode {mode} not supported"
        
        if mode == 'classification': 
            assert num_classes is not None, f"`num_classes` cannot be `None` when mode is `classification`"
            assert num_classes > 0, f"`num_classes` cannot be less than or equal to zero"
            self._ff = nn.Linear(in_features=transformer_kwargs['emb-dim'], out_features=num_classes)
        
        self._mode = mode
        self._padding_value = padding_value
        
    def _add_start_embedding(self, dec_input_ids, dec_attn_mask):
        if self._dec_start_token_emb is None:
            # dec_start_token_emb has to be of size (1, emb_dim)
            self._dec_start_token_emb = torch.randn((1, self._emb_dim), dtype=torch.float, device=dec_input_ids.device)
        # Adding a dummy start embedding (for regression) or index (for classification)
        # dec_input_ids.size() == (bsz, seq_len, emb_dim)
        if dec_input_ids.size(1) == 0: return self._dec_start_token_emb.expand((dec_input_ids.size(0), 1, self._emb_dim)), torch.ones((dec_input_ids.size(0), 1), dtype=torch.int, device=dec_input_ids.device)
        dec_input_ids = torch.cat((self._dec_start_token_emb.expand((dec_input_ids.size(0), 1, self._emb_dim)), dec_input_ids), dim=1)
        dec_attn_mask = torch.cat((torch.ones((dec_input_ids.size(0), 1), dtype=torch.int, device=dec_input_ids.device), dec_attn_mask), dim=1)
        return dec_input_ids, dec_attn_mask
    
    def _get_attention_mask(self, tensor):
        assert len(tensor.size()) in [2, 3], f"Expected elements in batch to be 1D or 2D, but found tensor of shape: {tensor.size()}"
        if len(tensor.size()) == 3: # each item in the batch is a 2D tensor
            return 1 - torch.eq(tensor.mean(dim=-1).to(torch.int), int(self._padding_value)).to(torch.int)
        elif len(tensor.size()) == 2: # each item in the batch is a 1D tensor
            return 1 - torch.eq(tensor, int(self._padding_value)).to(torch.int)
    
    def _mse_loss(self, outputs, labels):
        # outputs.size() == labels.size() == (bsz, seq_len_out, emb_dim)
        labels_mask = self._get_attention_mask(labels) # (bsz, seq_len_out)
        
        return torch.sum(
            torch.sum(
                torch.sum(
                    torch.square(outputs - labels), dim=-1
                    ) / labels.size(2) * labels_mask, dim=-1
                ) / labels_mask.sum(dim=-1)
            ) / labels.size(0)
        
    def _ce_loss(self, outputs, labels):
        # outputs.size() == (bsz, seq_len_out, num_classes)
        # labels.size() == (bsz, seq_len_out)
        outputs = outputs.reshape(-1, outputs.size(-1))
        labels = labels.reshape(-1)
        
        return nn.functional.cross_entropy(input=outputs, target=labels, ignore_index=self._padding_value)
        
    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, labels, return_loss=True):
        '''
            encoder_input_ids: Input to the encoder, size == (bsz, seq_len_in, emb_dim)
            encoder_attention_mask: Mask for input to the encoder, size == (bsz, seq_len_in)
            decoder_input_ids: Input to the decoder, size == (bsz, seq_len_out, emb_dim)
            decoder_attention_mask: Mask for input to the decoder, size == (bsz, seq_len_out)
            labels: Ground truth labels, size == (bsz, seq_len_out) for `classification` else (bsz, seq_len_out, emb_dim)
            
            The transformer predicts y_{2:n} given x_{1:n} and y_{1:n-1}
        '''
        decoder_input_ids, decoder_attention_mask = self._add_start_embedding(decoder_input_ids, decoder_attention_mask)
        encoder_input_ids = self._pe(encoder_input_ids)
        decoder_input_ids = self._pe(decoder_input_ids)
        
        transformer_output, attention_weights = self._model(
            src=encoder_input_ids,
            src_key_padding_mask=encoder_attention_mask,
            tgt=decoder_input_ids,
            tgt_key_padding_mask=decoder_attention_mask,
        ) # (bsz, seq_len_out, emb_dim)
                
        if self._mode == 'regression' and return_loss:
            assert labels is not None, f"`labels` cannot be None if `return_loss` is set to `True`"
            loss = self._mse_loss(transformer_output, labels)
            return {'loss': loss, 'output': transformer_output, 'vector-output': transformer_output, 'attention-weights': attention_weights}
        elif self._mode == 'classification' and return_loss:
            assert labels is not None, f"`labels` cannot be None if `return_loss` is set to `True`"
            classification_output = self._ff(transformer_output)
            loss = self._ce_loss(classification_output, labels)
            return {'loss': loss, 'output': classification_output, 'vector-output': transformer_output, 'attention-weights': attention_weights}
        
        if self._mode == 'regression' and not return_loss:
            return {'output': transformer_output, 'attention-weights': attention_weights}
        elif self._mode == 'classification' and not return_loss:
            classification_output = self._ff(transformer_output)
            return {'output': classification_output, 'attention-weights': attention_weights}
            