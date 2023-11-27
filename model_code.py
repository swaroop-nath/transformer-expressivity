import torch
import torch.nn as nn

class TransformerCheck(nn.Module):
    def __init__(self, mode, num_classes=None, padding_value=-100, **transformer_kwargs):
        super().__init__()
        self._model = nn.Transformer(
                    d_model=transformer_kwargs['emb-dim'],
                    nhead=transformer_kwargs['num-heads'],
                    num_encoder_layers=transformer_kwargs['num-encoder-layers'],
                    num_decoder_layers=transformer_kwargs['num-decoder-layers'],
                    dim_feedforward=transformer_kwargs['pffn-dim'],
                    dropout=transformer_kwargs['dropout'],
                    activation=transformer_kwargs['activation'],
                    batch_first=True,
                    norm_first=transformer_kwargs['pre-ln'])
        
        assert mode in ['classification', 'regression'], f"Specified mode {mode} not supported"
        
        if mode == 'classification': 
            assert num_classes is not None, f"`num_classes` cannot be `None` when mode is `classification`"
            assert num_classes > 0, f"`num_classes` cannot be less than or equal to zero"
            self._ff = nn.Linear(in_features=transformer_kwargs['emb-dim'], out_features=num_classes)
        
        self._mode = mode
        self._padding_value = padding_value
    
    def _get_attention_mask(self, tensor):
        assert len(tensor.size()) in [2, 3], f"Expected elements in batch to be 1D or 2D, but found tensor of shape: {tensor.size()}"
        if len(tensor.size()) == 3: # each item in the batch is a 2D tensor
            return 1 - torch.eq(tensor.sum(dim=-1).to(torch.int), int(self._padding_value)).to(torch.int)
        elif len(tensor.size()) == 2: # each item in the batch is a 1D tensor
            return 1 - torch.eq(tensor, int(self._padding_value)).to(torch.int)
    
    def _mse_loss(self, outputs, labels):
        # outputs.size() == labels.size() == (bsz, seq_len_out, emb_dim)
        labels_mask = self._get_attention_mask(labels) # (bsz, seq_len_out)
        
        return torch.sum(torch.sum(torch.sum(torch.square(outputs - labels), dim=-1) / labels.size(2) * labels_mask) / labels_mask.sum(dim=-1)) / labels.size(0)
    
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
        transformer_output = self._model(
            src=encoder_input_ids,
            src_key_padding_mask=encoder_attention_mask,
            tgt=decoder_input_ids,
            tgt_key_padding_mask=decoder_attention_mask,
        ) # (bsz, seq_len_out, n_classes) for `classification` else (bsz, seq_len_out, emb_dim)
        
        if self._mode == 'regression' and return_loss:
            assert labels is not None, f"`labels` cannot be None if `return_loss` is set to `True`"
            loss = self._mse_loss(transformer_output, labels)
            return {'loss': loss, 'output': transformer_output}
        elif self._mode == 'classification' and return_loss:
            assert labels is not None, f"`labels` cannot be None if `return_loss` is set to `True`"
            classification_output = self._ff(transformer_output)
            loss = self._ce_loss(classification_output, labels)
            return {'loss': loss, 'output': classification_output}
        
        if self._mode == 'regression' and not return_loss:
            return {'output': transformer_output}
        elif self._mode == 'classification' and not return_loss:
            classification_output = self._ff(transformer_output)
            return {'output': classification_output}
            