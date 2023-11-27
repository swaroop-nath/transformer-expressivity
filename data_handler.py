from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class SyntheticDataset(Dataset):
    def __init__(self, data_path, mode, input_fname, output_fname, num_classes=None):
        assert mode in ['classification', 'regression'], f"Specified mode {mode} not supported"
        if mode == 'classification': assert num_classes is not None, f"`num_classes` cannot be None if `mode` is set to `classification`"
        self._input_data, self._output_data = self._read_data(data_path, mode, num_classes, input_fname, output_fname)
        assert len(self._input_data) == len(self._output_data), f"Input and Output differ in length | Input has {len(self._input_data)} rows and output has {len(self._output_data)} rows"
        self._len = len(self._input_data)
        self._splitter = ","
        self._mode = mode
            
    def _read_data(self, data_path, mode, num_classes, input_fname, output_fname):
        if mode == 'regression':
            input_data_path = data_path + f'/{mode}/{input_fname}'
            output_data_path = data_path + f'/{mode}/{output_fname}'
        elif mode == 'classification':
            input_data_path = data_path + f'/{mode}/{num_classes}/{input_fname}'
            output_data_path = data_path + f'/{mode}/{num_classes}/{output_fname}'
            
        with open(input_data_path, 'r') as file:
            input_data = file.readlines() # Each line in input is a 2D array --> first dimension represents sequence length, second dimension represents embedding
            
        with open(output_data_path, 'r') as file:
            output_data = file.readlines() # Each line in input is a 2D array for regression or a 2D array, 1D array for classification
            
        return input_data, output_data
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        input_seq = eval(self._input_data[idx]) # 2D array --> first dimension represents sequence length, second dimension represents embedding
        output_seq = eval(self._output_data[idx]) # 2D array for regression or a 2D array, 1D array for classification
        
        if self._mode == 'classification': 
            decoder_input_seq = output_seq[0]
            output_seq = output_seq[1]
        
        input_dtype = torch.float
        output_dtype = torch.float if self._mode == 'regression' else torch.long
        input_seq = torch.as_tensor(input_seq, dtype=input_dtype)
        if self._mode == 'classification': decoder_input_seq = torch.as_tensor(decoder_input_seq, dtype=input_dtype)
        output_seq = torch.as_tensor(output_seq, dtype=output_dtype)
        
        # Adding a dummy start embedding (for regression) or index (for classification)
        if len(output_seq.size()) == 1: output_seq = torch.cat((torch.tensor([0], dtype=output_seq.dtype), output_seq))
        elif len(output_seq.size()) == 2: output_seq = torch.cat((torch.zeros((1, output_seq.size(-1)), dtype=output_seq.dtype), output_seq), dim=0)
        if self._mode == 'classification': decoder_input_seq = torch.cat((torch.zeros((1, decoder_input_seq.size(-1)), dtype=decoder_input_seq.dtype), decoder_input_seq), dim=0)
        
        if self._mode == 'regression': return input_seq, output_seq # (seq_len_in, emb_dim), (seq_len_out, emb_dim)
        elif self._mode == 'classification': return input_seq, decoder_input_seq, output_seq
    
def _get_attention_mask(tensor, padding_value):
    assert len(tensor.size()) in [2, 3], f"Expected elements in batch to be 1D or 2D, but found tensor of shape: {tensor.size()}"
    if len(tensor.size()) == 3: # each item in the batch is a 2D tensor
        return 1 - torch.eq(tensor.sum(dim=-1).to(torch.int), int(padding_value)).to(torch.int)
    elif len(tensor.size()) == 2: # each item in the batch is a 1D tensor
        return 1 - torch.eq(tensor, int(padding_value)).to(torch.int)

def _data_collater_fn(batch, mode, padding_value=-100, return_loss=True):
    # batch is a list of tuples, where each element in the tuple is a 2D and/or 1D tensor
    if mode == 'regression': inputs, outputs = tuple(zip(*batch))
    elif mode == 'classification': inputs, dec_inputs, outputs = tuple(zip(*batch))
    inputs = list(inputs)
    outputs = list(outputs)
    if mode == 'classification': dec_inputs = list(dec_inputs)
        
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value) # (bsz, seq_len_in, emb_dim)
    padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=padding_value) # (bsz, seq_len_out, emb_dim) for `regression` else (bsz, seq_len_out)
    if mode == 'classification': dec_inputs = pad_sequence(dec_inputs, batch_first=True, padding_value=padding_value) # (bsz, seq_len_out, emb_dim)
    
    encoder_input_ids = padded_inputs
    encoder_attention_mask = _get_attention_mask(encoder_input_ids, padding_value)
    if mode == 'regression': decoder_input_ids = padded_outputs[:, :-1, :] # (bsz, seq_len_out, emb_dim)
    elif mode == 'classification': decoder_input_ids = dec_inputs[:, :-1, :] # (bsz, seq_len_out, emb_dim)
    decoder_attention_mask = _get_attention_mask(decoder_input_ids, padding_value)
    labels = padded_outputs[:, 1:] # (bsz, seq_len_out, emb_dim) for `regression` else (bsz, seq_len_out)
    
    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "return_loss": return_loss
    }
    
def get_synthetic_data_loader(data_path, mode, input_fname, output_fname, padding_value, return_loss, batch_size, shuffle, num_workers, num_classes):
    dataset = SyntheticDataset(data_path, mode, input_fname, output_fname, num_classes)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=lambda batch: _data_collater_fn(batch, mode, padding_value, return_loss),
                            drop_last=False)
    
    return data_loader