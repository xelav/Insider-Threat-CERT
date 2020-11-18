class HackTokenizer(PreTrainedTokenizer):
    
    def __init__(self, vocab_size, pad_token=0, mask_token=-1):
        
        self._pad_token = pad_token
        self.pad_token_id = pad_token
        
        self.mask_token = mask_token
        
        self.vocab_size = vocab_size
        
    def __len__(self):
        
        return self.vocab_size