import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

    
class FalsePositiveRate(Metric):

    def __init__(self, ignored_class, output_transform=lambda x: x, device="cpu"):
        self._false_positive_num = None
        self._num_examples = None
        super(FalsePositiveRate, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._false_positive_num = 0
        self._num_examples = 0
        super(FalsePositiveRate, self).reset()
        
    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y_pred, y = y_pred.numpy(), y.numpy()
        
        self._false_positive_num += np.logical_and(y_pred != y, y_pred == 1).sum()
        self._num_examples += (y == 0).sum()

    @sync_all_reduce("_num_examples", "_false_positive_num")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('FalsePositiveRate must have at least one example before it can be computed.')
        return self._false_positive_num/ self._num_examples


class AccuracyIgnoringPadding(Accuracy):
    """
    Same accuracy metric except that it
    ignores single class in calculations.
    This is neccessary so that the metric
    is not overly optimistic
    """

    def __init__(self, ignored_class, *args, **kwargs):
        self.ignored_class = ignored_class
        super(Accuracy, self).__init__(*args, **kwargs)

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)

        mask = (y != self.ignored_class)
        mask &= (indices != self.ignored_class)
        y = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]
    

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len = 80):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = max_seq_len
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))
                
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, input):
        
        return F.embedding(input, self.pe)
