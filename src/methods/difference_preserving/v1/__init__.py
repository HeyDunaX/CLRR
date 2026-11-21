from .loss import difference_alignment_loss
from .model import DifferencePreservingSeq2SeqLM

METHODS = {
    'difference-generic-v1': ('generic', False),
    'difference-candidate-v1': ('candidate', False),
    'difference-clrr-generic-v1': ('generic', True),
    'difference-clrr-candidate-v1': ('candidate', True),
}

__all__ = ['difference_alignment_loss', 'DifferencePreservingSeq2SeqLM', 'METHODS']
