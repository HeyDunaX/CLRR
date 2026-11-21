"""PLC and its frozen mathematical controls; native two-view inference wrapper."""
import json
from pathlib import Path
from transformers.modeling_outputs import Seq2SeqLMOutput
from ...paired_route.v1.model import PairedRouteMT
from .losses import prefix_loss

METHODS = {'prefix-consistency-v1': 'prefix',
           'diagonal-consistency-v1': 'diagonal',
           'endpoint-consistency-v1': 'endpoint',
           'calibrated-consistency-v1': 'disabled'}


class PrefixConsistencyMT(PairedRouteMT):
    def __init__(self, base_model, method, weight=.10, beta=1.25):
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method, self.mode = method, METHODS[method]
        self.path_weight, self.beta = float(weight), float(beta)
        if self.mode == 'disabled' and self.path_weight != 0.:
            raise ValueError('Calibrated W control must disable path loss')

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, decoder_input_ids=decoder_input_ids, **kwargs)
        decoder_input_ids = (self.prepare_decoder_input_ids_from_labels(labels)
                             if decoder_input_ids is None else decoder_input_ids)
        logits = []
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits)
        loss, ce, kl, auxiliary, _ = prefix_loss(*logits, labels,
            mode=self.mode, weight=self.path_weight, beta=self.beta)
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float(kl.detach()),
            path=float(auxiliary.detach()), beta=self.beta, path_weight=self.path_weight,
            mode=self.mode, views=2, second_view_rewired=False,
            checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(version=1, method=self.method, mode=self.mode,
            beta=self.beta, path_weight=self.path_weight,
            ce_kl='unchanged W valid-token microbatch means',
            path_reduction='mean sentences; packed supervised prefixes',
            masks='labels != -100 including supervised special tokens',
            arithmetic='FP32 log_softmax and cumsum', inference_route='native',
            extra_parameters=0, views=2, stop_gradient=False,
            attribution='R-Drop backbone objective; exploratory ordered gold-score consistency')

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'prefix_consistency_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'prefix_consistency_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Prefix consistency checkpoint recipe mismatch')
