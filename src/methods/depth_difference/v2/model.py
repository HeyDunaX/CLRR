"""U2 v2: detached per-example source norm matching for donor V input only."""
import torch
import json
from pathlib import Path
from transformers import MBartConfig, MBartForConditionalGeneration, GenerationConfig
from safetensors.torch import load_model
from methods.depth_difference.v1.model import PackedCrossAttention, DepthDifferencePrototype

METHODS = {
    'depth-donor-addition-v2': 'donor',
    'depth-donor-addition-v2': 'donor',
    'depth-ordinary-v2': 'ordinary',
    'depth-attention-v2': 'attention',
    'depth-value-v2': 'value',
    'depth-additive-v2': 'additive',
    'depth-difference-value-scaled-v2': 'mixed',
}


class ValueScaledCrossAttention(PackedCrossAttention):
    mode = 'mixed'

    def correction(self, top_probability, donor_probability, value_top, value_donor):
        if self.mode == 'donor': return donor_probability @ value_donor
        if self.mode == 'donor': return donor_probability @ value_donor
        da = donor_probability - top_probability
        dv = value_donor - value_top
        if self.mode == 'ordinary':
            return donor_probability @ value_donor - top_probability @ value_top
        if self.mode == 'attention': return da @ value_top
        if self.mode == 'value': return top_probability @ dv
        if self.mode == 'additive': return da @ value_top + top_probability @ dv
        if self.mode == 'mixed': return da @ dv
        raise ValueError(f'Unknown correction: {self.mode}')

    def donor_value_input(self, recipient, donor, attention_mask):
        # Cross-attention additive masks: [batch, 1, query or 1, source].
        # Every query must share source validity; causal masks belong to self-attention.
        if attention_mask is None:
            valid = torch.ones_like(recipient[..., :1], dtype=torch.bool)
        else:
            allowed = attention_mask[:, 0].eq(0)
            if not torch.equal(allowed, allowed[:, :1].expand_as(allowed)):
                raise ValueError('Value scaling requires query-independent source mask')
            valid = allowed[:, 0, :, None]
        with torch.no_grad():
            top_sq = (recipient.float().square() * valid).sum((1, 2))
            donor_sq = (donor.float().square() * valid).sum((1, 2))
            scale = (top_sq / donor_sq.clamp_min(1e-12)).sqrt()
        return donor.detach() * scale.to(donor.dtype)[:, None, None]


class ValueScaledDepthDifference(DepthDifferencePrototype):
    attention_class = ValueScaledCrossAttention
    method_id = 'depth-difference-value-scaled-v2'
    supported_backends = ('eager', 'sdpa')

    def __init__(self, base_model, donor_layer=6, decoder_layer=None, mode='mixed'):
        if mode not in METHODS.values(): raise ValueError('Unknown depth correction mode')
        super().__init__(base_model, donor_layer, decoder_layer)
        self.correction_attention.mode = mode
        self.method_id = next(key for key, value in METHODS.items() if value == mode)

    def save_pretrained(self, folder, state_dict=None, safe_serialization=None, **kwargs):
        if state_dict is None:
            return super().save_pretrained(folder)
        # Trainer's native checkpoint loader reads this full wrapper state strictly.
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(folder)
        self.generation_config.save_pretrained(folder)
        (folder/'prototype_config.json').write_text(json.dumps(dict(method=self.method_id,
            donor_layer=self.donor_layer, decoder_layer=self.decoder_layer,
            attention_implementation=self.config._attn_implementation),indent=2),encoding='utf-8')
        torch.save(state_dict, folder/'pytorch_model.bin')

    @classmethod
    def from_pretrained(cls, folder):
        folder = Path(folder)
        recipe = json.loads((folder/'prototype_config.json').read_text(encoding='utf-8'))
        if recipe['method'] not in METHODS: raise ValueError('Not a U2-v2 checkpoint')
        config = MBartConfig.from_pretrained(folder,local_files_only=True)
        config._attn_implementation = recipe.get('attention_implementation', 'eager')
        model = cls(MBartForConditionalGeneration(config),recipe['donor_layer'],
            recipe['decoder_layer'],mode=METHODS[recipe['method']])
        model.base_model.generation_config = GenerationConfig.from_pretrained(folder,local_files_only=True)
        if (folder/'pytorch_model.bin').exists():
            model.load_state_dict(torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
        else:
            load_model(model,str(folder/'prototype.safetensors'),strict=True)
        return model
