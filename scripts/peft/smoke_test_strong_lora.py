"""Check Strong LoRA gradients, shared embeddings, and checkpoint reload on CPU."""
from pathlib import Path
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import MBartConfig, MBartForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from peft_baselines.lora_adapter import apply_lora_to_model


def check(unfreeze: bool) -> None:
    torch.manual_seed(42)
    config = MBartConfig(
        vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        forced_bos_token_id=4,
    )
    base = MBartForConditionalGeneration(config)
    initial_state = {name: value.clone() for name, value in base.state_dict().items()}
    model, _ = apply_lora_to_model(
        base, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["shared", "embed_tokens", "lm_head"] if unfreeze else None,
    )
    ids = torch.randint(5, 64, (2, 8))
    mask = torch.ones_like(ids)
    embedding = model.get_base_model().model.encoder.embed_tokens.weight
    before = embedding.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(input_ids=ids, attention_mask=mask, labels=ids).loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if unfreeze:
        core = model.get_base_model()
        assert embedding is core.model.shared.weight
        assert embedding is core.model.decoder.embed_tokens.weight
        assert embedding is core.lm_head.weight
        assert embedding.grad is not None and embedding.grad.abs().sum() > 0
    else:
        assert not embedding.requires_grad and embedding.grad is None
    optimizer.step()
    assert (not torch.equal(before, embedding)) == unfreeze
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, labels=ids).logits
        expected_tokens = model.generate(input_ids=ids, attention_mask=mask, max_length=10)
    with tempfile.TemporaryDirectory() as folder:
        model.save_pretrained(folder, safe_serialization=False)
        restored_base = MBartForConditionalGeneration(config)
        restored_base.load_state_dict(initial_state)
        restored = PeftModel.from_pretrained(restored_base, folder)
        restored.eval()
        with torch.no_grad():
            actual = restored(input_ids=ids, attention_mask=mask, labels=ids).logits
            actual_tokens = restored.generate(input_ids=ids, attention_mask=mask, max_length=10)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual_tokens, expected_tokens)
    print(f"PASS: unfreeze_embeddings={unfreeze}, gradients/update/reload/generation", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    check(False)
    check(True)
