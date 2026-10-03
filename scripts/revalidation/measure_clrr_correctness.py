"""Offline CPU checks of actual CLRR/LSR wrappers; no trained-model claims."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import MBartConfig, MBartForConditionalGeneration, M2M100Config, M2M100ForConditionalGeneration
from transformers import T5Config, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM

OUT = ROOT / "outputs_rebuttal/easy_diagnostics_20261002"


def wrap(base, method, alpha=0.1, weight=0.1):
    if method in ("clrr", "full"):
        base = CrossLayerResidualRewire(base, distance=2, strength=alpha)
    if method in ("lsr", "full"):
        base = JEPAGuidedSeq2SeqLM(base, jepa_weight=weight)
    return base


def gradient_run(base, batch, method, checkpoint=None, alpha=0.1, weight=0.1):
    model = wrap(copy.deepcopy(base), method, alpha, weight)
    model.train()
    if checkpoint is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
    torch.manual_seed(123)
    outputs = model(**batch)
    outputs.loss.backward()
    gradients = {
        name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in model.named_parameters()
    }
    # Names gain wrapper prefixes; normalize by using ordered parameters.
    vector = torch.cat([g.flatten() for g in gradients.values()])
    return {"loss": float(outputs.loss.detach()), "logits": outputs.logits.detach(),
            "gradient": vector, "parameters": sum(p.numel() for p in model.parameters())}


def compare(a, b):
    diff = b["gradient"] - a["gradient"]
    return {"loss_abs_difference": abs(a["loss"] - b["loss"]),
            "logits_max_abs_difference": float((a["logits"] - b["logits"]).abs().max()),
            "gradient_max_abs_difference": float(diff.abs().max()),
            "gradient_relative_l2_difference": float(diff.norm() / a["gradient"].norm().clamp_min(1e-12)),
            "same_parameter_count": a["parameters"] == b["parameters"]}


def routing_check():
    # Invoke the production hooks on independent leaf tensors to isolate detach.
    class Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    wrapper = CrossLayerResidualRewire(Owner(), distance=2, strength=0.1)
    leaves = [torch.tensor([[[float(i + 1)]]], requires_grad=True) for i in range(4)]
    with wrapper._fresh_cache():
        values = [layer(x) for layer, x in zip(wrapper.base_model.encoder.layers, leaves)]
        source_grad, destination_grad = torch.autograd.grad(values[2].sum(), [leaves[0], leaves[2]], allow_unused=True)
        detached = all(not x.requires_grad for x in wrapper._cache["encoder"].values())
    actual = [float(x.detach().item()) for x in values]
    return {"expected_layer_values": [1.0, 2.0, 3.1, 4.2], "actual_layer_values": actual,
            "distance_two_routing_pass": all(abs(a-b) < 1e-6 for a,b in zip(actual, [1,2,3.1,4.2])),
            "skip_source_gradient_is_none": source_grad is None,
            "destination_gradient": float(destination_grad.item()), "cached_tensors_detached": detached,
            "cache_empty_after_forward": all(not x for x in wrapper._cache.values())}


def cache_and_lsr_check(base, batch):
    rewired = wrap(copy.deepcopy(base), "clrr").eval()
    with torch.no_grad():
        first = rewired(**batch).logits
        other = {k: v.flip(0) for k,v in batch.items()}
        rewired(**other)
        again = rewired(**batch).logits
        generated_first = rewired.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                           num_beams=2, max_new_tokens=4)
        rewired.generate(input_ids=other["input_ids"], attention_mask=other["attention_mask"],
                         num_beams=2, max_new_tokens=4)
        generated_again = rewired.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                           num_beams=2, max_new_tokens=4)
    cache_empty = all(not c for c in rewired._cache.values())
    full = wrap(copy.deepcopy(base), "full").train()
    captures = []
    def capture(_module, _inputs, output):
        captures.append(output.last_hidden_state)
    handle = full.get_encoder().register_forward_hook(capture)
    full_output = full(**batch)
    handle.remove()
    # Recover CE from the very same logits, avoiding a second dropout draw.
    ce = torch.nn.functional.cross_entropy(full_output.logits.reshape(-1, full.config.vocab_size),
                                           batch["labels"].reshape(-1), ignore_index=-100)
    weighted_aux = full_output.loss - ce
    source_grad = torch.autograd.grad(weighted_aux, captures[0], retain_graph=True)[0]
    encoder_parameters = list(full.get_encoder().parameters())
    aux_grads = torch.autograd.grad(weighted_aux, encoder_parameters, retain_graph=True, allow_unused=True)
    ce_grads = torch.autograd.grad(ce, encoder_parameters, allow_unused=True)
    def flatten(grads, params):
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(grads,params)])
    aux_vec, ce_vec = flatten(aux_grads,encoder_parameters), flatten(ce_grads,encoder_parameters)
    return {"intervening_batch_logits_max_abs_difference": float((first-again).abs().max()),
            "beam_generation_repeat_identical": bool(torch.equal(generated_first,generated_again)),
            "cache_empty_after_calls": cache_empty, "encoder_calls_source_and_target": len(captures),
            "source_hidden_requires_grad": captures[0].requires_grad,
            "target_hidden_requires_grad": captures[1].requires_grad,
            "source_hidden_weighted_lsr_gradient_norm": float(source_grad.norm()),
            "ce": float(ce.detach()), "weighted_lsr": float(weighted_aux.detach()),
            "encoder_ce_gradient_norm": float(ce_vec.norm()),
            "encoder_weighted_lsr_gradient_norm": float(aux_vec.norm()),
            "encoder_lsr_to_ce_gradient_norm_ratio": float(aux_vec.norm()/ce_vec.norm().clamp_min(1e-12)),
            "encoder_gradient_cosine": float(torch.nn.functional.cosine_similarity(aux_vec,ce_vec,dim=0)),
            "scope": "Random tiny model only; these norms do not diagnose trained checkpoints."}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    batch = {"input_ids": torch.tensor([[4,5,6,2,1],[7,8,9,10,2]]),
             "attention_mask": torch.tensor([[1,1,1,1,0],[1,1,1,1,1]]),
             "labels": torch.tensor([[11,12,2,-100],[13,14,15,2]])}
    kwargs = dict(vocab_size=48, d_model=32, encoder_layers=4, decoder_layers=2,
                  encoder_attention_heads=4, decoder_attention_heads=4,
                  encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=32,
                  pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                  dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
                  encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=False)
    results = {"protocol": {"torch": torch.__version__, "transformers": transformers.__version__,
                "device": "cpu", "dtype": "float32", "seed": 42, "dropout": 0,
                "trained_weights": False, "optimizer_steps": 0,
                "source_sha256": hashlib.sha256((ROOT/"src/amis_rewire/modeling.py").read_bytes()).hexdigest()},
                "routing": routing_check(), "backbones": {}}
    for family, config_class, model_class in [
        ("mBART", MBartConfig, MBartForConditionalGeneration),
        ("NLLB_architecture_M2M100", M2M100Config, M2M100ForConditionalGeneration),
        ("mT5", MT5Config, MT5ForConditionalGeneration),
        ("ByT5_architecture_T5", T5Config, T5ForConditionalGeneration)]:
        torch.manual_seed(42)
        if family in ("mT5","ByT5_architecture_T5"):
            config = config_class(vocab_size=48,d_model=32,d_kv=8,d_ff=64,num_layers=4,num_decoder_layers=2,
                                  num_heads=4,dropout_rate=0.0,pad_token_id=1,eos_token_id=2,
                                  decoder_start_token_id=1,use_cache=False)
        else:
            config = config_class(**kwargs)
        base = model_class(config)
        baseline = gradient_run(base,batch,"baseline")
        zero = gradient_run(base,batch,"full",alpha=0,weight=0)
        family_results = {"zero_strength_zero_weight_vs_baseline": compare(baseline,zero),
                          "cache_and_lsr": cache_and_lsr_check(base,batch), "checkpointing": []}
        for method in ("baseline","clrr","lsr","full"):
            reference = gradient_run(base,batch,method)
            for reentrant in (True,False):
                try:
                    candidate = gradient_run(base,batch,method,checkpoint=reentrant)
                    entry = {"method":method,"use_reentrant":reentrant, **compare(reference,candidate)}
                    entry["pass"] = entry["gradient_relative_l2_difference"] < 1e-5 and entry["logits_max_abs_difference"] < 1e-6
                except Exception as exc:
                    entry = {"method":method,"use_reentrant":reentrant,"pass":False,
                             "exception": type(exc).__name__ + ": " + str(exc)}
                family_results["checkpointing"].append(entry)
                print(f"[checkpoint] {family} {method} reentrant={reentrant}: {entry}",flush=True)
        # Actual training uses dropout: checkpoint recomputation must replay RNG.
        if family in ("mT5","ByT5_architecture_T5"):
            config.dropout_rate = 0.1
        else:
            config.dropout = 0.1
            config.attention_dropout = 0.1
            config.activation_dropout = 0.1
        torch.manual_seed(42)
        dropout_base = model_class(config)
        dropout_reference = gradient_run(dropout_base,batch,"full")
        family_results["checkpointing_with_dropout"] = []
        for reentrant in (True,False):
            try:
                candidate = gradient_run(dropout_base,batch,"full",checkpoint=reentrant)
                entry = {"method":"full","dropout":0.1,"use_reentrant":reentrant,**compare(dropout_reference,candidate)}
                entry["pass"] = entry["gradient_relative_l2_difference"] < 1e-5 and entry["logits_max_abs_difference"] < 1e-6
            except Exception as exc:
                entry = {"method":"full","dropout":0.1,"use_reentrant":reentrant,"pass":False,
                         "exception":type(exc).__name__+": "+str(exc)}
            family_results["checkpointing_with_dropout"].append(entry)
            print(f"[dropout checkpoint] {family}: {entry}",flush=True)
        results["backbones"][family] = family_results
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"correctness.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("[saved] " + str(OUT/"correctness.json"),flush=True)


if __name__ == "__main__":
    main()
