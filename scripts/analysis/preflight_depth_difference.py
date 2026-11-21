"""Forward/backward U2 preflight only: no optimizer, downloads or training."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import transformers
from transformers import AutoTokenizer, MBartConfig, MBartForConditionalGeneration
from methods.depth_difference.v1 import DepthDifferencePrototype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='tiny')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--src-lang', default='tl_XX')
    parser.add_argument('--tgt-lang', default='zh_CN')
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--donor-layer', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    lock = None
    free_mib = None
    if args.device == 'cuda':
        import fcntl
        if not Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda'):
            raise RuntimeError('RTX must use the conda environment inside this project')
        lock = (ROOT / 'tmp/training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free, _ = torch.cuda.mem_get_info()
        free_mib = free // 2**20
        if free_mib < args.minimum_free_gpu_mib:
            raise RuntimeError(f'Only {free_mib} MiB free; leave other users untouched')
        torch.cuda.reset_peak_memory_stats()
    if args.model == 'tiny':
        config = MBartConfig(vocab_size=43, d_model=16, encoder_layers=4,
            decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
            encoder_layerdrop=0., decoder_layerdrop=0., pad_token_id=1,
            bos_token_id=0, eos_token_id=2, decoder_start_token_id=2)
        config._attn_implementation = 'eager'
        base = MBartForConditionalGeneration(config)
        batch = dict(input_ids=torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]]),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            labels=torch.tensor([[0, 10, 11, 2], [0, 12, 13, 2]]))
    else:
        if args.data is None or args.cache_dir is None:
            parser.error('Cached pretrained preflight requires --data and --cache-dir')
        base = MBartForConditionalGeneration.from_pretrained(
            args.model, cache_dir=str(args.cache_dir), local_files_only=True,
            attn_implementation='eager', torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(args.cache_dir),
            local_files_only=True, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        with args.data.open(encoding='utf-8-sig', newline='') as stream:
            rows = []
            for row in csv.DictReader(stream):
                rows.append(row)
                if len(rows) == args.rows:
                    break
        if len(rows) != args.rows:
            raise ValueError('Not enough rows in supplied training CSV')
        encoded = tokenizer([r['source'] for r in rows], text_target=[r['target'] for r in rows],
            padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
        batch = {key: encoded[key] for key in ('input_ids', 'attention_mask', 'labels')}
        batch['labels'] = batch['labels'].masked_fill(batch['labels'] == tokenizer.pad_token_id, -100)
    base = base.to(args.device).eval()
    batch = {key: value.to(args.device) for key, value in batch.items()}
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask')}
    parameters_before = sum(p.numel() for p in base.parameters())
    with torch.no_grad():
        reference = base(**batch)
        reference_logits = reference.logits.detach().cpu()
        reference_loss = reference.loss.item()
        del reference
        reference_generated = base.generate(**inputs, max_new_tokens=8, num_beams=1, do_sample=False)
    model = DepthDifferencePrototype(base, args.donor_layer).eval()
    model.correction_attention.collect_diagnostics = True
    result = model(**batch)
    logits_error = (result.logits.detach().cpu() - reference_logits).abs().max().item()
    torch.testing.assert_close(result.logits.detach().cpu(), reference_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.loss.detach().cpu(), torch.tensor(reference_loss), atol=1e-5, rtol=1e-5)
    result.loss.backward()
    gate_gradient = model.correction_attention.gates.grad.detach().cpu()
    assert torch.isfinite(gate_gradient).all() and gate_gradient.norm() > 0
    diagnostic = dict(model.correction_attention.last_diagnostics)
    assert diagnostic['correction_norm'] > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    encoder_gradient = model.base_model.model.encoder.layers[0].fc1.weight.grad.norm().item()
    assert encoder_gradient > 0
    del result
    model.zero_grad(set_to_none=True)
    greedy = model.generate(**inputs, max_new_tokens=8, num_beams=1, do_sample=False)
    assert torch.equal(reference_generated, greedy)
    # A synthetic nonzero gate exercises donor K/V caching; it is never optimized.
    model.correction_attention.gates.data.fill_(.1)
    cached = model.generate(**inputs, max_new_tokens=8, num_beams=4, do_sample=False, use_cache=True)
    uncached = model.generate(**inputs, max_new_tokens=8, num_beams=4, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)
    model.correction_attention.gates.data.zero_()
    report = dict(status='PASS', scope='preflight, not translation-quality evaluation',
        model=args.model, model_revision=getattr(model.config, '_commit_hash', None),
        dtype='float32', device=args.device, torch=torch.__version__,
        transformers=transformers.__version__, python_prefix=sys.prefix,
        source_shape=list(batch['input_ids'].shape), target_shape=list(batch['labels'].shape),
        source_lang=args.src_lang, target_lang=args.tgt_lang, max_length=args.max_length,
        training_csv_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
        donor_layer=args.donor_layer, decoder_layer=model.decoder_layer,
        extra_parameters=sum(p.numel() for p in model.parameters()) - parameters_before,
        zero_gate_loss=reference_loss, zero_gate_logits_max_abs_error=logits_error,
        zero_gate_greedy_matches=True, nonzero_gate_beam4_cache_matches=True,
        gate_gradient=gate_gradient.tolist(), gate_gradient_norm=gate_gradient.norm().item(),
        first_encoder_fc1_gradient_norm=encoder_gradient,
        ungated_correction_to_base_norm=diagnostic['correction_norm'] / diagnostic['base_context_norm'],
        context_diagnostics=diagnostic, optimizer_steps=0, training_runs=0,
        free_gpu_mib_before=free_mib,
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None,
        elapsed_seconds=time.time() - started,
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method_sha256=hashlib.sha256((ROOT/'src/methods/depth_difference/v1/model.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
