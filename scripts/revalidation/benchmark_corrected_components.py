"""Benchmark computational cost and overhead across 4 corrected families on GPU.
Evaluates Baseline, CLRR-only, LSR-only (lambda 0.10), and Full (alpha 0.10, lambda 0.10) for Seed 42.
Measures:
1. Parameter counts (Total, Trainable, Non-trainable) -> Proves parameter neutrality.
2. Train-step latency (ms/step) with BS=32, Accum=4 (eff BS=128), BF16, Grad Checkpointing.
3. Inference latency (ms/sentence) and throughput (tokens/sec) with greedy beam 1, FP32 generation.
4. Peak VRAM allocation and reservation (MiB).
"""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire import train
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
from amis_rewire.metrics import safe_decode_inputs

DATA_DIR = ROOT / 'data_processed/amis_mandarin'

FAMILIES = {
    'baseline': {'method': 'baseline', 'alpha': 0.0, 'lambda': 0.0},
    'clrr':     {'method': 'clrr-enc', 'alpha': 0.10, 'lambda': 0.0},
    'lsr':      {'method': 'jepa',     'alpha': 0.0,  'lambda': 0.10},
    'full':     {'method': 'jepa-clrr-enc', 'alpha': 0.10, 'lambda': 0.10}
}

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def benchmark_train_step(model, tokenizer, device, train_dataset, num_warmup=3, num_timed=10):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    
    model.train()
    batch_size = 32
    accum_steps = 4
    
    prep = argparse.Namespace(source_prefix='', max_source_length=256, max_target_length=256)
    tokenized = train.tokenize_dataset(train_dataset, tokenizer, prep)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    loader = torch.utils.data.DataLoader(tokenized, batch_size=batch_size, collate_fn=collator, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Warmup steps
    loader_iter = iter(loader)
    for _ in range(num_warmup):
        optimizer.zero_grad()
        for _ in range(accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps
            loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    
    for _ in range(num_timed):
        optimizer.zero_grad()
        for _ in range(accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps
            loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    ms_per_step = (total_time / num_timed) * 1000.0
    
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return ms_per_step, peak_alloc_mb, peak_res_mb

def benchmark_inference(model, tokenizer, device, samples, num_warmup=2):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    
    model.eval()
    batch_size = 8
    
    # Prepare dataloader
    prep = argparse.Namespace(source_prefix='', max_source_length=256, max_target_length=256)
    tokenized = train.tokenize_dataset(samples, tokenizer, prep)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    
    loader = torch.utils.data.DataLoader(tokenized, batch_size=batch_size, collate_fn=collator, shuffle=False)
    
    # Warmup
    warmup_count = 0
    for batch in loader:
        if warmup_count >= num_warmup:
            break
        batch = {k: v.to(device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            _ = model.generate(**batch, max_new_tokens=64, num_beams=1)
        warmup_count += 1
        
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    
    total_generated_tokens = 0
    total_sentences = 0
    
    for batch in loader:
        b_size = batch['input_ids'].size(0)
        total_sentences += b_size
        batch = {k: v.to(device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=64, num_beams=1)
        total_generated_tokens += gen.numel()
        
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    ms_per_sentence = (total_time / total_sentences) * 1000.0
    tokens_per_sec = total_generated_tokens / total_time
    
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return ms_per_sentence, tokens_per_sec, peak_alloc_mb, peak_res_mb

def run_benchmark(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    revision = "e30b6cb8eb0d43a0b73cab73c7676b9863223a30"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    
    # Load datasets for benchmarks
    train_dataset = train.load_split(DATA_DIR, 'train').select(range(128))
    val_dataset = train.load_split(DATA_DIR, 'validation').select(range(64))
    
    results = []
    
    for fam_name, cfg in FAMILIES.items():
        print(f"\n>>> Benchmarking family: {fam_name} (method={cfg['method']}, alpha={cfg['alpha']}, lambda={cfg['lambda']})")
        model = train.load_model(
            model_name, cfg['method'],
            distance=2, strength=cfg['alpha'], stack='encoder',
            jepa_weight=cfg['lambda'], revision=revision
        )
        train.configure_mbart(tokenizer, model, 'tl_XX', 'zh_CN')
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.to(device)
        
        total_p, train_p = count_parameters(model)
        print(f"Parameters: Total = {total_p:,}, Trainable = {train_p:,}")
        
        # Train-step benchmark
        print("Running train-step benchmark (BS=32, Accum=4, BF16, GC)...")
        train_ms, train_alloc_mb, train_res_mb = benchmark_train_step(model, tokenizer, device, train_dataset)
        print(f"Train Step: {train_ms:.2f} ms/step | Peak VRAM: {train_alloc_mb:.1f} MiB alloc, {train_res_mb:.1f} MiB res")
        
        # Inference benchmark
        print("Running inference benchmark (Greedy beam 1, FP32, BS=8)...")
        inf_ms, inf_tps, inf_alloc_mb, inf_res_mb = benchmark_inference(model, tokenizer, device, val_dataset)
        print(f"Inference: {inf_ms:.2f} ms/sentence | {inf_tps:.1f} tokens/s | Peak VRAM: {inf_alloc_mb:.1f} MiB alloc")
        
        results.append({
            'family': fam_name,
            'method': cfg['method'],
            'alpha': cfg['alpha'],
            'lambda': cfg['lambda'],
            'total_parameters': total_p,
            'trainable_parameters': train_p,
            'param_overhead_vs_baseline': total_p - results[0]['total_parameters'] if results else 0,
            'train_step_ms': train_ms,
            'train_overhead_pct': ((train_ms - results[0]['train_step_ms']) / results[0]['train_step_ms'] * 100.0) if results else 0.0,
            'train_peak_alloc_mib': train_alloc_mb,
            'train_peak_res_mib': train_res_mb,
            'inference_ms_per_sentence': inf_ms,
            'inference_overhead_pct': ((inf_ms - results[0]['inference_ms_per_sentence']) / results[0]['inference_ms_per_sentence'] * 100.0) if results else 0.0,
            'inference_tokens_per_sec': inf_tps,
            'inference_peak_alloc_mib': inf_alloc_mb
        })
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    df = pd.DataFrame(results)
    df.to_csv(out / 'cost_benchmark_summary.csv', index=False)
    (out / 'cost_benchmark_results.json').write_text(json.dumps(results, indent=2))
    
    # Markdown summary
    md = [
        "# Benchmark Chi Phí & Overhead (Cost & Overhead Benchmark)",
        "",
        f"**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | **Precision:** BF16 Training + GC / FP32 Inference",
        "**Effective Batch Size:** 32 per-device x 4 accumulation = 128 (Length: 256/256)",
        "",
        "| Family | Total Params | Trainable Params | Train Step (ms) | Train Overhead vs FT | Inf Latency (ms/sent) | Inf Overhead vs FT | Throughput (tok/s) | Peak VRAM Train (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in results:
        md.append(f"| **{r['family']}** | {r['total_parameters']:,} | {r['trainable_parameters']:,} | {r['train_step_ms']:.2f} | {r['train_overhead_pct']:+.2f}% | {r['inference_ms_per_sentence']:.2f} | {r['inference_overhead_pct']:+.2f}% | {r['inference_tokens_per_sec']:.1f} | {r['train_peak_alloc_mib']:.1f} |")
        
    (out / 'BENCHMARK_REPORT.md').write_text('\n'.join(md) + '\n')
    print("\nBenchmark completed successfully! Artifacts written to:", out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='outputs_rebuttal/amis_cost_benchmark_20261005')
    args = p.parse_args()
    run_benchmark(args.output_dir)
