"""Reusable disposable correctness/VRAM gate; never updates or saves model weights."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import transformers
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.grounded_interaction.v1.model import GroundedInteractionSeq2SeqLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--scale', choices=['tiny', 'full'], default='tiny')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--length', type=int, default=16)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--minimum-free-mib', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.microbatch, args.length, args.accumulation) < 2:
        parser.error('Use batch/length/accumulation >=2 to exercise negatives and accumulated gradients')
    if args.output.exists():
        raise FileExistsError('Preserve existing receipt; choose a new output path')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial_free = None
    if args.device == 'cuda':
        initial_free = torch.cuda.mem_get_info()[0] / 2**20
        assert initial_free >= args.minimum_free_mib, ('Insufficient VRAM', initial_free)
        approved = Path('/workspace/ura/thanhdat/projects/CLRR')
        assert Path(sys.prefix).resolve().is_relative_to(approved / 'conda')
        assert ROOT.is_relative_to(approved)
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    full = args.scale == 'full'
    # mBART-large-50 geometry; synthetic random weights and token IDs only.
    config = MBartConfig(vocab_size=250054 if full else 40, d_model=1024 if full else 16,
        encoder_layers=12 if full else 4, decoder_layers=12 if full else 2,
        encoder_attention_heads=16 if full else 2, decoder_attention_heads=16 if full else 2,
        encoder_ffn_dim=4096 if full else 32, decoder_ffn_dim=4096 if full else 32,
        max_position_embeddings=1024 if full else 64, pad_token_id=1, bos_token_id=0,
        eos_token_id=2, decoder_start_token_id=2, dropout=.1, attention_dropout=.0,
        activation_dropout=.0, use_cache=False)
    config._attn_implementation = 'eager'
    core = MBartForConditionalGeneration(config)
    model = GroundedInteractionSeq2SeqLM(core).to(args.device)
    model.train(); model.gradient_checkpointing_enable()
    losses, cells = [], []
    for _ in range(args.accumulation):
        ids = torch.randint(4, config.vocab_size, (args.microbatch, args.length), device=args.device)
        ids[:, -1] = config.eos_token_id
        labels = torch.randint(4, config.vocab_size, ids.shape, device=args.device)
        labels[:, -1] = config.eos_token_id
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        assert torch.isfinite(loss)
        (loss / args.accumulation).backward()
        losses.append(loss.item()); cells.append(dict(model.last_diagnostics))
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            count += p.grad.numel()
    assert core.model.encoder.layers[0].fc1.weight.grad.norm() > 0
    assert core.model.decoder.layers[0].fc1.weight.grad.norm() > 0
    if args.device == 'cuda': torch.cuda.synchronize()
    sources = [Path(__file__), ROOT / 'src/methods/registry.py',
        ROOT / 'src/methods/grounded_interaction/v1/model.py', ROOT / 'src/methods/original/clrr.py']
    receipt = dict(status='PASS', scope='synthetic random-weight forward/backward only',
        optimizer_steps=0, dataset_rows_used=0, training_launched=False, arguments=vars(args),
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__, python=sys.version,
                     executable=sys.executable), device=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
        initial_free_mib=initial_free, checkpointing='whole-view non-reentrant', dtype='float32', tf32=False,
        parameters=sum(p.numel() for p in model.parameters()), gradient_parameters=count,
        loss=losses, cells=cells, seconds=time.time() - started,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if args.device == 'cuda' else None,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('source_sha256', 'cells', 'runtime')}, default=str))


if __name__ == '__main__':
    main()
