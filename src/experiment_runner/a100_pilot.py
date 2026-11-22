"""One-command resumable A100 pilot runner: build -> CPU gate -> GPU gates -> Step-0 -> train -> summary."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT, build, read_config, verify_frozen, workspace
from .integrity import sha256


REUSED_A100_COMPARATORS = {
    'BASELINE_S42': {'bleu': 17.5264, 'chrf0': 16.7632, 'chrf++': 14.5382, 'archive_sha256': '985f9fc4'},
    'W_B125_S42': {'bleu': 17.7069, 'chrf0': 17.0817, 'chrf++': 14.8852, 'archive_sha256': '4df107c0'},
    'GOC_SYM_B250_S42': {'bleu': 18.5143, 'chrf0': 17.5405, 'chrf++': 15.3266, 'archive_sha256': '47f6f925'},
}


def _ensure_hf_token(need_token: bool) -> dict:
    env = os.environ.copy()
    if env.get('HF_TOKEN'):
        return env
    for candidate in (
        ROOT / '.cache/cr_dual_credentials/token',
        ROOT / '.cache/goc_pilot_credentials/token',
    ):
        if candidate.is_file():
            token = candidate.read_text(encoding='utf-8').strip()
            if token:
                env['HF_TOKEN'] = token
                return env
    if need_token:
        raise RuntimeError(
            'HF_TOKEN is required for verified checkpoint backup during training. '
            'Run: export HF_TOKEN="hf_..." (or pass --gates-only to run gates without training).'
        )
    return env


def _latest_resumable_checkpoint(run_dir: Path):
    ckpts = []
    for folder in run_dir.glob('checkpoint-*'):
        if not folder.is_dir():
            continue
        if all((folder / n).exists() for n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json')):
            try:
                step = int(folder.name.split('-')[-1])
            except ValueError:
                continue
            ckpts.append((step, folder))
    if not ckpts:
        return None
    ckpts.sort(key=lambda pair: pair[0])
    return ckpts[-1][1]


def _run_subcommand(args_list, env):
    cmd = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'scripts/experiment.py'), *args_list]
    print('>>>', ' '.join(args_list), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _write_cr_dual_summary(out: Path, plan: dict):
    rows = []
    scores_by_id = {}
    for spec in plan['runs']:
        ident = spec['id']
        val_path = out / 'runs' / ident / 'VAL_COMPLETE.json'
        if not val_path.exists():
            continue
        val = json.loads(val_path.read_text(encoding='utf-8'))
        sc = val.get('scores', {})
        scores_by_id[ident] = sc
        rows.append(dict(
            id=ident,
            method=spec['method'],
            bleu=round(float(sc.get('bleu', 0.0)), 4),
            chrf0=round(float(sc.get('chrf0', 0.0)), 4),
            chrfpp=round(float(sc.get('chrf++', 0.0)), 4),
            archive_sha256=val.get('archive_sha256', '')[:12],
        ))

    gates = {}
    verdict = 'INCOMPLETE'
    if all(k in scores_by_id for k in ('BIT_CE_S42', 'CR_DUAL_S42', 'CR_SHUFFLED_S42', 'UNI_CONTRAST_S42')):
        b_bit = scores_by_id['BIT_CE_S42']['bleu']
        c_bit = scores_by_id['BIT_CE_S42']['chrf0']
        b_cr = scores_by_id['CR_DUAL_S42']['bleu']
        c_cr = scores_by_id['CR_DUAL_S42']['chrf0']
        b_sh = scores_by_id['CR_SHUFFLED_S42']['bleu']
        b_un = scores_by_id['UNI_CONTRAST_S42']['bleu']
        b_sym = REUSED_A100_COMPARATORS['GOC_SYM_B250_S42']['bleu']

        gates = {
            'gate1_vs_bit_ce_ge_050': (b_cr - b_bit) >= 0.50 and c_cr > c_bit,
            'gate2_vs_shuffled_ge_030': (b_cr - b_sh) >= 0.30,
            'gate3_vs_uni_contrast_ge_040': (b_cr - b_un) >= 0.40,
            'gate4_full_go_ge_sym_sg': b_cr >= b_sym,
            'gate4_orthogonality_go_ge_1825': b_cr >= 18.25,
            'deltas': {
                'cr_minus_bit_bleu': round(b_cr - b_bit, 4),
                'cr_minus_shuffled_bleu': round(b_cr - b_sh, 4),
                'cr_minus_uni_contrast_bleu': round(b_cr - b_un, 4),
                'cr_minus_baseline_bleu': round(b_cr - REUSED_A100_COMPARATORS['BASELINE_S42']['bleu'], 4),
                'cr_minus_sym_sg_bleu': round(b_cr - b_sym, 4),
            },
        }
        mech_pass = (
            gates['gate1_vs_bit_ce_ge_050']
            and gates['gate2_vs_shuffled_ge_030']
            and gates['gate3_vs_uni_contrast_ge_040']
        )
        if mech_pass and gates['gate4_full_go_ge_sym_sg']:
            verdict = 'FULL_GO'
        elif mech_pass and gates['gate4_orthogonality_go_ge_1825']:
            verdict = 'CONDITIONAL_ORTHOGONALITY_GO'
        else:
            verdict = 'NO_GO'

    step0_path = out / 'step0_diagnostic.json'
    step0 = json.loads(step0_path.read_text(encoding='utf-8')) if step0_path.exists() else None
    summary = dict(
        status='COMPLETE' if len(rows) == len(plan['runs']) else 'PARTIAL',
        verdict=verdict,
        updated_unix=time.time(),
        reused_a100_comparators=REUSED_A100_COMPARATORS,
        new_runs=rows,
        gates=gates,
        step0_diagnostic=step0,
    )
    (out / 'PILOT_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    md = [
        f'# Pilot Summary: `{out.name}`',
        '',
        f'- **Verdict:** `{verdict}`',
        '- **Protocol:** Locked A100 BF16 `B32xGA4` `eval_bleu` (`beam 1` FP32 `zh`), validation-only (no test split).',
        '',
        '## Reused A100 Comparators (Same Protocol)',
        '| Run | BLEU | chrF0 | chrF++ | SHA prefix |',
        '|---|---:|---:|---:|---|',
    ]
    for name, rec in REUSED_A100_COMPARATORS.items():
        md.append(f"| `{name}` | {rec['bleu']:.4f} | {rec['chrf0']:.4f} | {rec['chrf++']:.4f} | `{rec['archive_sha256']}` |")
    md += [
        '',
        '## New Pilot Runs',
        '| Run | Method | BLEU | chrF0 | chrF++ | SHA prefix |',
        '|---|---|---:|---:|---:|---|',
    ]
    for r in rows:
        md.append(f"| `{r['id']}` | `{r['method']}` | {r['bleu']:.4f} | {r['chrf0']:.4f} | {r['chrfpp']:.4f} | `{r['archive_sha256']}` |")
    if gates:
        md += ['', '## Pre-Locked GO/NO-GO Gates', '```json', json.dumps(gates, indent=2), '```']
    (out / 'PILOT_SUMMARY.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return summary


def run_a100_pilot(config_path, gates_only=False, step0_only=False, skip_step0=False):
    path, cfg = read_config(config_path)
    out = workspace(path)
    env = _ensure_hf_token(need_token=not gates_only)

    # 1) Build or verify frozen bundle
    if not (out / 'PLAN.json').exists():
        print('=== [1/5] BUILD FROZEN WORKSPACE ===', flush=True)
        build(path)
    out, plan = verify_frozen(path)
    bundle_sha = sha256(out / 'bundle_manifest.json')

    # 2) CPU unit tests + CPU gate
    cpu_path = out / 'CPU_GATE.json'
    cpu_ok = False
    if cpu_path.exists():
        rec = json.loads(cpu_path.read_text(encoding='utf-8'))
        cpu_ok = bool(rec.get('passed') and rec.get('bundle_sha256') == bundle_sha)
    if not cpu_ok:
        print('=== [2/5] CPU UNIT TESTS & CPU GATE ===', flush=True)
        _run_subcommand(['cpu', '--config', str(path)], env)
    else:
        print('=== [2/5] CPU GATE ALREADY PASSED (skipping) ===', flush=True)

    # 3) All GPU 2-step gates
    print('=== [3/5] GPU 2-STEP GATES (ALL RUNS) ===', flush=True)
    for spec in plan['runs']:
        ident = spec['id']
        gate_path = out / 'gates' / ident / 'PASS.json'
        if gate_path.exists():
            rec = json.loads(gate_path.read_text(encoding='utf-8'))
            if rec.get('passed') and rec.get('bundle_sha256') == bundle_sha:
                print(f'  [SKIP] GPU gate already PASSED: {ident}', flush=True)
                continue
        _run_subcommand(['run', '--config', str(path), '--id', ident, '--stage', 'gate'], env)
    _run_subcommand(['watch', '--config', str(path)], env)

    if gates_only:
        return {'status': 'ALL_GATES_PASSED', 'workspace': str(out)}

    # 4) Train runs in locked order, inserting Step-0 Diagnostic right after BIT_CE_S42
    print('=== [4/5] VALIDATION-ONLY A100 TRAINING ===', flush=True)
    from methods.cross_ratio_dual import METHODS as CR_DUAL_METHODS
    is_cr_dual_pilot = any(r['method'] in CR_DUAL_METHODS for r in plan['runs'])

    for spec in plan['runs']:
        ident = spec['id']
        run_dir = out / 'runs' / ident
        val_path = run_dir / 'VAL_COMPLETE.json'
        if val_path.exists():
            print(f'  [SKIP] Already validated: {ident}', flush=True)
        else:
            cmd = ['run', '--config', str(path), '--id', ident, '--stage', 'train']
            resume_ckpt = _latest_resumable_checkpoint(run_dir)
            if resume_ckpt is not None:
                print(f'  [RESUME] Resuming {ident} from {resume_ckpt.name}', flush=True)
                cmd += ['--resume', str(resume_ckpt)]
            _run_subcommand(cmd, env)
        _run_subcommand(['watch', '--config', str(path)], env)

        if is_cr_dual_pilot and ident == 'BIT_CE_S42' and not skip_step0:
            step0_path = out / 'step0_diagnostic.json'
            if not step0_path.exists():
                print('=== [STEP-0] ZERO-OPTIMIZER-STEP DIAGNOSTIC ON BIT_CE_S42 ===', flush=True)
                _run_subcommand(['cr-dual-diagnose', '--config', str(path)], env)
            if step0_only:
                summary = _write_cr_dual_summary(out, plan)
                return {'status': 'STEP0_COMPLETE', 'workspace': str(out), 'summary': summary}

    # 5) Write final summary + GO/NO-GO gates
    print('=== [5/5] FINAL PILOT SUMMARY & GO/NO-GO GATES ===', flush=True)
    summary = _write_cr_dual_summary(out, plan)
    return summary
