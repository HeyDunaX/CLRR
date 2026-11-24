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


def _compute_length_stats(pred_csv: Path, bleu_tokenizer: str):
    if not pred_csv.is_file():
        return {}
    import pandas as pd
    import sacrebleu

    df = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    bleu = sacrebleu.metrics.BLEU(tokenize=bleu_tokenizer).corpus_score(
        [s.strip() for s in df['prediction'].tolist()],
        [[s.strip() for s in df['target'].tolist()]],
    )
    return dict(
        bp=round(float(bleu.bp), 4),
        lr=round(float(bleu.sys_len) / float(max(1, bleu.ref_len)), 4),
        sys_len=int(bleu.sys_len),
        ref_len=int(bleu.ref_len),
    )


def _write_cocc_summary(out: Path, plan: dict):
    bleu_tok = plan.get('protocol', {}).get('bleu_tokenizer', 'zh')
    rows = []
    scores_by_id = {}
    length_by_id = {}
    for spec in plan['runs']:
        ident = spec['id']
        run_dir = out / 'runs' / ident
        val_path = run_dir / 'VAL_COMPLETE.json'
        if not val_path.exists():
            continue
        val = json.loads(val_path.read_text(encoding='utf-8'))
        sc = val.get('scores', {})
        lstats = _compute_length_stats(run_dir / 'validation_predictions.csv', bleu_tok)
        scores_by_id[ident] = sc
        length_by_id[ident] = lstats
        rows.append(dict(
            id=ident,
            method=spec['method'],
            beta=spec.get('consistency_beta', 0.0),
            bleu=round(float(sc.get('bleu', 0.0)), 4),
            chrf0=round(float(sc.get('chrf0', 0.0)), 4),
            chrfpp=round(float(sc.get('chrf++', 0.0)), 4),
            bp=lstats.get('bp'),
            lr=lstats.get('lr'),
            archive_sha256=val.get('archive_sha256', '')[:12],
        ))

    gates = {}
    verdict = 'INCOMPLETE'
    req = ('CONCAT_CE_S42', 'CONCAT_SYMSG_S42', 'COCC_SYM_S42', 'COCC_ISO_TEACHER_S42')
    if all(k in scores_by_id for k in req):
        b_ce, c_ce = scores_by_id['CONCAT_CE_S42']['bleu'], scores_by_id['CONCAT_CE_S42']['chrf0']
        b_cs = scores_by_id['CONCAT_SYMSG_S42']['bleu']
        b_cocc, c_cocc = scores_by_id['COCC_SYM_S42']['bleu'], scores_by_id['COCC_SYM_S42']['chrf0']
        b_v2, c_v2 = scores_by_id['COCC_ISO_TEACHER_S42']['bleu'], scores_by_id['COCC_ISO_TEACHER_S42']['chrf0']
        b_sym = REUSED_A100_COMPARATORS['GOC_SYM_B250_S42']['bleu']
        c_sym = REUSED_A100_COMPARATORS['GOC_SYM_B250_S42']['chrf0']
        lr_cocc = length_by_id.get('COCC_SYM_S42', {}).get('lr', 0.0)
        lr_v2 = length_by_id.get('COCC_ISO_TEACHER_S42', {}).get('lr', 0.0)

        g1 = (b_cocc - b_ce) >= 0.50 and c_cocc > c_ce
        g2 = (b_cocc - b_cs) >= 0.30
        g3 = (b_cocc - b_sym) >= 0.40 and c_cocc > c_sym
        g4 = 0.93 <= lr_cocc <= 1.05
        v2_gates = (
            (b_v2 - b_ce) >= 0.50 and c_v2 > c_ce
            and (b_v2 - b_cs) >= 0.30
            and (b_v2 - b_sym) >= 0.40 and c_v2 > c_sym
            and (0.93 <= lr_v2 <= 1.05)
        )
        gates = {
            'gate1_vs_concat_ce_ge_050': g1,
            'gate2_vs_concat_symsg_ge_030': g2,
            'gate3_vs_sym_sg_ge_040': g3,
            'gate4_lr_in_093_to_105': g4,
            'gate5_sym_vs_iso_teacher_delta': round(b_cocc - b_v2, 4),
            'deltas': {
                'cocc_minus_concat_ce_bleu': round(b_cocc - b_ce, 4),
                'cocc_minus_concat_symsg_bleu': round(b_cocc - b_cs, 4),
                'cocc_minus_sym_sg_bleu': round(b_cocc - b_sym, 4),
                'cocc_minus_baseline_bleu': round(b_cocc - REUSED_A100_COMPARATORS['BASELINE_S42']['bleu'], 4),
                'cocc_minus_iso_teacher_bleu': round(b_cocc - b_v2, 4),
                'cocc_lr': lr_cocc,
                'iso_teacher_lr': lr_v2,
            },
        }
        if g1 and g2 and g3 and g4:
            verdict = 'FULL_GO'
        elif v2_gates and (b_v2 - b_cocc) >= 0.30:
            verdict = 'V2_REVERSAL_REVIEW'
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
        '## New CoCC Pilot Runs',
        '| Run | Method | beta | BLEU | chrF0 | chrF++ | BP | LR | SHA prefix |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in rows:
        bp_s = f"{r['bp']:.4f}" if r['bp'] is not None else 'n/a'
        lr_s = f"{r['lr']:.4f}" if r['lr'] is not None else 'n/a'
        md.append(
            f"| `{r['id']}` | `{r['method']}` | {r['beta']:.2f} | {r['bleu']:.4f} | "
            f"{r['chrf0']:.4f} | {r['chrfpp']:.4f} | {bp_s} | {lr_s} | `{r['archive_sha256']}` |"
        )
    if gates:
        md += ['', '## Pre-Locked GO/NO-GO Gates', '```json', json.dumps(gates, indent=2), '```']
    (out / 'PILOT_SUMMARY.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return summary


def _write_off_eos_summary(out: Path, plan: dict):
    bleu_tok = plan.get('protocol', {}).get('bleu_tokenizer', 'zh')
    rows = []
    scores_by_id = {}
    length_by_id = {}
    for spec in plan['runs']:
        ident = spec['id']
        run_dir = out / 'runs' / ident
        val_path = run_dir / 'VAL_COMPLETE.json'
        if not val_path.exists():
            continue
        val = json.loads(val_path.read_text(encoding='utf-8'))
        sc = val.get('scores', {})
        lstats = _compute_length_stats(run_dir / 'validation_predictions.csv', bleu_tok)
        scores_by_id[ident] = sc
        length_by_id[ident] = lstats
        rows.append(dict(
            id=ident,
            method=spec['method'],
            stop_weight=spec.get('stop_weight', 0.10),
            bleu=round(float(sc.get('bleu', 0.0)), 4),
            chrf0=round(float(sc.get('chrf0', 0.0)), 4),
            chrfpp=round(float(sc.get('chrf++', 0.0)), 4),
            bp=lstats.get('bp'),
            lr=lstats.get('lr'),
            archive_sha256=val.get('archive_sha256', '')[:12],
        ))

    gates = {}
    verdict = 'INCOMPLETE'
    if all(k in scores_by_id for k in ('OFF_EOS_LENGTH_S42', 'OFF_EOS_CHRF_S42')):
        b_len = scores_by_id['OFF_EOS_LENGTH_S42']['bleu']
        c_len = scores_by_id['OFF_EOS_LENGTH_S42']['chrf0']
        b_chrf = scores_by_id['OFF_EOS_CHRF_S42']['bleu']
        c_chrf = scores_by_id['OFF_EOS_CHRF_S42']['chrf0']
        cpp_chrf = scores_by_id['OFF_EOS_CHRF_S42']['chrf++']
        lr_len = length_by_id.get('OFF_EOS_LENGTH_S42', {}).get('lr', 0.0)
        lr_chrf = length_by_id.get('OFF_EOS_CHRF_S42', {}).get('lr', 0.0)

        b_base = REUSED_A100_COMPARATORS['BASELINE_S42']['bleu']
        c_base = REUSED_A100_COMPARATORS['BASELINE_S42']['chrf0']
        b_w125 = REUSED_A100_COMPARATORS['W_B125_S42']['bleu']
        c_w125 = REUSED_A100_COMPARATORS['W_B125_S42']['chrf0']
        cpp_w125 = REUSED_A100_COMPARATORS['W_B125_S42']['chrf++']
        b_sym = REUSED_A100_COMPARATORS['GOC_SYM_B250_S42']['bleu']

        g1 = (b_chrf - b_base) >= 0.70 and c_chrf > c_base
        g2 = (b_chrf - b_w125) >= 0.50 and c_chrf > c_w125 and cpp_chrf > cpp_w125
        g3 = (b_chrf - b_len) >= 0.30
        g4 = 0.93 <= lr_chrf <= 1.05
        gates = {
            'gate1_vs_baseline_ge_070': g1,
            'gate2_vs_rdrop_ge_050': g2,
            'gate3_vs_length_control_ge_030': g3,
            'gate4_lr_in_093_to_105': g4,
            'deltas': {
                'chrf_minus_baseline_bleu': round(b_chrf - b_base, 4),
                'chrf_minus_rdrop_bleu': round(b_chrf - b_w125, 4),
                'chrf_minus_length_control_bleu': round(b_chrf - b_len, 4),
                'chrf_minus_length_control_chrf0': round(c_chrf - c_len, 4),
                'chrf_minus_sym_sg_bleu_ref_only': round(b_chrf - b_sym, 4),
                'chrf_lr': lr_chrf,
                'length_control_lr': lr_len,
            },
        }
        if g1 and g2 and g3 and g4:
            verdict = 'FULL_GO'
        elif (b_chrf - b_w125) >= 0.20 and c_chrf > c_w125 and g3 and g4:
            verdict = 'PROMISING_MECHANISM_REVIEW'
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
        '## New Off-EOS Pilot Runs',
        '| Run | Method | stop_weight | BLEU | chrF0 | chrF++ | BP | LR | SHA prefix |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in rows:
        bp_s = f"{r['bp']:.4f}" if r['bp'] is not None else 'n/a'
        lr_s = f"{r['lr']:.4f}" if r['lr'] is not None else 'n/a'
        md.append(
            f"| `{r['id']}` | `{r['method']}` | {r['stop_weight']:.2f} | {r['bleu']:.4f} | "
            f"{r['chrf0']:.4f} | {r['chrfpp']:.4f} | {bp_s} | {lr_s} | `{r['archive_sha256']}` |"
        )
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

    # 4) Step-0 diagnostics & validation-only training in locked order
    from methods.compositional_consistency import METHODS as COCC_METHODS
    from methods.cross_ratio_dual import METHODS as CR_DUAL_METHODS
    from methods.off_eos_stopping import METHODS as OFF_EOS_METHODS
    is_cocc_pilot = any(r['method'] in COCC_METHODS for r in plan['runs'])
    is_cr_dual_pilot = any(r['method'] in CR_DUAL_METHODS for r in plan['runs'])
    is_off_eos_pilot = any(r['method'] in OFF_EOS_METHODS for r in plan['runs'])

    if is_cocc_pilot and not skip_step0:
        step0_path = out / 'step0_diagnostic.json'
        if not step0_path.exists():
            print('=== [STEP-0] ZERO-OPTIMIZER-STEP CoCC DIAGNOSTIC ===', flush=True)
            _run_subcommand(['cocc-diagnose', '--config', str(path)], env)
        if step0_only:
            summary = _write_cocc_summary(out, plan)
            return {'status': 'STEP0_COMPLETE', 'workspace': str(out), 'summary': summary}

    if is_off_eos_pilot and not skip_step0:
        step0_path = out / 'step0_diagnostic.json'
        if not step0_path.exists():
            print('=== [STEP-0] ZERO-OPTIMIZER-STEP OFF-EOS ORACLE & LATENCY DIAGNOSTIC ===', flush=True)
            _run_subcommand(['off-eos-diagnose', '--config', str(path)], env)
        if step0_only:
            summary = _write_off_eos_summary(out, plan)
            return {'status': 'STEP0_COMPLETE', 'workspace': str(out), 'summary': summary}

    print('=== [4/5] VALIDATION-ONLY A100 TRAINING ===', flush=True)
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
    if is_cocc_pilot:
        summary = _write_cocc_summary(out, plan)
    elif is_off_eos_pilot:
        summary = _write_off_eos_summary(out, plan)
    else:
        summary = _write_cr_dual_summary(out, plan)
    return summary
