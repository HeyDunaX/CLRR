import json,shutil
from pathlib import Path
root=Path('/content/CLRR_controls/outputs_rebuttal/amis_standardized_controls')
assert not list(root.glob('runs/*/progress.json')), 'A research run started; inspect rather than delete'
assert not list(root.glob('runs/*/VAL_COMPLETE.json')), 'A validated research run exists'
pid=Path('/content/standardized_controls.pid')
if pid.exists():
 stat=Path('/proc')/pid.read_text().strip()/'stat'
 assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Queue still active'
gate=root/'gates'
assert gate.resolve().is_relative_to(root.resolve())
if gate.exists():shutil.rmtree(gate)
log=Path('/content/standardized_controls.log')
if log.exists():
 target=Path('/content/standardized_controls_gate_attempt1.log')
 assert not target.exists()
 log.rename(target)
print('PRETRAINING_GATES_RESET_NO_RESEARCH_RUN_REMOVED')
