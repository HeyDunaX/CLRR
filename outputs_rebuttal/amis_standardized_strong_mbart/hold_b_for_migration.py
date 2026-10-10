"""Colab3: pause only the dispatcher; the active A05 training child continues."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/content/CLRR_controls/outputs_rebuttal/amis_standardized_strong_mbart')


def process(pid):
    folder = Path('/proc') / str(pid)
    if not (folder / 'stat').exists():
        return None
    stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    return stat[0], int(stat[1]), (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


queue = json.loads((ROOT / 'queue_account3.json').read_text())
assert queue['status'] == 'TRAIN' and queue['current'] == 'A05'
assert not (ROOT / 'runs/A06/progress.json').exists(), 'B already started; do not interrupt'
controller = int(Path('/content/strong_mbart_supplement.pid').read_text())
assert process(controller) and 'controller.py' in process(controller)[2]
candidates = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    info = process(int(entry.name))
    if info and info[1] == controller and 'run_standardized_strong_mbart.py' in info[2] and '--stage gate-train' in info[2]:
        candidates.append(int(entry.name))
assert len(candidates) == 1, candidates
dispatcher = candidates[0]
children = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    info = process(int(entry.name))
    if info and info[1] == dispatcher and '--id A05 --stage train' in info[2]:
        children.append(int(entry.name))
assert len(children) == 1, children
training = children[0]
os.kill(dispatcher, signal.SIGSTOP)
time.sleep(.1)
assert process(dispatcher)[0] == 'T'
assert process(training) and process(training)[0] not in ('T', 'Z', 'X')
receipt = {'status': 'A05_TRAINING_B_HELD_FOR_ACCOUNT1', 'dispatcher_pid': dispatcher,
           'training_pid': training, 'destination_account': 1, 'updated_unix': time.time(),
           'reason': 'User approved migration if Colab3 budget insufficient; 17.40CU remains, A05 plus A06 forecast exceeds budget'}
write(ROOT / 'migration_guard.json', receipt)
watcher = ROOT / 'finish_a05_migration.py'
watcher.write_text('''import json,os,signal,time
from pathlib import Path
root=Path('/content/CLRR_controls/outputs_rebuttal/amis_standardized_strong_mbart')
record=json.loads((root/'migration_guard.json').read_text())
def alive(pid):
    p=Path('/proc',str(pid),'stat')
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
while alive(record['training_pid']): time.sleep(10)
receipt=root/'runs/A05/VAL_COMPLETE.json'
if not receipt.exists():
    record.update(status='A05_FAILED_WEIGHTS_RETAINED_B_HELD',updated_unix=time.time())
    (root/'migration_guard.json').write_text(json.dumps(record)); raise SystemExit(1)
assert json.loads(receipt.read_text())['status']=='VALIDATION_COMPLETE_VERIFIED'
dispatcher=record['dispatcher_pid']
if alive(dispatcher):
    command=Path('/proc',str(dispatcher),'cmdline').read_bytes()
    assert b'run_standardized_strong_mbart.py' in command and b'gate-train' in command
    os.kill(dispatcher,signal.SIGTERM); os.kill(dispatcher,signal.SIGCONT)
controller=int(Path('/content/strong_mbart_supplement.pid').read_text())
while alive(controller): time.sleep(1)
record.update(status='A05_VERIFIED_B_MIGRATED_TO_ACCOUNT1',updated_unix=time.time())
(root/'migration_guard.json').write_text(json.dumps(record,indent=2))
(root/'schedule_status.json').write_text(json.dumps(record,indent=2))
(root/'queue_account3.json').write_text(json.dumps({'account':3,'status':'VALIDATION_ASSIGNED_QUEUE_COMPLETE','completed':['A05'],'migrated':['A06'],'updated_unix':time.time()},indent=2))
''')
with open('/content/strong_mbart_migration_guard.log', 'w') as log:
    child = subprocess.Popen(['/content/miniforge3/envs/clrr/bin/python', '-u', str(watcher)],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path('/content/strong_mbart_migration_guard.pid').write_text(str(child.pid))
print(json.dumps(receipt))
