"""Deploy/collect the frozen diagnostic queue on Colab 1/2; never train."""
import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import call_pilot, ensure_session, get_session_name, get_hf_token
OUT=ROOT/'outputs_rebuttal/method_candidate_diagnostics'


def call(account,args,timeout=180):
    code,stdout,stderr=call_pilot(str(account),args,timeout=timeout)
    if code: raise RuntimeError(f'Account{account} {args[0]} failed: {stdout}\n{stderr}')
    return stdout


def launch(account):
    ensure_session(str(account))
    session=get_session_name(str(account))
    bundle=OUT/'bundle.zip'
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/revalidation'/n for n in ('diagnose_method_candidates.py','amis_v4_measurements.py')]
    files += [OUT/'PLAN.json',ROOT/'data_processed/amis_mandarin/validation.csv']
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,f.relative_to(ROOT).as_posix())
    check=OUT/f'check{account}.py'
    check.write_text("import subprocess\nr=subprocess.run(['pgrep','-af','python.*scripts/revalidation/(run_|diagnose_)'],capture_output=True,text=True)\nassert not r.stdout.strip(), 'Another worker active'\nprint('NO_ACTIVE_WORKER')\n")
    print(call(account,['exec','-s',session,'-f',str(check),'--timeout','60']),flush=True)
    call(account,['upload',str(bundle),'/content/diagnostics_bundle.zip','-s',session])
    fd,tmp=tempfile.mkstemp(prefix='clrr-diagnostic-token-')
    try:
        with os.fdopen(fd,'w') as f:f.write(get_hf_token())
        call(account,['upload',tmp,'/content/.diagnostic_hf_token','-s',session])
    finally:Path(tmp).unlink(missing_ok=True)
    # Reuse pinned installation instructions but replace the training command.
    original=(ROOT/'scripts/revalidation/launch_method_candidates.py').read_text()
    shell=original.split("shell = '''",1)[1].split("'''.replace",1)[0]
    shell=shell.replace('method_candidates_bundle.zip','diagnostics_bundle.zip').replace('CLRR_candidates','CLRR_diagnostics')
    shell=shell.replace('.standardized_hf_token','.diagnostic_hf_token')
    shell=shell[:shell.index('"$PY" -u scripts/revalidation/run_method_candidates.py')]
    ids=['OLD','U1F','U1C'] if account==1 else ['U2F','U2L','BASE']
    for ident in ids:
        shell+=f'"$PY" -u scripts/revalidation/diagnose_method_candidates.py --stage gpu --id {ident}\n'
    shell+='touch /content/diagnostic_queue_complete\n'
    path=OUT/f'account{account}.sh';path.write_text(shell,encoding='utf-8',newline='\n')
    call(account,['upload',str(path),'/content/diagnostics.sh','-s',session])
    start=OUT/f'start{account}.py'
    start.write_text("import subprocess\nfrom pathlib import Path\nPath('/content/diagnostic_queue_complete').unlink(missing_ok=True)\nwith open('/content/diagnostics.log','w') as log:\n p=subprocess.Popen(['bash','/content/diagnostics.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nPath('/content/diagnostics.pid').write_text(str(p.pid))\nprint('DIAGNOSTICS_LAUNCHED',p.pid)\n")
    print(call(account,['exec','-s',session,'-f',str(start),'--timeout','60']),flush=True)
    print('ACCOUNT',account,'ASSIGNED',ids,flush=True)


def monitor(account):
    session=get_session_name(str(account))
    probe=OUT/f'monitor{account}.py'
    probe.write_text("import json,subprocess\nfrom pathlib import Path\np=Path('/content/diagnostics.pid');stat=Path('/proc')/p.read_text().strip()/'stat' if p.exists() else Path('/nonexistent')\nalive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')\nroot=Path('/content/CLRR_diagnostics/outputs_rebuttal/method_candidate_diagnostics')\nresults={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/COMPLETE.json')}\nlog=Path('/content/diagnostics.log')\nprint(json.dumps({'alive':alive,'queue_complete':Path('/content/diagnostic_queue_complete').exists(),'results':results,'tail':log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []}))\n")
    stdout=call(account,['exec','-s',session,'-f',str(probe),'--timeout','60'])
    result=next(json.loads(line) for line in stdout.splitlines() if line.startswith('{'))
    for ident,record in result.pop('results').items():
        dest=OUT/ident;dest.mkdir(exist_ok=True)
        (dest/'COMPLETE.json').write_text(json.dumps(record,indent=2))
        for condition in record['scores']:
            filename=f'validation_{condition}.csv'
            if not (dest/filename).exists():
                call(account,['download',f'/content/CLRR_diagnostics/outputs_rebuttal/method_candidate_diagnostics/{ident}/{filename}',str(dest/filename),'-s',session])
    (OUT/f'status{account}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'account':account,**result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--account',type=int,choices=(1,2),required=True);p.add_argument('--stage',choices=('launch','monitor'),required=True)
    a=p.parse_args();(launch if a.stage=='launch' else monitor)(a.account)
