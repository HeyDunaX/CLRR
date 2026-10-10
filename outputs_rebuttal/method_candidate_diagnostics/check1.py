import subprocess
r=subprocess.run(['pgrep','-af','python.*scripts/revalidation/(run_|diagnose_)'],capture_output=True,text=True)
assert not r.stdout.strip(), 'Another worker active'
print('NO_ACTIVE_WORKER')
