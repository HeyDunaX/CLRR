import subprocess
r=subprocess.run(['pgrep','-af','python.*scripts/revalidation/run_'],capture_output=True,text=True)
assert not r.stdout.strip(), 'A worker is already active'
print('NO_ACTIVE_RESEARCH_WORKER')
