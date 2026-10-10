"""Colab command under account1/2 without changing HOME or exposing credentials."""
import argparse,os,subprocess,sys,json
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--account',choices=('1','2','3','4'),required=True);p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args();env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
cmd=['colab']
if a.account=='2':
    env['USERPROFILE']=r'C:\Users\Lenovo\.colab_acc2'
    cmd+=['--config',r'C:\Users\Lenovo\.colab_acc2\.config\colab-cli\sessions.json']
elif a.account=='3':
    env['USERPROFILE']=r'C:\Users\Lenovo\.colab_acc3'
    cmd+=['--config',r'C:\Users\Lenovo\.colab_acc3\.config\colab-cli\sessions.json']
elif a.account=='4':
    env['USERPROFILE']=r'C:\Users\Lenovo\.colab_acc4'
    cmd+=['--config',r'C:\Users\Lenovo\.colab_acc4\.config\colab-cli\sessions.json']
r=subprocess.run(cmd+a.args,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
