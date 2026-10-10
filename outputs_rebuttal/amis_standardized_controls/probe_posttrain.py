from pathlib import Path
import json, subprocess
root = Path('/content/CLRR_controls/outputs_rebuttal/amis_standardized_controls')
for name in ['/content/standardized_controls.pid', str(root/'queue_account2.json'), str(root/'runs/A04/progress.json'), str(root/'runs/A04/VAL_COMPLETE.json'), str(root/'runs/A04/FAILED.json')]:
    p = Path(name)
    print('\nFILE', name, p.exists())
    if p.exists():
        print(p.read_text()[-6000:])
p=Path('/content/standardized_controls.log')
print('\nLOGTAIL\n', '\n'.join(p.read_text(errors='replace').splitlines()[-70:]) if p.exists() else 'missing')
print('\nPROCESSES\n', subprocess.run(['ps','-eo','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout[-8000:])
