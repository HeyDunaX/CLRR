from pathlib import Path
p=Path('/content/diagnostics.pid');s=Path('/proc')/p.read_text().strip()/'stat'
assert not s.exists() or s.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Worker still alive'
print('INACTIVE')
