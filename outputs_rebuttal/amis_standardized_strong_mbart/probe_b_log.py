from pathlib import Path
p=Path('/content/strong_mbart_b_account1.log')
print('\n'.join(p.read_text(errors='replace').splitlines()[-24:]))
