"""Compile and check the upload project with pdfLaTeX and BibTeX, in conda clrr."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex-bin', type=Path, help='Directory containing pdflatex and bibtex')
    args = parser.parse_args()
    engine = shutil.which('pdflatex') if args.tex_bin is None else str(args.tex_bin/'pdflatex.exe' if os.name == 'nt' else args.tex_bin/'pdflatex')
    bibtex = shutil.which('bibtex') if args.tex_bin is None else str(args.tex_bin/'bibtex.exe' if os.name == 'nt' else args.tex_bin/'bibtex')
    if not engine or not bibtex:
        raise SystemExit('pdfLaTeX and BibTeX are required; pass --tex-bin when they are not on PATH.')
    project = ROOT/'docs/paper_upload'
    report = ROOT/'outputs_rebuttal/pdftex_compile_20261003'
    report.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((project/'provenance.json').read_text(encoding='utf-8'))
    for name, expected in provenance['files'].items():
        assert hashlib.sha256((project/name).read_bytes()).hexdigest() == expected, name
    env = dict(os.environ)
    if args.tex_bin:
        env['PATH'] = str(args.tex_bin) + os.pathsep + env.get('PATH', '')
    command = [engine, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', 'clrr_main.tex']
    steps = [('pdflatex_1', command), ('bibtex', [bibtex, 'clrr_main']),
             ('pdflatex_2', command), ('pdflatex_3', command)]
    result = {'compiler': 'pdfLaTeX + BibTeX', 'project': str(project.relative_to(ROOT)), 'steps': []}
    for name, cmd in steps:
        with (report/(name+'.log')).open('w', encoding='utf-8') as log:
            process = subprocess.run(cmd, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        result['steps'].append({'step': name, 'exit_code': process.returncode})
        (report/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(name, process.returncode, flush=True)
        if process.returncode:
            print('\n'.join((report/(name+'.log')).read_text(encoding='utf-8', errors='replace').splitlines()[-25:]))
            raise SystemExit(process.returncode)
    final_log = (project/'clrr_main.log').read_text(encoding='utf-8', errors='replace')
    bib_log = (project/'clrr_main.blg').read_text(encoding='utf-8', errors='replace')
    for pattern in (r'LaTeX Error:', r'Citation .* undefined', r'There were undefined references',
                    r'Label\(s\) may have changed', r'Overfull \\hbox'):
        assert not re.search(pattern, final_log), pattern
    assert "I didn't find a database entry" not in bib_log
    import pymupdf
    with pymupdf.open(project/'clrr_main.pdf') as pdf:
        text = '\n'.join(page.get_text() for page in pdf)
        assert '??' not in text
        assert not re.search(r'ashaninka|asháninka|layerskip', text, re.I)
        assert len(re.findall(r'Table \d+:', text)) == 10
        result['pages'] = len(pdf)
    result.update(tables=10, unresolved_citations=0, missing_inputs=0,
                  pdf_sha256=hashlib.sha256((project/'clrr_main.pdf').read_bytes()).hexdigest(),
                  source_files=provenance['files'])
    (report/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for suffix in ('.log', '.blg', '.aux', '.bbl', '.out'):
        shutil.copy2(project/('clrr_main'+suffix), report/('clrr_main'+suffix))
    target = ROOT/'docs/clrr_main.pdf'
    backup = report/'before/clrr_main.pdf'
    backup.parent.mkdir(exist_ok=True)
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(project/'clrr_main.pdf', target)
    for suffix in ('.log', '.blg', '.aux', '.bbl', '.out', '.pdf'):
        (project/('clrr_main'+suffix)).unlink()
    print('Verified:', result['pages'], 'pages, 10 tables, no missing inputs or unresolved citations.')


if __name__ == '__main__':
    main()
