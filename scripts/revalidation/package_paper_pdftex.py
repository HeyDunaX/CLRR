"""Export a complete pdfLaTeX upload bundle, preserving the conference template.

Run in conda clrr after build_paper_tables.py. Tables are embedded unchanged;
the full current bibliography and original ACL style files accompany the paper.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/paper' if (ROOT / 'docs/paper/clrr_main.tex').exists() else ROOT / 'docs'
OUT = DOCS / 'paper_upload'


def embed_paper_tables(manuscript):
    """Refresh inline tables, also accepting the older input-based manuscript."""
    sources = {}
    table_dir = (DOCS / 'paper_tables') if (DOCS / 'paper_tables').exists() else (ROOT / 'docs/paper_tables')
    tables = sorted(table_dir.glob('*.tex'))
    if len(tables) != 10:
        raise ValueError('Expected exactly ten generated paper tables.')
    for path in tables:
        relative = 'paper_tables/' + path.name

        begin = '% BEGIN embedded ' + relative
        end = '% END embedded ' + relative
        pattern = (re.escape(begin) + r'\n.*?\n' + re.escape(end)
                   + '|' + re.escape(r'\input{' + relative[:-4] + '}'))
        block = begin + '\n' + path.read_text(encoding='utf-8').rstrip() + '\n' + end
        manuscript, count = re.subn(pattern, lambda match: block, manuscript, flags=re.S)
        if count != 1:
            raise ValueError(f'Expected one table location for {relative}; found {count}.')
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not manuscript.startswith('% !TeX program = pdflatex\n'):
        manuscript = '% !TeX program = pdflatex\n' + manuscript
    if re.search(r'\\(?:input|include)\{', manuscript):
        raise ValueError('Manuscript still contains external LaTeX inputs.')
    return manuscript, sources


def main():
    OUT.mkdir(exist_ok=True)
    manuscript = (DOCS/'clrr_main.tex').read_text(encoding='utf-8')
    bibliography = (DOCS/'clrr_references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bibliography))
    citations = {key.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', manuscript)
                 for key in group.split(',')}
    missing = citations - keys
    if missing:
        raise ValueError('Bibliography entries missing: ' + ', '.join(sorted(missing)))
    manuscript, sources = embed_paper_tables(manuscript)
    (OUT/'clrr_main.tex').write_text(manuscript, encoding='utf-8')
    for name in ('clrr_references.bib', 'acl.sty', 'acl_natbib.bst'):
        shutil.copy2(DOCS/name, OUT/name)
    (OUT/'README.md').write_text('''# Upload and compile

Upload the entire `clrr_pdflatex_upload.zip` to a new project, or replace all four
source/style files in your existing project. Select `clrr_main.tex` as the main
document and **pdfLaTeX** as the compiler. Recompile from scratch after replacing
the bibliography so cached references are rebuilt.

All ten tables are embedded in `clrr_main.tex`; no `paper_tables/` directory is
required for this upload version. Use the included `clrr_references.bib`, which
contains every citation key in this revision. The two ACL style files are
identical to the author's original conference template.

Local build, from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error clrr_main.tex
bibtex clrr_main
pdflatex -interaction=nonstopmode -halt-on-error clrr_main.tex
pdflatex -interaction=nonstopmode -halt-on-error clrr_main.tex
```

Source of truth: `docs/clrr_main.tex`, `docs/clrr_references.bib`, and the frozen
generated table files. Regenerate with `scripts/revalidation/package_paper_pdftex.py`
in the existing conda `clrr` environment. No model evaluation or training occurs.
''', encoding='utf-8')
    files = ('clrr_main.tex', 'clrr_references.bib', 'acl.sty', 'acl_natbib.bst', 'README.md')
    provenance = {'table_sources': sources, 'citation_keys': sorted(citations), 'missing_keys': [],
                  'files': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files},
                  'compiler': 'pdfLaTeX', 'embedded_tables': 10}
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    archive = DOCS/'clrr_pdflatex_upload.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (*files, 'provenance.json'):
            bundle.write(OUT/name, arcname=name)
    print('Ready:', archive.relative_to(ROOT), '— 10 embedded tables;', len(citations), 'citation keys verified.')


if __name__ == '__main__':
    main()
