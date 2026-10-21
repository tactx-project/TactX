"""Release structure checks for removals, reference integrity, and data exclusion."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote

from conftest import ROOT


def test_cleanup_manifest_matches_paths():
    manifest = json.loads((ROOT/'maintenance/cleanup_manifest.json').read_text())
    assert len(manifest['removed']) == 274
    assert len(manifest['renamed']) == 39
    for old in manifest['removed']:
        assert not (ROOT/old).exists(), old
    for old,new in manifest['renamed'].items():
        assert not (ROOT/old).exists(), old
        assert (ROOT/new).is_file(), new


def test_documentation_relative_links_exist():
    docs = ['README.md','policy_learning/README.md','THIRD_PARTY_NOTICES.md','maintenance/CLEANUP.md']
    for doc in docs:
        path = ROOT/doc
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            target = unquote(target.split('#')[0])
            if not target or '://' in target: continue
            assert (path.parent/target).exists(), f'{doc}: {target}'


def test_active_sources_have_no_retired_names_and_compile():
    for directory in ['joint_embedding_learning','scripts','policy_learning/tactile_policy','config']:
        for path in (ROOT/directory).rglob('*'):
            if path.suffix not in ['.py','.yaml','.json']: continue
            text = path.read_text()
            assert 'sachin' not in text.lower(), path
            assert 'tactile_benchmark' not in text, path
            if path.suffix == '.py': ast.parse(text, filename=str(path))
