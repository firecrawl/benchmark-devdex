"""Load the actual standalone dataset-selection statements without a paid SDK import.

The runner reads argv and builds an SDK jail on import. Like existing _resume_shim,
extract only production selection/load_items logic and execute it on shipped ground truth.
"""
import ast
import json
import os
from pathlib import Path
import pytest
from devdex.scorer import suite

RUNNER = Path(__file__).resolve().parents[1]/'harness/runner_sdk.py'


def selection(track, monkeypatch, override=None):
    monkeypatch.delenv('GT_FILE', raising=False)
    if override is not None: monkeypatch.setenv('GT_FILE', str(override))
    wanted = {'GT_FILE', 'INTENT', 'load_items'}
    nodes = [n for n in ast.parse(RUNNER.read_text()).body
             if (isinstance(n, ast.FunctionDef) and n.name in wanted)
             or (isinstance(n, ast.Assign) and any(getattr(t,'id',None) in wanted for t in n.targets))]
    ns = dict(os=os, json=json, TRACK=track, DATASETS=str(suite.DATASETS), N='all')
    exec(compile(ast.Module(body=nodes, type_ignores=[]),str(RUNNER),'exec'),ns)
    return Path(ns['GT_FILE']), ns['load_items']()

@pytest.mark.parametrize('track', ['repo','fix','docs'])
def test_standalone_uses_the_selected_tracks_shipped_corpus(track, monkeypatch):
    path, rows = selection(track, monkeypatch)
    expected = suite.DATASETS/suite.TRACKS[track]
    expected_rows = [json.loads(line) for line in expected.read_text().splitlines() if line.strip()]
    assert rows, 'a supported track cannot silently run zero questions'
    assert path == expected
    assert [r['id'] for r in rows] == [r['id'] for r in expected_rows]


def test_explicit_gt_file_override_is_preserved(tmp_path, monkeypatch):
    custom=tmp_path/'custom.jsonl'
    custom.write_text(json.dumps({'id':'custom','intent':'find_docs','query':'fixture'})+'\n')
    path, rows = selection('docs', monkeypatch, custom)
    assert path == custom
    assert [r['id'] for r in rows] == ['custom']
