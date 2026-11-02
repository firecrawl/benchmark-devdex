"""Exercise the real pass/checkpoint function without starting a paid SDK session.

Like the existing _resume_shim, load only the function: runner import starts an SDK jail
and reads argv. The async item producer is synthetic; persistence is production code.
"""
import ast
import gzip
import json
import os
import time
from pathlib import Path
import anyio

RUNNER = Path(__file__).resolve().parents[1] / 'harness/runner_sdk.py'


def run_resumed_pass(tmp_path, old_rows, new_rows, history=None):
    tasks = tmp_path / 'tasks'; tasks.mkdir()
    archive = tasks / 'p1_repo_fixture.raw.jsonl.gz'
    old_raw = [{'qid': r['qid'], 'searches': [{'original': r['qid']}],
                'transcript': [{'text': 'retained-' + r['qid']}]} for r in (old_rows if history is None else history)]
    with gzip.open(archive, 'wt') as f:
        for entry in old_raw: f.write(json.dumps(entry) + '\n')
    async def run_one(item): return dict(new_rows[item['id']])
    ns = dict(json=json, os=os, time=time, anyio=anyio, TRACK='repo', ARM_NAME='fixture',
              _resume=lambda *args, **kwargs: (old_rows, [{'id': key} for key in new_rows]),
              _out=lambda name: str(tasks / name), run_one=run_one,
              _start_reaper=lambda: None, _flush=lambda *args: None,
              _CFG_SNAPSHOT={}, REAPED=[], GT_SHA='fixture-hash', norm=lambda x: x)
    for key in ('MAX_SEARCH','MAX_READ','MAX_TURNS','AGENT_SNIPPET','PAGE_FULL','GROUND_CHARS',
                'K','CITE_K','TIMEOUT','REAP_GRACE','RESTRICT','MODEL','SEARCH_TOOLS','MANIFEST',
                'RUNS','SKILL_NAME','SKILL_SHA','SKILL_PROV'):
        ns[key] = None
    ns.update(CHUNK=25, CONCURRENCY=1, ARM=type('Arm', (), {'no_depth':False,'no_scope':False})())
    fn = next(n for n in ast.parse(RUNNER.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == '_one_pass')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(RUNNER), 'exec'), ns)
    ns['_one_pass']([], 'p1', 1)
    with gzip.open(archive, 'rt') as f: return [json.loads(line) for line in f]


def row(qid, **extra):
    return dict(qid=qid, citations=[], gold=[], cost=0, tool_calls=0, error=None, **extra)


def test_resume_preserves_transcripts_of_inherited_completed_items(tmp_path):
    saved = run_resumed_pass(tmp_path, [row('old')], {
        'new': row('new', _raw=[{'fresh':1}], _transcript=[{'text':'new'}])})
    by_id = {r['qid']:r for r in saved}
    assert by_id['old']['searches'] == [{'original':'old'}]
    assert by_id['old']['transcript'] == [{'text':'retained-old'}]
    assert by_id['new']['searches'] == [{'fresh':1}]


def test_rerun_replaces_prior_attempt_evidence_instead_of_inheriting_it(tmp_path):
    saved = run_resumed_pass(tmp_path, [row('keep')], {
        'retry': row('retry', _raw=[], _transcript=[{'text':'replacement'}])},
        history=[row('keep'), row('retry')])
    by_id = {r['qid']:r for r in saved}
    assert by_id['retry']['searches'] == []
    assert by_id['retry']['transcript'] == [{'text':'replacement'}]
