"""A readable checkpoint must not turn a failed driver into successful automation."""
import json
import sys
import pytest
from devdex.scorer import suite


def test_real_failed_subprocess_remains_failed_when_checkpoint_is_scoreable(tmp_path, monkeypatch):
    script = tmp_path / 'driver.py'
    script.write_text('''import json,os,pathlib,sys
p=pathlib.Path(os.environ['OUT_DIR'])/'tasks';p.mkdir(exist_ok=True)
(p/'p1_repo_fixture.json').write_text(json.dumps([{'qid':'a','gold':['x/y'],'citations':[],'committed':False}]))
sys.exit(7)
''')
    monkeypatch.setitem(suite.DRIVERS, 'fixture', (str(script), None))
    result = suite.run_cell('repo', 'fixture', 'fixture', 1, 1, tmp_path / 'runs')
    assert result['aggregate']['precision']['mean'] == 0
    assert result.get('failed_passes') == [{'pass':1, 'returncode':7}]


def test_run_eval_entrypoint_exits_nonzero_after_driver_failure(monkeypatch):
    from devdex import run_eval
    monkeypatch.setattr(sys, 'argv', ['devdex', '--track','repo','--arm','no-tool',
                                     '--driver','opus','--skip-preflight'])
    monkeypatch.setattr(run_eval, 'load_env', lambda: None)
    monkeypatch.setattr(run_eval, 'run_cell', lambda **kw: {'failed_passes':[{'pass':1,'returncode':7}]})
    with pytest.raises(SystemExit) as error:
        run_eval.main()
    assert error.value.code != 0


def test_public_benchmark_entrypoint_preserves_failure_while_reporting(monkeypatch):
    from benchmark import run_benchmark
    import register_arm
    monkeypatch.setattr(sys, 'argv', ['benchmark','--name','fixture','--mcp-url','http://127.0.0.1:1'])
    monkeypatch.setattr(register_arm, 'run_agent_mode', lambda **kw: 7)
    reported = []
    monkeypatch.setattr(run_benchmark, 'report', lambda *args: reported.append(args) or 0)
    assert run_benchmark.main() != 0
    assert reported, 'failure should not suppress available diagnostic scores'
