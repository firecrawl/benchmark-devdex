"""Invalid execution counts must fail before filesystem or paid driver activity."""
import sys
import pytest
from devdex.scorer import suite

@pytest.mark.parametrize('field,value', [('n_runs',0),('n_runs',-1),('concurrency',0),
    ('concurrency',-2),('limit',0),('limit',-3)])
def test_run_cell_rejects_nonpositive_counts_before_output(tmp_path, field, value):
    args = dict(track='repo', arm='no-tool', driver='opus', n_runs=1,
                concurrency=1, out_root=tmp_path/'must-not-exist')
    args[field] = value
    with pytest.raises(SystemExit, match='must be positive'):
        suite.run_cell(**args)
    assert not args['out_root'].exists()

@pytest.mark.parametrize('option', ['--n-runs','--max-workers','--limit'])
@pytest.mark.parametrize('value', ['0','-2'])
def test_run_eval_parser_rejects_before_loading_env(monkeypatch, option, value):
    from devdex import run_eval
    monkeypatch.setattr(sys, 'argv', ['devdex','--track','repo','--arm','no-tool',
                                     '--driver','opus',option,value])
    monkeypatch.setattr(run_eval, 'load_env', lambda: pytest.fail('must reject before loading credentials'))
    with pytest.raises(SystemExit) as error: run_eval.main()
    assert error.value.code == 2

@pytest.mark.parametrize('option', ['--n-runs','--workers','--limit'])
@pytest.mark.parametrize('value', ['0','-2'])
def test_public_benchmark_parser_rejects_before_reporting(monkeypatch, option, value):
    from benchmark import run_benchmark
    monkeypatch.setattr(sys, 'argv', ['benchmark','--report-only',option,value])
    monkeypatch.setattr(run_benchmark, 'report', lambda *args: pytest.fail('must reject before reporting'))
    with pytest.raises(SystemExit) as error: run_benchmark.main()
    assert error.value.code == 2
