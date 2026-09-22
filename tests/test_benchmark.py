import json
from open_jev.benchmark import run_benchmark, save_report


def test_benchmark_records_latency_and_writes_report(tmp_path):
    result = run_benchmark("toy", [1, 2], lambda x: x + 1, lambda rows, outs: {"accuracy": 1.0})
    assert result.count == 2 and result.latency_ms_p95 is not None
    path = tmp_path / "report.json"
    save_report(path, [result])
    assert json.loads(path.read_text())["results"][0]["name"] == "toy"


def test_compare_baselines_keeps_dimensions():
    from open_jev.benchmark import compare_results

    result = compare_results({"jev": {"accuracy": 0.8}, "baseline": {"accuracy": 0.7}})
    assert result["jev"]["accuracy"] == 0.8 and result["baseline"]["accuracy"] == 0.7
