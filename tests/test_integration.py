import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "ps-server"


def run(tmp_path, mode, staleness=0, compression=None):
    cfg = yaml.safe_load((ROOT / "tests/configs/tiny.yaml").read_text())
    cfg["ps"] = {"mode": mode, "staleness": staleness, "shards": 2}
    cfg["compression"] = compression or {"kind": "none"}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "result.json"
    subprocess.run([sys.executable, str(ROOT / "experiments/run.py"), "--config", str(path), "--out", str(out),
                    "--server-bin", str(BIN)], check=True, timeout=60)
    return json.loads(out.read_text())


@pytest.mark.skipif(not BIN.exists(), reason="run `make build` first")
@pytest.mark.parametrize("mode,staleness,compression", [
    ("bsp", 0, None),
    ("ssp", 2, {"kind": "topk", "ratio": 0.05}),
    ("asp", 0, {"kind": "int8"}),
])
def test_two_workers_converge(tmp_path, mode, staleness, compression):
    r = run(tmp_path, mode, staleness, compression)
    assert r["final_acc"] > 0.85
    assert r["curve"][-1]["train_loss"] < 0.5
    assert all(w["pushes"] == r["steps"] for w in r["server_stats"])
    assert max(int(k) for k in r["staleness_hist"]) <= max(staleness, 0) or mode == "asp"


@pytest.mark.skipif(not BIN.exists(), reason="run `make build` first")
def test_compression_cuts_push_bytes(tmp_path):
    dense = run(tmp_path, "bsp")
    sparse = run(tmp_path, "bsp", compression={"kind": "topk", "ratio": 0.01})
    assert dense["bytes_push"] / sparse["bytes_push"] > 10
