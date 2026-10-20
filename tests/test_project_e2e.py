"""End to end with the project's real config.yaml on synthetic snapshots (no network)."""

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from parkwaits.cli import main
from parkwaits.features import FEATURES
from parkwaits.lake import upsert_partitioned
from parkwaits.registry import load_current
from parkwaits.serve import create_app
from parkwaits.storage import connect, read_table
from tests.synth import TZ, make_snapshots


@pytest.fixture
def project(tmp_path):
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg["lake"]["root"] = str(tmp_path / "raw")
    cfg["data"]["db_path"] = str(tmp_path / "wh.duckdb")
    cfg["model"]["registry_dir"] = str(tmp_path / "models")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg["queue_times"]["parks"] = [{"id": k, "name": f"park{k}", "timezone": tz} for k, tz in TZ.items()]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    upsert_partitioned(
        cfg["lake"]["root"],
        "queue_times",
        make_snapshots(days=14, rides_per_park=8, seed=5),
        key_cols=["park_id", "ride_id", "observed_at"],
        partition_col="observed_at",
    )
    return cfg, str(path)


def test_build_train_serve_and_monitor(project, tmp_path, capsys):
    cfg, path = project
    assert main(["--config", path, "build"]) == 0
    assert "Persistence baseline AUC" in capsys.readouterr().out
    con = connect(cfg["data"]["db_path"])
    table = read_table(con, cfg["data"]["table"])
    con.close()
    assert set(FEATURES) <= set(table.columns) and {"label", "event_ts"} <= set(table.columns)
    assert 0.05 < table["label"].mean() < 0.95

    assert main(["--config", path, "train"]) == 0
    _, card, _ = load_current(cfg["model"]["registry_dir"])
    assert card["features"] == FEATURES
    best = card["metrics"]["all_candidates"][card["champion"]]
    assert best["roc_auc"] > 0.8  # waits are strongly autocorrelated: the signal is learnable
    assert "60 minutes" in card["label_definition"]
    assert any("seasonality" in item for item in card["limitations"])

    record = {f: (0.0 if pd.isna(v) else float(v)) for f, v in table.iloc[-1][FEATURES].items()}
    with TestClient(create_app(cfg["model"]["registry_dir"])) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = client.post("/predict", json={"records": [record]}).json()
        assert 0.0 <= body["predictions"][0]["probability"] <= 1.0

    assert main(["--config", path, "drift", "--out", str(tmp_path / "drift.json")]) in (0, 2)


def test_build_names_the_missing_source(project):
    cfg, path = project
    cfg["lake"]["root"] = str(project[0]["lake"]["root"]) + "_empty"
    other = path + ".empty.yaml"
    open(other, "w").write(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="queue_times"):
        main(["--config", other, "build"])
