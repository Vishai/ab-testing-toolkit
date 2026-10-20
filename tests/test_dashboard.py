import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from abkit.workflow import PipelineConfig, run_pipeline

APP = Path(__file__).resolve().parents[1] / "app" / "posterior_dashboard.py"


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashboard_runs")
    config = PipelineConfig.from_dict({
        "run_id": "dash", "output_root": str(root), "draws": 200, "tune": 300, "chains": 2,
        "prior_sensitivity": True, "uplift_boot": 50, "uplift_perm": 10,
        "bundled_scenarios": ["travel_booking"], "experiment": {"n_users": 30_000},
    })
    run_pipeline(config)
    return root


def open_app(monkeypatch, runs: Path) -> AppTest:
    monkeypatch.setenv("ABKIT_RUNS_DIR", str(runs))
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_dashboard_shows_a_finished_run(monkeypatch, runs_dir):
    app = open_app(monkeypatch, runs_dir)
    assert app.title[0].value == "Experiment readout · dash"
    assert len(app.tabs) == 5
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Trust checks"] == "PASS"
    assert metrics["Targeted policy lift"].startswith("+")
    assert {"Max R-hat", "Min bulk ESS", "Divergences", "Posterior predictive"} <= set(metrics)
    assert any("Sending each user" in m.value for m in app.markdown)


def test_recommendations_follow_the_posterior(monkeypatch, runs_dir):
    app = open_app(monkeypatch, runs_dir)
    table = app.dataframe[0].value.set_index("Segment")
    assert len(table) == 6
    bundle = json.loads((runs_dir / "dash" / "dashboard.json").read_text())
    segments = bundle["hierarchical"]["segments"]["device_tenure"]
    for name, row in table.iterrows():
        device, tenure = name.split(" / ")
        here = [s for s in segments if s["device"] == device and s["tenure"] == tenure]
        positive = sorted((s for s in here if s["verdict"] == "positive"),
                          key=lambda s: -s["rel_lift_mean"])
        negative = {s["arm"] for s in here if s["verdict"] == "negative"}
        if positive:
            assert row["Action"] == f"Show {positive[0]['arm']}"
        else:
            assert row["Action"] == "Keep control, or explore"
        assert set(filter(None, row["Avoid"].replace("-", "").split(", "))) == negative


def test_every_segment_view_renders(monkeypatch, runs_dir):
    app = open_app(monkeypatch, runs_dir)
    for label in ("Segment by", "Offer"):
        box = next(s for s in app.selectbox if s.label == label)
        for option in box.options:
            box.set_value(option).run()
            assert not app.exception, (label, option)


def test_dashboard_explains_what_to_do_without_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("ABKIT_RUNS_DIR", str(tmp_path / "empty"))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception
    assert app.title[0].value == "No pipeline runs yet"
    assert "abkit.workflow run" in app.info[0].value
