"""Milestone 9: the demo interface.

No test here touches a network or a live model, which is the same property the
demo mode itself claims -- if these tests needed a provider, the mode would not
be doing its job.

The tests that matter most are the negative ones: that demo mode cannot fall
through to a paid provider, that a preset button cannot be offered without a
recording behind it, and that the two fields the milestone calls the point of
the system are actually surfaced rather than merely present in the JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent.contracts import ErrorCode
from src.api import config as api_config
from src.api import demo
from src.api.app import STATIC_DIR, create_app

REPO = Path(__file__).resolve().parents[1]
TRANSCRIPTS = REPO / "evals" / "transcripts"
SCENARIOS = REPO / "evals" / "scenarios.yaml"

pytestmark = pytest.mark.skipif(
    not TRANSCRIPTS.exists() or not any(TRANSCRIPTS.glob("*.json")),
    reason="recorded transcripts are not present",
)


@pytest.fixture
def demo_client(built_db, tmp_path):
    settings = api_config.Settings(
        database=str(built_db),
        run_store=str(tmp_path / "runs"),
        demo_mode=True,
        transcripts=str(TRANSCRIPTS),
        scenarios=str(SCENARIOS),
    )
    return TestClient(create_app(settings))


# ----------------------------------------------------------------------
# The page itself
# ----------------------------------------------------------------------


def test_the_page_is_served_at_the_root(demo_client):
    response = demo_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>" in response.text


def test_the_script_is_served_and_is_the_only_static_file_reachable(demo_client):
    assert demo_client.get("/app.js").status_code == 200
    # Nothing else in the package is exposed. A directory mount would have made
    # this a question about traversal; naming the one file makes it not one.
    for hostile in ("/static/app.js", "/app.py", "/../config.py", "/index.html"):
        assert demo_client.get(hostile).status_code in (404, 405)


def test_the_page_loads_nothing_from_a_network():
    """No build step, no framework, no CDN -- so it works offline.

    Asserted against the files rather than trusted, because a single
    `<script src="https://...">` added later would break the one property the
    demo is meant to have.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for external in ("http://", "https://", "//cdn", "@import"):
        assert external not in html, f"{external!r} in index.html"
        assert external not in script, f"{external!r} in app.js"

    # Every src/href is same-origin and relative.
    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert ref.startswith("/"), f"non-relative reference: {ref}"


def test_the_page_is_one_html_file_and_one_script():
    files = sorted(p.name for p in STATIC_DIR.iterdir() if p.is_file())
    assert files == ["app.js", "index.html"], files


def test_the_script_builds_the_dom_rather_than_interpolating_html():
    """Tool results and model output are untrusted text, not markup.

    The hazard is *assigning* to innerHTML, so that is what is checked -- an
    earlier version of this test banned the word and failed on its own comment.
    """
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"innerHTML\s*(\+?=)[^=]", script)
    assert not re.search(r"outerHTML\s*(\+?=)[^=]", script)
    assert "insertAdjacentHTML" not in script


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------


def test_the_presets_cover_every_shape_the_visitor_should_see(demo_client):
    body = demo_client.get("/v1/demo/presets").json()
    assert body["demo_mode"] is True
    presets = body["presets"]
    assert 6 <= len(presets) <= 8

    kinds = {p["kind"] for p in presets}
    # The six shapes the milestone asks for: a lookup, risk with an adequate
    # warning, risk with an insufficient one, parts, fleet-level, and one the
    # system cannot answer.
    assert {"lookup", "risk", "parts", "fleet", "refusal"} <= kinds


def test_every_offered_preset_has_a_recording_behind_it(demo_client):
    """A button that errors when pressed is worse than an absent button."""
    from evals.transcript import transcript_path

    for preset in demo_client.get("/v1/demo/presets").json()["presets"]:
        path = transcript_path(preset["scenario_id"], demo.DEMO_SEED, TRANSCRIPTS)
        assert path.exists(), f"{preset['scenario_id']} has no transcript"


def test_a_preset_without_a_transcript_is_dropped_not_offered(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    assert demo.available(transcripts=empty, scenarios_path=SCENARIOS) == []


def test_every_preset_names_a_real_scenario():
    known = demo.scenarios_by_id(SCENARIOS)
    for preset in demo.PRESETS:
        assert preset.scenario_id in known, preset.scenario_id


def test_each_preset_carries_a_takeaway():
    """The point of a refusal preset is lost on a visitor who reads it as a bug."""
    for preset in demo.PRESETS:
        assert preset.takeaway.strip()


# ----------------------------------------------------------------------
# Demo mode does not call a model
# ----------------------------------------------------------------------


def test_demo_mode_never_builds_a_provider_client(demo_client, monkeypatch):
    """The guarantee the page advertises: no key needed, nothing spent.

    `build_client` is made to explode. A replayed answer proves the live path
    was never entered -- not merely that it happened to work without a key.
    """
    def explode(self):
        raise AssertionError("demo mode built a provider client")

    monkeypatch.setattr(api_config.Settings, "build_client", explode)
    response = demo_client.post(
        "/v1/ask", json={"scenario_id": "lookup-machine-age-01"}
    )
    assert response.status_code == 200
    assert response.json()["replayed"] is True


def test_free_text_in_demo_mode_is_refused_rather_than_billed(demo_client):
    """It must not quietly fall back to a paid provider."""
    response = demo_client.post("/v1/ask", json={"question": "What is the risk on machine 5?"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == ErrorCode.INVALID_INPUT.value
    assert "PDM_DEMO_MODE" in body["message"]


def test_an_unknown_scenario_is_404_not_a_traceback(demo_client):
    response = demo_client.post("/v1/ask", json={"scenario_id": "no-such-thing"})
    assert response.status_code == 404
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value


def test_a_request_with_neither_question_nor_scenario_is_rejected(demo_client):
    assert demo_client.post("/v1/ask", json={}).status_code == 422


def test_demo_mode_is_the_default():
    """A default that spends money on first click would be the wrong default."""
    assert api_config.Settings().demo_mode is True
    assert "PDM_DEMO_MODE" in api_config.ENV_VARS


# ----------------------------------------------------------------------
# The trace, and the accounting behind it
# ----------------------------------------------------------------------


def test_a_replayed_run_returns_its_tool_calls_with_arguments_and_results(demo_client):
    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-comp1-01"}
    ).json()

    assert body["answer"].strip()
    assert body["tool_calls"], "the reasoning trace is the point of the page"
    for call in body["tool_calls"]:
        assert call["tool"]
        assert call["status"] in {"ok", "error"}
        assert isinstance(call["arguments"], dict)
        assert call["result"] is not None


def test_accounting_comes_from_the_obs_module_not_from_the_ui(demo_client):
    """Milestone 9 item 2: pull it from `src.obs.accounting`, do not recompute.

    Checked by rebuilding the record from the persisted spans with that module
    and requiring the served figures to match it exactly.
    """
    from src.obs import accounting

    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-comp1-01"}
    ).json()
    served = body["accounting"]

    store = Path(demo_client.app.state.settings.run_store)
    spans = json.loads((store / f"{body['run_id']}.spans.json").read_text(encoding="utf-8"))
    rebuilt = accounting.from_spans(spans, run_id=body["run_id"])[0]

    assert served["tokens_in"] == rebuilt.tokens_in
    assert served["tokens_out"] == rebuilt.tokens_out
    assert served["iterations"] == rebuilt.iterations
    assert served["tool_calls"] == rebuilt.tool_calls


def test_the_accounting_is_populated_rather_than_zeroed(demo_client):
    """A replayed run carries the recorded usage; it must not read as free."""
    served = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-comp1-01"}
    ).json()["accounting"]

    assert served["tokens_in"] > 0
    assert served["tokens_out"] > 0
    assert served["estimated_cost_usd"] > 0
    assert served["max_iterations"] > 0
    assert served["iterations"] <= served["max_iterations"]


def test_a_replayed_run_can_be_fetched_back_by_its_id(demo_client):
    posted = demo_client.post(
        "/v1/ask", json={"scenario_id": "lookup-machine-age-01"}
    ).json()
    fetched = demo_client.get(f"/v1/runs/{posted['run_id']}").json()
    assert fetched["answer"] == posted["answer"]
    assert fetched["replayed"] is True


# ----------------------------------------------------------------------
# The two fields that are the point of the system
# ----------------------------------------------------------------------


def test_warning_adequacy_and_calibrated_are_surfaced_not_buried(demo_client):
    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-comp1-01"}
    ).json()

    assert body["highlights"], "a risk question must surface its risk fields"
    for h in body["highlights"]:
        assert h["warning_adequacy"] in {"sufficient", "marginal", "insufficient"}
        assert isinstance(h["calibrated"], bool)
        assert 0.0 <= h["probability"] <= 1.0


def test_the_highlights_are_copied_from_the_tool_result_never_computed(demo_client):
    """Derived, not authored: every badge must trace to a returned field."""
    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-comp1-01"}
    ).json()

    risk = next(
        c for c in body["tool_calls"]
        if c["tool"] == "get_failure_risk" and c["status"] == "ok"
    )
    components = {c["component"]: c for c in risk["result"]["data"]["components"]}

    for h in body["highlights"]:
        source = components[h["component"]]
        assert h["calibrated"] == source["calibrated"]
        assert h["warning_adequacy"] == source["warning_adequacy"]
        assert h["probability"] == source["calibrated_probability"]


def test_an_uncalibrated_probability_is_flagged_as_such(demo_client):
    """comp3 reads 1.000 and is not established as better than the base rate.

    A visitor seeing that number without the flag would draw exactly the wrong
    conclusion, which is why the flag is a badge and not a footnote.
    """
    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "risk-inadequate-uncalibrated-01"}
    ).json()
    comp3 = next(h for h in body["highlights"] if h["component"] == "comp3")
    assert comp3["calibrated"] is False


def test_the_interval_is_labelled_as_the_models_not_this_machines():
    """`confidence_interval_*` is the model's PR-AUC interval.

    `src/agent/risk.py` says so where it sets the field, and on three of four
    components it does not bracket the probability it would sit beside. Showing
    it as a plain "95% CI" next to that probability is the misreading the
    contract warns about, so the page must name what the interval is of.
    """
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "PR-AUC" in script
    assert "'95% CI [" not in script


def test_a_parts_question_surfaces_no_risk_and_calls_no_risk_tool(demo_client):
    """The separation, visible on the page as well as in the import graph."""
    body = demo_client.post(
        "/v1/ask", json={"scenario_id": "parts-position-comp3-01"}
    ).json()
    assert [c["tool"] for c in body["tool_calls"]] == ["get_parts_position"]
    assert body["highlights"] == []


def test_highlights_ignore_a_failed_risk_call():
    """A badge built from an error body would be a fabricated reassurance."""
    failed = [{
        "tool": "get_failure_risk",
        "status": "error",
        "result": {"status": "error", "code": "timeout", "message": "x", "tool": "get_failure_risk"},
    }]
    assert demo.highlights(failed) == []
