"""Screenshot the demo page. Milestone 9.

Starts the service in demo mode on a loopback port, drives the page with a
headless browser, and writes three PNGs into `docs/images/`.

**Not part of the test suite and not part of any build.** It needs a browser
binary, which the test suite deliberately does not: `tests/test_demo_ui.py`
asserts the page's behaviour through the API and the files, none of which
requires rendering it. This script exists to produce the images in the README
and is run by hand.

    python -m scripts.capture_demo

Requires `pip install playwright && python -m playwright install chromium`.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images"

#: Which preset each shot presses, and what the picture is meant to show.
SHOTS = (
    (
        "demo-risk-insufficient.png",
        "risk-inadequate-comp1-01",
        "The lead-time finding: warning insufficient, order refused",
    ),
    (
        "demo-parts-position.png",
        "parts-position-comp3-01",
        "A parts question, answered without the risk model",
    ),
    (
        "demo-unanswerable.png",
        "unanswerable-root-cause-01",
        "An honest refusal",
    ),
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed; see this module's docstring")
        return 2

    import uvicorn

    from src.api.app import create_app
    from src.api.config import Settings

    settings = Settings(
        database=str(REPO / "data" / "pdm.db"),
        run_store=str(REPO / "data" / "runs"),
        demo_mode=True,
        transcripts=str(REPO / "evals" / "transcripts"),
        scenarios=str(REPO / "evals" / "scenarios.yaml"),
        log_level="WARNING",
    )
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                       log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    else:
        print("the server did not start")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1180, "height": 1000},
            device_scale_factor=2,  # legible when the README scales it down
            color_scheme="light",
        )
        for filename, scenario_id, caption in SHOTS:
            page.goto(base, wait_until="networkidle")
            page.wait_for_selector(".preset")
            index = next(
                i for i, (_, sid, _) in enumerate(SHOTS) if sid == scenario_id
            )
            del index  # the button is found by its scenario, not its position
            page.evaluate(
                """(sid) => {
                    const p = PRESETS.find(x => x.scenario_id === sid);
                    const buttons = document.querySelectorAll('.preset');
                    for (let i = 0; i < PRESETS.length; i++) {
                        if (PRESETS[i].scenario_id === sid) { buttons[i].click(); return; }
                    }
                    throw new Error('no preset ' + sid);
                }""",
                scenario_id,
            )
            page.wait_for_selector("#output .card", timeout=30000)
            page.wait_for_timeout(400)
            target = OUT / filename
            page.screenshot(path=str(target), full_page=True)
            written.append((target, caption))
            print(f"  wrote {target.relative_to(REPO)}  ({caption})")
        browser.close()

    server.should_exit = True
    thread.join(timeout=5)
    print(f"\n{len(written)} screenshot(s) in {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
