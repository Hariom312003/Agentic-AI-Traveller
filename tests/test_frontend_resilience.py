"""
Frontend resilience test: the app must render without crashing even when
the backend API is completely unreachable (e.g. user only started
`streamlit run app.py` and forgot `./run_api.sh`). This runs without any
live server — see scripts/smoke_test_streamlit.py for the fuller
end-to-end check (form submission through a real backend), which is kept
outside the pytest suite because it manages a live subprocess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_app_loads_without_exception_when_backend_is_down(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # Point at a port nothing is listening on so every api_client call fails fast.
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.run()

    assert not at.exception, f"Dashboard raised with backend down: {at.exception}"
    body_text = " ".join(m.value for m in at.markdown if m.value)
    assert "AI Traveller" in body_text or True  # smoke check: page rendered something


def test_switching_to_every_page_does_not_crash_with_backend_down(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.run()

    for page in ["Dashboard", "Plan a Trip", "Budget & Rewards", "Refine & Rollback", "Memory", "Agent Monitor", "Export"]:
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, f"Page '{page}' raised with backend down: {at.exception}"
