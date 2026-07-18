"""
Structured logging setup.

Two handlers: a Rich console handler for local dev readability, and a
rotating file handler (`logs/app.log`) that the Streamlit Agent Monitor page
tails for the APM-style console. Every call site uses `extra={...}` for
structured fields instead of string-formatting them into the message, so the
same log line is both human-readable in a terminal and machine-parseable in
the monitor view.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False
_LOG_DIR = Path("logs")
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFileFormatter(logging.Formatter):
    """Renders each record as one JSON line, folding in any `extra=` fields,
    so the monitoring dashboard can `json.loads` each line directly."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in payload:
                try:
                    json.dumps(value)
                    payload[key] = value
                except TypeError:
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("ai_traveller")
    root.setLevel(level)
    root.propagate = False

    try:
        from rich.logging import RichHandler

        console_handler: logging.Handler = RichHandler(show_path=False, rich_tracebacks=True)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:  # rich is a soft dependency for the console; file logging still works
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFileFormatter())

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"ai_traveller.{name}")


def read_recent_logs(limit: int = 200) -> list[dict]:
    """Used by the Streamlit Agent Monitor page to render the APM console
    without re-implementing log parsing there."""
    path = _LOG_DIR / "app.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
