from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github/workflows/install-e2e.yml"
_UPSTREAM_SCHEDULE_GUARD = (
    "github.event_name != 'schedule' || "
    "github.repository == 'NousResearch/hermes-agent'"
)


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.load(_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_install_e2e_schedule_is_inert_on_forks_without_disabling_other_triggers():
    workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_dispatch", "schedule", "push"}

    jobs = workflow["jobs"]
    assert jobs["pick-releases"]["if"] == _UPSTREAM_SCHEDULE_GUARD
    assert jobs["leg-player"]["if"] == _UPSTREAM_SCHEDULE_GUARD
    assert jobs["report"]["if"] == f"always() && ({_UPSTREAM_SCHEDULE_GUARD})"
