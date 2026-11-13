"""pytest -q tests/test_guidance.py"""
import re, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from guidance.rules import get_guidance, PHASES, GOALS      # noqa: E402
from api.main import app                                     # noqa: E402

client = TestClient(app)
FORBIDDEN = re.compile(r"\\bmg\\b|\\bmcg\\b|kcal|calorie|lose weight|detox|cleanse", re.I)


def all_text(g):
    return " ".join(i["title"] + " " + i["text"] for i in g["items"]) + " ".join(g["when_to_seek_care"])


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("goal", GOALS)
@pytest.mark.parametrize("band", [None, "low", "moderate", "elevated"])
def test_safe_content_everywhere(phase, goal, band):
    g = get_guidance(phase, goal, band)
    assert not FORBIDDEN.search(all_text(g)), "no doses, calorie talk or detox claims"
    assert len(g["items"]) >= 4 and g["disclaimer"] and g["when_to_seek_care"]
    assert all(i["evidence"] in ("guideline", "general", "limited", "safety") for i in g["items"])
    cited = {i["source"] for i in g["items"] if i["source"]}
    assert cited <= {s["key"] for s in g["sources"]}


def test_goal_specific_and_pattern_specific():
    assert any("contraception" in i["title"].lower() for i in get_guidance("baseline", "avoiding_pregnancy")["items"])
    assert not any("contraception" in i["title"].lower() for i in get_guidance("baseline", "tracking")["items"])
    assert any("fertile" in i["text"].lower() for i in get_guidance("surge_window", "trying_to_conceive")["items"])
    assert any("folic" in i["text"].lower() for i in get_guidance("post_surge", "trying_to_conceive")["items"])
    el = get_guidance("baseline", "tracking", "elevated")
    assert any("clinician" in i["category"].lower() for i in el["items"])
    assert any("PCOS is confirmed" in i["title"] for i in el["items"])
    low = get_guidance("baseline", "tracking", "low")
    assert not any("clinician" in i["category"].lower() for i in low["items"])


def test_surge_window_says_diet_does_not_trigger_surge():
    g = get_guidance("surge_window")
    assert any("no good evidence" in i["text"].lower() for i in g["items"])


def test_api_guidance_and_validation():
    r = client.post("/guidance", json={"phase": "surge_window", "goal": "trying_to_conceive"})
    assert r.status_code == 200 and r.json()["items"] and r.json()["headline"]
    assert client.post("/guidance", json={"phase": "nonsense"}).status_code == 422
