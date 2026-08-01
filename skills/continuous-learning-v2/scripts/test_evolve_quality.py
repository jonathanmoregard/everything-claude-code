"""Tests for evolve-quality changes in instinct-cli.py

Part A: evolved-skill dedupe — Jaccard overlap append-vs-create, frontmatter
        evolved_from merge, 50KB size cap, 20-line evidence cap.
"""

import importlib.util
import os
from pathlib import Path

import pytest

# Load instinct-cli.py (hyphenated filename requires importlib)
_spec = importlib.util.spec_from_file_location(
    "instinct_cli_evolve_quality",
    os.path.join(os.path.dirname(__file__), "instinct-cli.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


# ─────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────

def _make_instinct(iid, action="Do the thing.", confidence=0.8, domain="testing"):
    return {
        "id": iid,
        "trigger": f"when {iid}",
        "confidence": confidence,
        "domain": domain,
        "content": f"## Action\n{action}",
    }


def _make_candidate(instincts, domain="testing", prefix="foo-bar"):
    return {
        "trigger": f"{domain}:{prefix}-*",
        "instincts": instincts,
        "avg_confidence": sum(i.get("confidence", 0.5) for i in instincts) / len(instincts),
        "domains": [domain],
        "scopes": ["project"],
    }


def _write_existing_skill(skills_dir, name, evolved_from, evidence_lines, filler=""):
    """Author an existing generated skill file in the current output format."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    text = "---\n"
    text += f"name: {name}\n"
    text += "evolved_from:\n"
    for iid in evolved_from:
        text += f"  - {iid}\n"
    text += "---\n"
    text += f"# {name}\n\n"
    text += "Evolved from instincts (avg confidence: 80%)\n\n"
    if filler:
        text += filler + "\n\n"
    text += "## When to Apply\n\n"
    text += f"Trigger: testing:{name}-*\n\n"
    text += "## Actions\n\n"
    for line in evidence_lines:
        text += line + "\n"
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir / "SKILL.md"


@pytest.fixture
def evolved_dir(tmp_path):
    d = tmp_path / "evolved"
    for sub in ["skills", "commands", "agents"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────
# Part A: Jaccard overlap computation
# ─────────────────────────────────────────────

def test_jaccard_identical_sets():
    assert _mod._jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint_sets():
    assert _mod._jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_empty_sets():
    assert _mod._jaccard(set(), set()) == 0.0


def test_jaccard_point_eight():
    shared = {f"s{i}" for i in range(4)}
    assert _mod._jaccard(shared | {"extra"}, shared) == pytest.approx(0.8)


def test_jaccard_point_seven_nine():
    shared = {f"s{i}" for i in range(79)}
    existing = shared | {f"e{i}" for i in range(21)}
    assert _mod._jaccard(shared, existing) == pytest.approx(0.79)


# ─────────────────────────────────────────────
# Part A: append-target selection at the 0.8 threshold
# ─────────────────────────────────────────────

def test_find_append_target_below_threshold_returns_none(evolved_dir):
    shared = [f"s{i}" for i in range(79)]
    existing_ids = shared + [f"e{i}" for i in range(21)]  # jaccard 79/100 = 0.79
    _write_existing_skill(evolved_dir / "skills", "existing-skill", existing_ids, ["- old evidence"])
    assert _mod._find_append_target(evolved_dir / "skills", set(shared)) is None


def test_find_append_target_at_threshold_returns_path(evolved_dir):
    shared = [f"s{i}" for i in range(80)]
    existing_ids = shared + [f"e{i}" for i in range(20)]  # jaccard 80/100 = 0.8
    path = _write_existing_skill(evolved_dir / "skills", "existing-skill", existing_ids, ["- old evidence"])
    assert _mod._find_append_target(evolved_dir / "skills", set(shared)) == path


def test_find_append_target_empty_dir(evolved_dir):
    assert _mod._find_append_target(evolved_dir / "skills", {"a"}) is None


# ─────────────────────────────────────────────
# Part A: _generate_evolved append vs create behavior
# ─────────────────────────────────────────────

def test_generate_appends_on_high_overlap(evolved_dir, capsys):
    existing_ids = ["foo-bar-a", "foo-bar-b", "foo-bar-c", "foo-bar-d"]
    path = _write_existing_skill(
        evolved_dir / "skills", "testing-foo-bar", existing_ids,
        ["- Old evidence line."])

    # intersection 4, union 5 -> 0.8 -> append
    instincts = [_make_instinct(i) for i in existing_ids + ["foo-bar-e"]]
    cand = _make_candidate(instincts)

    generated = _mod._generate_evolved([cand], [], [], evolved_dir)

    # No sibling created
    subdirs = [d for d in (evolved_dir / "skills").iterdir() if d.is_dir()]
    assert len(subdirs) == 1
    assert str(path) in generated

    text = path.read_text(encoding="utf-8")
    parsed = _mod._parse_evolved_skill_text(text)
    assert parsed["evolved_from"] == existing_ids + ["foo-bar-e"]
    assert "- Old evidence line." in text
    assert "Do the thing." in text


def test_generate_creates_on_low_overlap(evolved_dir):
    _write_existing_skill(
        evolved_dir / "skills", "testing-x-y",
        ["x-y-a", "x-y-b", "x-y-c"], ["- Old evidence line."])

    # intersection 1, union 5 -> 0.2 -> create sibling
    instincts = [_make_instinct(i) for i in ["x-y-a", "q-r-one", "q-r-two"]]
    cand = _make_candidate(instincts, prefix="q-r")

    _mod._generate_evolved([cand], [], [], evolved_dir)

    subdirs = [d for d in (evolved_dir / "skills").iterdir() if d.is_dir()]
    assert len(subdirs) == 2


def test_generate_new_skill_writes_evolved_from_frontmatter(evolved_dir):
    instincts = [_make_instinct(i) for i in ["foo-bar-a", "foo-bar-b"]]
    cand = _make_candidate(instincts)

    generated = _mod._generate_evolved([cand], [], [], evolved_dir)
    assert len(generated) == 1

    text = Path(generated[0]).read_text(encoding="utf-8")
    parsed = _mod._parse_evolved_skill_text(text)
    assert parsed["evolved_from"] == ["foo-bar-a", "foo-bar-b"]


def test_append_merge_dedupes_ids(evolved_dir):
    existing_ids = ["foo-bar-a", "foo-bar-b", "foo-bar-c", "foo-bar-d"]
    path = _write_existing_skill(
        evolved_dir / "skills", "testing-foo-bar", existing_ids,
        ["- Old evidence line."])

    instincts = [_make_instinct(i) for i in existing_ids + ["foo-bar-e"]]
    cand = _make_candidate(instincts)
    _mod._generate_evolved([cand], [], [], evolved_dir)

    parsed = _mod._parse_evolved_skill_text(path.read_text(encoding="utf-8"))
    assert len(parsed["evolved_from"]) == len(set(parsed["evolved_from"]))
    assert sorted(parsed["evolved_from"]) == sorted(set(existing_ids + ["foo-bar-e"]))


def test_append_refused_over_size_cap(evolved_dir, capsys):
    existing_ids = ["foo-bar-a", "foo-bar-b", "foo-bar-c", "foo-bar-d"]
    filler = "X" * (_mod.EVOLVED_SKILL_MAX_BYTES + 100)
    path = _write_existing_skill(
        evolved_dir / "skills", "testing-foo-bar", existing_ids,
        ["- Old evidence line."], filler=filler)
    before = path.read_text(encoding="utf-8")

    instincts = [_make_instinct(i) for i in existing_ids + ["foo-bar-e"]]
    cand = _make_candidate(instincts)
    generated = _mod._generate_evolved([cand], [], [], evolved_dir)

    assert generated == []
    assert path.read_text(encoding="utf-8") == before  # unchanged
    out = capsys.readouterr().out
    assert "refus" in out.lower()
    # No sibling was created either
    subdirs = [d for d in (evolved_dir / "skills").iterdir() if d.is_dir()]
    assert len(subdirs) == 1


def test_append_trims_evidence_to_cap_keeping_newest(evolved_dir, capsys):
    # 40 existing ids, cluster carries 36 of them + 4 new -> jaccard 36/44 ≈ 0.818
    existing_ids = [f"foo-bar-{i:02d}" for i in range(40)]
    old_evidence = [f"- old evidence {i:02d}" for i in range(18)]
    path = _write_existing_skill(
        evolved_dir / "skills", "testing-foo-bar", existing_ids, old_evidence)

    new_ids = [f"foo-bar-new-{i}" for i in range(4)]
    instincts = [_make_instinct(i) for i in existing_ids[:36]]
    instincts += [_make_instinct(i, action=f"Do new thing {i}.") for i in new_ids]
    cand = _make_candidate(instincts)
    _mod._generate_evolved([cand], [], [], evolved_dir)

    text = path.read_text(encoding="utf-8")
    evidence = _mod._extract_evidence_lines(_mod._parse_evolved_skill_text(text)["body"])
    assert len(evidence) == _mod.EVOLVED_SKILL_MAX_EVIDENCE_LINES
    # Newest kept: the 4 new lines are present, the 2 oldest were dropped
    for iid in new_ids:
        assert f"- Do new thing {iid}." in evidence
    assert "- old evidence 00" not in text
    assert "- old evidence 01" not in text
    assert "- old evidence 02" in text
    out = capsys.readouterr().out
    assert "capped" in out.lower() or "trim" in out.lower()


def test_cap_evidence_lines_keeps_newest():
    lines = [f"- line {i}" for i in range(25)]
    capped = _mod._cap_evidence_lines(lines)
    assert len(capped) == _mod.EVOLVED_SKILL_MAX_EVIDENCE_LINES
    assert capped == lines[-_mod.EVOLVED_SKILL_MAX_EVIDENCE_LINES:]


def test_cap_evidence_lines_noop_under_cap():
    lines = [f"- line {i}" for i in range(5)]
    assert _mod._cap_evidence_lines(lines) == lines


def test_create_refused_over_size_cap(evolved_dir, capsys):
    # A single enormous action pushes the rendered new file over the cap
    big_action = "Y" * (_mod.EVOLVED_SKILL_MAX_BYTES + 100)
    instincts = [
        _make_instinct("foo-bar-a", action=big_action),
        _make_instinct("foo-bar-b"),
    ]
    cand = _make_candidate(instincts)
    generated = _mod._generate_evolved([cand], [], [], evolved_dir)
    assert generated == []
    out = capsys.readouterr().out
    assert "refus" in out.lower()
