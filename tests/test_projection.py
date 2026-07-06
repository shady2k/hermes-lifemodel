from __future__ import annotations

import re

from lifemodel.core.projection import project_contact

THETA = 1.0


def test_bands_map_to_distinct_phrasings() -> None:
    light, _ = project_contact(1.1, theta=THETA, seed="a")
    strong, _ = project_contact(3.0, theta=THETA, seed="a")
    assert light != strong


def test_choice_is_deterministic_in_seed() -> None:
    assert project_contact(2.0, theta=THETA, seed="corr-1") == project_contact(
        2.0, theta=THETA, seed="corr-1"
    )


def test_different_seed_can_vary_phrasing() -> None:
    # across several seeds within a band, more than one synonym is reachable
    outs = {project_contact(2.0, theta=THETA, seed=f"s{i}")[0] for i in range(20)}
    assert len(outs) >= 2


def test_phrasing_contains_no_raw_numbers() -> None:
    for v in (1.1, 2.0, 3.5):
        phrasing, _ = project_contact(v, theta=THETA, seed="x")
        assert not re.search(r"\d", phrasing)  # no digits — never leaks values/hours


def test_projection_id_identifies_choice() -> None:
    phrasing, pid = project_contact(2.0, theta=THETA, seed="k")
    assert isinstance(pid, str) and pid
    # same seed+value -> same id
    assert project_contact(2.0, theta=THETA, seed="k")[1] == pid
