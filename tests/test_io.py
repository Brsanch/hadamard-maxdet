import json
from pathlib import Path

from hadamard_maxdet import Matrix
from hadamard_maxdet.io import (
    matrix_to_submission,
    read_submission,
    submission_from_dict,
    write_submission,
)


def test_round_trip(tmp_path: Path):
    M = Matrix.sylvester(4)
    p = tmp_path / "H4.json"
    write_submission(M, "H4", p)
    oid, n, entries = read_submission(p)
    assert oid == "H4"
    assert n == 4
    assert entries == M.entries


def test_matrix_to_submission_shape():
    M = Matrix.sylvester(4)
    payload = matrix_to_submission(M, "H4")
    assert payload["order_id"] == "H4"
    assert payload["n"] == 4
    assert payload["abs_det"] == 16
    assert len(payload["entries"]) == 4
    assert all(len(r) == 4 for r in payload["entries"])


def test_submission_rejects_missing_keys():
    import pytest

    with pytest.raises(ValueError, match="missing"):
        submission_from_dict({"order_id": "H4", "n": 4})


def test_submission_rejects_wrong_n():
    import pytest

    data = {"order_id": "H4", "n": 5, "entries": [[1, 1], [1, -1]]}
    with pytest.raises(ValueError, match="n=5"):
        submission_from_dict(data)


def test_submission_json_is_human_readable(tmp_path: Path):
    M = Matrix.sylvester(2)
    p = tmp_path / "H2.json"
    write_submission(M, "H2", p)
    # Ensure it parses as JSON with expected top-level keys.
    data = json.loads(p.read_text())
    assert data["order_id"] == "H2"
    assert data["n"] == 2
    assert data["abs_det"] == 2
