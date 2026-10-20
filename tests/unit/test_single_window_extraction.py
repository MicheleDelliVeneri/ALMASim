"""The single-window extraction's table handling, on a fake casacore (MADIV-28 T2.7)."""

from __future__ import annotations

import sys
import types

import pytest

from almasim.services.imaging.ms import single_window

pytestmark = pytest.mark.unit


class FakeTable:
    def __init__(self, columns=None, rows=0):
        self.columns = dict(columns or {})
        self.rows = rows
        self.closed = False
        self.removed: list[int] = []

    def getcol(self, name):
        return self.columns[name]

    def putcol(self, name, values):
        self.columns[name] = list(values)

    def nrows(self):
        return self.rows

    def removerows(self, rows):
        self.removed = list(rows)

    def close(self):
        self.closed = True


class FakeSelection(FakeTable):
    def __init__(self, rows):
        super().__init__(rows=rows)

    def copy(self, path, deep=False):
        import pathlib

        pathlib.Path(path).mkdir(parents=True)
        (pathlib.Path(path) / "table.dat").write_bytes(b"x" * 10)


@pytest.fixture
def fake_casacore(monkeypatch, tmp_path):
    tables: dict[str, FakeTable] = {}
    selections: list[FakeSelection] = []
    counts = {"all": 5, "kept": 3}

    def table(path, **kwargs):
        return tables[str(path)]

    def taql(query, tables=()):
        rows = counts["kept"] if "max(WEIGHT)" in query else counts["all"]
        selection = FakeSelection(rows)
        selections.append(selection)
        return selection

    module = types.ModuleType("casacore.tables")
    module.table = table
    module.taql = taql
    package = types.ModuleType("casacore")
    package.tables = module
    monkeypatch.setitem(sys.modules, "casacore", package)
    monkeypatch.setitem(sys.modules, "casacore.tables", module)

    src, out = tmp_path / "in.ms", tmp_path / "out.ms"
    tables[f"{src}/DATA_DESCRIPTION"] = FakeTable(
        {"SPECTRAL_WINDOW_ID": [3, 5, 5], "POLARIZATION_ID": [0, 1, 1]}
    )
    tables[str(src)] = FakeTable(rows=5)
    tables[f"{out}/DATA_DESCRIPTION"] = FakeTable(rows=3)
    tables[f"{out}/SPECTRAL_WINDOW"] = FakeTable(rows=6)
    tables[str(out)] = FakeTable({"DATA_DESC_ID": [1, 2, 1]})
    return types.SimpleNamespace(
        tables=tables, selections=selections, counts=counts, src=src, out=out
    )


def test_extracts_the_window_and_closes_every_table(fake_casacore):
    fc = fake_casacore
    result = single_window._extract_single_window_ms_once(fc.src, 5, [2], fc.out, max_weight=1000.0)
    assert result["rows"] == 3 and result["dropped_rows"] == 2 and result["data_desc_ids"] == [1, 2]
    assert fc.tables[f"{fc.out}/DATA_DESCRIPTION"].removed == [0]
    assert fc.tables[f"{fc.out}/DATA_DESCRIPTION"].columns["POLARIZATION_ID"] == [1, 1]
    assert fc.tables[f"{fc.out}/SPECTRAL_WINDOW"].removed == [0, 1, 2, 3, 4]
    assert fc.tables[str(fc.out)].columns["DATA_DESC_ID"] == [0, 1, 0]
    assert all(t.closed for t in fc.tables.values()) and all(s.closed for s in fc.selections)


def test_a_window_without_rows_or_data_description_is_an_error(fake_casacore):
    fc = fake_casacore
    with pytest.raises(RuntimeError, match="no DATA_DESCRIPTION row"):
        single_window._extract_single_window_ms_once(fc.src, 9, None, fc.out, max_weight=None)
    fc.counts["all"] = 0
    with pytest.raises(RuntimeError, match="No visibilities"):
        single_window._extract_single_window_ms_once(fc.src, 5, None, fc.out, max_weight=None)
    fc.counts.update(all=5, kept=0)
    with pytest.raises(RuntimeError, match="have WEIGHT >"):
        single_window._extract_single_window_ms_once(fc.src, 5, None, fc.out, max_weight=10.0)
    assert fc.tables[f"{fc.src}/DATA_DESCRIPTION"].closed and fc.tables[str(fc.src)].closed
    assert all(s.closed for s in fc.selections)
