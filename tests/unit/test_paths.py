"""safe_join keeps user-supplied paths inside their base (MADIV-28 T1.9)."""

from __future__ import annotations

import pytest

from almasim.services.paths import UnsafePathError, safe_join

pytestmark = pytest.mark.unit


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "data2").mkdir()  # shares the base's name as a prefix
    return root


@pytest.mark.parametrize("raw", ["run1", "run1/../run2", "./a/b", ""])
def test_relative_paths_resolve_under_the_base(base, raw):
    assert safe_join(base, raw).is_relative_to(base.resolve())


def test_absolute_path_inside_the_base_is_kept(base):
    assert safe_join(base, str(base / "x")) == (base / "x").resolve()


def test_the_base_itself_is_allowed(base):
    assert safe_join(base, str(base)) == base.resolve()


@pytest.mark.parametrize("raw", ["..", "../data2", "a/../../data2/x", "/etc", "{sibling}"])
def test_paths_escaping_the_base_are_refused(base, raw):
    with pytest.raises(UnsafePathError):
        safe_join(base, raw.format(sibling=base.parent / "data2"))


def test_symlink_out_of_the_base_is_refused(base, tmp_path):
    (base / "link").symlink_to(tmp_path / "data2")
    with pytest.raises(UnsafePathError):
        safe_join(base, "link/file")


def test_null_byte_is_refused(base):
    with pytest.raises(UnsafePathError):
        safe_join(base, "a\x00b")


def test_home_is_expanded(base, monkeypatch):
    monkeypatch.setenv("HOME", str(base))
    assert safe_join(base, "~/logs") == (base / "logs").resolve()
    assert safe_join("~", "logs") == (base / "logs").resolve()


def test_unsafe_path_error_is_a_value_error():
    assert issubclass(UnsafePathError, ValueError)
