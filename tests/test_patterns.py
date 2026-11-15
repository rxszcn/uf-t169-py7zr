"""Tests for pattern-based member selection on extraction.

The matrix under test combines three axes:

* selection mode: exact ``targets`` only, wildcard ``patterns`` only, and the
  two united in a single call;
* member shape: a root-level file, a file nested in a directory, a directory
  member itself, and a file whose name literally contains wildcard chars;
* the ``recursive`` switch, which extends a selected directory name to its
  whole subtree.

Every selection scenario is exercised twice, through the on-disk path and
through the in-memory ``WriterFactory`` path, since both share the same
filtering code in ``SevenZipFile._extract``.
"""

from io import BytesIO

import pytest

import py7zr
from py7zr.helpers import member_is_selected

MEMBERS: dict[str, bytes] = {
    "logs/app.log": b"app log\n",
    "logs/system.log": b"system log\n",
    "data/manifest.json": b"{}\n",
    "data/notes.txt": b"notes\n",
    "readme.txt": b"readme\n",
    "weird[1].dat": b"literal\n",
}

DIRECTORY_MEMBERS = ("logs", "data")


@pytest.fixture
def archive_buffer(tmp_path_factory):
    """Build an in-memory archive with root, nested, directory and odd members."""
    src = tmp_path_factory.mktemp("src")
    for name, payload in MEMBERS.items():
        member = src / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(payload)
    buf = BytesIO()
    with py7zr.SevenZipFile(buf, "w") as archive:
        for directory in DIRECTORY_MEMBERS:
            archive.write(src / directory, directory)
        for name, payload in MEMBERS.items():
            archive.writef(BytesIO(payload), name)
    buf.seek(0)
    return buf


def _extracted_names(root) -> set:
    names = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        names.add(rel)
    return names


def _factory_names(archive_buffer, **kwargs) -> set:
    factory = py7zr.io.BytesIOFactory(1024)
    with py7zr.SevenZipFile(archive_buffer) as archive:
        archive.extract(factory=factory, **kwargs)
    return set(factory.products)


class TestMemberIsSelected:
    """Unit tests of the selection predicate shared by both extraction paths."""

    def test_no_selector_selects_everything(self):
        assert member_is_selected("any/name", None, None, False) is True
        assert member_is_selected("any/name", None, None, True) is True

    @pytest.mark.parametrize(
        "name,pattern,expected",
        [
            ("readme.txt", "*.txt", True),
            ("readme.txt", "*.log", False),
            ("logs/app.log", "*.log", True),
            ("logs/app.log", "logs/*.log", True),
            ("logs/app.log", "logs/app.l?g", True),
            ("logs/app.log", "logs/app.X?g", False),
            ("data/manifest.json", "data/manifest.[jJ]son", True),
            ("data/manifest.json", "data/manifest.[!j]son", False),
            ("data/manifest.json", "data/manifest.[a-j]son", True),
            # A pattern is a glob, so a bracket expression denotes a class.
            ("weird[1].dat", "weird[1].dat", False),
            ("weird1.dat", "weird[1].dat", True),
        ],
    )
    def test_wildcard_semantics_non_recursive(self, name, pattern, expected):
        assert member_is_selected(name, None, [pattern], False) is expected

    def test_case_sensitive_matching(self):
        assert member_is_selected("Readme.TXT", None, ["*.txt"], False) is False
        assert member_is_selected("Readme.TXT", None, ["*.TXT"], False) is True

    def test_targets_are_always_literal(self):
        # The literal name is selected by targets ...
        assert member_is_selected("weird[1].dat", ["weird[1].dat"], None, False) is True
        # ... and is not mistaken for a glob that matches "weird1.dat".
        assert member_is_selected("weird1.dat", ["weird[1].dat"], None, False) is False
        # A wildcard string in targets is literal too.
        assert member_is_selected("readme.txt", ["*.txt"], None, False) is False

    @pytest.mark.parametrize("recursive", [False, True])
    def test_exact_target_recursive_equality(self, recursive):
        assert member_is_selected("readme.txt", ["readme.txt"], None, recursive) is True

    def test_exact_target_non_recursive_keeps_subtree_out(self):
        assert member_is_selected("logs/app.log", ["logs"], None, False) is False

    def test_exact_target_recursive_selects_subtree(self):
        assert member_is_selected("logs/app.log", ["logs"], None, True) is True
        assert member_is_selected("logs", ["logs"], None, True) is True
        # A sibling sharing the prefix is not inside the selected directory.
        assert member_is_selected("logs2/app.log", ["logs"], None, True) is False

    def test_pattern_non_recursive_does_not_descend(self):
        assert member_is_selected("logs/app.log", None, ["logs"], False) is False
        assert member_is_selected("logs", None, ["logs"], False) is True

    def test_pattern_recursive_selects_matching_directory_subtree(self):
        assert member_is_selected("logs/app.log", None, ["logs"], True) is True
        assert member_is_selected("logs/system.log", None, ["logs"], True) is True
        assert member_is_selected("logs2/app.log", None, ["logs"], True) is False
        assert member_is_selected("logs/app.log", None, ["log?"], True) is True
        assert member_is_selected("data/manifest.json", None, ["*data*"], True) is True

    def test_targets_and_patterns_are_united(self):
        selected = lambda name: member_is_selected(name, ["readme.txt"], ["logs/*.log"], False)  # noqa: E731
        assert selected("readme.txt") is True
        assert selected("logs/app.log") is True
        assert selected("data/notes.txt") is False


class TestExtractPatterns:
    """End-to-end extraction using patterns, on disk and through a factory."""

    def test_star_extension_root_and_nested(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["*.log"])
        names = _extracted_names(tmp_path)
        # "logs" is created as the parent of the selected members.
        assert names == {"logs", "logs/app.log", "logs/system.log"}

    def test_star_extension_factory(self, archive_buffer):
        names = _factory_names(archive_buffer, patterns=["*.log"])
        assert names == {"logs/app.log", "logs/system.log"}

    def test_question_mark(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["logs/app.l?g"])
        assert _extracted_names(tmp_path) == {"logs", "logs/app.log"}

    def test_bracket_character_class(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["data/manifest.[j]son"])
        assert _extracted_names(tmp_path) == {"data", "data/manifest.json"}

    def test_bracket_negated_class_excludes_member(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["*.tx?"])
        assert _extracted_names(tmp_path) == {"readme.txt", "data", "data/notes.txt"}
        assert not tmp_path.joinpath("data/manifest.json").exists()

    def test_several_patterns_are_united(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["*.txt", "*.json"])
        names = _extracted_names(tmp_path)
        assert names == {"readme.txt", "data/manifest.json", "data", "data/notes.txt"}

    def test_patterns_factory_matches_disk_selection(self, archive_buffer):
        names = _factory_names(archive_buffer, patterns=["*.txt", "*.json"])
        assert names == {"readme.txt", "data/manifest.json", "data/notes.txt"}

    def test_pattern_on_directory_name_non_recursive_selects_only_dir(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["logs"], recursive=False)
        assert tmp_path.joinpath("logs").is_dir()
        assert _extracted_names(tmp_path) == {"logs"}

    def test_pattern_on_directory_name_recursive_selects_subtree(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["logs"], recursive=True)
        assert _extracted_names(tmp_path) == {"logs", "logs/app.log", "logs/system.log"}

    def test_wildcard_directory_pattern_recursive(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["dat?"], recursive=True)
        assert _extracted_names(tmp_path) == {"data/notes.txt", "data", "data/manifest.json"}

    def test_recursive_pattern_factory_path(self, archive_buffer):
        names = _factory_names(archive_buffer, patterns=["dat?"], recursive=True)
        assert names == {"data/manifest.json", "data/notes.txt"}

    def test_recursive_star_selects_everything(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["*"], recursive=True)
        assert _extracted_names(tmp_path) == set(DIRECTORY_MEMBERS) | set(MEMBERS)

    def test_exact_target_non_recursive(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, targets=["readme.txt"])
        assert _extracted_names(tmp_path) == {"readme.txt"}

    def test_exact_directory_target_non_recursive(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, targets=["logs"])
        assert _extracted_names(tmp_path) == {"logs"}

    def test_exact_directory_target_recursive(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, targets=["logs"], recursive=True)
        assert _extracted_names(tmp_path) == {"logs", "logs/app.log", "logs/system.log"}

    def test_recursive_target_keeps_sibling_prefix_directory_out(self, tmp_path):
        buf = BytesIO()
        with py7zr.SevenZipFile(buf, "w") as archive:
            archive.writef(BytesIO(b"inside"), "logs/a.log")
            archive.writef(BytesIO(b"sibling"), "logs2/b.log")
        buf.seek(0)
        out = tmp_path / "out"
        with py7zr.SevenZipFile(buf) as archive:
            archive.extract(path=out, targets=["logs"], recursive=True)
        names = _extracted_names(out)
        assert "logs/a.log" in names
        assert "logs2/b.log" not in names
        assert "logs2" not in names

    def test_exact_target_factory_path(self, archive_buffer):
        names = _factory_names(archive_buffer, targets=["logs"], recursive=True)
        assert names == {"logs/app.log", "logs/system.log"}

    def test_targets_and_patterns_united_disk(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, targets=["readme.txt"], patterns=["logs/*.log"])
        assert _extracted_names(tmp_path) == {"logs", "readme.txt", "logs/app.log", "logs/system.log"}

    def test_targets_and_patterns_united_factory(self, archive_buffer):
        names = _factory_names(archive_buffer, targets=["readme.txt"], patterns=["logs/*.log"])
        assert names == {"readme.txt", "logs/app.log", "logs/system.log"}

    def test_literal_wildcard_filename_needs_target(self, archive_buffer, tmp_path):
        # A name containing wildcard characters is selected literally via targets.
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, targets=["weird[1].dat"])
        assert _extracted_names(tmp_path) == {"weird[1].dat"}

    def test_literal_wildcard_filename_via_factory(self, archive_buffer):
        names = _factory_names(archive_buffer, targets=["weird[1].dat"])
        assert names == {"weird[1].dat"}

    def test_bracket_pattern_matches_glob_style_name_instead(self, archive_buffer, tmp_path):
        # As a pattern, "[1]" is a character class: it matches "weird1.dat",
        # which this archive does not contain, so nothing matches.
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError):
                archive.extract(path=tmp_path, patterns=["weird[1].dat"])
        assert _extracted_names(tmp_path) == set()

    def test_no_match_raises_keyerror_disk(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError, match="No archive member matches"):
                archive.extract(path=tmp_path, patterns=["missing/*.bin"])
        assert _extracted_names(tmp_path) == set()

    def test_no_match_raises_keyerror_factory(self, archive_buffer):
        factory = py7zr.io.BytesIOFactory(1024)
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError):
                archive.extract(patterns=["missing/*.bin"], factory=factory)
        assert factory.products == {}

    def test_no_match_unknown_exact_target_raises(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError):
                archive.extract(path=tmp_path, targets=["does/not/exist.txt"])

    def test_no_match_does_not_consume_archive(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError):
                archive.extract(path=tmp_path, patterns=["missing/*.bin"])
            archive.extract(path=tmp_path, patterns=["readme.txt"])
        assert tmp_path.joinpath("readme.txt").read_bytes() == MEMBERS["readme.txt"]

    def test_no_match_raises_before_callback_starts(self, archive_buffer, tmp_path):
        class RecordingCallback(py7zr.callbacks.ExtractCallback):
            def __init__(self):
                self.events = []

            def report_start_preparation(self):
                self.events.append("pre")

            def report_start(self, processing_file_path, processing_bytes):
                self.events.append("start")

            def report_update(self, decompressed_bytes):
                self.events.append("update")

            def report_end(self, processing_file_path, wrote_bytes):
                self.events.append("end")

            def report_postprocess(self):
                self.events.append("post")

            def report_warning(self, message):
                self.events.append("warning")

        callback = RecordingCallback()
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(KeyError):
                archive.extract(path=tmp_path, patterns=["missing/*.bin"], callback=callback)
        assert callback.events == []
        assert archive.reporterd is None or not archive.reporterd.is_alive()

    @pytest.mark.parametrize("bad", ["*.txt", ("*.txt",), 1])
    def test_patterns_reject_string_tuple_and_scalar(self, archive_buffer, bad):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            with pytest.raises(TypeError, match="patterns must be a list or set"):
                archive.extract(patterns=bad)

    def test_patterns_accept_set(self, archive_buffer):
        names = _factory_names(archive_buffer, patterns={"*.json"})
        assert names == {"data/manifest.json"}

    def test_trailing_slash_stripped_from_patterns(self, archive_buffer, tmp_path):
        with py7zr.SevenZipFile(archive_buffer) as archive:
            archive.extract(path=tmp_path, patterns=["logs/"], recursive=True)
        assert _extracted_names(tmp_path) == {"logs", "logs/app.log", "logs/system.log"}
