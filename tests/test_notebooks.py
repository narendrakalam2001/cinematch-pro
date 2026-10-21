"""Static and executable checks for notebooks/Kaggle_Training.ipynb.

The notebook shells out to the project's scripts, so a renamed flag would break it silently
(only on Kaggle, hours into a run). These tests keep the notebook and the scripts in sync.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "Kaggle_Training.ipynb"


@pytest.fixture(scope="module")
def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _shell_commands(code_cells: list[str]) -> list[str]:
    """All `!python -m ...` commands, with backslash continuations joined."""
    commands = []
    for cell in code_cells:
        for command in re.findall(r"^!python -m (.+?)(?<!\\)$", cell.replace("\\\n", " "), flags=re.M):
            commands.append(command)
    return commands


def test_code_cells_are_numbered_sequentially(code_cells: list[str]):
    """Numbered cells ('# CELL n') count 1..N in order, matching the instructions given to users.

    Sub-cells such as '# CELL 8b' may follow their parent, and un-numbered helper cells
    ('%%writefile ...' or '# CELL: patch ...') are ignored.
    """
    labels = [m.groups() for c in code_cells if (m := re.match(r"# CELL (\d+)([a-z]?) ", c))]
    base_numbers = [int(number) for number, suffix in labels if not suffix]
    assert base_numbers == list(range(1, len(base_numbers) + 1))
    order = [(int(number), suffix) for number, suffix in labels]
    assert order == sorted(order), "sub-cells must come right after their parent cell"


def test_every_script_called_by_the_notebook_exists_and_accepts_its_flags(code_cells: list[str]):
    """`!python -m module --flag ...` must refer to a real module that defines each flag."""
    checked_modules: set[str] = set()
    for command in _shell_commands(code_cells):
        module, *rest = command.split()
        if module == "pytest":
            continue
        source_file = ROOT / (module.replace(".", "/") + ".py")
        assert source_file.exists(), f"notebook runs missing module {module}"
        source = source_file.read_text(encoding="utf-8")
        for flag in re.findall(r"(--[a-z][a-z-]*)", " ".join(rest)):
            assert f'"{flag}"' in source, f"{module} has no {flag} option, but the notebook passes it"
        checked_modules.add(module)
    core = {
        "src.preprocessing", "scripts.build_content_model", "scripts.ingest_descriptions", "scripts.build_cf_model",
        "scripts.build_faiss_index", "scripts.train_sentiment", "scripts.check_artifacts",
    }
    assert core <= checked_modules  # `scripts.enrich_credits` (TMDB credits fix) may be run in addition


def test_notebook_only_interpolates_variables_it_defines(code_cells: list[str]):
    """Every {NAME} used inside a `!` command is assigned in an earlier cell."""
    defined: set[str] = set()
    for cell in code_cells:
        for line in cell.splitlines():
            if line.startswith("!"):
                for name in re.findall(r"\{([A-Za-z_]\w*)\}", line):
                    assert name in defined, f"{{{name}}} is used before it is defined"
        defined.update(re.findall(r"^([A-Z_]+)\s*=", cell, flags=re.M))


def _with_run_mode(cell: str, mode: str) -> str:
    """Return CELL 3's code with RUN_MODE set to `mode` (whatever value the notebook ships with).

    The shipped FULL notebook also locks itself with `assert RUN_MODE == "FULL"`; that safety line
    is dropped here so every mode can be exercised.
    """
    cell = re.sub(r'RUN_MODE = "[A-Za-z]*"', f'RUN_MODE = "{mode}"', cell, count=1)
    return "\n".join(line for line in cell.splitlines() if not line.startswith("assert RUN_MODE =="))


@pytest.mark.parametrize("mode, sentiment_args, poster_args", [
    ("QUICK", "--sample 5000", "--limit 2000"),
    ("FULL", "", ""),
])
def test_run_mode_switch_controls_sizes(code_cells: list[str], mode: str, sentiment_args: str, poster_args: str):
    """QUICK caps sentiment samples and posters; FULL passes no cap flags at all."""
    cell = next(c for c in code_cells if c.startswith("# CELL 3"))
    namespace: dict = {}
    exec(_with_run_mode(cell, mode), namespace)
    assert namespace["SENT_ARGS"] == sentiment_args
    assert namespace["POSTER_ARGS"] == poster_args


def test_invalid_run_mode_is_rejected(code_cells: list[str]):
    """A typo in RUN_MODE fails immediately instead of mid-run."""
    cell = next(c for c in code_cells if c.startswith("# CELL 3"))
    with pytest.raises(AssertionError):
        exec(_with_run_mode(cell, "MEDIUM"), {})


@pytest.fixture
def isolated_project_import():
    """Run notebook cells that `import src` from an unpacked copy without leaking into other tests."""
    saved_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    saved_path, saved_cwd = list(sys.path), os.getcwd()
    for name in saved_modules:
        del sys.modules[name]
    yield
    for name in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[name]
    sys.modules.update(saved_modules)
    sys.path[:] = saved_path
    os.chdir(saved_cwd)


@pytest.mark.parametrize("layout", ["folder", "zip", "nested_zip"])
def test_data_discovery_and_project_unpack_cells_work_on_kaggle_layouts(
    code_cells: list[str], tmp_path: Path, isolated_project_import, layout: str
):
    """CELL 4 + CELL 5 find all data files and unpack the project however Kaggle presents it."""
    dataset = tmp_path / "input" / "my-data"
    (dataset / "ml-25m").mkdir(parents=True)
    for name in ("ratings.csv", "movies.csv", "links.csv"):
        (dataset / "ml-25m" / name).write_text("x")
    # A decoy MovieLens-like folder that must NOT win over the 25M one.
    (tmp_path / "input" / "other" / "ml-latest-small").mkdir(parents=True)
    (tmp_path / "input" / "other" / "ml-latest-small" / "ratings.csv").write_text("decoy")
    (dataset / "IMDB Dataset.csv").write_text("x")
    (dataset / "IMDB TMDB Movie Metadata Big Dataset (1M).csv").write_text("x")
    (dataset / "MovieGenre.csv").write_text("x")

    if layout == "folder":
        for part in ("src", "scripts"):
            shutil.copytree(ROOT / part, dataset / part, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        prefix = Path("cinematch-pro-complete") if layout == "nested_zip" else Path("")
        with zipfile.ZipFile(dataset / "cinematch-pro-complete.zip", "w") as archive:
            for part in ("src", "scripts"):
                for file in (ROOT / part).rglob("*.py"):
                    archive.write(file, prefix / file.relative_to(ROOT))

    working = tmp_path / "working"
    working.mkdir()
    namespace: dict = {}
    discover = next(c for c in code_cells if c.startswith("# CELL 4"))
    unpack = next(c for c in code_cells if c.startswith("# CELL 5"))
    exec(discover.replace("/kaggle/input", (tmp_path / "input").as_posix()), namespace)
    exec(unpack.replace("/kaggle/working", working.as_posix()), namespace)

    assert "ml-25m" in str(namespace["RATINGS"])  # chose the 25M file, not the decoy
    assert namespace["PROJECT"] == working / "cinematch-pro"
    assert str(namespace["BASE_DIR"]) == str(working / "cinematch-pro")
