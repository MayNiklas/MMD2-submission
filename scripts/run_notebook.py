"""Execute a Jupyter notebook (or a cell-index range) headless via nbclient.

Built for long, unattended runs inside a tmux session: it executes the chosen cells in a single
kernel, streams each cell's output to stdout as it finishes, and re-saves the executed notebook
after every cell so a crash or kill still leaves partial results on disk.

Examples
--------
List the cells with their 0-based indices so you can pick a range::

    python scripts/run_notebook.py notebooks/brighter_emotion_mvp_qwen.ipynb --list

Run the whole notebook in place (overwrites the original with executed outputs)::

    python scripts/run_notebook.py notebooks/brighter_emotion_mvp_qwen.ipynb --all --inplace

Run only "everything after training" (e.g. cells 10..end) into a timestamped copy::

    python scripts/run_notebook.py notebooks/brighter_emotion_mvp_qwen.ipynb --start 10

Overnight pattern in tmux::

    tmux new -s mmd2
    conda activate mmd2-emotion
    python scripts/run_notebook.py notebooks/brighter_emotion_mvp_qwen.ipynb --all \
        --inplace --timeout 0 2>&1 | tee outputs/run_$(date +%Y%m%d_%H%M%S).log
    # detach with Ctrl-b d; reattach later with: tmux attach -t mmd2

Important  -  kernel state caveat
-------------------------------
All selected code cells run in ONE fresh kernel, in document order. A cell range only sees state
created by *earlier cells in that same range*. So running "everything after training" in a fresh
kernel will fail on names like ``trainer`` / ``tokenizer`` unless you recreate them first. Use
``--bootstrap path/to/setup.py`` to inject a setup cell (run before the range) that, for instance,
loads the saved fine-tuned model from ``outputs/qwen3.5-0.8B_model`` and rebinds ``trainer``,
``tokenizer``, ``config`` and ``splits``. If you need the *trained* weights, the range must either
include the training cells or your bootstrap must load the saved model  -  re-running the load-model
cell alone gives the untrained base model.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Execute a notebook or a cell-index range headless via nbclient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb to execute.")
    parser.add_argument("--list", action="store_true", help="List cells with indices and exit.")
    parser.add_argument("--all", action="store_true", help="Execute every code cell.")
    parser.add_argument("--start", type=int, default=None, help="First cell index to run (0-based, inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last cell index to run (0-based, inclusive).")
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=None,
        help="Optional .py file injected as a setup cell that runs in the same kernel before the range.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the executed notebook. Default: a timestamped copy next to the input.",
    )
    parser.add_argument("--inplace", action="store_true", help="Write executed outputs back into the input file.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-cell timeout in seconds. 0 or negative means no timeout (for long training cells).",
    )
    parser.add_argument("--allow-errors", action="store_true", help="Keep going after a cell raises.")
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Kernel name to use. Default: the notebook's own kernelspec, falling back to python3.",
    )
    return parser.parse_args(argv)


def resolve_kernel_name(notebook: nbformat.NotebookNode, override: str | None) -> str:
    """Pick the kernel name, preferring an explicit override then the notebook's kernelspec."""

    if override:
        return override
    kernelspec: dict = notebook.get("metadata", {}).get("kernelspec", {})
    return str(kernelspec.get("name", "python3"))


def first_source_line(cell: nbformat.NotebookNode) -> str:
    """Return a one-line preview of a cell's source for the listing."""

    source: str = cell.get("source", "")
    for line in source.splitlines():
        stripped: str = line.strip()
        if stripped:
            return stripped[:90]
    return "(empty)"


def list_cells(notebook: nbformat.NotebookNode) -> None:
    """Print every cell with its index, type, and a source preview."""

    for index, cell in enumerate(notebook.cells):
        print(f"[{index:>3}] {cell.cell_type:<8} {first_source_line(cell)}")


def select_indices(notebook: nbformat.NotebookNode, arguments: argparse.Namespace) -> list[int]:
    """Resolve the requested cell range into a sorted list of indices to execute."""

    cell_count: int = len(notebook.cells)
    if arguments.all:
        start_index, end_index = 0, cell_count - 1
    else:
        start_index = arguments.start if arguments.start is not None else 0
        end_index = arguments.end if arguments.end is not None else cell_count - 1
    if start_index < 0 or end_index >= cell_count or start_index > end_index:
        raise SystemExit(
            f"Invalid range: start={start_index}, end={end_index} for a notebook with {cell_count} cells."
        )
    return list(range(start_index, end_index + 1))


def build_bootstrap_cell(bootstrap_path: Path) -> nbformat.NotebookNode:
    """Read a .py file into a fresh code cell used to seed kernel state before the range runs."""

    source: str = bootstrap_path.read_text(encoding="utf-8")
    return nbformat.v4.new_code_cell(source=source)


def resolve_output_path(arguments: argparse.Namespace) -> Path:
    """Decide where the executed notebook is written."""

    if arguments.inplace:
        return arguments.notebook
    if arguments.output is not None:
        return arguments.output
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return arguments.notebook.with_name(f"{arguments.notebook.stem}.executed_{timestamp}.ipynb")


def print_cell_outputs(cell: nbformat.NotebookNode) -> None:
    """Echo a finished cell's textual outputs so tmux logs stay informative."""

    for output in cell.get("outputs", []):
        output_type: str = output.get("output_type", "")
        if output_type == "stream":
            sys.stdout.write(output.get("text", ""))
        elif output_type in ("execute_result", "display_data"):
            text: str = output.get("data", {}).get("text/plain", "")
            if text:
                print(text)
        elif output_type == "error":
            print("\n".join(output.get("traceback", [])))
    sys.stdout.flush()


def execute_notebook(arguments: argparse.Namespace) -> int:
    """Execute the selected cells and write the result. Returns a process exit code."""

    notebook_path: Path = arguments.notebook
    notebook: nbformat.NotebookNode = nbformat.read(notebook_path, as_version=4)

    if arguments.list:
        list_cells(notebook)
        return 0

    indices: list[int] = select_indices(notebook=notebook, arguments=arguments)
    kernel_name: str = resolve_kernel_name(notebook=notebook, override=arguments.kernel)
    output_path: Path = resolve_output_path(arguments=arguments)
    timeout: int | None = arguments.timeout if arguments.timeout and arguments.timeout > 0 else None

    # Resolve relative paths (datasets, outputs/, reports/) from the notebook's own directory,
    # matching how the notebook behaves when launched interactively from notebooks/.
    run_path: str = str(notebook_path.resolve().parent)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=arguments.allow_errors,
        resources={"metadata": {"path": run_path}},
    )

    bootstrap_cell: nbformat.NotebookNode | None = (
        build_bootstrap_cell(arguments.bootstrap) if arguments.bootstrap is not None else None
    )

    print(
        f"Notebook: {notebook_path}\n"
        f"Kernel: {kernel_name}\n"
        f"Cells to run: {indices[0]}..{indices[-1]} ({sum(notebook.cells[i].cell_type == 'code' for i in indices)} code cells)\n"
        f"Timeout: {'none' if timeout is None else f'{timeout}s'} | allow_errors={arguments.allow_errors}\n"
        f"Output: {output_path}\n",
        flush=True,
    )

    executed_count: int = 0
    failed: bool = False
    with client.setup_kernel():
        if bootstrap_cell is not None:
            print(f"[bootstrap] running {arguments.bootstrap}", flush=True)
            client.execute_cell(cell=bootstrap_cell, cell_index=0)
            print_cell_outputs(bootstrap_cell)

        for index in indices:
            cell: nbformat.NotebookNode = notebook.cells[index]
            if cell.cell_type != "code" or not cell.get("source", "").strip():
                continue
            print(f"\n{'=' * 70}\n[cell {index}] {first_source_line(cell)}\n{'=' * 70}", flush=True)
            try:
                client.execute_cell(cell=cell, cell_index=index)
                print_cell_outputs(cell)
            except CellExecutionError as error:
                failed = True
                print(f"\n[cell {index}] FAILED: {error}", file=sys.stderr, flush=True)
                nbformat.write(notebook, output_path)
                if not arguments.allow_errors:
                    break
            executed_count += 1
            # Persist after every cell so an interrupted overnight run keeps its progress.
            nbformat.write(notebook, output_path)

    print(f"\nExecuted {executed_count} code cell(s). Wrote {output_path}.", flush=True)
    return 1 if failed and not arguments.allow_errors else 0


def main() -> None:
    """CLI entry point."""

    arguments: argparse.Namespace = parse_arguments()
    raise SystemExit(execute_notebook(arguments=arguments))


if __name__ == "__main__":
    main()
