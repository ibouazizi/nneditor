"""Run NNEditor with ``python -m nneditor``."""

from nneditor.ui.app import cli


def main() -> None:
    """Launch the Flet application."""
    cli()


if __name__ == "__main__":  # pragma: no cover - interpreter entry point
    main()
