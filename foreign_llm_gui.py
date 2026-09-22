"""Compatibility launcher for the refactored Qt desktop application.

The previous Tkinter implementation and its hard-coded API keys were removed.
Run ``python3 -m mama --gui`` (or this file) to launch the new interface.
"""
from mama.interface.gui_interface import main


if __name__ == "__main__":
    raise SystemExit(main())
