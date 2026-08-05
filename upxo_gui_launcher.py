import os
import sys
import tkinter as tk
from tkinter import messagebox

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from PIL import Image, ImageTk
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Dependency Error",
        "Pillow (PIL) library is required to run the UPXO GUI.\n"
        "Please install it using:\npip install Pillow"
    )
    sys.exit(1)

use_customtkinter = False
try:
    import customtkinter
    use_customtkinter = True
except ImportError:
    pass

from upxo.gui.root import launch_gui

if __name__ == "__main__":
    launch_gui(use_customtkinter=use_customtkinter)
