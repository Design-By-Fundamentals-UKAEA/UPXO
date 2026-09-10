import os
import sys
import tkinter as tk
from tkinter import messagebox

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Set Windows High-DPI awareness before creating Tkinter / CustomTkinter windows
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def check_core_dependencies():
    required = [("Pillow", "PIL"), ("NumPy", "numpy"), ("SciPy", "scipy"), ("Pandas", "pandas"), ("Matplotlib", "matplotlib")]
    missing = []
    for pkg_name, module_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        root = tk.Tk()
        root.withdraw()
        pip_pkgs = " ".join(pkg.lower() for pkg in missing)
        messagebox.showerror(
            "Dependency Error",
            "The following required libraries are missing from your Python environment:\n\n"
            + "\n".join(f" • {pkg}" for pkg in missing)
            + f"\n\nPlease install them using:\npip install {pip_pkgs}"
        )
        sys.exit(1)

check_core_dependencies()

use_customtkinter = False
try:
    import customtkinter
    use_customtkinter = True
except ImportError:
    pass

from upxo.gui.root import launch_gui

if __name__ == "__main__":
    launch_gui(use_customtkinter=use_customtkinter)
