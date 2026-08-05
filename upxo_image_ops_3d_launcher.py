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


class ImageOpsApp(customtkinter.CTk if use_customtkinter else tk.Tk):
    def __init__(self, use_customtkinter=False):
        super().__init__()
        self.use_customtkinter = use_customtkinter
        self.title("UPXO 3D Image Operations Dashboard")
        self.geometry("1100x650")
        self.shared_state = {}
        self.running_processes = {}
        self._visited_pages = set()

        if use_customtkinter:
            self.configure(fg_color="#F0F2F5")
            self.container = customtkinter.CTkFrame(self, fg_color="transparent")
        else:
            self.container = tk.Frame(self, bg="#F0F2F5")
        
        self.container.pack(fill=tk.BOTH, expand=True)

        # Load OpsWelcomePage as starting page
        from upxo.gui.GSImageOperations3D.pages import OpsWelcomePage
        self.current_frame = OpsWelcomePage(self.container, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_page_by_class(self, page_cls):
        if self.current_frame:
            self.current_frame.destroy()
        self._visited_pages.add(page_cls)
        self.current_frame = page_cls(self.container, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def next_page(self):
        if not self.current_frame.validate_inputs():
            return
        self.current_frame.save_state()
        from upxo.gui.GSImageOperations3D.pages import (
            OpsWelcomePage, OpsImportPage, OpsLabellingPage, OpsCleaningPage,
            OpsManipulatePage, OpsGraphPage, OpsCharacterisationPage, OpsVisualisationPage
        )
        cls = self.current_frame.__class__
        if cls == OpsWelcomePage:
            self.show_page_by_class(OpsImportPage)
        elif cls == OpsImportPage:
            self.show_page_by_class(OpsLabellingPage)
        elif cls == OpsLabellingPage:
            self.show_page_by_class(OpsCleaningPage)
        elif cls == OpsCleaningPage:
            self.show_page_by_class(OpsManipulatePage)
        elif cls == OpsManipulatePage:
            self.show_page_by_class(OpsGraphPage)
        elif cls == OpsGraphPage:
            self.show_page_by_class(OpsCharacterisationPage)
        elif cls == OpsCharacterisationPage:
            self.show_page_by_class(OpsVisualisationPage)
        elif cls == OpsVisualisationPage:
            self.destroy()

    def back_page(self):
        from upxo.gui.GSImageOperations3D.pages import (
            OpsWelcomePage, OpsImportPage, OpsLabellingPage, OpsCleaningPage,
            OpsManipulatePage, OpsGraphPage, OpsCharacterisationPage, OpsVisualisationPage
        )
        cls = self.current_frame.__class__
        if cls == OpsImportPage:
            self.show_page_by_class(OpsWelcomePage)
        elif cls == OpsLabellingPage:
            self.show_page_by_class(OpsImportPage)
        elif cls == OpsCleaningPage:
            self.show_page_by_class(OpsLabellingPage)
        elif cls == OpsManipulatePage:
            self.show_page_by_class(OpsCleaningPage)
        elif cls == OpsGraphPage:
            self.show_page_by_class(OpsManipulatePage)
        elif cls == OpsCharacterisationPage:
            self.show_page_by_class(OpsGraphPage)
        elif cls == OpsVisualisationPage:
            self.show_page_by_class(OpsCharacterisationPage)

    def go_home(self):
        from upxo.gui.GSImageOperations3D.pages import OpsWelcomePage
        self.show_page_by_class(OpsWelcomePage)


if __name__ == "__main__":
    app = ImageOpsApp(use_customtkinter=use_customtkinter)
    app.mainloop()
