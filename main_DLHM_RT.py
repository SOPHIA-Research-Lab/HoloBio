
import numpy as np
import customtkinter as ctk
import time
import os
from multiprocessing import Process, Queue
from settings import *
from parallel_rc import *
from PIL import ImageTk, Image, ImageOps
from scipy import ndimage
import tkinter as tk
from matplotlib.widgets import RectangleSelector
from pandastable import Table
import warnings 
from tkinter import simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module, reload
import tools_GUI as tGUI
import functions_GUI as fGUI
import cv2

class App(ctk.CTk):

    
    def __init__(self):
        ctk.set_appearance_mode("Light")
        super().__init__()
        if not hasattr(tGUI, "ImageTk"):
            tGUI.ImageTk = ImageTk   
        self.title('DLHM GUI')
        self.attributes('-fullscreen', False)
        self.state('normal')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE)/1.8

        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L
        self.L = INIT_L
        self.Z = INIT_Z
        self.r = self.L - self.Z
        self.wavelength = DEFAULT_WAVELENGTH
        self.dxy = DEFAULT_DXY
        self.scale_factor = self.L/self.Z if self.Z != 0 else 1.0
        self.cosine_period = DEFAULT_COSINE_PERIOD

        self.fix_r = ctk.BooleanVar(self, value=False)
        self.square_field = ctk.BooleanVar(self, value=False)
        self.Processed_Image_r = ctk.BooleanVar(self, value=False)
        self.algorithm_var = ctk.StringVar(self, value='AS')
        self.filter_image_var = ctk.StringVar(self, value='CA')

        self.file_path = ''
        self.ref_path = ''

        self.gamma_checkbox_var = ctk.BooleanVar(self, value=False)
        self.contrast_checkbox_var = ctk.BooleanVar(self, value=False)
        self.adaptative_eq_checkbox_var = ctk.BooleanVar(self, value=False)
        self.highpass_checkbox_var = ctk.BooleanVar(self, value=False)
        self.lowpass_checkbox_var = ctk.BooleanVar(self, value=False)

        self.manual_gamma_c_var = ctk.BooleanVar(self, value=False)
        self.manual_gamma_r_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_c_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_r_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_c_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_r_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_r_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_r_var = ctk.BooleanVar(self, value=False)

        self.filters_c = []
        self.filters_r = []
        self.filter_params_c = []
        self.filter_params_r = []

        self.gamma_c = 0
        self.gamma_r = 0
        self.contrast_c = 0
        self.contrast_r = 0
        self.adaptative_eq_c = False
        self.adaptative_eq_r = False
        self.highpass_c = 0
        self.highpass_r = 0
        self.lowpass_c = 0
        self.lowpass_r = 0

        # Initialize arrays as black placeholders
        self.arr_c = np.zeros((400, 300), dtype=np.uint8)
        self.arr_r = np.zeros((400, 300), dtype=np.uint8)

        self.viewbox_width = 400
        self.viewbox_height = 300

        # Convert them to PIL + CTkImage
        im_c = Image.fromarray(self.arr_c)
        im_r = Image.fromarray(self.arr_r)
        self.img_c = ctk.CTkImage(light_image=im_c, size=(self.viewbox_width, self.viewbox_height))
        self.img_r = ctk.CTkImage(light_image=im_r, size=(self.viewbox_width, self.viewbox_height))

        self.arr_c_orig = self.arr_c.copy()     # last *unfiltered* capture
        self.arr_r_orig = self.arr_r.copy()     # last *unfiltered* processed
        self.arr_c_view = self.arr_c.copy()     # what is shown on the left
        self.arr_r_view = self.arr_r.copy()     # what is shown on the right

        self.w_fps = 0
        self.c_fps = 0
        self.r_fps = 0
        self.max_w_fps = 0
        self.settings = False

        warnings.filterwarnings("ignore",
                                category=RuntimeWarning,
                                module="skimage.filters._fft_based")

        self.speckle_lock: bool = False
        self.speckle_k_last: int = 0
        self.speckle_applied: bool = False   
        # We keep only the reconstruction process, so real-time capturing is removed:
        self.queue_manager = {
            "capture": {
                "input": Queue(1),
                "output": Queue(1),
            },
            "reconstruction": {
                "input": Queue(1),
                "output": Queue(1),
            },
        }

        # Start ONLY the reconstruction process, skip capture:
        self.reconstruction = Process(target=reconstruct, args=(self.queue_manager,))
        self.reconstruction.start()

        self.capture_input = {'path': None, 'reference path': None, 'settings': None, 'filters': None, 'filter': None}
        self.capture_output = {'image': None, 'filtered': None, 'fps': 0, 'size': (0, 0)}
        self.recon_input = {'image': None, 'filters': None, 'filter': False, 'algorithm': None, 'L': 0, 'Z': 0, 'r': 0,
                            'wavelength': 0, 'dxy': 0, 'scale_factor': 0, 'squared': False, 'Processed_Image': False}
        self.recon_output = {'image': None, 'filtered': None, 'fps': 0}

        self.speckle_checkbox_var = tk.BooleanVar(value=False)

        self.multi_holo_arrays: list[np.ndarray] = []
        self.hologram_frames:    list[ctk.CTkImage] = []
        self.current_left_index: int = 0

        self.ft_arrays:  list[np.ndarray]   = []
        self.ft_frames:  list[ctk.CTkImage] = []
        self.current_ft_index: int          = 0

        # keep wavelength unit and add separated pitch-units
        self.wavelength_unit = "µm"
        self.pitch_x_unit    = "µm"
        self.pitch_y_unit    = "µm"
        self.distance_unit   = "µm"
    
        self._dist_unit_var = tk.StringVar(
        value=getattr(self, "distance_unit", "µm"))

        self.param_entries = {}
        self.ft_mode_var = tk.StringVar(self, value="With logarithmic scale")

        # checkboxes for speckle panel
        self.compare_side_by_side_var = tk.BooleanVar()
        self.compare_speckle_plot_var = tk.BooleanVar()
        self.compare_line_profile_var = tk.BooleanVar()

        # Phase ( _r_ )
        self.manual_lowpass_r_var = ctk.BooleanVar(self, value=False)

        #self._add_amplitude_filter_vars()
        self.capture = Process(target=capture, args=(self.queue_manager,))
        self.capture.start()
   
        # recording
        self.is_recording = False
        self.record_type = None
        self.record_frames = []

        # Initialize frames
        self._sync_canvas_and_frame_bg()
        self.init_viewing_frame()
        self._build_fps_indicators()
        self.init_saving_frame()
        self.update_inputs()
        self.after(0, self.after_idle_setup)
        self.after(0, self.draw)
        self._init_data_containers()


    def update_inputs(self, process: str = ''):
        # We won't do a real capture pipeline, so only the reconstruction matters
        if process == 'capture' or not process:
            self.capture_input['path'] = self.file_path
            self.capture_input['reference path'] = self.ref_path
            self.capture_input['settings'] = self.settings
            self.capture_input['filters'] = (self.filters_c, self.filter_params_c)
            self.capture_input['filter'] = True

        if process in ("reconstruction", ""):
            self.recon_input = {
                "image":        self.arr_c,
                "filters":      (self.filters_r, self.filter_params_r),
                "filter":       True,
                "algorithm":    self.algorithm_var.get(),
                "L":            self.L,
                "Z":            self.Z,
                "r":            self.r,
                "wavelength":   self.wavelength,
                "dxy":          self.dxy,
                "scale_factor": self.scale_factor,
                "squared":      self.square_field.get(),
                "phase":        self.Processed_Image_r.get()
            }

    
    def update_outputs(self, process: str = ""):
        # capture side
        if process in ("capture", ""):
            self.arr_c_orig = self.capture_output["image"]

            if self.arr_c_orig is not None:
                # update the left-pane *buffers* …
                self._recompute_and_show(left=True)
                # …then paint whichever view the user actually chose
                self.update_left_view()

            self.c_fps              = self.capture_output["fps"]
            self.width, self.height = self.capture_output["size"]

        # rconstruction side
        if process in ("reconstruction", ""):
            self._update_recon_arrays()

    def _reset_toolbar_labels(self) -> None:
        """Restore the original captions of the toolbar OptionMenus."""
        for attr, caption in (
            ("load_menu",  "Load"),
            ("tools_menu", "Tools"),
            ("save_menu",  "Save"),
            ("theme_menu", "Theme"),
        ):
            m = getattr(self, attr, None)
            if m is not None:
                m.set(caption)

    def _show_popup_image(self, arr: np.ndarray, title: str = "Speckle filtered"):
     """Show a static image in a non‑blocking Toplevel window."""
     win = tk.Toplevel(self)
     win.title(title)
     im  = Image.fromarray(arr)
     tk_img = ImageTk.PhotoImage(im)
     lbl = tk.Label(win, image=tk_img)
     lbl.image = tk_img     # keep a reference
     lbl.pack()

    def _init_data_containers(self) -> None:
        """
        Creates all attributes that tools_GUI relies on, with safe defaults.
        Call this ONCE from __init__ before any frame is built.
        """

        # --- holograms -------------------------------------------------
        self.current_holo_array          = np.zeros((1, 1), dtype=np.uint8)
        self.original_multi_holo_arrays  = []   # pristine captures
        self.multi_holo_arrays           = []   # filtered captures
        self.hologram_frames             = []   # CTkImages for display
        self.current_left_index          = 0

        # --- reconstructions (amplitude & phase, 8-bit) ----------------
        self.amplitude_arrays          = []
        self.original_amplitude_arrays = []
        self.amplitude_frames          = []
        self.current_amp_index         = 0

        self.phase_arrays              = []
        self.original_phase_arrays     = []
        self.phase_frames              = []
        self.current_phase_index       = 0

        # *** NUEVO BLOQUE – INTENSIDADES ***
        self.intensity_arrays          = []
        self.original_intensity_arrays = []
        self.intensity_frames          = []
        self.current_int_index         = 0

        # --- complex fields for SPP filter -----------------------------
        self.complex_fields              = []

        # --- per-image filter state memory (Filters panel) -------------
        self.filter_states_dim0 = []     # hologram
        self.filter_states_dim1 = []     # amplitude
        self.filter_states_dim2 = []     # phase

        # --- last results of a speckle filter --------------------------
        self.filtered_amp_array   = None
        self.filtered_phase_array = None
    

    def _map_ui_to_mpl_cmap(self, ui_name: str) -> str:
        """Translate the UI name into a valid Matplotlib identifier."""
        table = {
            "Viridis":  "viridis",
            "Plasma":   "plasma",
            "Inferno":  "inferno",
            "Magma":    "magma",
            "Cividis":  "cividis",
            "Hot":      "hot",
            "Cool":     "cool",
            "Wistia":   "Wistia",   # needs capital “W”
        }
        return table.get(ui_name, ui_name.lower())

    def _apply_ui_colormap(self, arr8u: np.ndarray, ui_name: str) -> np.ndarray:
        """
        Return *arr8u* converted to RGB with the UI-selected colormap.
        If ui_name is “Original” or the array is already RGB, it is
        returned unchanged.
        """
        if ui_name == "Original" or arr8u.ndim == 3:
            return arr8u

        from matplotlib import colormaps as mpl_cmaps
        cmap = mpl_cmaps[self._map_ui_to_mpl_cmap(ui_name)]

        norm = arr8u.astype(np.float32) / 255.0
        rgb  = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        return rgb

    # ------------------------------------------------------------------
    # 2.  Safe replacement for tools_GUI.apply_matplotlib_colormap (new)
    # ------------------------------------------------------------------
    def _safe_apply_matplotlib_colormap(self,
                                        arr8u : np.ndarray,
                                        ui_name: str):
        """
        Drop-in replacement for the old helper.
        Returns a PIL.Image – never crashes if the array is already RGB.
        """
        rgb = self._apply_ui_colormap(arr8u, ui_name)

        # guarantee an RGB PIL image
        if rgb.ndim == 2:                       # greyscale → stack channels
            rgb = np.stack([rgb]*3, axis=-1)

        from PIL import Image
        return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


    def _preserve_aspect_ratio_right(self, pil_image: Image.Image) -> ctk.CTkImage:
        """Return a CTkImage letterboxed to (viewbox_width, viewbox_height)."""
        max_w, max_h = self.viewbox_width, self.viewbox_height
        orig_w, orig_h = pil_image.size

        # Scale down/up while preserving aspect ratio
        ratio_w = max_w / float(orig_w)
        ratio_h = max_h / float(orig_h)
        scale_factor = min(ratio_w, ratio_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)

        # Letterbox
        final_img = Image.new("RGB", (max_w, max_h), color=(0, 0, 0))
        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        offset_x = (max_w - new_w) // 2
        offset_y = (max_h - new_h) // 2
        final_img.paste(resized, (offset_x, offset_y))

        # Return as CTkImage with the exact desired size
        return ctk.CTkImage(light_image=final_img, size=(max_w, max_h))
    

    def init_viewing_frame(self) -> None:
        self.init_navigation_frame()
        # placeholders so the helper modules have something to draw
        self.holo_views  = [("init", self.img_c)]
        self.recon_views = [("init", self.img_r)]
        # toolbar
        fGUI.build_toolbar(self)
        # ── NEW: lock the Tools menu ───────────────────────
        if hasattr(self, "tools_menu"):
            try:
                self.tools_menu.configure(state="disabled")
            except Exception:
                pass
        fGUI.build_two_views_panel(self)



    def _ensure_frame_lists_length(self) -> None:
        """Pad every *…_frames* list with a 1 × 1 dummy so index
        assignments can never raise IndexError."""
        def _pad(lst, target_len):
            dummy = ctk.CTkImage(light_image=Image.new("RGB", (1, 1)),
                                 size=(1, 1))
            while len(lst) < target_len:
                lst.append(dummy)

        _pad(self.hologram_frames,   len(self.multi_holo_arrays))
        _pad(self.amplitude_frames,  len(self.amplitude_arrays))
        _pad(self.phase_frames,      len(self.phase_arrays))
        _pad(self.intensity_frames,  len(self.intensity_arrays))
        _pad(self.ft_frames,         len(self.ft_arrays))       


    # Values that must appear in the “Load” drop-down
    def get_load_menu_values(self):
        return ["Load image", "Select reference", "Reset reference"]

    # When the user picks something in the drop-downs:
    def _on_load_select(self, choice: str):
        {"Load image":       self.selectfile,
         "Select reference": self.selectref,
         "Reset reference":  self.resetref}.get(choice, lambda: None)()
        self.after(100, self._reset_toolbar_labels)   # ← NEW

    def _on_tools_select(self, choice: str):
        if choice == "Filters":
            self.change_menu_to("filters")
        elif choice == "Speckle":
            self.change_menu_to("speckle")
            if hasattr(self, "speckles_canvas"):
                self.speckles_canvas.yview_moveto(0.0)
        elif choice == "Bio-Analysis":
            self.change_menu_to("bio")
        self.after(100, self._reset_toolbar_labels)
  
    def _on_save_select(self, choice: str):
        {"Save FT":        self.save_capture,
         "Save Phase":     self.save_processed,
         "Save Amplitude": self.save_processed}.get(choice, lambda: None)()
        self.after(100, self._reset_toolbar_labels) 

    def _on_theme_select(self, mode: str):
        ctk.set_appearance_mode(mode)
        self._sync_canvas_and_frame_bg()


    def _compute_ft(self, arr: np.ndarray) -> np.ndarray:
        """Returns log-magnitude FT (uint8) of *arr*."""
        f  = np.fft.fftshift(np.fft.fft2(arr.astype(np.float32)))
        mag = np.log1p(np.abs(f))
        mag = (mag / mag.max() * 255).astype(np.uint8)
        return mag


    def update_left_view(self):
        """
        Uses the Fourier-transform image prepared by the background
        process instead of computing a fresh FFT in the GUI thread.
        """
        view_choice = self.holo_view_var.get()

        if view_choice == "Hologram":
            disp_arr = self.arr_c_view
            title    = "Hologram"

        else:  # ---------- Fourier Transform ------------------------
            # fall back to local FFT only if the worker has not yet
            # delivered anything (rare, first few frames)
            if self.ft_arrays:
                disp_arr = self.ft_arrays[self.current_ft_index]
            else:
                # one-off emergency local FFT (small & fast, 8-bit)
                disp_arr = self._generate_ft_display(self.arr_c_view,
                                                     log_scale=True)
            title = "Fourier Transform"
            # remember for zoom helpers etc.
            self._last_ft_display = disp_arr.copy()

        pil  = Image.fromarray(disp_arr)
        self.img_c = self._preserve_aspect_ratio_right(pil)
        self.captured_label.configure(image=self.img_c)
        self.captured_title_label.configure(text=title)

        

    def _get_current_array(self, what: str) -> np.ndarray | None:
        """
        Return the numpy array that corresponds to *what*:
        'Hologram', 'FT', 'Phase', 'Amplitude', etc.
        """
        if what in ("Hologram", "Hologram "):
            return self.arr_c_view
        if what in ("Fourier Transform", "FT"):
            # rebuild FT if it wasn’t cached yet
            if not hasattr(self, "_last_ft_display"):
                log = self.ft_mode_var.get().startswith("With")
                self._last_ft_display = self._generate_ft_display(
                    self.arr_c_view, log_scale=log)
            return self._last_ft_display
        if what == "Phase":
            if self.phase_arrays:
                return self.phase_arrays[self.current_phase_index]
        if what == "Amplitude":
            if self.amplitude_arrays:
                return self.amplitude_arrays[self.current_amp_index]
        return None 
    

    # ------ wrappers de zoom ---------------------
    def zoom_holo_view(self, *args, **kwargs):
        tGUI.zoom_holo_view(self, *args, **kwargs)

    def zoom_recon_view(self, *args, **kwargs):
        tGUI.zoom_recon_view(self, *args, **kwargs)

    def _open_zoom_view(self, *args, **kwargs):
        tGUI._open_zoom_view(self, *args, **kwargs)

    def _refresh_zoom_view(self, *args, **kwargs):
        tGUI._refresh_zoom_view(self, *args, **kwargs)

    def _show_ft_mode_menu(self):
        menu = tk.Menu(self, tearoff=0)
        opts = ["With logarithmic scale", "Without logarithmic scale"]
        for opt in opts:
            menu.add_radiobutton(
                label=opt, value=opt,
                variable=self.ft_mode_var,
                command=self._on_ft_mode_changed
            )
        menu.tk_popup(self.ft_mode_button.winfo_rootx(),
                      self.ft_mode_button.winfo_rooty() + self.ft_mode_button.winfo_height())
        
    def update_right_view(self):
        view_name = self.recon_view_var.get().strip()
        amp_mode  = getattr(self, "amp_mode_var",
                            tk.StringVar(value="Amplitude")).get()

        if view_name.startswith("Phase"):
            src = self.phase_arrays[self.current_phase_index]

        else:                                # “Amplitude Reconstruction”
            if amp_mode == "Amplitude":
                src = self.amplitude_arrays[self.current_amp_index]
            else:                            # “Intensities”
                src = self.intensity_arrays[self.current_int_index]

        disp = self._run_filters_pipeline(src, use_left_side=False)

        self.arr_r_view = self.arr_r = disp
        pil = Image.fromarray(disp)
        self.img_r = self._preserve_aspect_ratio_right(pil)
        self.processed_label.configure(image=self.img_r)
        self.processed_title_label.configure(text=view_name)

    def _on_ft_mode_changed(self):
     self._refresh_all_ft_views()

    def _show_amp_mode_menu(self):
     """Menú ‘Amplitude / Intensities’ para la reconstrucción."""
     menu = tk.Menu(self, tearoff=0)
     for opt in ("Amplitude", "Intensities"):
         menu.add_radiobutton(
             label=opt, value=opt,
             variable=self.amp_mode_var,
             command=self._on_amp_mode_changed
         )
     menu.tk_popup(self.amp_mode_button.winfo_rootx(),
                   self.amp_mode_button.winfo_rooty()+self.amp_mode_button.winfo_height())
 
    def _on_amp_mode_changed(self, *_):
        if self.recon_view_var.get().startswith("Amplitude"):
            self.update_right_view()

    # ---------- utilidades de imagen ----------
    def _generate_ft_display(self, holo_array: np.ndarray, log_scale: bool = True) -> np.ndarray:
     """Devuelve la magnitud (uint8) de la FT centrada; opcionalmente en escala log."""
     ft_cplx = np.fft.fftshift(np.fft.fft2(holo_array.astype(np.float32)))
     mag = np.abs(ft_cplx)
     if log_scale:
         mag = np.log1p(mag)
     mag = mag / (mag.max() + 1e-9) * 255.0
     return mag.astype(np.uint8)

    def _generate_intensity_display(self, amp_array_8bit: np.ndarray) -> np.ndarray:
     """Convierte amplitud 8-bit en intensidades (|A|²) también en 8-bit."""
     amp_f = amp_array_8bit.astype(np.float32) / 255.0
     intens = amp_f ** 2
     intens = intens / (intens.max() + 1e-9) * 255.0
     return intens.astype(np.uint8)

    # ---------- navegación entre hologramas ----------
    def previous_hologram_view(self):
        """Show the previous hologram and restore its filter UI state."""
        if not getattr(self, "multi_holo_arrays", []):
            return
        self.current_left_index = (self.current_left_index - 1) % len(self.multi_holo_arrays)
        self.arr_c_orig = self.multi_holo_arrays[self.current_left_index]
        self._recompute_and_show(left=True)
        self.update_left_view()

        # restore sliders / check-boxes for *this* hologram
        self.load_ui_from_filter_state(0, self.current_left_index)
        self.update_image_filters()


    def next_hologram_view(self):
        """Show the next hologram and restore its filter UI state."""
        if not getattr(self, "multi_holo_arrays", []):
            return
        self.current_left_index = (self.current_left_index + 1) % len(self.multi_holo_arrays)
        self.arr_c_orig = self.multi_holo_arrays[self.current_left_index]
        self._recompute_and_show(left=True)
        self.update_left_view()

        # restore sliders / check-boxes for *this* hologram
        self.load_ui_from_filter_state(0, self.current_left_index)
        self.update_image_filters()
     
    def _place_holo_arrows(self) -> None:
        """Ensure arrows are gridded in row-4 if they were removed."""
        self.left_arrow_holo.grid(row=4, column=0, sticky="w",
                                  padx=20, pady=5)
        self.right_arrow_holo.grid(row=4, column=1, sticky="e",
                                   padx=20, pady=5)

    def show_holo_arrows(self) -> None:
        """Show the navigation arrows when >1 hologram is loaded."""
        self._place_holo_arrows()          # put them in the grid

    def hide_holo_arrows(self) -> None:
        """Hide the navigation arrows."""
        self.left_arrow_holo.grid_remove()
        self.right_arrow_holo.grid_remove()

    # functions for ft coordinates
    def _activate_ft_coordinate_display(self) -> None:
        """Bind mouse-motion to the FT image and show the label."""
        self.captured_label.bind("<Motion>", self._on_ft_mouse_move)
        self.captured_label.bind("<Leave>",
                                 lambda e: self.ft_coord_label.configure(text=""))

        # top-left corner of *left_frame* with a small margin
        self.ft_coord_label.place(relx=0.5, rely=1.0, x=0, y=-8, anchor="s")

    def _deactivate_ft_coordinate_display(self) -> None:
        """Remove bindings and hide the label when FT is not shown."""
        self.captured_label.unbind("<Motion>")
        self.captured_label.unbind("<Leave>")
        self.ft_coord_label.place_forget()

    def _refresh_all_ft_views(self):
     """Re-dibuja las vistas que estén mostrando la FT con el modo elegido."""
     if self.holo_view_var.get() == "Fourier Transform":
         self.update_left_view()

    def _hide_parameters_nav_button(self) -> None:
        if hasattr(self, "param_button"):
            self.param_button.destroy()         # make it disappear
        # make “Parameters” the default view on the left‑hand column
        self.change_menu_to("parameters")


    def _make_unit_button(self,
                          parent      : ctk.CTkFrame,
                          row         : int,
                          column      : int,
                          unit_var    : tk.StringVar,
                          label_target: ctk.CTkLabel) -> None:

        btn = ctk.CTkButton(parent, width=28, text="▼")
        btn.grid(row=row, column=column, sticky="e", padx=(0, 2))

        def _on_click(event=None):
            m = tk.Menu(self, tearoff=0, font=("Helvetica", 14))
            for u in ("µm", "nm", "mm", "cm", "m", "in"):
                m.add_command(
                    label=u,
                    command=lambda unit=u: (
                        unit_var.set(unit),
                        self._set_unit_in_label(label_target, unit)
                    )
                )
            m.post(btn.winfo_rootx(),
                   btn.winfo_rooty() + btn.winfo_height())

        btn.bind("<Button-1>", _on_click)


    def _set_unit_in_label(self, lbl: ctk.CTkLabel, unit: str) -> None:

        base = lbl.cget("text").split("(")[0].strip()
        lbl.configure(text=f"{base} ({unit})")

        if   "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pixel pitch" in base:
            self.pitch_unit      = unit
        elif "Distance" in base or base.endswith("(L)") \
             or base.endswith("(Z)") or base.endswith("(r)"):
            self.distance_unit   = unit

    def get_value_in_micrometers(self, value: str, unit: str) -> float:
        """Converts *value* (given in *unit*) into micrometres (µm)."""
        conversion = {
            "µm": 1.0,          "Micrometers": 1.0,
            "nm": 1e-3,
            "mm": 1e3,
            "cm": 1e4,
            "m" : 1e6,
            "in": 2.54e4,
        }
        val = value.strip().replace(",", ".")
        if not val:
            return 0.0
        try:
            val_f = float(val)
        except ValueError:
            raise ValueError(f"Cannot convert “{value}” into float.")
        return val_f * conversion.get(unit, 1.0)


    def _setup_unit_buttons(self) -> None:

        # 1) first‑time initialisation of unit attributes
        if not hasattr(self, "wavelength_unit"):
            self.wavelength_unit = "µm"
            self.pitch_unit      = "µm"
            self.distance_unit   = "µm"

        # 2) StringVars that will track the current unit for each group
        self._wave_unit_var  = tk.StringVar(value=self.wavelength_unit)
        self._pitch_unit_var = tk.StringVar(value=self.pitch_unit)
        self._dist_unit_var  = tk.StringVar(value=self.distance_unit)

        # 3) ▼ next to “Wavelength” and “Pixel pitch”
        self._make_unit_button(self.variables_frame, row=0, column=0,
                               unit_var=self._wave_unit_var,
                               label_target=self.lambda_label)
        self._make_unit_button(self.variables_frame, row=0, column=2,
                               unit_var=self._pitch_unit_var,
                               label_target=self.dxy_label)

        # 4) one ▼ that will apply to *all* distances (L, Z, r)
        #    we place it on the same row as the L‑title label
        self._make_unit_button(self.L_frame, row=0, column=3,
                               unit_var=self._dist_unit_var,
                               label_target=self.L_slider_title)


    def choose_load_option(self, selected_option):
        """Callback for the Load OptionMenu."""
        if selected_option == "Load image":
            self.selectfile()
        elif selected_option == "Select reference":
            self.selectref()
        elif selected_option == "Reset reference":
            self.resetref()

    def show_save_options(self):
        """Creates an OptionMenu with two choices: save capture or save reconstruction."""
        if hasattr(self, 'save_options_menu') and self.save_options_menu.winfo_ismapped():
            self.save_options_menu.grid_forget()
            return
        self.save_options_menu = ctk.CTkOptionMenu(
            self.options_frame,
            values=["Save capture", "Save reconstruction"],
            command=self.choose_save_option,
            width=270
        )
        self.save_options_menu.grid(row=0, column=2, padx=5, pady=5,sticky='ew')

    def choose_save_option(self, selected_option):
        """Callback for the Save OptionMenu."""
        if selected_option == "Save capture":
            self.save_capture()
        elif selected_option == "Save reconstruction":
            self.save_processed()


   
    def _sync_canvas_bg(self):
     col = self.Tools_frame.cget("fg_color")
     if isinstance(col, (list, tuple)):          # ('#ebebeb', '#2b2b2b')
         col = col[0] if ctk.get_appearance_mode() == "Light" else col[1]
 
     for cv in (self.tools_canvas,
               self.filters_canvas,
               self.param_canvas):
         if cv:   # por si alguno no existe aún
             cv.configure(background=col,
                          highlightthickness=0,
                          borderwidth=0)
 
    def init_navigation_frame(self) -> None:
        """
        Re-creates the Parameters column using the modular helper
        fGUI.create_param_with_arrow() for wavelength + pitch X/Y, y
        mueve la línea de Magnification y el selector de unidad ▼ al header de ‘L’
        exactamente como se pidió.
        """
        # ——— container, canvas & scroll-bar (sin cambios) ———
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8, width=MENU_FRAME_WIDTH)
        self.navigation_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(0, weight=1)
        self.navigation_frame.grid_propagate(False)

        self.param_container = ctk.CTkFrame(self.navigation_frame, corner_radius=8, width=420)
        self.param_container.grid_propagate(False)
        self.param_container.pack(fill="both", expand=True)

        self.param_scrollbar = ctk.CTkScrollbar(self.param_container, orientation="vertical")
        self.param_scrollbar.grid(row=0, column=0, sticky="ns")

        self.param_canvas = ctk.CTkCanvas(self.param_container, width=PARAMETER_FRAME_WIDTH)
        self.param_canvas.grid(row=0, column=1, sticky="nsew")

        self.param_container.grid_rowconfigure(0, weight=1)
        self.param_container.grid_columnconfigure(1, weight=1)

        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        self.param_scrollbar.configure(command=self.param_canvas.yview)

        self.parameters_inner_frame = ctk.CTkFrame(self.param_canvas)
        self.param_canvas.create_window((0, 0), window=self.parameters_inner_frame, anchor="nw")

        title_lbl = ctk.CTkLabel(self.parameters_inner_frame,
            text="Parameters",
            font=ctk.CTkFont(weight="bold"))
        title_lbl.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # ============================================================
        #  A) THREE-PARAMETER HEADER  (row-1)
        # ============================================================
        self.variables_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.variables_frame.grid(row=1, column=0, sticky="ew", pady=2)
        self.variables_frame.grid_propagate(False)

        # three equal columns
        for c in range(3):
            self.variables_frame.columnconfigure(c, weight=1)

        units = ["µm", "nm", "mm", "cm", "m", "in"]

        # -- wavelength, pitch-X, pitch-Y ----------------------------
        fGUI.create_param_with_arrow(
            parent=self.variables_frame, row=0, col=0,
            label_text=f"Wavelength ({self.wavelength_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries,
            entry_key="wavelength",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.variables_frame, row=0, col=1,
            label_text=f"Pitch X ({self.pitch_x_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries,
            entry_key="pitch_x",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.variables_frame, row=0, col=2,
            label_text=f"Pitch Y ({self.pitch_y_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries,
            entry_key="pitch_y",
            unit_update_callback=self._set_unit_in_label
        )

        # keep handy references
        self.wave_entry   = self.param_entries["wavelength"]
        self.pitchx_entry = self.param_entries["pitch_x"]
        self.pitchy_entry = self.param_entries["pitch_y"]

        # ============================================================
        #  B) L-FRAME  (row-2)  – incluye Magnification & unit ▼
        # ============================================================
        self.L_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.L_frame.grid(row=3, column=0, sticky="ew", pady=2)
        self.L_frame.grid_propagate(False)

        # 3 columns:  magnif | Distances(label) | ▼
        for c in range(3):
            self.L_frame.columnconfigure(c, weight=(1 if c == 0 else 0))

        # ---------- L slider & entry / set button -------------------
        self.L_frame.rowconfigure(1, weight=1)
        self.L_frame.columnconfigure(0, weight=2)
        self.L_slider_title = ctk.CTkLabel(
            self.L_frame,
            text=f"Distance between camera and source L "
                 f"({self.distance_unit}): {round(self.L, 4)}"
        )
        self.L_slider_title.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.L_slider = ctk.CTkSlider(
            self.L_frame, height=SLIDER_HEIGHT, corner_radius=8,
            from_=self.MIN_L, to=self.MAX_L, command=self.update_L
        )
        self.L_slider.grid(row=2, column=0, sticky="ew")
        self.L_slider.set(round(self.L, 4))

        self.L_slider_entry = ctk.CTkEntry(
            self.L_frame, width=PARAMETER_ENTRY_WIDTH,
            placeholder_text=f"{round(self.L, 4)}"
        )
        self.L_slider_entry.grid(row=2, column=1, sticky="ew", padx=5)

        self.L_slider_button = ctk.CTkButton(
            self.L_frame, width=PARAMETER_BUTTON_WIDTH,
            text="Set", command=self.set_value_L
        )
        self.L_slider_button.grid(row=2, column=2, sticky="ew", padx=10)

        # ================== resto de filas originales =================
        #    Z-frame  → row 3
        #    r-frame  → row 4
        #    (todo lo de abajo es IGUAL, así que mantenlo igual)
        # ============================================================

        # *** IMPORTANTE ***
        # actualiza el número de fila (+1 offset) para Z-frame, r-frame, etc.
        start_row = 3  # primera fila después del nuevo L-frame
        self._build_Z_r_and_remaining_frames(start_row)

        # Final scroll-region update
        self.parameters_inner_frame.update_idletasks()
        self.param_canvas.config(scrollregion=self.param_canvas.bbox("all"))

        # RIGHT column – vacío por ahora: los helpers lo rellenan.
        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky="nsew")
        self.viewing_frame.grid_rowconfigure(0, weight=0)   # toolbar
        self.viewing_frame.grid_rowconfigure(1, weight=1)   # viewers
        self.viewing_frame.grid_columnconfigure(0, weight=1)

    def _build_Z_r_and_remaining_frames(self, first_row: int):
        # ============ adit_options_frame  (¡NUEVO LAYOUT!) ==============
        self.adit_options_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.adit_options_frame.grid(row=2, column=0,
                                     sticky="ew", pady=2)
        self.adit_options_frame.grid_propagate(False)

        # rejilla flexible: 5 columnas

        for c in range(5):
            weight = 1 if c in (0, 4) else 0
            self.adit_options_frame.columnconfigure(c, weight=weight)
        for r in range(3):
            self.adit_options_frame.rowconfigure(r, weight=0)

        # ── magnification─────────────────────────
        if hasattr(self, "magnification_label") and self.magnification_label.winfo_exists():
            self.magnification_label.grid_forget()
        self.magnification_label = ctk.CTkLabel(
            self.adit_options_frame,
            text=f"Magnification: {round(self.scale_factor,4)}"
        )
        self.magnification_label.grid(row=0, column=0, columnspan=5,
                                      sticky="nsew", pady=(4, 2))

        # ── fila-1:  Fix r        Distances (µm) ▼ ────────────────────
        if hasattr(self, "fix_r_checkbox") and self.fix_r_checkbox.winfo_exists():
            self.fix_r_checkbox.grid_forget()
        self.fix_r_checkbox = ctk.CTkCheckBox(
            self.adit_options_frame,
            text="Fix r",
            variable=self.fix_r
        )
        self.fix_r_checkbox.grid(row=1, column=1, sticky="w",
                                 padx=10, pady=5)

        # Distances label
        if hasattr(self, "dist_label") and self.dist_label.winfo_exists():
            self.dist_label.grid_forget()
        self.dist_label = ctk.CTkLabel(
            self.adit_options_frame,
            text=f"Distances ({self.distance_unit})"
        )
        self.dist_label.grid(
            row=1, column=2,
            sticky="w",
            padx=(2, 0),
            pady=5
        )

        # ▼ botón de unidades
        if hasattr(self, "dist_unit_btn") and self.dist_unit_btn.winfo_exists():
            self.dist_unit_btn.grid_forget()
        self._make_unit_button(
            parent=self.adit_options_frame,
            row=1, column=3,
            unit_var=self._dist_unit_var,
            label_target=self.dist_label
        )

        # ============ Z-frame =================================================
        self.Z_frame = ctk.CTkFrame(self.parameters_inner_frame,
                                    width=PARAMETER_FRAME_WIDTH,
                                    height=PARAMETER_FRAME_HEIGHT)
        self.Z_frame.grid(row=first_row+1, column=0, sticky="ew", pady=2)
        self.Z_frame.grid_propagate(False)
        self.Z_frame.columnconfigure(0, weight=2)

        self.Z_slider_title = ctk.CTkLabel(
            self.Z_frame,
            text=f"Distance between the sample and source Z "
                 f"({self.distance_unit}): {round(self.Z, 4)}"
        )
        self.Z_slider_title.grid(row=0, column=0, columnspan=3,
                                 sticky="ew", pady=5)

        self.Z_slider = ctk.CTkSlider(self.Z_frame, height=SLIDER_HEIGHT,
                                      corner_radius=8,
                                      from_=self.MIN_Z, to=self.MAX_Z,
                                      command=self.update_Z)
        self.Z_slider.grid(row=1, column=0, sticky="ew")
        self.Z_slider.set(round(self.Z, 4))

        self.Z_slider_entry = ctk.CTkEntry(self.Z_frame,
                                           width=PARAMETER_ENTRY_WIDTH,
                                           placeholder_text=f"{round(self.Z, 4)}")
        self.Z_slider_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.Z_slider_entry.setvar(value=f"{round(self.Z, 4)}")

        self.Z_slider_button = ctk.CTkButton(self.Z_frame,
                                             width=PARAMETER_BUTTON_WIDTH,
                                             text="Set",
                                             command=self.set_value_Z)
        self.Z_slider_button.grid(row=1, column=2, sticky="ew", padx=10)

        # ============ r-frame =================================================
        self.r_frame = ctk.CTkFrame(self.parameters_inner_frame,
                                    width=PARAMETER_FRAME_WIDTH,
                                    height=PARAMETER_FRAME_HEIGHT)
        self.r_frame.grid(row=first_row+2, column=0, sticky="ew", pady=2)
        self.r_frame.grid_propagate(False)
        self.r_frame.columnconfigure(0, weight=2)

        self.r_slider_title = ctk.CTkLabel(
            self.r_frame,
            text=f"Reconstruction distance r "
                 f"({self.distance_unit}): {round(self.r, 4)}"
        )
        self.r_slider_title.grid(row=0, column=0, columnspan=3,
                                 sticky="ew", pady=5)

        self.r_slider = ctk.CTkSlider(self.r_frame, height=SLIDER_HEIGHT,
                                      corner_radius=8,
                                      from_=self.MIN_R, to=self.MAX_R,
                                      command=self.update_r)
        self.r_slider.grid(row=1, column=0, sticky="ew")
        self.r_slider.set(round(self.r, 4))

        self.r_slider_entry = ctk.CTkEntry(self.r_frame,
                                           width=PARAMETER_ENTRY_WIDTH,
                                           placeholder_text=f"{round(self.r, 4)}")
        self.r_slider_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.r_slider_entry.setvar(value=f"{round(self.r, 4)}")

        self.r_slider_button = ctk.CTkButton(self.r_frame,
                                             width=PARAMETER_BUTTON_WIDTH,
                                             text="Set",
                                             command=self.set_value_r)
        self.r_slider_button.grid(row=1, column=2, sticky="ew", padx=10)


        # ============ algorithm_frame (sin cambios) ===================
        self.algorithm_frame = ctk.CTkFrame(self.parameters_inner_frame,
                                            width=PARAMETER_FRAME_WIDTH,
                                            height=PARAMETER_FRAME_HEIGHT)
        self.algorithm_frame.grid(row=first_row+3, column=0, sticky="ew", pady=2)
        self.algorithm_frame.grid_propagate(False)

        for idx in range(4):
            self.algorithm_frame.columnconfigure(
                idx, weight=(1 if idx in (0, 3) else 0))

        self.algorithm_title = ctk.CTkLabel(self.algorithm_frame, text="Algorithm")
        self.algorithm_title.grid(row=0, column=1, columnspan=2,
                                  sticky="w", pady=5)

        self.as_algorithm_radio = ctk.CTkRadioButton(
            self.algorithm_frame, text="Angular Spectrum",
            variable=self.algorithm_var, value="AS")
        self.as_algorithm_radio.grid(row=1, column=0, sticky="w",
                                     padx=5, pady=5)

        self.kr_algorithm_radio = ctk.CTkRadioButton(
            self.algorithm_frame, text="Kreuzer Method",
            variable=self.algorithm_var, value="KR")
        self.kr_algorithm_radio.grid(row=1, column=1, sticky="w",
                                     padx=5, pady=5)

        self.dl_algorithm_radio = ctk.CTkRadioButton(
            self.algorithm_frame, text="DLHM",
            variable=self.algorithm_var, value="DL")
        self.dl_algorithm_radio.grid(row=1, column=2, sticky="w",
                                     padx=5, pady=5)       


        # limits_frame
        self.limits_frame = ctk.CTkFrame(self.parameters_inner_frame,
                                         width=PARAMETER_FRAME_WIDTH,
                                         height=PARAMETER_FRAME_HEIGHT + LIMITS_FRAME_EXTRA_SPACE)
        self.limits_frame.grid(row=first_row+4, column=0, sticky="ew", pady=2)
        self.limits_frame.grid_propagate(False)

        for idx in range(5):
            self.limits_frame.columnconfigure(
                idx, weight=(1 if idx in (0, 4) else 0))
        for idx in range(4):
            self.limits_frame.rowconfigure(
                idx, weight=(1 if idx in (0, 3) else 0))

        self.limit_min_label = ctk.CTkLabel(self.limits_frame, text="Minimum")
        self.limit_min_label.grid(row=1, column=0, sticky="ew", padx=5)
        self.limit_max_label = ctk.CTkLabel(self.limits_frame, text="Maximum")
        self.limit_max_label.grid(row=2, column=0, sticky="ew", padx=5)

        self.limit_L_label = ctk.CTkLabel(self.limits_frame,
                                          text=f"L")
        self.limit_L_label.grid(row=0, column=1, sticky="ew", padx=5)
        self.limit_Z_label = ctk.CTkLabel(self.limits_frame,
                                          text=f"Z")
        self.limit_Z_label.grid(row=0, column=2, sticky="ew", padx=5)
        self.limit_R_label = ctk.CTkLabel(self.limits_frame,
                                          text=f"r")
        self.limit_R_label.grid(row=0, column=3, sticky="ew", padx=5)

        self.limit_min_L_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MIN_L, 4)}")
        self.limit_min_L_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.limit_max_L_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MAX_L, 4)}")
        self.limit_max_L_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        self.limit_min_Z_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MIN_Z, 4)}")
        self.limit_min_Z_entry.grid(row=1, column=2, sticky="ew", padx=5, pady=2)
        self.limit_max_Z_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MAX_Z, 4)}")
        self.limit_max_Z_entry.grid(row=2, column=2, sticky="ew", padx=5, pady=2)

        self.limit_min_R_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MIN_R, 4)}")
        self.limit_min_R_entry.grid(row=1, column=3, sticky="ew", padx=5, pady=2)
        self.limit_max_R_entry = ctk.CTkEntry(self.limits_frame,
                                              width=PARAMETER_ENTRY_WIDTH,
                                              placeholder_text=f"{round(self.MAX_R, 4)}")
        self.limit_max_R_entry.grid(row=2, column=3, sticky="ew", padx=5, pady=2)

        self.set_limits_button = ctk.CTkButton(self.limits_frame,
                                               width=PARAMETER_BUTTON_WIDTH,
                                               text="Set all",
                                               command=self.set_limits)
        self.set_limits_button.grid(row=1, column=4, sticky="ew", padx=10)

        self.restore_limits_button = ctk.CTkButton(self.limits_frame,
                                                   width=PARAMETER_BUTTON_WIDTH,
                                                   text="Restore all",
                                                   command=self.restore_limits)
        self.restore_limits_button.grid(row=2, column=4, sticky="ew", padx=10)

        # ====================================================================
        #   RECORD SECTION  (replaces the old “Theme” OptionMenu)
        # ====================================================================

        self.record_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.record_frame.grid(row=first_row + 5, column=0,
                               sticky="ew", pady=2)
        self.record_frame.grid_propagate(False)

        for col in (0, 1, 2, 3):
            self.record_frame.columnconfigure(col, weight=1)

        ctk.CTkLabel(self.record_frame, text="Record").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )

        self.record_var = ctk.StringVar(value="Phase")
        ctk.CTkOptionMenu(
            self.record_frame,
            values=["Phase", "Amplitude", "Hologram"],
            variable=self.record_var,
            width=120
        ).grid(row=0, column=1, padx=(0, 5), pady=10, sticky="w")

        ctk.CTkButton(
            self.record_frame, text="Start", width=70,
            command=self.start_record
        ).grid(row=0, column=2, padx=(0, 5), pady=10, sticky="ew")

        ctk.CTkButton(
            self.record_frame, text="Stop", width=70,
            command=self.stop_recording
        ).grid(row=0, column=3, padx=(0, 10), pady=10, sticky="ew")

        # red “● REC” sign (hidden until recording starts)
        self.record_indicator = ctk.CTkLabel(
            self.record_frame, text="●  REC", text_color="red",
            font=ctk.CTkFont(weight="bold")
        )
        self.record_indicator.grid(row=1, column=0, columnspan=4,
                                   pady=(0, 6))
        self.record_indicator.grid_remove()   

    def start_record(self):
        """Begin capturing frames from the chosen source."""
        if self.is_recording:
            return                               # already running

        # make sure something is actually visible first
        if (self.record_var.get() == "Amplitude" and not self.amplitude_arrays) or \
           (self.record_var.get() == "Phase"     and not self.phase_arrays)     or \
           (self.record_var.get() == "Hologram"  and self.arr_c_view is None):
            ctk.messagebox.showwarning("Nothing to record",
                "Load an image / run a reconstruction before recording.")
            return

        self.record_type   = self.record_var.get()
        self.record_frames = []
        self.is_recording  = True
        self.record_indicator.grid()             # show the red dot
        print(f"[Record] started → {self.record_type}")

    def stop_recording(self):
        """Finish capture and write an .mp4 / .avi with consistent BGR frames."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_indicator.grid_remove()

        if not self.record_frames:
            print("[Record] nothing captured – aborted.")
            return

        # ── ask target file ───────────────────────────────────────
        dst = ctk.filedialog.asksaveasfilename(
            title="Save recorded video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"),
                       ("All files", "*.*")]
        )
        if not dst:                      # user cancelled
            self.record_frames.clear()
            return

        # ── homogenise all frames to BGR uint8 ─────────────────────
        bgr_frames = []
        for fr in self.record_frames:
            if fr.ndim == 2:                        # gray → BGR
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            elif fr.shape[2] == 3:                  # RGB → BGR
                fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("Unsupported frame shape "
                                 f"{fr.shape} for video output.")
            bgr_frames.append(fr)

        h, w = bgr_frames[0].shape[:2]

        # ── OpenCV writer (always isColor=True) ────────────────────
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if dst.lower().endswith(".mp4")
                                          else "XVID"))
        writer = cv2.VideoWriter(dst, fourcc, 20, (w, h), isColor=True)

        for fr in bgr_frames:
            writer.write(fr)
        writer.release()

        self.record_frames.clear()
        print(f"[Record] saved → {dst}")

    def _distance_unit_update(self, _lbl, unit: str) -> None:

        # keep a reference for later automatic updates
        self.dist_label = _lbl

        # keep StringVar in-sync  ➜  conversions elsewhere rely on it
        self._dist_unit_var.set(unit)

        # update every caption / placeholder in the UI
        self._on_distance_unit_change(unit)


    def _reset_all_images(self) -> None:
        """Forget every capture/reconstruction currently stored."""
        # left-hand
        self.original_multi_holo_arrays.clear()
        self.multi_holo_arrays.clear()
        self.hologram_frames.clear()
        # right-hand
        self.original_amplitude_arrays.clear()
        self.amplitude_arrays.clear()
        self.amplitude_frames.clear()
        self.original_phase_arrays.clear()
        self.phase_arrays.clear()
        self.phase_frames.clear()
        # filter state & indices
        self.filter_states_dim0.clear()
        self.filter_states_dim1.clear()
        self.filter_states_dim2.clear()
        self.current_left_index = self.current_amp_index = self.current_phase_index = 0
        # black placeholders
        self.arr_c_orig = np.zeros((1, 1), dtype=np.uint8)
        self.arr_r_orig = np.zeros((1, 1), dtype=np.uint8)
        # UI niceties
        self.hide_holo_arrows()


    def _sync_filter_state_from_ui(self) -> None:
        """
        Keep the internal *manual_* BooleanVars and the numeric filter
        parameters perfectly in-sync with the current state of the
        check-boxes and sliders **on every frame**.  
        Doing the synchronisation here guarantees that once a filter is
        enabled it will *stay* enabled, eliminating the “filters appear
        for a second and then vanish” problem.
        """
        # Which side of the UI are the widgets currently targeting?
        left_side_selected = self.filter_image_var.get() == "CA"

        # (checkbox, manual-var-capture, manual-var-processed,
        #  slider-widget, handler-method)
        controls = (
            (self.gamma_checkbox_var,
             self.manual_gamma_c_var,  self.manual_gamma_r_var,
             self.gamma_slider,        self.adjust_gamma),

            (self.contrast_checkbox_var,
             self.manual_contrast_c_var,  self.manual_contrast_r_var,
             self.contrast_slider,        self.adjust_contrast),

            (self.adaptative_eq_checkbox_var,
             self.manual_adaptative_eq_c_var,  self.manual_adaptative_eq_r_var,
             None,                             self.adjust_adaptative_eq),

            (self.highpass_checkbox_var,
             self.manual_highpass_c_var,  self.manual_highpass_r_var,
             self.highpass_slider,        self.adjust_highpass),

            (self.lowpass_checkbox_var,
             self.manual_lowpass_c_var,   self.manual_lowpass_r_var,
             self.lowpass_slider,         self.adjust_lowpass),   # ← fixed!
        )

        for ui_chk, man_cap, man_proc, slider, handler in controls:
            manual_var = man_cap if left_side_selected else man_proc
            manual_var.set(ui_chk.get())           # 1) copy check-box → var

            # 2) if that filter is active, refresh its numeric value
            if manual_var.get() and slider is not None:
                handler(slider.get())



    def _update_recon_arrays(self,
                             amp_arr:   np.ndarray | None = None,
                             int_arr:   np.ndarray | None = None,
                             phase_arr: np.ndarray | None = None,
                             ft_arr:    np.ndarray | None = None) -> None:
        """
        Refresh internal buffers for Amplitude, Phase, Intensity **and**
        the live Fourier Transform coming from the worker process.
        """

        # -------------------------------------------------- fetch fresh data
        out = self.recon_output
        if amp_arr   is None: amp_arr   = out.get("amp")
        if phase_arr is None: phase_arr = out.get("phase")
        if int_arr   is None: int_arr   = out.get("int")
        if ft_arr    is None: ft_arr    = out.get("ft")

        if amp_arr is None or phase_arr is None:          # nothing usable
            return

        # -------------------------------------------------- keep originals
        self.original_amplitude_arrays = [amp_arr.copy()]
        self.original_phase_arrays     = [phase_arr.copy()]

        # intensity fallback (worker may omit it)
        if int_arr is None:
            tmp = (amp_arr.astype(np.float32) / 255.0) ** 2
            int_arr = (tmp / (tmp.max() + 1e-9) * 255).astype(np.uint8)

        self.original_intensity_arrays = [int_arr.copy()]

        # -------------------------------------------------- working copies
        self.amplitude_arrays = [amp_arr.copy()]
        self.phase_arrays     = [phase_arr.copy()]
        self.intensity_arrays = [int_arr.copy()]

        # -------------------------------------------------- filter recall
        self._ensure_filter_state_lists_length()
        st_amp   = self.filter_states_dim1[0]
        st_phase = self.filter_states_dim2[0]
        """ 
        if self._filters_enabled(st_amp):
            self.amplitude_arrays[0] = self._apply_filters_from_state(
                self.amplitude_arrays[0], st_amp)

        if self._filters_enabled(st_phase):
            self.phase_arrays[0] = self._apply_filters_from_state(
                self.phase_arrays[0], st_phase)

        # -------------------------------------------------- colour-maps
        self.amplitude_arrays[0] = self._apply_ui_colormap(
            self.amplitude_arrays[0], self._active_cmap_amp)
        self.phase_arrays[0] = self._apply_ui_colormap(
            self.phase_arrays[0], self._active_cmap_phase)

        # -------------------------------------------------- live speckle
        if self._current_speckle_method() is not None:
            self._apply_live_speckle_if_active()
        """
        # -------------------------------------------------- live Fourier T
        if ft_arr is not None:
            self.ft_arrays   = [ft_arr.copy()]
            pil_ft           = Image.fromarray(ft_arr)
            self.ft_frames   = [self._preserve_aspect_ratio_right(pil_ft)]
            self.current_ft_index = 0

            # refresh left viewer if FT is currently displayed
            if self.holo_view_var.get() == "Fourier Transform":
                self.img_c = self.ft_frames[0]
                self.captured_label.configure(image=self.img_c)
                self.captured_label.configure(image=self.img_c)
        
    def _remove_legacy_show_checkboxes(self):
        """Hide the old ‘Show Intensity’ and ‘Show Phase’ tick-boxes."""
        for widget in (
            getattr(self, "square_field_checkbox", None),
            getattr(self, "Processed_Image_r_checkbox", None),
        ):
            if widget is not None:
                widget.grid_remove()

    def _unit_factor(self, unit: str) -> float:
        """Return how many µm correspond to *1 unit*."""
        return {
            "µm": 1.0,        "Micrometers": 1.0,
            "nm": 1e-3,
            "mm": 1e3,
            "cm": 1e4,
            "m" : 1e6,
            "in": 2.54e4,
        }.get(unit, 1.0)

    def _convert_dist_selector(self) -> None:
        if not hasattr(self, "dist_dummy_entry"):
            return                                          
        container = self.dist_dummy_entry.master
        for child in container.winfo_children():
            child.destroy()

        self._distance_unit_menu = ctk.CTkOptionMenu(
            container,
            values=["µm", "nm", "mm", "cm", "m", "in"],
            variable=self._dist_unit_var,
            command=self._on_distance_unit_change,
            width=90
        )
        self._distance_unit_menu.grid(row=0, column=0, sticky="ew")

    def _on_distance_unit_change(self, new_unit: str) -> None:
        """Triggered by the ▼ in ‘Distances’. Refresh everything."""
        self.distance_unit = new_unit
        self._dist_unit_var.set(new_unit)      # keep StringVar in-sync
        self._refresh_distance_unit_labels()   # redraw captions


    def _refresh_distance_unit_labels(self) -> None:
        u      = self.distance_unit
        factor = self._unit_factor(u)

        # Distances header
        self.dist_label.configure(text=f"Distances ({u})")

        # Slider captions
        self.L_slider_title.configure(
            text=f"Distance between camera and source L ({u}): "
                 f"{round(self.L / factor, 4)}")
        self.Z_slider_title.configure(
            text=f"Distance between sample and source Z ({u}): "
                 f"{round(self.Z / factor, 4)}")
        self.r_slider_title.configure(
            text=f"Reconstruction distance r ({u}): "
                 f"{round(self.r / factor, 4)}")

        # Limits-frame headings
        self.limit_L_label.configure(text=f"L ({u})")
        self.limit_Z_label.configure(text=f"Z ({u})")
        self.limit_R_label.configure(text=f"r ({u})")

        # Limits-frame placeholders
        self.limit_min_L_entry.configure(
            placeholder_text=f"{round(self.MIN_L / factor, 4)}")
        self.limit_max_L_entry.configure(
            placeholder_text=f"{round(self.MAX_L / factor, 4)}")
        self.limit_min_Z_entry.configure(
            placeholder_text=f"{round(self.MIN_Z / factor, 4)}")
        self.limit_max_Z_entry.configure(
            placeholder_text=f"{round(self.MAX_Z / factor, 4)}")
        self.limit_min_R_entry.configure(
            placeholder_text=f"{round(self.MIN_R / factor, 4)}")
        self.limit_max_R_entry.configure(
            placeholder_text=f"{round(self.MAX_R / factor, 4)}")

        # Force the per-slider numeric update as well
        self.update_parameters()

    def _run_filters_pipeline(self, img: np.ndarray,
                              use_left_side: bool) -> np.ndarray:

        if use_left_side:
            active = (self.manual_gamma_c_var.get()      or
                      self.manual_contrast_c_var.get()   or
                      self.manual_adaptative_eq_c_var.get() or
                      self.manual_highpass_c_var.get()   or
                      self.manual_lowpass_c_var.get())
        else:
            active = (self.manual_gamma_r_var.get()      or
                      self.manual_contrast_r_var.get()   or
                      self.manual_adaptative_eq_r_var.get() or
                      self.manual_highpass_r_var.get()   or
                      self.manual_lowpass_r_var.get())
        if not active:                     # nothing selected → keep original
            return img.copy()


        out = img.astype(np.float32) / 255.0

        # -------------------------------------------------------------- γ
        gamma_on  = self.manual_gamma_c_var.get()  if use_left_side else self.manual_gamma_r_var.get()
        gamma_val = self.gamma_c                  if use_left_side else self.gamma_r
        if gamma_on:
            gamma_val = max(gamma_val, 1e-3)
            out = np.power(out, gamma_val)

        # -------------------------------------------------------- contrast
        contrast_on  = self.manual_contrast_c_var.get()  if use_left_side else self.manual_contrast_r_var.get()
        contrast_val = self.contrast_c                  if use_left_side else self.contrast_r
        if contrast_on:
            mean = np.mean(out)
            out  = np.clip((out - mean) * contrast_val + mean, 0.0, 1.0)

        # ------------------------------------------------ adaptive EQ (simple HE)
        adapt_on = self.manual_adaptative_eq_c_var.get() if use_left_side else self.manual_adaptative_eq_r_var.get()
        if adapt_on:
            hist, bins = np.histogram(out.flatten(), 256, [0, 1])
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
            out = np.interp(out.flatten(), bins[:-1], cdf).reshape(out.shape)

        # --------------------------------------------------------- low‑pass
        low_on  = self.manual_lowpass_c_var.get()  if use_left_side else self.manual_lowpass_r_var.get()
        low_cut = self.lowpass_c                  if use_left_side else self.lowpass_r
        if low_on:
            sigma = max(low_cut, 0.5)
            out = ndimage.gaussian_filter(out, sigma=sigma)

        # -------------------------------------------------------- high‑pass
        high_on  = self.manual_highpass_c_var.get()  if use_left_side else self.manual_highpass_r_var.get()
        high_cut = self.highpass_c                  if use_left_side else self.highpass_r
        if high_on:
            sigma = max(high_cut, 0.5)
            low   = ndimage.gaussian_filter(out, sigma=sigma)
            out   = np.clip(out - low + 0.5, 0.0, 1.0)    # simple HP boost

        return (out * 255).astype(np.uint8)


    def _recompute_and_show(self, left: bool = False, right: bool = False):
        """Build *display* images from pristine copies + checked filters."""
        if left:
            self.arr_c_view = self._run_filters_pipeline(self.arr_c_orig,
                                                         use_left_side=True)
            self.arr_c = self.arr_c_view            # keep public reference
            im = Image.fromarray(self.arr_c_view)
            self.img_c = self._preserve_aspect_ratio_right(im)
            self.captured_label.configure(image=self.img_c)

        if right:
            self.arr_r_view = self._run_filters_pipeline(self.arr_r_orig,
                                                         use_left_side=False)
            self.arr_r = self.arr_r_view
            im = Image.fromarray(self.arr_r_view)
            self.img_r = self._preserve_aspect_ratio_right(im)
            self.processed_label.configure(image=self.img_r)

    # ------------------------------------------------------------------
    # apply_filters – now performs the work immediately on screen
    # ------------------------------------------------------------------
 
    def init_saving_frame(self):
        # Frame to activate and configure image enhancement filters
        self.so_frame = ctk.CTkFrame(self, corner_radius=8, width=SAVING_FRAME_WIDTH)
        self.so_frame.grid_propagate(False)

        self.main_title_so= ctk.CTkLabel(self.so_frame, text='Saving Options')
        self.main_title_so.grid(row=0, column=0, padx=20, pady=40, sticky='nsew')

        self.static_frame = ctk.CTkFrame(self.so_frame, width=SAVING_FRAME_WIDTH, height=SAVING_FRAME_HEIGHT)
        self.static_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.static_frame.grid_propagate(False)

        self.static_frame.columnconfigure(0, weight=1)
        self.static_frame.columnconfigure(1, weight=0)
        self.static_frame.columnconfigure(2, weight=0)
        self.static_frame.columnconfigure(3, weight=1)

        self.static_button = ctk.CTkButton(self.static_frame, text='Use static image', command=self.selectfile)
        self.static_button.grid(row=0, column=1, padx=20, pady=20)

        self.real_button = ctk.CTkButton(self.static_frame, text='Real time view', command=self.return_to_stream)
        self.real_button.grid(row=0, column=2, padx=20, pady=20)

        self.nofilter_frame = ctk.CTkFrame(self.so_frame, width=SAVING_FRAME_WIDTH, height=SAVING_FRAME_HEIGHT)
        self.nofilter_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.nofilter_frame.grid_propagate(False)

        self.nofilter_frame.columnconfigure(0, weight=1)
        self.nofilter_frame.columnconfigure(1, weight=0)
        self.nofilter_frame.columnconfigure(2, weight=0)
        self.nofilter_frame.columnconfigure(3, weight=1)

        self.nf_title_label = ctk.CTkLabel(self.nofilter_frame, text='Saved without filters')
        self.nf_title_label.grid(row=0, column=1, columnspan=2, padx=20, pady=5, sticky='nsew')

        self.nf_c_button = ctk.CTkButton(self.nofilter_frame, text='Save capture', command=self.no_filter_save_c)
        self.nf_c_button.grid(row=1, column=1, padx=20, pady=20)
        self.nf_r_button = ctk.CTkButton(self.nofilter_frame, text='Save reconstruction', command=self.no_filter_save_r)
        self.nf_r_button.grid(row=1, column=2, padx=20, pady=20)


        self.so_frame.rowconfigure(8, weight=1)
        
        self.home_button = ctk.CTkButton(self.so_frame, text='Home', command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=8, column=0, pady=20, sticky='s')

    def no_filter_save_c(self):
        '''Saves a capture with an increasing number'''
        i = 0
        while os.path.exists("saves_DLHM/capture/capture%s.bmp" % i):
            i += 1
        im_c = arr2im(self.arr_c)
        im_c.save('saves_DLHM/capture/capture%s.bmp' % i)

    def no_filter_save_r(self):
        '''Saves a reconstruction with an increasing number'''
        i = 0
        while os.path.exists("saves_DLHM/reconstruction/reconstruction%s.bmp" % i):
            i += 1

        im_r = arr2im(self.arr_r)
        im_r.save('saves_DLHM/reconstruction/reconstruction%s.bmp' % i)

    def save_reference(self):
        '''Saves a reference with an increasing number'''
        i = 0
        while os.path.exists("references/reference%s.bmp" % i):
            i += 1

        im_r = arr2im(self.arr_r)
        im_r.save('references/reference%s.bmp' % i)

    def open_settings(self):
        self.settings = True
        self.after(1000, self.close_settings)
    
    def close_settings(self):
        self.settings = False

    def _update_unit_in_label(self, lbl: ctk.CTkLabel, unit: str) -> None:
        base = lbl.cget("text").split("(")[0].strip()
        lbl.configure(text=f"{base} ({unit})")

        if   "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pitch X" in base:
            self.pitch_x_unit = unit
        elif "Pitch Y" in base:
            self.pitch_y_unit = unit
        else:  # all distances share the same selector
            self.distance_unit = unit
    
    def set_variables(self):
        """Reads wavelength + pitch X/Y and computes average pixel pitch."""
        try:
            self.wavelength = self.get_value_in_micrometers(
                self.wave_entry.get(), self.wavelength_unit
            )
        except Exception:
            self.wavelength = DEFAULT_WAVELENGTH
            print("Invalid Wavelength.")

        try:
            px = self.get_value_in_micrometers(
                self.pitchx_entry.get(), self.pitch_x_unit
            )
            py = self.get_value_in_micrometers(
                self.pitchy_entry.get(), self.pitch_y_unit
            )
            self.dxy = (px + py) / 2.0     # simple average
        except Exception:
            self.dxy = DEFAULT_DXY
            print("Invalid Pitch values.")

        print(f"Wavelength: {self.wavelength} µm    "
              f"Pixel pitch: {self.dxy} µm")

    def update_image_filters(self):
        """
        Refreshes all check-boxes and sliders so they always show the
        values that belong to the *currently selected* dimension:

            0 → Hologram (Capture)  
            1 → Amplitude (Reconstruction)  
            2 → Phase     (Reconstruction)
        """
        dim = self.filters_dimensions_var.get()

        if dim == 0:      # ── Hologram ───────────────────────────
            self.gamma_checkbox_var.set(self.manual_gamma_c_var.get())
            self.gamma_slider.set(self.gamma_c)

            self.contrast_checkbox_var.set(self.manual_contrast_c_var.get())
            self.contrast_slider.set(self.contrast_c)

            self.adaptative_eq_checkbox_var.set(self.manual_adaptative_eq_c_var.get())

            self.highpass_checkbox_var.set(self.manual_highpass_c_var.get())
            self.highpass_slider.set(self.highpass_c)

            self.lowpass_checkbox_var.set(self.manual_lowpass_c_var.get())
            self.lowpass_slider.set(self.lowpass_c)

        elif dim == 1:    # ── Amplitude ──────────────────────────
            self.gamma_checkbox_var.set(self.manual_gamma_a_var.get())
            self.gamma_slider.set(self.gamma_a)

            self.contrast_checkbox_var.set(self.manual_contrast_a_var.get())
            self.contrast_slider.set(self.contrast_a)

            self.adaptative_eq_checkbox_var.set(self.manual_adaptative_eq_a_var.get())

            self.highpass_checkbox_var.set(self.manual_highpass_a_var.get())
            self.highpass_slider.set(self.highpass_a)

            self.lowpass_checkbox_var.set(self.manual_lowpass_a_var.get())
            self.lowpass_slider.set(self.lowpass_a)

        else:             # ── Phase ──────────────────────────────
            self.gamma_checkbox_var.set(self.manual_gamma_r_var.get())
            self.gamma_slider.set(self.gamma_r)

            self.contrast_checkbox_var.set(self.manual_contrast_r_var.get())
            self.contrast_slider.set(self.contrast_r)

            self.adaptative_eq_checkbox_var.set(self.manual_adaptative_eq_r_var.get())

            self.highpass_checkbox_var.set(self.manual_highpass_r_var.get())
            self.highpass_slider.set(self.highpass_r)

            self.lowpass_checkbox_var.set(self.manual_lowpass_r_var.get())
            self.lowpass_slider.set(self.lowpass_r)

    def update_manual_filter(self):
        # mirror UI → manual_* flags
        self._sync_filter_state_from_ui()

        # persist new state right away so background refreshes respect it
        if self.filter_image_var.get() == "CA":                     # hologram pane
            self.store_filter_state(0, self.current_left_index)
        elif self.recon_view_var.get().startswith("Phase"):         # phase view
            self.store_filter_state(2, self.current_phase_index)
        else:                                                       # amplitude/intensity
            self.store_filter_state(1, self.current_amp_index)
    '''
    def update_parameters(self):
         
        self.Z_slider_title.configure(text=f'Distance between the sample and the source (Z): {round(self.Z, 4)}')
        self.Z_slider.set(round(self.Z, 4))
        self.Z_slider_entry.configure(placeholder_text=f'{round(self.Z, 4)}')
        self.L_slider_title.configure(text=f'Distance between the camera and the source (L):{round(self.L, 4)}')
        self.L_slider.set(round(self.L, 4))
        self.L_slider_entry.configure(placeholder_text=f'{round(self.L, 4)}')
        self.r_slider_title.configure(text=f'Reconstruction distance r ({self.wavelength_unit}): {round(self.r, 4)}')
        self.r_slider.set(round(self.r, 4))
        self.r_slider_entry.configure(placeholder_text=f'{round(self.r, 4)}')
        self.magnification_label.configure(text=f'Magnification:{round(self.scale_factor, 4)}')
        self.scale_factor = self.L/self.Z if self.Z!=0 else self.L/MIN_DISTANCE
    '''
    def update_parameters(self):
        u = self.distance_unit                                  
        self.Z_slider_title.configure(text=f"Distance between sample and source Z ({u}): {round(self.Z,4)}")
        self.Z_slider.set(round(self.Z, 4))
        self.Z_slider_entry.configure(placeholder_text=f"{round(self.Z,4)}")
        self.L_slider_title.configure(text=f"Distance between camera and source L ({u}): {round(self.L,4)}")
        self.L_slider.set(round(self.L, 4))
        self.L_slider_entry.configure(placeholder_text=f"{round(self.L,4)}")
        self.r_slider_title.configure(text=f"Reconstruction distance r ({u}): {round(self.r,4)}")
        self.r_slider.set(round(self.r, 4))
        self.r_slider_entry.configure(placeholder_text=f"{round(self.r,4)}")
        self.scale_factor = self.L / self.Z if self.Z != 0 else self.L / MIN_DISTANCE
        self.magnification_label.configure(text=f"Magnification: {round(self.scale_factor, 4)}")


    def update_L(self, val):
        '''Updates the value of L based on the slider'''
        self.L = val

        # Z depends on r and L, if r is fixed, Z and L move together
        if self.fix_r.get():
            self.Z = self.L-self.r
        else:
            # neither Z nor r can be larger than L
            if self.L<=self.Z:
                self.Z = self.L

            self.r = self.L-self.Z

        self.update_parameters()

    def update_Z(self, val):
        '''Updates the value of Z based on the slider'''

        self.Z = val

        # L depends on Z and r, if r is fixed L and Z move together
        # if not, r is just the difference between L and Z
        if self.fix_r.get():
            self.L = self.Z+self.r
        else:

            # L cannot be lower than Z
            if self.Z >= self.L:
                self.L = self.Z
        
            self.r = self.L-self.Z


        self.update_parameters()

    def update_r(self, val):
        '''Updates the value of r based on the slider'''

        self.r = val

        # If r is fixed, Z will be fixed since it's more probable to be correct
        if self.fix_r.get():
            self.L = self.Z+self.r
        else:
            self.Z = self.L-self.r

        self.update_parameters()


    def set_value_L(self):
        try:
            user_val = self.get_value_in_micrometers(
                           self.L_slider_entry.get(),
                           self._dist_unit_var.get())
        except Exception:
            user_val = self.L      # keep the previous value on error

        user_val = max(self.MIN_L, min(self.MAX_L, user_val))
        self.update_L(user_val)

    def set_value_Z(self):
        try:
            user_val = self.get_value_in_micrometers(
                           self.Z_slider_entry.get(),
                           self._dist_unit_var.get())
        except Exception:
            user_val = self.Z

        user_val = max(self.MIN_Z, min(self.MAX_Z, user_val))
        self.update_Z(user_val)

    def set_value_r(self):
        try:
            user_val = self.get_value_in_micrometers(
                           self.r_slider_entry.get(),
                           self._dist_unit_var.get())
        except Exception:
            user_val = self.r

        user_val = max(self.MIN_R, min(self.MAX_R, user_val))
        self.update_r(user_val)

    def set_limits(self):
     """Lee los límites introducidos y actualiza los sliders."""
     try: self.MIN_L = float(self.limit_min_L_entry.get())
     except ValueError: print("self.MIN_L received invalid value.")
     try: self.MAX_L = float(self.limit_max_L_entry.get())
     except ValueError: print("self.MAX_L received invalid value.")
     try: self.MIN_Z = float(self.limit_min_Z_entry.get())
     except ValueError: print("self.MIN_Z received invalid value.")
     try: self.MAX_Z = float(self.limit_max_Z_entry.get())
     except ValueError: print("self.MAX_Z received invalid value.")
     try: self.MIN_R = float(self.limit_min_R_entry.get())     # ← R mayúscula
     except ValueError: print("self.MIN_R received invalid value.")
     try: self.MAX_R = float(self.limit_max_R_entry.get())
     except ValueError: print("self.MAX_R received invalid value.")
 
     self.L_slider.configure(from_=self.MIN_L, to=self.MAX_L)
     self.Z_slider.configure(from_=self.MIN_Z, to=self.MAX_Z)
     self.r_slider.configure(from_=self.MIN_R, to=self.MAX_R)


    def _ensure_filter_state_lists_length(self) -> None:
        """Rellena las tablas de estado de filtros con diccionarios vacíos."""
        import tools_GUI as tGUI          # para usar default_filter_state

        def _pad(lst, target_len):
            while len(lst) < target_len:
                lst.append(tGUI.default_filter_state())

        _pad(self.filter_states_dim0, len(self.multi_holo_arrays))   # hologramas
        _pad(self.filter_states_dim1, len(self.amplitude_arrays))    # amplitud
        _pad(self.filter_states_dim2, len(self.phase_arrays))        # fase

    def restore_limits(self):
        '''Sets the parameters to their initial values'''
        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L

        self.L_slider.configure(from_=self.MIN_L, to=self.MAX_L)
        self.Z_slider.configure(from_=self.MIN_Z, to=self.MAX_Z)
        self.r_slider.configure(from_=self.MIN_R, to=self.MAX_R)

    
    def change_menu_to(self, name: str):
     """Shows one left-column pane and hides the rest."""
     if name == "home":
         name = "parameters"

     # Parameters
     self.navigation_frame.grid(row=0, column=0, sticky="nsew", padx=5) \
         if name == "parameters" else self.navigation_frame.grid_forget()
     
 
    def update_im_size(self, size):
        '''Updates scale from slider'''
        self.scale = size

    def save_capture(self, ext: str = "bmp"):
        """Open a ‘Save as…’ dialog and store the *current* hologram frame."""
        filetypes = [
            ("Bitmap", "*.bmp"),
            ("PNG",     "*.png"),
            ("TIFF",    "*.tif"),
            ("JPEG",    "*.jpg"),
            ("All",     "*.*"),
        ]
        target = ctk.filedialog.asksaveasfilename(
            title="Save hologram image",
            defaultextension=f".{ext}",
            filetypes=filetypes,
        )
        if not target:        # user bailed out
            return

        # grab the PIL.Image living inside the CTkImage and save it
        pil_img: Image.Image = self.img_c.cget("light_image")
        pil_img.save(target)
        print(f"[save_capture] Hologram saved →  {target}")


    def save_processed(self, ext: str = "bmp"):
        """Open a ‘Save as…’ dialog and store the *current* reconstruction."""
        filetypes = [
            ("Bitmap", "*.bmp"),
            ("PNG",     "*.png"),
            ("TIFF",    "*.tif"),
            ("JPEG",    "*.jpg"),
            ("All",     "*.*"),
        ]
        target = ctk.filedialog.asksaveasfilename(
            title="Save reconstruction image",
            defaultextension=f".{ext}",
            filetypes=filetypes,
        )
        if not target:
            return

        pil_img: Image.Image = self.img_r.cget("light_image")
        pil_img.save(target)
        print(f"[save_processed] Reconstruction saved →  {target}")

    def _sync_canvas_and_frame_bg(self):
     mode = ctk.get_appearance_mode()
     color = "gray15" if mode == "Dark" else "gray85"

     # Update all CTkCanvas backgrounds
     for canvas_attr in [
         "filters_canvas", "tools_canvas", "param_canvas"
     ]:
         canvas = getattr(self, canvas_attr, None)
         if canvas is not None:
             canvas.configure(background=color)
 
     # Update all CTkFrame fg_color backgrounds
     for frame_attr in [
         "filters_frame", "filters_container", "filters_inner_frame",
         "tools_frame", "tools_container", "tools_inner_frame",
         "navigation_frame", "param_container", "parameters_inner_frame",
         "viewing_frame", "navigation_frame", "image_frame",
         "options_frame", "dimensions_frame", "speckle_filters_frame",
         "Tools_frame",  # QPI main frame
         # Add any other frames you want to sync here
     ]:
         frame = getattr(self, frame_attr, None)
         if frame is not None:
             frame.configure(fg_color=color)
   
    def after_idle_setup(self):
        self._hide_parameters_nav_button()
        self._convert_dist_selector()
        self._sync_canvas_and_frame_bg()
        self._remove_legacy_show_checkboxes()
        self._customize_bio_analysis() 

    def _customize_bio_analysis(self) -> None:
        """
        • Remove the whole “QPI Measurements” section.
        • Hide BOTH the entry *and* the label that said
          “Lateral Magnification”.
        • Re-pack remaining widgets and refresh the scroll area.
        """

        # ---------- 1. delete QPI pane --------------------------------
        if hasattr(self, "QPI_frame"):
            self.QPI_frame.destroy()
            delattr(self, "QPI_frame")

        # ---------- 2. delete magnification widgets ------------------
        # 2-A  entry field
        if hasattr(self, "magnification_entry"):
            try:
                self.magnification_entry.destroy()
            except tk.TclError:
                pass
            delattr(self, "magnification_entry")

        # 2-B  accompanying label – may live inside nested frames, so
        #      walk the tree recursively and nuke the first match
        def _remove_mag_label(widget):
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkLabel) and \
                   "Lateral Magnification" in child.cget("text"):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
                    return True
                # descend into frames/containers
                if isinstance(child, (ctk.CTkFrame, tk.Frame, ctk.CTkCanvas)):
                    if _remove_mag_label(child):
                        return True
            return False

        if hasattr(self, "dimensions_frame"):
            _remove_mag_label(self.dimensions_frame)

        # 2-C  expand pixel-size widgets to reclaim space (optional)
        if hasattr(self, "pixel_size_entry"):
            try:
                self.pixel_size_entry.grid_configure(columnspan=3)
            except tk.TclError:
                pass

        # ---------- 3. refresh scroll region -------------------------
        if hasattr(self, "bio_inner_frame") and hasattr(self, "bio_canvas"):
            self.bio_inner_frame.update_idletasks()
            self.bio_canvas.configure(
                scrollregion=self.bio_canvas.bbox("all")
            )

    def change_appearance_mode_event(self, new_appearance_mode):
     if new_appearance_mode == "🏠 Main Menu":
         self.open_main_menu()
     else:
         ctk.set_appearance_mode(new_appearance_mode)
         self._sync_canvas_and_frame_bg()

    def open_main_menu(self):
     if hasattr(self, '_draw_after_id'):
            self.after_cancel(self._draw_after_id)
     self.destroy()
     # replace 'main_menu' with the actual module name where MainMenu lives
     main_mod = import_module("Main_")
     reload(main_mod)

     MainMenu = getattr(main_mod, "MainMenu")
     MainMenu().mainloop()



    def selectfile(self):
        """Load a hologram image from disk and trigger one reconstruction."""
        fp = ctk.filedialog.askopenfilename(title="Select an image file")
        if not fp:
            return

        # -------- raw hologram ----------------------------------------
        im = Image.open(fp).convert("L")
        self.arr_c_orig         = np.array(im)
        self.current_holo_array = self.arr_c_orig.copy()

        # pristine & working copies
        self.original_multi_holo_arrays.append(self.arr_c_orig.copy())
        self.multi_holo_arrays.append(self.arr_c_orig.copy())

        # CTkImage for left viewer
        self._recompute_and_show(left=True)              # honours sliders
        self.hologram_frames.append(self.img_c)
        self.current_left_index = len(self.multi_holo_arrays) - 1

        # -------- synchronous reconstruction --------------------------
        self.update_inputs("reconstruction")
        self.queue_manager["reconstruction"]["input"].put(self.recon_input)
        try:
            out = self.queue_manager["reconstruction"]["output"].get(timeout=1)
            self.recon_output.update(out)
        except Exception as e:
            print("[selectfile] reconstruction timed-out →", e)
            return

        # Assume reconstructor returns an 8-bit image in ‘image’
        self.arr_r_orig = self.recon_output["image"]
        if self.arr_r_orig is None:
            print("Reconstruction returned no data."); return

        # For now we store the same array as both amp & phase placeholders
        amp_8u  = self.arr_r_orig.copy()
        phase_8u = self.arr_r_orig.copy()

        # --- amplitude ------------------------------------------------
        self.original_amplitude_arrays.append(amp_8u.copy())
        self.amplitude_arrays.append(amp_8u.copy())
        pil_amp = tGUI.apply_matplotlib_colormap(self, amp_8u,
                                                 self.colormap_amp_var.get())
        self.amplitude_frames.append(self._preserve_aspect_ratio_right(pil_amp))

        # --- phase ----------------------------------------------------
        self.original_phase_arrays.append(phase_8u.copy())
        self.phase_arrays.append(phase_8u.copy())
        pil_ph = tGUI.apply_matplotlib_colormap(self, phase_8u,
                                                self.colormap_phase_var.get())
        self.phase_frames.append(self._preserve_aspect_ratio_right(pil_ph))

        # show first reconstruction on the right viewer
        self._recompute_and_show(right=True)

        # resize filter-state tables if necessary
        while len(self.filter_states_dim1) < len(self.amplitude_arrays):
            self.filter_states_dim1.append(tGUI.default_filter_state())
        while len(self.filter_states_dim2) < len(self.phase_arrays):
            self.filter_states_dim2.append(tGUI.default_filter_state())

        # show / hide navigation arrows on left pane
        if len(self.multi_holo_arrays) > 1:
            self.show_holo_arrows()
        else:
            self.hide_holo_arrows()

    def selectfile(self):
        """Load a hologram image from disk and trigger one reconstruction."""
        fp = ctk.filedialog.askopenfilename(title="Select an image file")
        if not fp:
            return

        # -------- raw hologram ----------------------------------------
        im = Image.open(fp).convert("L")
        self.arr_c_orig         = np.array(im)
        self.current_holo_array = self.arr_c_orig.copy()

        # pristine & working copies
        self.original_multi_holo_arrays.append(self.arr_c_orig.copy())
        self.multi_holo_arrays.append(self.arr_c_orig.copy())

        # CTkImage for left viewer
        self._recompute_and_show(left=True)              # honours sliders
        self.hologram_frames.append(self.img_c)
        self.current_left_index = len(self.multi_holo_arrays) - 1

        # -------- synchronous reconstruction --------------------------
        self.update_inputs("reconstruction")
        self.queue_manager["reconstruction"]["input"].put(self.recon_input)
        try:
            out = self.queue_manager["reconstruction"]["output"].get(timeout=1)
            self.recon_output.update(out)
        except Exception as e:
            print("[selectfile] reconstruction timed-out →", e)
            return

        # Robust pick: prefer 'image', fall back to 'amp'
        self.arr_r_orig = self.recon_output.get("image") \
                          or self.recon_output.get("amp")

        if self.arr_r_orig is None:
            print("Reconstruction returned no data."); return

        # For now we store the same array as both amp & phase placeholders
        amp_8u  = self.arr_r_orig.copy()
        phase_8u = self.arr_r_orig.copy()

        # --- amplitude ------------------------------------------------
        self.original_amplitude_arrays.append(amp_8u.copy())
        self.amplitude_arrays.append(amp_8u.copy())
        pil_amp = tGUI.apply_matplotlib_colormap(self, amp_8u,
                                                 self.colormap_amp_var.get())
        self.amplitude_frames.append(self._preserve_aspect_ratio_right(pil_amp))

        # --- phase ----------------------------------------------------
        self.original_phase_arrays.append(phase_8u.copy())
        self.phase_arrays.append(phase_8u.copy())
        pil_ph = tGUI.apply_matplotlib_colormap(self, phase_8u,
                                                self.colormap_phase_var.get())
        self.phase_frames.append(self._preserve_aspect_ratio_right(pil_ph))

        # show first reconstruction on the right viewer
        self._recompute_and_show(right=True)

        # resize filter-state tables if necessary
        while len(self.filter_states_dim1) < len(self.amplitude_arrays):
            self.filter_states_dim1.append(tGUI.default_filter_state())
        while len(self.filter_states_dim2) < len(self.phase_arrays):
            self.filter_states_dim2.append(tGUI.default_filter_state())

        # show / hide navigation arrows on left pane
        if len(self.multi_holo_arrays) > 1:
            self.show_holo_arrows()
        else:
            self.hide_holo_arrows()


    def selectref(self):
        self.ref_path = ctk.filedialog.askopenfilename(title='Selecciona un archivo de imagen')

    def resetref(self):
        self.ref_path = ''

    def return_to_stream(self):
        self.file_path = ''

    
    def draw(self):
        """Main refresh loop (~20 fps): capture ➜ reconstruct ➜ display."""
        start = time.time()

        # ---------- A) CAPTURE PIPELINE --------------------------------
        # build filter list for the live hologram
        self.filters_c, self.filter_params_c = [], []
        if self.manual_contrast_c_var.get():
            self.filters_c += ["contrast"];  self.filter_params_c += [self.contrast_c]
        if self.manual_gamma_c_var.get():
            self.filters_c += ["gamma"];     self.filter_params_c += [self.gamma_c]
        if self.manual_adaptative_eq_c_var.get():
            self.filters_c += ["adaptative_eq"]; self.filter_params_c += [[]]
        if self.manual_highpass_c_var.get():
            self.filters_c += ["highpass"];  self.filter_params_c += [self.highpass_c]
        if self.manual_lowpass_c_var.get():
            self.filters_c += ["lowpass"];   self.filter_params_c += [self.lowpass_c]

        # send current settings to the capture worker
        self.update_inputs("capture")
        if not self.queue_manager["capture"]["input"].full():
            self.queue_manager["capture"]["input"].put(self.capture_input)

        # pull the latest frame (if any) from the worker
        if not self.queue_manager["capture"]["output"].empty():
            self.capture_output.update(self.queue_manager["capture"]["output"].get())
            self.update_outputs("capture")

        # ---------- B) RECONSTRUCTION PIPELINE -------------------------
        # build filter list for reconstruction
        self.filters_r, self.filter_params_r = [], []
        if self.manual_contrast_r_var.get():
            self.filters_r += ["contrast"];  self.filter_params_r += [self.contrast_r]
        if self.manual_gamma_r_var.get():
            self.filters_r += ["gamma"];     self.filter_params_r += [self.gamma_r]
        if self.manual_adaptative_eq_r_var.get():
            self.filters_r += ["adaptative_eq"]; self.filter_params_r += [[]]
        if self.manual_highpass_r_var.get():
            self.filters_r += ["highpass"];  self.filter_params_r += [self.highpass_r]
        if self.manual_lowpass_r_var.get():
            self.filters_r += ["lowpass"];   self.filter_params_r += [self.lowpass_r]

        # push job
        self.update_inputs("reconstruction")
        if not self.queue_manager["reconstruction"]["input"].full():
            self.queue_manager["reconstruction"]["input"].put(self.recon_input)

        # pull result
        if not self.queue_manager["reconstruction"]["output"].empty():
            self.recon_output.update(self.queue_manager["reconstruction"]["output"].get())
            self._update_recon_arrays()          # refresh internal buffers
            self.update_right_view()             # repaint right viewer

        # ---------- C) PROFILING --------------------------------------
        elapsed = time.time() - start
        fps = round(1.0 / elapsed, 1) if elapsed else 0.0
        self.max_w_fps = max(self.max_w_fps, min(fps, 144))
        self.w_fps     = fps or self.max_w_fps

        # update on-screen counters
        self.w_fps_label.configure(text=f"FPS: {self.w_fps}")
        self.c_fps_label.configure(text=f"FPS: {self.c_fps}")
        self.r_fps_label.configure(text=f"FPS: {self.r_fps}")

        # ---------- C) RECORD THE CURRENT FRAME  ---------------------
        self._record_current_frame()

        # ---------- D) PROFILING -------------------------------------
        elapsed = time.time() - start
        fps = round(1.0 / elapsed, 1) if elapsed else 0.0
        self.max_w_fps = max(self.max_w_fps, min(fps, 144))
        self.w_fps     = fps or self.max_w_fps

        self.w_fps_label.configure(text=f"FPS: {self.w_fps}")
        self.c_fps_label.configure(text=f"FPS: {self.c_fps}")
        self.r_fps_label.configure(text=f"FPS: {self.r_fps}")

        # schedule next loop
        self._draw_after_id = self.after(50, self.draw)


    def _record_current_frame(self):
        """Store the current viewer frame (always uint8) into the buffer."""
        if not self.is_recording:
            return

        if self.record_type == "Hologram":
            src = self.arr_c_view
        else:                               # Amplitude or Phase
            src = self.arr_r_view

        if src is not None and src.size:
            self.record_frames.append(src.copy())

    def check_current_FC(self):
        self.FC = filtcosenoF(self.cosine_period, np.array((self.width, self.height)))
        plt.imshow(self.FC, cmap='gray')
        plt.show()

    def set_FC_param(self, cosine_period):
        self.cosine_period = cosine_period

    def reset_FC_param(self):
        self.cosine_period = DEFAULT_COSINE_PERIOD


    def release(self):
        """Safely terminate background processes before closing."""
        try:
            self.capture.terminate()
            self.capture.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.reconstruction.terminate()
            self.reconstruction.join(timeout=0.5)
        except Exception:
            pass
        # fallback   (Windows only, same as before)
        os.system("taskkill /f /im python.exe")

    def _build_fps_indicators(self) -> None:

        # Called too soon? – postpone and return
        if not hasattr(self, "viewing_frame"):
            self.after_idle(self._build_fps_indicators)
            return

        # one extra row at the bottom of the viewing column
        self.viewing_frame.grid_rowconfigure(2, weight=0)

        # transparent so it adopts the current theme colour
        self.fps_frame = ctk.CTkFrame(self.viewing_frame, fg_color="transparent")
        self.fps_frame.grid(row=2, column=0, sticky="sew", padx=4, pady=(0, 6))
        self.fps_frame.grid_columnconfigure((0, 1, 2), weight=1)

        #      ▸ Whole loop FPS   ▸ Capture FPS   ▸ Recon FPS
        self.w_fps_label = ctk.CTkLabel(self.fps_frame, text="FPS: 0.0", anchor="w")
        self.c_fps_label = ctk.CTkLabel(self.fps_frame, text="FPS: 0.0")
        self.r_fps_label = ctk.CTkLabel(self.fps_frame, text="FPS: 0.0", anchor="e")

        self.w_fps_label.grid(row=0, column=0, sticky="w", padx=(2, 0))
        self.c_fps_label.grid(row=0, column=1)
        self.r_fps_label.grid(row=0, column=2, sticky="e", padx=(0, 2))



if __name__=='__main__':
    app = App()
    #app.check_current_FC()
    app.draw()
    app.mainloop()
    app.release()