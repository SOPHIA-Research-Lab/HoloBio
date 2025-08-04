# Standard Library
import customtkinter as ctk
import os
from multiprocessing import Process
from matplotlib import colormaps as mpl_cmaps
from parallel_rc import *
from PIL import ImageTk, Image
from scipy import ndimage
import tkinter as tk
from matplotlib.widgets import RectangleSelector
import warnings
import matplotlib.pyplot as plt
from importlib import import_module, reload
import tools_GUI as tGUI
import functions_GUI as fGUI


class App(ctk.CTk):

    class _DummyEntry:
        #removed “LateralMagnification” entry.
        def __init__(self, value: float):
            self._txt = str(value)          # stored as text – matches real CTkEntry.get()
        def get(self) -> str:               # tools_GUI calls .get().strip()
            return self._txt

    @staticmethod
    def _patched_apply_dimensions(app):
        # Always provide a fresh dummy entry with the *current* magnification
        app.magnification_entry = App._DummyEntry(app.scale_factor)
        # Delegate to the original implementation preserved below
        return tGUI._orig_apply_dimensions(app)

    if not hasattr(tGUI, "_orig_apply_dimensions"):
        # Keep an untouched copy of the factory version
        tGUI._orig_apply_dimensions = tGUI.apply_dimensions
        # Replace it globally – all existing callbacks keep working
        tGUI.apply_dimensions = _patched_apply_dimensions


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

        self.arr_c_orig = self.arr_c.copy()
        self.arr_r_orig = self.arr_r.copy()
        self.arr_c_view = self.arr_c.copy()
        self.arr_r_view = self.arr_r.copy()

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

        # keep wavelength unit and add separated pitch-units
        self.wavelength_unit = "µm"
        self.pitch_x_unit = "µm"
        self.pitch_y_unit = "µm"
        self.distance_unit = "µm"
    
        self._dist_unit_var = tk.StringVar(
        value=getattr(self, "distance_unit", "µm"))

        self.param_entries = {}
        self.ft_mode_var = tk.StringVar(self, value="With logarithmic scale")

        # checkboxes for speckle panel
        self.compare_side_by_side_var = tk.BooleanVar()
        self.compare_speckle_plot_var = tk.BooleanVar()
        self.compare_line_profile_var = tk.BooleanVar()

        # Phase (r) manual flags already existed here
        self.manual_lowpass_r_var = ctk.BooleanVar(self, value=False)

        # NEW: give Tools-GUI the “_a_” flags it expects
        self._add_amplitude_filter_vars()        

        # Initialize frames
        self._sync_canvas_and_frame_bg()
        self.init_viewing_frame()
        self.init_saving_frame()
        self.update_inputs()
        self.after(0, self.after_idle_setup)
        self.after(0, self.draw)
        self._init_colormap_settings()
        self._init_data_containers()
        self.init_all_frames()
        fGUI.init_speckles_frame(self)
        tGUI.apply_matplotlib_colormap = App._safe_apply_matplotlib_colormap


    def update_inputs(self, process: str = ''):
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
     im = Image.fromarray(arr)
     tk_img = ImageTk.PhotoImage(im)
     lbl = tk.Label(win, image=tk_img)
     lbl.image = tk_img
     lbl.pack()


    def _init_data_containers(self) -> None:
        """
        Creates all attributes that tools_GUI relies on, with safe defaults.
        Call this ONCE from __init__ before any frame is built.
        """
        # Holograms
        self.current_holo_array = np.zeros((1, 1), dtype=np.uint8)
        self.original_multi_holo_arrays = []
        self.multi_holo_arrays = []
        self.hologram_frames = []
        self.current_left_index = 0

        # Reconstructions
        self.amplitude_arrays = []
        self.original_amplitude_arrays = []
        self.amplitude_frames = []
        self.current_amp_index = 0

        self.phase_arrays  = []
        self.original_phase_arrays = []
        self.phase_frames = []
        self.current_phase_index = 0

        self.intensity_arrays          = []
        self.original_intensity_arrays = []
        self.intensity_frames          = []
        self.current_int_index         = 0

        # Complex fields for SPP filter
        self.complex_fields = []

        # Per-image filter state memory (Filters panel)
        self.filter_states_dim0 = []
        self.filter_states_dim1 = []
        self.filter_states_dim2 = []

        # Last results of a speckle filter
        self.filtered_amp_array = None
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
            "Wistia":   "Wistia",
        }
        return table.get(ui_name, ui_name.lower())


    def _apply_ui_colormap(self, arr8u: np.ndarray, ui_name: str) -> np.ndarray:
        """
        Return *arr8u* converted to RGB with the UI-selected colormap.
        If `ui_name` is “Original” or the array is already RGB, it is
        returned unchanged.
        """
        if ui_name == "Original" or arr8u.ndim == 3:
            return arr8u
        cmap = mpl_cmaps[self._map_ui_to_mpl_cmap(ui_name)]
        norm = arr8u.astype(np.float32) / 255.0
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        return rgb


    def _safe_apply_matplotlib_colormap(self,arr8u : np.ndarray,ui_name: str):
        """
        Drop-in replacement for the old helper.
        Returns a PIL.Image – never crashes if the array is already RGB.
        """
        rgb = self._apply_ui_colormap(arr8u, ui_name)
 
        if rgb.ndim == 2:
            rgb = np.stack([rgb]*3, axis=-1)

        from PIL import Image
        return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


    def _add_amplitude_filter_vars(self) -> None:
        """
        Tools-GUI expects a full set of “manual_*_a_var” flags and
        numeric parameters for the *Amplitude* dimension (index 1).
        They did not exist, which is why every access raised
        AttributeError.  
        We create them **and** keep them *wired* to the Phase ones
        so both views (Amplitude / Phase) stay consistent.
        """
        # Boolean flags – we simply alias them to the Phase (r) ones
        self.manual_gamma_a_var = self.manual_gamma_r_var
        self.manual_contrast_a_var = self.manual_contrast_r_var
        self.manual_adaptative_eq_a_var = self.manual_adaptative_eq_r_var
        self.manual_highpass_a_var = self.manual_highpass_r_var
        self.manual_lowpass_a_var = self.manual_lowpass_r_var

        # Numeric parameters – mirror Phase values initially
        self.gamma_a = self.gamma_r
        self.contrast_a = self.contrast_r
        self.adaptative_eq_a = self.adaptative_eq_r
        self.highpass_a = self.highpass_r
        self.lowpass_a = self.lowpass_r


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
    

    def init_all_frames(self) -> None:
        self.apply_dimensions = lambda: tGUI.apply_dimensions(self)
        self.apply_QPI = lambda: tGUI.apply_QPI(self)
        self.apply_microstructure = lambda: tGUI.apply_microstructure(self)

        # Callbacks needed by the helper modules
        self.apply_dimensions  = lambda: tGUI.apply_dimensions(self)
        self.apply_QPI = lambda: tGUI.apply_QPI(self)
        self.apply_microstructure = lambda: tGUI.apply_microstructure(self)

        # use the SAFE wrappers defined above
        self.apply_filters = self.apply_filters
        self.apply_colormap = self.apply_colormap

        # sliders / check-boxes helpers (unchanged)
        self.adjust_gamma = lambda v: tGUI.adjust_gamma(self, v)
        self.adjust_contrast = lambda v: tGUI.adjust_contrast(self, v)
        self.adjust_highpass = lambda v: tGUI.adjust_highpass(self, v)
        self.adjust_lowpass  = lambda v: tGUI.adjust_lowpass(self, v)
        self.adjust_adaptative_eq = lambda: tGUI.adjust_adaptative_eq(self)
        self.default_filter_state = lambda: tGUI.default_filter_state()
        self.store_filter_state = lambda d, i: tGUI.store_current_ui_filter_state(self, d, i)
        self.load_ui_from_filter_state = lambda d, i: tGUI.load_ui_from_filter_state(self, d, i)
        self.update_colormap_display = lambda: tGUI.update_colormap_display(self)

        self.speckle_exclusive_callback = \
            lambda i: tGUI.speckle_exclusive_callback(self, i)
        self.apply_speckle = lambda: tGUI.apply_speckle(self)
        self.apply_speckle_filter = self._apply_speckle_filter
        fGUI.init_filters_frame(self)

        if not hasattr(self, "filters_dimensions_var"):
            self.filters_dimensions_var = tk.IntVar(self, value=0)

        # React whenever the user toggles between Holo / Amp / Phase
        self.filters_dimensions_var.trace_add(
            "write",
            lambda *_: self.on_filters_dimensions_change()
        )

        # Keep track of the dimension we start in (Hologram = 0)
        self._last_filters_dimension = 0
        # Initialise the sliders/checkboxes for the very first image
        self.load_ui_from_filter_state(0, 0)
        self.update_image_filters()

        fGUI.init_speckles_frame(self)
        
        fGUI.init_bio_analysis_frame(
            parent=self,
            apply_dimensions_callback=self.apply_dimensions,
            apply_qpi_callback=self.apply_QPI,
            update_qpi_placeholder_callback=self.update_qpi_placeholder,
            apply_microstructure_callback=self.apply_microstructure,
            add_structure_quantification_callback=self.apply_microstructure
        )
 

    def on_filters_dimensions_change(self, *_):

        new_dim = self.filters_dimensions_var.get()
        prev_dim = getattr(self, "_last_filters_dimension", 0)

        # save state of the panel we are leaving
        if prev_dim == 0:
            self.store_filter_state(0, self.current_left_index)
        elif prev_dim == 1:
            self.store_filter_state(1, self.current_amp_index)
        else:
            self.store_filter_state(2, self.current_phase_index)

        # Tell the rest of the GUI which side on
        self.filter_image_var.set("CA" if new_dim == 0 else "PR")

        # Restore saved widgets for the new dimension
        if new_dim == 0:
            self.load_ui_from_filter_state(0, self.current_left_index)
        elif new_dim == 1:
            self.load_ui_from_filter_state(1, self.current_amp_index)
        else:
            self.load_ui_from_filter_state(2, self.current_phase_index)

        self.update_image_filters()
        self._recompute_and_show(left=(new_dim == 0), right=(new_dim != 0))
        self._last_filters_dimension = new_dim


    def _current_speckle_method(self) -> int | None:
        """Return the index (0-3) of the active speckle check-box, or None."""
        if not hasattr(self, "spk_vars"):
            return None
        for i, var in enumerate(self.spk_vars):
            if var.get():
                return i
        return None


    def _refresh_after_speckle(self) -> None:
        """Swap in/out filtered arrays and repaint the right viewer."""
        active = self.speckle_applied
        idx_amp = getattr(self, "current_amp_index",   0)
        idx_phase = getattr(self, "current_phase_index", 0)

        # Amplitude
        if active and self.filtered_amp_array is not None:
            if not hasattr(self, "_amp_backup"):
                self._amp_backup = self.amplitude_arrays[idx_amp].copy()
            self.amplitude_arrays[idx_amp] = self.filtered_amp_array.copy()
        elif not active and hasattr(self, "_amp_backup"):
            self.amplitude_arrays[idx_amp] = self._amp_backup
            delattr(self, "_amp_backup")

        # Phase
        if active and self.filtered_phase_array is not None:
            if not hasattr(self, "_phase_backup"):
                self._phase_backup = self.phase_arrays[idx_phase].copy()
            self.phase_arrays[idx_phase] = self.filtered_phase_array.copy()
        elif not active and hasattr(self, "_phase_backup"):
            self.phase_arrays[idx_phase] = self._phase_backup
            delattr(self, "_phase_backup")

        # Regenerate CTkImages with the committed colour-maps
        if idx_amp < len(self.amplitude_arrays):
            pil = self._safe_apply_matplotlib_colormap(
                self.amplitude_arrays[idx_amp], self._active_cmap_amp)
            self.amplitude_frames[idx_amp] = self._preserve_aspect_ratio_right(pil)

        if idx_phase < len(self.phase_arrays):
            pil = self._safe_apply_matplotlib_colormap(
                self.phase_arrays[idx_phase], self._active_cmap_phase)
            self.phase_frames[idx_phase] = self._preserve_aspect_ratio_right(pil)

        self.update_right_view()


    def _apply_live_speckle_if_active(self) -> None:
        if not self.speckle_applied:
            return
        self._refresh_after_speckle()


    def _apply_speckle_filter(self) -> None:
        """Called by the ‘Apply’ button in the Speckle pane."""
        if self._current_speckle_method() is None:
            self.speckle_applied = False
            self.filtered_amp_array = None
            self.filtered_phase_array = None
            self._refresh_after_speckle()
            return

        # Run the heavy lifting ONCE
        tGUI.apply_speckle_filter(self)
        self.speckle_applied = True
        self._refresh_after_speckle()


    def update_qpi_placeholder(self) -> None:
        """
        Enable or disable input fields based on the selected QPI mode:
        - If mode is 2 (Thickness): enable thickness input, disable index fields.
        - Otherwise (Index mode): enable index fields, disable thickness input.
        """
        mode = self.option_meas_var.get()


    def init_viewing_frame(self) -> None:
        """
        Wrapper that assembles the whole UI.

        • `init_navigation_frame()` builds the left strip (*navigation_frame*)
          and creates –empty– `self.viewing_frame` on the right.
        • `fGUI.build_toolbar()` drops the top toolbar inside `viewing_frame`.
        • `fGUI.build_two_views_panel()` puts the twin image viewers below.
        """
        # LEFT strip  (Parameters + scrollbar)
        self.init_navigation_frame()
        self.holo_views  = [("init", self.img_c)]
        self.recon_views = [("init", self.img_r)]

        # RIGHT column: toolbar + two viewers
        fGUI.build_toolbar(self)
        fGUI.build_two_views_panel(self)


    def _ensure_frame_lists_length(self) -> None:
        def _pad(lst, target_len):
            dummy = ctk.CTkImage(light_image=Image.new("RGB", (1, 1)),
                                 size=(1, 1))
            while len(lst) < target_len:
                lst.append(dummy)

        _pad(self.hologram_frames,   len(self.multi_holo_arrays))
        _pad(self.amplitude_frames,  len(self.amplitude_arrays))
        _pad(self.phase_frames,      len(self.phase_arrays))
        _pad(self.intensity_frames,  len(self.intensity_arrays))


    def get_load_menu_values(self):
        return ["Load image", "Select reference", "Reset reference"]

    
    def _on_load_select(self, choice: str):
        {"Load image":       self.selectfile,
         "Select reference": self.selectref,
         "Reset reference":  self.resetref}.get(choice, lambda: None)()
        self.after(100, self._reset_toolbar_labels)


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
        Show either the hologram or its Fourier Transform.
        • honours the log-scale toggle
        • stores the array shown on-screen so zoom gets the right image
        """
        view = self.holo_view_var.get()
        if view == "Hologram":
            disp_arr = self.arr_c_view
            title = "Hologram"
        else:
            log = self.ft_mode_var.get().startswith("With")
            disp_arr = self._generate_ft_display(self.arr_c_view, log_scale=log)
            title = f"Fourier Transform ({'log' if log else 'linear'})"
        self._last_ft_display = disp_arr.copy()
        pil = Image.fromarray(disp_arr)
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
        amp_mode = getattr(self, "amp_mode_var",
                            tk.StringVar(value="Amplitude")).get()

        if view_name.startswith("Phase"):
            src = self.phase_arrays[self.current_phase_index]
        else:
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


    def _generate_ft_display(self, holo_array: np.ndarray, log_scale: bool = True) -> np.ndarray:
        ft_cplx = np.fft.fftshift(np.fft.fft2(holo_array.astype(np.float32)))
        mag = np.abs(ft_cplx)
        if log_scale:
            mag = np.log1p(mag)
        mag = mag / (mag.max() + 1e-9) * 255.0
        return mag.astype(np.uint8)


    def _generate_intensity_display(self, amp_array_8bit: np.ndarray) -> np.ndarray:
        amp_f = amp_array_8bit.astype(np.float32) / 255.0
        intens = amp_f ** 2
        intens = intens / (intens.max() + 1e-9) * 255.0
        return intens.astype(np.uint8)


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
        if self.holo_view_var.get() == "Fourier Transform":
            self.update_left_view()


    def _hide_parameters_nav_button(self) -> None:
        if hasattr(self, "param_button"):
            self.param_button.destroy()
        # Make “Parameters” the default view on the left‑hand column
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
        if "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pixel pitch" in base:
            self.pitch_unit = unit
        elif "Distance" in base or base.endswith("(L)") \
             or base.endswith("(Z)") or base.endswith("(r)"):
            self.distance_unit = unit


    def get_value_in_micrometers(self, value: str, unit: str) -> float:
        """Converts *value* (given in *unit*) into micrometres (µm)."""
        conversion = {
            "µm": 1.0,
            "Micrometers": 1.0,
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
        if not hasattr(self, "wavelength_unit"):
            self.wavelength_unit = "µm"
            self.pitch_unit      = "µm"
            self.distance_unit   = "µm"

        # StringVars that will track the current unit for each group
        self._wave_unit_var  = tk.StringVar(value=self.wavelength_unit)
        self._pitch_unit_var = tk.StringVar(value=self.pitch_unit)
        self._dist_unit_var  = tk.StringVar(value=self.distance_unit)

        # Wavelength and Pixel pitch
        self._make_unit_button(self.variables_frame, row=0, column=0,
                               unit_var=self._wave_unit_var,
                               label_target=self.lambda_label)
        self._make_unit_button(self.variables_frame, row=0, column=2,
                               unit_var=self._pitch_unit_var,
                               label_target=self.dxy_label)

        self._make_unit_button(self.L_frame, row=0, column=3,
                               unit_var=self._dist_unit_var,
                               label_target=self.L_slider_title)


    def _init_colormap_settings(self):
        """Centralise all colour-map related state."""
        self.available_colormaps = [
            "Original", "Viridis", "Plasma", "Inferno",
            "Magma", "Cividis", "Hot", "Cool", "Wistia"
        ]
        # user-facing (pending) choice
        self.colormap_amp_var = tk.StringVar(self, value="Original")
        self.colormap_phase_var = tk.StringVar(self, value="Original")

        # NEW → committed colormaps; images are rendered with these
        self._active_cmap_amp = "Original"
        self._active_cmap_phase = "Original"


    def show_filters_menu(self):
        if hasattr(self, "filters_choice_menu") and self.filters_choice_menu.winfo_ismapped():
            self.filters_choice_menu.grid_forget()
            return
        self.filters_choice_menu = ctk.CTkOptionMenu(
        self.options_frame,
        values=["Filters", "Bio-Analysis"],
        command=self.choose_filters_menu,
        width=270
        )
        self.filters_choice_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")


    def choose_filters_menu(self, selection: str):
     self.change_menu_to("bio" if selection == "Bio-Analysis" else "filters")


    def show_load_options(self):
        """Creates an OptionMenu with two choices: select reference or reset reference."""
        if hasattr(self, 'load_options_menu') and self.load_options_menu.winfo_ismapped():
            self.load_options_menu.grid_forget()
            return
        self.load_options_menu = ctk.CTkOptionMenu(
            self.options_frame,
            values=["Load image","Select reference", "Reset reference"],
            command=self.choose_load_option,
            width=270
        )
        # place it in the same row to the right of the Load button
        self.load_options_menu.grid(row=0, column=0, padx=5, pady=5,sticky='ew')


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
    

    def toggle_tools(self):
        if self.tools_frame.winfo_ismapped():
          self.tools_frame.grid_remove()
        else:
          self.tools_frame.grid()


    def toggle_options(self):
        if self.options_frame.winfo_ismapped():
          self.options_frame.grid_remove()
        else:
          self.options_frame.grid()


    def init_navigation_frame(self) -> None:
        """
        Re-creates the Parameters column using the modular helper
        `fGUI.create_param_with_arrow()` for wavelength + pitch X/Y, y
        mueve la línea de Magnification y el selector de unidad ▼ al header de ‘L’
        exactamente como se pidió.
        """
        # Container, canvas & scroll-bar
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
                                 font=ctk.CTkFont(size=15, weight="bold"))
        title_lbl.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        #  Three parameters header
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

        # Wavelength, pitch-X, pitch-Y
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

        # L-FRAME
        self.L_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.L_frame.grid(row=3, column=0, sticky="ew", pady=2)
        self.L_frame.grid_propagate(False)

        # Columns magnification
        for c in range(3):
            self.L_frame.columnconfigure(c, weight=(1 if c == 0 else 0))

        # L slider & entry / set button
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

        start_row = 3
        self._build_Z_r_and_remaining_frames(start_row)

        # Final scroll-region update
        self.parameters_inner_frame.update_idletasks()
        self.param_canvas.config(scrollregion=self.param_canvas.bbox("all"))
        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky="nsew")
        self.viewing_frame.grid_rowconfigure(0, weight=0)
        self.viewing_frame.grid_rowconfigure(1, weight=1)
        self.viewing_frame.grid_columnconfigure(0, weight=1)

    def _build_Z_r_and_remaining_frames(self, first_row: int):
        self.adit_options_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.adit_options_frame.grid(row=2, column=0,
                                     sticky="ew", pady=2)
        self.adit_options_frame.grid_propagate(False)

        for c in range(5):
            weight = 1 if c in (0, 4) else 0
            self.adit_options_frame.columnconfigure(c, weight=weight)
        for r in range(3):
            self.adit_options_frame.rowconfigure(r, weight=0)

        if hasattr(self, "magnification_label") and self.magnification_label.winfo_exists():
            self.magnification_label.grid_forget()
        self.magnification_label = ctk.CTkLabel(
            self.adit_options_frame,
            text=f"Magnification: {round(self.scale_factor,4)}"
        )
        self.magnification_label.grid(row=0, column=0, columnspan=5,
                                      sticky="nsew", pady=(4, 2))

        if hasattr(self, "fix_r_checkbox") and self.fix_r_checkbox.winfo_exists():
            self.fix_r_checkbox.grid_forget()
        self.fix_r_checkbox = ctk.CTkCheckBox(
            self.adit_options_frame,
            text="Fix r",
            variable=self.fix_r
        )
        self.fix_r_checkbox.grid(row=1, column=1, sticky="w",
                                 padx=10, pady=5)

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

        if hasattr(self, "dist_unit_btn") and self.dist_unit_btn.winfo_exists():
            self.dist_unit_btn.grid_forget()
        self._make_unit_button(
            parent=self.adit_options_frame,
            row=1, column=3,
            unit_var=self._dist_unit_var,
            label_target=self.dist_label
        )

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

        # r-frame
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


        self.compensate_button = ctk.CTkButton(
            self.parameters_inner_frame,
            text="Compensate",
            command=self._on_compensate,
            width=PARAMETER_BUTTON_WIDTH
        )
        # one row below the limits‑frame
        self.compensate_button.grid(row=first_row + 5,
                                    column=0,
                                    pady=(18, 8),
                                    sticky="ew")


    def _on_compensate(self) -> None:
     """Triggered by the *Compensate* button."""
     # 0 ▏make sure a worker is alive ------------------------------------------
     self._ensure_reconstruction_worker()

     # 1 ▏fresh parameters ------------------------------------------------------
     self.set_variables()          # reads wavelength / pitches
     self.set_value_L()            # reads sliders / entries
     self.set_value_Z()
     self.set_value_r()

     # 2 ▏sanity checks ---------------------------------------------------------
     if self.wavelength <= 0 or self.dxy <= 0:
         from tkinter import messagebox
         messagebox.showerror(
             "Missing parameters",
             "Please enter valid (non–zero) values\nfor Wavelength and Pixel pitch\n"
             "before pressing *Compensate*."
         )
         return 

     # 3 ▏prepare hologram (reference subtraction) -----------------------------
     holo = self.arr_c_orig.copy()
     if self.ref_path:
         try:
             ref = np.asarray(Image.open(self.ref_path).convert("L"))
             #ref = np.asarray(Image.open(self.ref_path))
             if ref.shape == holo.shape:
                 holo = holo - ref
             else:
                 print("Reference ignored → size mismatch.")
         except Exception as exc:
             print("Could not load reference →", exc)
 
     # 4 ▏send job to the worker -----------------------------------------------
     self.arr_c = holo
     self.update_inputs("reconstruction")
     self.recon_input["image"] = holo
 
     if not self.queue_manager["reconstruction"]["input"].full():
         self.queue_manager["reconstruction"]["input"].put(self.recon_input)
 
     # 5 ▏await result (gracefully handle time‑out) -----------------------------
     try:
         out = self.queue_manager["reconstruction"]["output"].get(timeout=3)
     except Exception as exc:
         print("Compensate: reconstruction timed‑out →", exc)
         self.need_recon = True          # still pending
         return
 
     # 6 ▏success → refresh viewers --------------------------------------------
     self.recon_output = out
     self._update_recon_arrays()
     self.update_right_view()
     self.need_recon = False
 

    def _distance_unit_update(self, _lbl, unit: str) -> None:
        # keep a reference for later automatic updates
        self.dist_label = _lbl

        # keep StringVar in-sync  ➜  conversions elsewhere rely on it
        self._dist_unit_var.set(unit)

        # update every caption / placeholder in the UI
        self._on_distance_unit_change(unit)


    def _reset_all_images(self) -> None:
        """Forget every capture/reconstruction currently stored."""
        # left‑hand
        self.original_multi_holo_arrays.clear()
        self.multi_holo_arrays.clear()
        self.hologram_frames.clear()

        # right‑hand
        self.original_amplitude_arrays.clear()
        self.amplitude_arrays.clear()
        self.amplitude_frames.clear()
        self.original_phase_arrays.clear()
        self.phase_arrays.clear()
        self.phase_frames.clear()
        self.original_intensity_arrays.clear()
        self.intensity_arrays.clear()
        self.intensity_frames.clear()
        self.complex_fields.clear()         

        # filter state & indices
        self.filter_states_dim0.clear()
        self.filter_states_dim1.clear()
        self.filter_states_dim2.clear()
        self.current_left_index = self.current_amp_index = self.current_phase_index = 0

        # black placeholders
        self.arr_c_orig = np.zeros((1, 1), dtype=np.uint8)
        self.arr_r_orig = np.zeros((1, 1), dtype=np.uint8)

        self.hide_holo_arrows()


    def _sync_filter_state_from_ui(self) -> None:
        """
        Keep the internal *manual_* BooleanVars and the numeric filter
        parameters perfectly in-sync with the current state of the
        check-boxes and sliders **on every frame.
        """
        # Which side of the UI are the widgets currently targeting?
        left_side_selected = self.filter_image_var.get() == "CA"

        # (checkbox, manual-var-capture, manual-var-processed, slider-widget, handler-method)
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
             self.lowpass_slider,         self.adjust_lowpass),
        )

        for ui_chk, man_cap, man_proc, slider, handler in controls:
            manual_var = man_cap if left_side_selected else man_proc
            manual_var.set(ui_chk.get())

            # If that filter is active, refresh its numeric value
            if manual_var.get() and slider is not None:
                handler(slider.get())


    def _update_recon_arrays(self,
                             amp_arr:   np.ndarray | None = None,
                             int_arr:   np.ndarray | None = None,
                             phase_arr: np.ndarray | None = None):
        """
        Refresh internal buffers (amplitude / phase / intensity) and keep
        the user-selected colormap applied, with no IndexError and no
        Matplotlib warnings.
        """

        # ─── 1 ▏pull fresh data from the worker ───────────────────
        if amp_arr is None or phase_arr is None:
            if not hasattr(self, "recon_output"):
                return
            amp_arr   = self.recon_output.get("amp")
            phase_arr = self.recon_output.get("phase")
            int_arr   = self.recon_output.get("int")
            field     = self.recon_output.get("field")  # optional
        else:
            field = None                    # user passed arrays manually

        if amp_arr is None or phase_arr is None:
            return                          # nothing to show

        # ─── 2 ▏store pristine / working copies (8‑bit) ───────────
        self.original_amplitude_arrays = [amp_arr.copy()]
        self.original_phase_arrays     = [phase_arr.copy()]
        self.amplitude_arrays          = [amp_arr.copy()]
        self.phase_arrays              = [phase_arr.copy()]

        if int_arr is not None:
            self.original_intensity_arrays = [int_arr.copy()]
            self.intensity_arrays          = [int_arr.copy()]

        # ─── 3 ▏(re)build complex field for SPP filter ────────────
        if field is None:
            # Worker did not send a complex field → synthesise it
            amp_f   = amp_arr.astype(np.float32) / 255.0          # 0…1
            phase_r = phase_arr.astype(np.float32) / 255.0 * 2*np.pi
            field   = amp_f * np.exp(1j * phase_r)
        self.complex_fields = [field]

        # ─── 4 ▏resize filter‑state tables, re‑apply saved filters ─
        self._ensure_filter_state_lists_length()

        st_amp   = self.filter_states_dim1[0]
        st_phase = self.filter_states_dim2[0]

        if self._filters_enabled(st_amp):
            self.amplitude_arrays[0] = self._apply_filters_from_state(
                self.amplitude_arrays[0], st_amp)
        if self._filters_enabled(st_phase):
            self.phase_arrays[0] = self._apply_filters_from_state(
                self.phase_arrays[0], st_phase)

        # ─── 5 ▏commit colour‑maps and refresh thumbnails ─────────
        self.amplitude_arrays[0] = self._apply_ui_colormap(
            self.amplitude_arrays[0], self._active_cmap_amp)
        self.phase_arrays[0] = self._apply_ui_colormap(
            self.phase_arrays[0], self._active_cmap_phase)

        self._apply_live_speckle_if_active()   # keep speckle preview live


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
        u = self.distance_unit
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
            active = (self.manual_gamma_c_var.get() or
                      self.manual_contrast_c_var.get() or
                      self.manual_adaptative_eq_c_var.get() or
                      self.manual_highpass_c_var.get() or
                      self.manual_lowpass_c_var.get())
        else:
            active = (self.manual_gamma_r_var.get() or
                      self.manual_contrast_r_var.get() or
                      self.manual_adaptative_eq_r_var.get() or
                      self.manual_highpass_r_var.get() or
                      self.manual_lowpass_r_var.get())
        if not active:
            return img.copy()

        # Gamma
        out = img.astype(np.float32) / 255.0
        gamma_on = self.manual_gamma_c_var.get() if use_left_side else self.manual_gamma_r_var.get()
        gamma_val = self.gamma_c if use_left_side else self.gamma_r
        if gamma_on:
            gamma_val = max(gamma_val, 1e-3)
            out = np.power(out, gamma_val)

        # Contrast
        contrast_on = self.manual_contrast_c_var.get() if use_left_side else self.manual_contrast_r_var.get()
        contrast_val = self.contrast_c if use_left_side else self.contrast_r
        if contrast_on:
            mean = np.mean(out)
            out = np.clip((out - mean) * contrast_val + mean, 0.0, 1.0)

        # Adaptive EQ
        adapt_on = self.manual_adaptative_eq_c_var.get() if use_left_side else self.manual_adaptative_eq_r_var.get()
        if adapt_on:
            hist, bins = np.histogram(out.flatten(), 256, [0, 1])
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
            out = np.interp(out.flatten(), bins[:-1], cdf).reshape(out.shape)

        # Low‑pass
        low_on  = self.manual_lowpass_c_var.get() if use_left_side else self.manual_lowpass_r_var.get()
        low_cut = self.lowpass_c if use_left_side else self.lowpass_r
        if low_on:
            sigma = max(low_cut, 0.5)
            out = ndimage.gaussian_filter(out, sigma=sigma)

        # High‑pass
        high_on = self.manual_highpass_c_var.get() if use_left_side else self.manual_highpass_r_var.get()
        high_cut = self.highpass_c if use_left_side else self.highpass_r
        if high_on:
            sigma = max(high_cut, 0.5)
            low = ndimage.gaussian_filter(out, sigma=sigma)
            out = np.clip(out - low + 0.5, 0.0, 1.0)

        return (out * 255).astype(np.uint8)


    def _recompute_and_show(self, left: bool = False, right: bool = False):
        """Build *display* images from pristine copies + checked filters."""
        if left:
            self.arr_c_view = self._run_filters_pipeline(self.arr_c_orig,
                                                         use_left_side=True)
            self.arr_c = self.arr_c_view
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
    # apply_filters
    # ------------------------------------------------------------------
    def apply_filters(self):
        """
        Runs the Tools-GUI filter pipeline **and** stores the current
        UI-filter configuration so it is recalled the next time this
        image/dimension is selected.
        """
        # Make sure the state-tables are long enough
        self._ensure_frame_lists_length()

        # Apply the filters visually (delegated to tools_GUI)
        tGUI.apply_filters(self)

        # Persist the just-applied UI state
        dim = self.filters_dimensions_var.get()
        if dim == 0:
            idx = self.current_left_index
        elif dim == 1:
            idx = self.current_amp_index
        else:
            idx = self.current_phase_index

        self.store_filter_state(dim, idx)

    def apply_colormap(self):
        # Move the *pending* values ➜ active
        self._active_cmap_amp = self.colormap_amp_var.get()
        self._active_cmap_phase = self.colormap_phase_var.get()

        # Rebuild the display buffers from the pristine copies
        for i, src in enumerate(self.original_amplitude_arrays):
            self.amplitude_arrays[i] = self._apply_ui_colormap(
                src, self._active_cmap_amp)

        for i, src in enumerate(self.original_phase_arrays):
            self.phase_arrays[i] = self._apply_ui_colormap(
                src, self._active_cmap_phase)

        # Refresh the viewer that is currently displayed
        self.update_right_view()

    def on_filters_dimensions_change(self, *args):
        """
        When the user toggles “Hologram / Amplitude / Phase” in the
        Filters pane:
          1.  Save the current widgets’ state to the *previous* image.
          2.  Restore the widgets for the *new* selection.
        """
        new_dim  = self.filters_dimensions_var.get()
        prev_dim = getattr(self, "_last_filters_dimension", None)

        # Persist the UI state of the dimension
        if prev_dim is not None:
            if prev_dim == 0:
                prev_idx = self.current_left_index
            elif prev_dim == 1:
                prev_idx = self.current_amp_index
            else:
                prev_idx = self.current_phase_index
            self.store_filter_state(prev_dim, prev_idx)

        # Update helper flag used elsewhere in the GUI
        self.filter_image_var.set("CA" if new_dim == 0 else "PR")
        # Restore the saved UI state for the newly-selected img
        if new_dim == 0:
            new_idx = self.current_left_index
        elif new_dim == 1:
            new_idx = self.current_amp_index
        else:
            new_idx = self.current_phase_index
        self.load_ui_from_filter_state(new_dim, new_idx)
        # keep the sliders / check-boxes mirror in sync
        self.update_image_filters()
        # Remember this selection for the next toggle
        self._last_filters_dimension = new_dim

    @staticmethod
    def _filters_enabled(state: dict) -> bool:
        return any((
            state["gamma_on"], state["contrast_on"],
            state["highpass_on"], state["lowpass_on"],
            state["adapt_eq_on"], state.get("speckle_on", False)
        ))


    def _apply_filters_from_state(self, arr: np.ndarray, st: dict) -> np.ndarray:
        """
        Re-implements tools_GUI.apply_all_filters_to_array() but drives it
        from a saved *state* dictionary instead of the live widgets.
        Absolutely no CTk/Tk variables are touched here.
        """
        out = arr.astype(np.float32)
        # Gamma
        if st["gamma_on"]:
            gamma = max(float(st["gamma_val"]), 1e-8)
            norm = out / (out.max() + 1e-9)
            out = np.power(norm, 1.0 / gamma) * 255.0

        # Contrast
        if st["contrast_on"]:
            cval = float(st["contrast_val"])
            mu = np.mean(out)
            out = (out - mu) * cval + mu

        # High-pass
        if st["highpass_on"]:
            cutoff = float(st["highpass_val"])
            f = np.fft.fftshift(np.fft.fft2(out))
            rows, cols = out.shape
            crow, ccol = rows // 2, cols // 2
            rad = int(min(rows, cols) * cutoff * 0.5)
            Y, X = np.ogrid[:rows, :cols]
            mask = (X - ccol)**2 + (Y - crow)**2 > rad**2
            out = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))

        # Low-pass
        if st["lowpass_on"]:
            cutoff = float(st["lowpass_val"])
            f = np.fft.fftshift(np.fft.fft2(out))
            rows, cols = out.shape
            crow, ccol = rows // 2, cols // 2
            rad = int(min(rows, cols) * cutoff * 0.5)
            Y, X = np.ogrid[:rows, :cols]
            mask = (X - ccol)**2 + (Y - crow)**2 <= rad**2
            out = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))

        # Adaptive histogram equalisation
        if st["adapt_eq_on"]:
            lo, hi = out.min(), out.max()
            span = hi - lo + 1e-9
            scaled = (out - lo) / span
            hist, bins = np.histogram(scaled.flatten(), 256, [0, 1])
            cdf = hist.cumsum() / hist.sum()
            eq = np.interp(scaled.flatten(), bins[:-1], cdf)
            out = eq.reshape(out.shape) * span + lo

        return np.clip(out, 0, 255).astype(np.uint8)

    def measure_speckle(self, sample):
        smin, smax = sample.min(), sample.max()
        if abs(smax - smin) < 1e-9:
            disp = np.zeros_like(sample, dtype=np.uint8)
        else:
            disp = ((sample - smin) / (smax - smin) * 255).astype(np.uint8)

        fig, ax = plt.subplots()
        ax.imshow(disp, cmap='gray')
        ax.set_title("Select ROI for Speckle measurement")

        # Store ROI corners in a small list
        coords = [0, 0, 0, 0]
        user_has_selected = [False]

        def onselect(eclick, erelease):
            coords[0], coords[1] = int(eclick.xdata), int(eclick.ydata)
            coords[2], coords[3] = int(erelease.xdata), int(erelease.ydata)
            user_has_selected[0] = True

        toggle_selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        plt.show()

        if not user_has_selected[0]:
            print("No ROI selected. Using default return => 0.0, 0,0,0,0")
            return 0.0, 0, 0, 0, 0, sample

        # Coordinates from list
        x1, y1, x2, y2 = coords
        # reorder if user dragged in reverse
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Measure std/mean inside that ROI from 'sample'
        roi = sample[y1:y2, x1:x2]
        std_val = np.std(roi)
        mean_val = np.mean(roi)
        if mean_val == 0:
            max_speckle_contrast = 0.0
        else:
            max_speckle_contrast = std_val / mean_val

        return max_speckle_contrast, x1, y1, x2, y2, sample

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

        if "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pitch X" in base:
            self.pitch_x_unit = unit
        elif "Pitch Y" in base:
            self.pitch_y_unit = unit
        else:  # all distances share the same selector
            self.distance_unit = unit
    

    def _ensure_reconstruction_worker(self) -> None:
     """
     If the reconstruction process crashed (e.g. ZeroDivision in propagate),
     build a brand‑new pair of Queues and spawn a fresh worker so the GUI
     keeps working without having to restart the whole application.
     """
     if getattr(self, "reconstruction", None) is not None and \
        self.reconstruction.is_alive():
         return                                    # still running – nothing to do
 
     # Kill the corpse, if any --------------------------------------------------
     try:
         if self.reconstruction is not None:
             self.reconstruction.terminate()
     except Exception:
         pass
 
     # Brand‑new, empty queues --------------------------------------------------
     from multiprocessing import Queue, Process
     self.queue_manager["reconstruction"] = {
         "input":  Queue(1),
         "output": Queue(1),
     }
 
     # Respawn the worker -------------------------------------------------------
     self.reconstruction = Process(
         target=reconstruct, args=(self.queue_manager,)
     )
     self.reconstruction.start()
 
 
    def set_variables(self) -> None:
     """
     Read the three parameter entries.  
     *Empty* or non‑numeric fields are **ignored** – we keep the previous
     (last valid) value instead of overwriting it with zero.
     """
      # --- Wavelength -----------------------------------------------------------
     txt = self.wave_entry.get().strip()
     if txt:                                               # ignore blanks
         try:
             val = self.get_value_in_micrometers(txt, self.wavelength_unit)
             if val > 0:
                 self.wavelength = val
         except Exception:
             print("Invalid Wavelength ignored → keeping previous value.")
 
     # --- Pixel pitch X / Y ----------------------------------------------------
     pxt = self.pitchx_entry.get().strip()
     pyt = self.pitchy_entry.get().strip()
     if pxt or pyt:                                        # at least one non‑blank
         try:
             px = self.get_value_in_micrometers(pxt or "0", self.pitch_x_unit)
             py = self.get_value_in_micrometers(pyt or "0", self.pitch_y_unit)
             if px > 0 and py > 0:
                 self.dxy = (px + py) * 0.5
         except Exception:
             print("Invalid pixel pitch ignored → keeping previous value.")
 


    def update_image_filters(self):
        """
        Refreshes all check-boxes and sliders so they always show the
        values that belong to the *currently selected* dimension:

            0 → Hologram (Capture)  
            1 → Amplitude (Reconstruction)  
            2 → Phase     (Reconstruction)
        """
        dim = self.filters_dimensions_var.get()

        if dim == 0:
            self.gamma_checkbox_var.set(self.manual_gamma_c_var.get())
            self.gamma_slider.set(self.gamma_c)

            self.contrast_checkbox_var.set(self.manual_contrast_c_var.get())
            self.contrast_slider.set(self.contrast_c)

            self.adaptative_eq_checkbox_var.set(self.manual_adaptative_eq_c_var.get())

            self.highpass_checkbox_var.set(self.manual_highpass_c_var.get())
            self.highpass_slider.set(self.highpass_c)

            self.lowpass_checkbox_var.set(self.manual_lowpass_c_var.get())
            self.lowpass_slider.set(self.lowpass_c)

        elif dim == 1:
            self.gamma_checkbox_var.set(self.manual_gamma_a_var.get())
            self.gamma_slider.set(self.gamma_a)

            self.contrast_checkbox_var.set(self.manual_contrast_a_var.get())
            self.contrast_slider.set(self.contrast_a)

            self.adaptative_eq_checkbox_var.set(self.manual_adaptative_eq_a_var.get())

            self.highpass_checkbox_var.set(self.manual_highpass_a_var.get())
            self.highpass_slider.set(self.highpass_a)

            self.lowpass_checkbox_var.set(self.manual_lowpass_a_var.get())
            self.lowpass_slider.set(self.lowpass_a)

        else:
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
        # Mirror UI manual_ flags
        self._sync_filter_state_from_ui()

        # Persist new state right away so background refreshes respect it
        if self.filter_image_var.get() == "CA":
            self.store_filter_state(0, self.current_left_index)
        elif self.recon_view_var.get().startswith("Phase"):
            self.store_filter_state(2, self.current_phase_index)
        else:
            self.store_filter_state(1, self.current_amp_index)

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
            if self.L <= self.Z:
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
     raw = self.L_slider_entry.get().strip()
     if raw == "":
         return                            # keep current value
     try:
         user_val = self.get_value_in_micrometers(raw, self._dist_unit_var.get())
     except Exception:
         return
     user_val = max(self.MIN_L, min(self.MAX_L, user_val))
     self.update_L(user_val)

    def set_value_Z(self):
     raw = self.Z_slider_entry.get().strip()
     if raw == "":
         return
     try:
         user_val = self.get_value_in_micrometers(raw, self._dist_unit_var.get())
     except Exception:
         return
     user_val = max(self.MIN_Z, min(self.MAX_Z, user_val))
     self.update_Z(user_val)

    def set_value_r(self):
     raw = self.r_slider_entry.get().strip()
     if raw == "":
         return
     try:
         user_val = self.get_value_in_micrometers(raw, self._dist_unit_var.get())
     except Exception:
         return
     user_val = max(self.MIN_R, min(self.MAX_R, user_val))
     self.update_r(user_val)

    def set_limits(self):
         """
         Reads the entered limits and updates the sliders.
         """
         try: self.MIN_L = float(self.limit_min_L_entry.get())
         except ValueError: print("self.MIN_L received invalid value.")
         try: self.MAX_L = float(self.limit_max_L_entry.get())
         except ValueError: print("self.MAX_L received invalid value.")
         try: self.MIN_Z = float(self.limit_min_Z_entry.get())
         except ValueError: print("self.MIN_Z received invalid value.")
         try: self.MAX_Z = float(self.limit_max_Z_entry.get())
         except ValueError: print("self.MAX_Z received invalid value.")
         try: self.MIN_R = float(self.limit_min_R_entry.get())
         except ValueError: print("self.MIN_R received invalid value.")
         try: self.MAX_R = float(self.limit_max_R_entry.get())
         except ValueError: print("self.MAX_R received invalid value.")
 
         self.L_slider.configure(from_=self.MIN_L, to=self.MAX_L)
         self.Z_slider.configure(from_=self.MIN_Z, to=self.MAX_Z)
         self.r_slider.configure(from_=self.MIN_R, to=self.MAX_R)

    def _ensure_filter_state_lists_length(self) -> None:
        """Fills filter state tables with empty dictionaries."""

        def _pad(lst, target_len):
            while len(lst) < target_len:
                lst.append(tGUI.default_filter_state())

        _pad(self.filter_states_dim0, len(self.multi_holo_arrays))
        _pad(self.filter_states_dim1, len(self.amplitude_arrays))
        _pad(self.filter_states_dim2, len(self.phase_arrays))

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
        # Filters
        self.filters_frame.grid(row=0, column=0, sticky="nsew", padx=5) \
            if name == "filters" else self.filters_frame.grid_forget()
        # Speckle
        self.speckles_frame.grid(row=0, column=0, sticky="nsew", padx=5) \
            if name == "speckle" else self.speckles_frame.grid_forget()
        # Bio-Analysis
        self.bio_frame.grid(row=0, column=0, sticky="nsew", padx=5) \
            if name == "bio" else self.bio_frame.grid_forget()
        # Saving Options
        self.so_frame.grid(row=0, column=0, sticky="nsew", padx=5) \
            if name == "so" else self.so_frame.grid_forget()
     

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
        if not target:
            return

        # Grab living inside the CTkImage and save it
        pil_img: Image.Image = self.img_c.cget("light_image")
        pil_img.save(target)

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
        self.algorithm_var.trace_add("write", self._require_new_compensation)

    def _customize_bio_analysis(self) -> None:
        """
        Remove the whole “QPI Measurements” section.
        """
        # Delete QPI pane
        if hasattr(self, "QPI_frame"):
            self.QPI_frame.destroy()
            delattr(self, "QPI_frame")

        # Delete magnification widgets
        if hasattr(self, "magnification_entry"):
            try:
                self.magnification_entry.destroy()
            except tk.TclError:
                pass
            delattr(self, "magnification_entry")

        # Accompanying label – may live inside nested frames
        def _remove_mag_label(widget):
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkLabel) and \
                   "Lateral Magnification" in child.cget("text"):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
                    return True
                # Descend into frames/containers
                if isinstance(child, (ctk.CTkFrame, tk.Frame, ctk.CTkCanvas)):
                    if _remove_mag_label(child):
                        return True
            return False

        if hasattr(self, "dimensions_frame"):
            _remove_mag_label(self.dimensions_frame)

        # Expand pixel-size widgets to reclaim space (optional)
        if hasattr(self, "pixel_size_entry"):
            try:
                self.pixel_size_entry.grid_configure(columnspan=3)
            except tk.TclError:
                pass

        # Refresh scroll region
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

        # Replace 'main_menu' with the actual module name where MainMenu lives
        main_mod = import_module("Main_")
        reload(main_mod)

        MainMenu = getattr(main_mod, "MainMenu")
        MainMenu().mainloop()


    def _require_new_compensation(self, *_):
     """
     Any change that invalidates the current reconstruction (for instance,
     picking a different algorithm) ends up here.  
     We mark the reconstruction as *out‑of‑date* and blank the right view
     until the user presses **Compensate** again.
     """
     self.need_recon = True
     #self._show_waiting_for_compensate()
 
 
    def _show_waiting_for_compensate(self):
     """
     Clear every reconstruction buffer and put a black placeholder on the
     right‑hand viewer **without** touching the titles – so they never
     change to “waiting for compensate”.
     """
     if self.arr_c_orig.size == 0:
         return                             # nothing loaded yet
 
     blank = np.zeros_like(self.arr_c_orig, dtype=np.uint8)
 
     # One neutral entry per dimension – prevents IndexError everywhere
     self.original_amplitude_arrays = [blank.copy()]
     self.original_phase_arrays     = [blank.copy()]
     self.original_intensity_arrays = [blank.copy()]
 
     self.amplitude_arrays  = [blank.copy()]
     self.phase_arrays      = [blank.copy()]
     self.intensity_arrays  = [blank.copy()]
 
     pil_blank = self._safe_apply_matplotlib_colormap(blank, "Original")
     blank_frame = self._preserve_aspect_ratio_right(pil_blank)
 
     self.amplitude_frames  = [blank_frame]
     self.phase_frames      = [blank_frame]
     self.intensity_frames  = [blank_frame]
 
     self.current_amp_index = self.current_phase_index = 0
 
     self.img_r = blank_frame
     self.processed_label.configure(image=self.img_r)
 
    def selectfile(self):
     """
     Load **one** hologram and show it on the left viewer.
     All previous reconstructions are cleared; the right viewer will stay
     blank until the user hits **Compensate**.
     """
     # Fresh start --------------------------------------------------------------
     self._reset_all_images()
 
     fp = ctk.filedialog.askopenfilename(title="Select an image file")
     if not fp:
         return
 
     # Left viewer – hologram ---------------------------------------------------
     im = Image.open(fp).convert("L")
     self.arr_c_orig = np.asarray(im)
     self.current_holo_array = self.arr_c_orig.copy()
 
     self.original_multi_holo_arrays = [self.arr_c_orig.copy()]
     self.multi_holo_arrays          = [self.arr_c_orig.copy()]
     self._recompute_and_show(left=True)
 
     self.hologram_frames   = [self.img_c]
     self.current_left_index = 0
 
     # Right viewer – completely blank -----------------------------------------
     self._show_waiting_for_compensate()
 
     # One default filter‑state per dimension
     self.filter_states_dim0 = [tGUI.default_filter_state()]
     self.filter_states_dim1 = [tGUI.default_filter_state()]
     self.filter_states_dim2 = [tGUI.default_filter_state()]
 
     # Force the user to press Compensate
     self.need_recon = True
 
     self.hide_holo_arrows()



    def selectref(self):
        self.ref_path = ctk.filedialog.askopenfilename(title='Select an image file')

    def resetref(self):
        self.ref_path = ''

    def return_to_stream(self):
        self.file_path = ''

    def draw(self):
        """Main refresh loop (≈ 20 fps)."""
        start = time.time()

        # 1) New reconstructions
        if not self.queue_manager["reconstruction"]["output"].empty():
            out = self.queue_manager["reconstruction"]["output"].get()
            self.recon_output = out
            self._update_recon_arrays()
            self.update_right_view()

        # 2) (no change) – build filter lists to send to the worker
        self.filters_c, self.filter_params_c = [], []
        if self.arr_c.size:
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

        # 3) send work to the reconstruction process
        self.update_inputs("reconstruction")
        if not getattr(self, "need_recon", False):
            if not self.queue_manager["reconstruction"]["input"].full():
                self.queue_manager["reconstruction"]["input"].put(self.recon_input)

        # 4) FPS
        elapsed = time.time() - start
        fps = round(1 / elapsed, 1) if elapsed else 0.0
        self.max_w_fps = max(getattr(self, "max_w_fps", 0), min(fps, 144))
        self.w_fps = fps or getattr(self, "w_fps", 0)
        self._draw_after_id = self.after(50, self.draw)


    def check_current_FC(self):
        self.FC = filtcosenoF(self.cosine_period, np.array((self.width, self.height)))
        plt.imshow(self.FC, cmap='gray')
        plt.show()

    def set_FC_param(self, cosine_period):
        self.cosine_period = cosine_period

    def reset_FC_param(self):
        self.cosine_period = DEFAULT_COSINE_PERIOD

    def release(self):
        # Safer
        os.system("taskkill /f /im python.exe")

if __name__=='__main__':
    app = App()
    #app.check_current_FC()
    app.draw()
    app.mainloop()
    app.release()

# # End of main_DLHM_PP.py