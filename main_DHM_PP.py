
# Standard Library
import os, zipfile, io
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from importlib import import_module, reload
import cv2
import matplotlib.pyplot as plt

# Third-Party Libraries
import customtkinter as ctk
from matplotlib.widgets import RectangleSelector
from PIL import ImageTk
from pandastable import Table
from scipy import ndimage

# Custom Modules
import pyDHM_methods as pyDHM
import functions_GUI as fGUI
import tools_GUI as tGUI
from pyDHM_methods import spatialFilteringCF, draw_manual_circle, draw_manual_rectangle
from pyDHM import utilities
from parallel_rc import *
from phaseShifting import PS5, PS4, PS3, SOSR, BPS2, BPS3

try:
    import utilities as ut
except ImportError:
    import utilities as ut


class App(ctk.CTk):
    def __init__(self):
        self.amp_mode_button = None
        self.main_menu_toolbar = None
        ctk.set_appearance_mode("Light")
        super().__init__()
        self.title('HoloBio: DHM - Offline ')
        self.attributes('-fullscreen', False)
        self.state('normal')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Get original screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Check if one dimension is twice the other
        if screen_width >= 2 * screen_height:
            # Screen is too wide, make it rectangular
            self.width = int(screen_height * 1.6)  #
            self.height = screen_height
        elif screen_height >= 2 * screen_width:
            # Screen is too tall, make it rectangular
            self.width = screen_width
            self.height = int(screen_width * 1.25)
        else:
            # Normal proportions, use original screen size
            self.width = screen_width
            self.height = screen_height

        # Set the window geometry and force update
        self.geometry(f'{self.width}x{self.height}')

        # Force window to update and then maximize/fit properly
        self.update_idletasks()

        # If using original screen size, maximize the window
        if self.width == screen_width and self.height == screen_height:
            self.state('zoomed')  # Windows maximized
        else:
            # Center the rectangular window
            x = (screen_width - self.width) // 2
            y = (screen_height - self.height) // 2
            self.geometry(f'{self.width}x{self.height}+{x}+{y}')

        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE) / 1.8
        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L

        self.L = INIT_L
        self.Z = INIT_Z
        self.r = self.L - self.Z
        self.wavelength = DEFAULT_WAVELENGTH  # µm
        self.dxy = DEFAULT_DXY  # µm
        self.scale_factor = self.L / self.Z

        # Booleans y strings
        self.fix_r = ctk.BooleanVar(self, value=False)
        self.square_field = ctk.BooleanVar(self, value=False)
        self.phase_r = ctk.BooleanVar(self, value=False)
        self.algorithm_var = ctk.StringVar(self, value='AS')

        # Paths
        self.file_path = ''
        self.ref_path = ''
        self.settings = False

        # Arrays and “photos”
        self.arr_hologram = np.zeros((int(self.width), int(self.height)))
        self.arr_phase = np.zeros((int(self.width), int(self.height)))
        self.arr_ft = np.zeros((int(self.width), int(self.height)))
        self.arr_amplitude = np.zeros((int(self.width), int(self.height)))

        im_hologram = arr2im(self.arr_hologram)
        im_phase = arr2im(self.arr_phase)
        im_ft = arr2im(self.arr_hologram)
        im_amplitude = arr2im(self.arr_phase)

        self.img_hologram = create_image(im_hologram, self.width, self.height)
        self.img_phase = create_image(im_phase, self.width, self.height)
        self.img_ft = create_image(im_ft, self.width, self.height)
        self.img_amplitude = create_image(im_amplitude, self.width, self.height)

        black_image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        self.img_black = create_image(black_image, self.width, self.height)

        self.img_hologram._size = (self.width * self.scale, self.height * self.scale)
        self.img_phase._size = (self.width * self.scale, self.height * self.scale)
        self.img_ft._size = (self.width * self.scale, self.height * self.scale)
        self.img_amplitude._size = (self.width * self.scale, self.height * self.scale)
        self.img_black._size = (self.width * self.scale, self.height * self.scale)

        self.holo_views = [
            ("Hologram", self.img_hologram),
            ("Fourier Transform", self.img_ft)
        ]
        self.current_holo_index = 0

        self.recon_views = [
            ("Phase Reconstruction ", self.img_phase),
            ("Amplitude Reconstruction ", self.img_amplitude)
        ]
        self.current_recon_index = 0

        self.current_holo_array = None
        self.current_ft_array = None
        self.current_phase_array = None
        self.current_amplitude_array = None

        # Filters
        self.gamma_checkbox_var = ctk.BooleanVar(self, value=False)
        self.contrast_checkbox_var = ctk.BooleanVar(self, value=False)
        self.adaptative_eq_checkbox_var = ctk.BooleanVar(self, value=False)
        self.highpass_checkbox_var = ctk.BooleanVar(self, value=False)
        self.lowpass_checkbox_var = ctk.BooleanVar(self, value=False)

        self.manual_gamma_c_var = ctk.BooleanVar(self, value=False)
        self.manual_gamma_ft_var = ctk.BooleanVar(self, value=False)
        self.manual_gamma_r_var = ctk.BooleanVar(self, value=False)
        self.manual_gamma_a_var = ctk.BooleanVar(self, value=False)

        self.manual_contrast_c_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_ft_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_r_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_a_var = ctk.BooleanVar(self, value=False)

        self.manual_adaptative_eq_c_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_ft_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_r_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_a_var = ctk.BooleanVar(self, value=False)

        self.manual_highpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_ft_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_r_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_a_var = ctk.BooleanVar(self, value=False)

        self.manual_lowpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_ft_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_r_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_a_var = ctk.BooleanVar(self, value=False)

        self.filters_c = []
        self.filters_r = []
        self.filter_params_c = []
        self.filter_params_r = []

        self.gamma_c = 0
        self.gamma_ft = 0
        self.gamma_r = 0
        self.gamma_a = 0

        self.contrast_c = 0
        self.contrast_ft = 0
        self.contrast_r = 0
        self.contrast_a = 0

        self.adaptative_eq_c = False
        self.adaptative_eq_ft = False
        self.adaptative_eq_r = False
        self.adaptative_eq_a = False

        self.highpass_c = 0
        self.highpass_ft = 0
        self.highpass_r = 0
        self.highpass_a = 0

        self.lowpass_c = 0
        self.lowpass_ft = 0
        self.lowpass_r = 0
        self.lowpass_a = 0

        self.speckle_checkbox_var = tk.BooleanVar(value=False)

        self.wavelength_unit = "µm"
        self.pitch_x_unit = "µm"
        self.pitch_y_unit = "µm"
        self.distance_unit = "µm"

        self.unit_symbols = {
            "Micrometers": "µm",
            "Nanometers": "nm",
            "Millimeters": "mm",
            "Centimeters": "cm",
            "Meters": "m",
            "Inches": "in"
        }

        self.unit_var = tk.StringVar(value="µm")
        self.spatial_filter_var = tk.StringVar(value="Circular")

        self.original_hologram = None
        self.phase_shift_imgs = []
        self.amplitude_arrays = []
        self.phase_arrays = []
        self.amplitude_frames = []
        self.phase_frames = []
        self.original_amplitude_arrays = []
        self.original_phase_arrays = []
        self.complex_fields = []

        self.multi_ft_arrays = []
        self.multi_holo_arrays = []
        self.original_multi_holo_arrays = []
        self.hologram_frames = []
        self.ft_frames = []

        # Keep track of last applied filter settings (so repeated Apply does nothing):
        self.last_filter_settings = None
        self.speckle_kernel_var = tk.IntVar(self, value=5)

        self.filter_states_dim0 = []  # for dimension=0 => Hologram or FT
        self.filter_states_dim1 = []  # for dimension=1 => Amplitude
        self.filter_states_dim2 = []  # for dimension=2 => Phase

        # checkboxes for speckle panel
        self.compare_side_by_side_var = tk.BooleanVar()
        self.compare_speckle_plot_var = tk.BooleanVar()
        self.compare_line_profile_var = tk.BooleanVar()

        # Initialize frames
        self._init_colormap_settings()
        self.init_viewing_frame()
        self.init_phase_shifting_frame()
        self.init_numerical_propagation_frame()
        self.init_phase_compensation_frame()
        self.init_all_frames()
        self._sync_canvas_and_frame_bg()
        fGUI.init_speckles_frame(self)

    def init_all_frames(self):
        self.apply_dimensions = lambda: tGUI.apply_dimensions(self)
        self.apply_QPI = lambda: tGUI.apply_QPI(self)
        self.apply_microstructure = lambda: tGUI.apply_microstructure(self)
        self.apply_filters = lambda: tGUI.apply_filters(self)
        self.adjust_gamma = lambda val: tGUI.adjust_gamma(self, val)
        self.adjust_contrast = lambda val: tGUI.adjust_contrast(self, val)
        self.adjust_highpass = lambda val: tGUI.adjust_highpass(self, val)
        self.adjust_lowpass = lambda val: tGUI.adjust_lowpass(self, val)
        self.adjust_adaptative_eq = lambda: tGUI.adjust_adaptative_eq(self)
        self.default_filter_state = lambda: tGUI.default_filter_state()
        self.store_filter_state = lambda dim, idx: tGUI.store_current_ui_filter_state(self, dim, idx)
        self.apply_colormap = lambda: tGUI.apply_colormap(self)
        self.update_colormap_display = lambda: tGUI.update_colormap_display(self)
        self.speckle_exclusive_callback = lambda idx: tGUI.speckle_exclusive_callback(self, idx)
        self.apply_speckle = lambda: tGUI.apply_speckle(self)
        self.apply_speckle_filter = lambda: tGUI.apply_speckle_filter(self)

        # Create Bio-Analysis frame and others
        fGUI.init_bio_analysis_frame(
            parent=self,
            apply_dimensions_callback=self.apply_dimensions,
            apply_qpi_callback=self.apply_QPI,
            update_qpi_placeholder_callback=self.update_qpi_placeholder,
            apply_microstructure_callback=self.apply_microstructure,
            add_structure_quantification_callback=self.apply_microstructure
        )

        fGUI.init_filters_frame(self)
        fGUI.init_speckles_frame(self)

    # ─────────────────────────────────────────────
    # Init view window
    # ─────────────────────────────────────────────
    def init_viewing_frame(self) -> None:
        """
        Wrapper that constructs the complete “Viewing Window” UI.

        Internally it delegates the work to two helpers so the layout
        is cleanly split:

        • `_build_nav_and_toolbar()`   → left strip and the top toolbar
        • `_build_two_views_panel()`   → side-by-side image viewers
        """
        self._build_navigation_strip()  # creates navigation_frame & viewing_frame
        fGUI.build_toolbar(self)
        fGUI.build_two_views_panel(self)

    def _build_navigation_strip(self) -> None:
        """Left column with the ‘DHM Processing Methods’ buttons."""
        # Shared geometry
        self.viewbox_width = 600
        self.viewbox_height = 500

        # Left strip
        self.navigation_frame = ctk.CTkFrame(
            self, corner_radius=8, width=MENU_FRAME_WIDTH,
            fg_color=("gray85", "gray15")
        )
        self.navigation_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)
        self.navigation_frame.grid_propagate(False)

        # Container on the right that will hold toolbar + viewers
        self.viewing_frame = ctk.CTkFrame(
            self, corner_radius=8, fg_color=("gray85", "gray15")
        )
        self.viewing_frame.grid(row=0, column=1, sticky="nsew")
        self.viewing_frame.grid_rowconfigure(1, weight=1)  # row-1 → viewers
        self.viewing_frame.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            self.navigation_frame, text="DHM Processing Methods",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=20)

        # Visual config for every button
        btn_cfg = dict(
            corner_radius=6, height=MENU_BUTTONS_HEIGHT,
            width=MENU_FRAME_WIDTH, border_spacing=10,
            fg_color=("gray75", "gray25"),
            text_color=("gray10", "gray90"),
            hover_color=("gray80", "gray20"), anchor="c"
        )
        # Grid config (NOT passed to the widget!)
        grid_cfg = dict(sticky="ew", padx=1, pady=0)

        ctk.CTkButton(
            self.navigation_frame, text="Phase-Shifting",
            command=lambda: self.change_menu_to("phase_shifting"), **btn_cfg
        ).grid(row=1, column=0, **grid_cfg)

        ctk.CTkButton(
            self.navigation_frame, text="Phase Compensation",
            command=lambda: self.change_menu_to("phase_compensation"), **btn_cfg
        ).grid(row=2, column=0, **grid_cfg)

        ctk.CTkButton(
            self.navigation_frame, text="Numerical Propagation",
            command=lambda: self.change_menu_to("numerical_propagation"), **btn_cfg
        ).grid(row=3, column=0, **grid_cfg)

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

    def _on_ft_mouse_move(self, event) -> None:
        """Translate widget coordinates → FT-array indices and update label."""
        if self.current_ft_array is None:
            return

        # geometry of the *displayed* (possibly rescaled) image
        disp_w = self.captured_label.winfo_width()
        disp_h = self.captured_label.winfo_height()
        arr_h, arr_w = self.current_ft_array.shape

        if disp_w == 0 or disp_h == 0:
            return

        # Map cursor position (event.x, event.y) → raw FT indices (u,v)
        u = int(event.x * arr_w / disp_w)
        v = int(event.y * arr_h / disp_h)

        if 0 <= u < arr_w and 0 <= v < arr_h:
            self.ft_coord_label.configure(text=f"({u}, {v})")
            self.ft_coord_label.configure(text=f"(fx:{u}, fy:{v})")
            self.ft_coord_label.lift()  # keep it on top

    def _on_theme_select(self, choice: str) -> None:
        """Handle Light / Dark selection and restore placeholder text."""
        if choice in ("Light", "Dark"):
            ctk.set_appearance_mode(choice)
            self._sync_canvas_and_frame_bg()
        self.theme_menu.set("Theme")  # keep the label constant

    def _on_tools_select(self, choice: str) -> None:
        """Direct handler for the toolbar Tools-menu."""
        if choice == "Bio-Analysis":
            self.change_menu_to("bio")
        if choice == "Filters":
            self.change_menu_to("filters")
        elif choice == "Speckle":
            self.change_menu_to("speckle")
        # Reset placeholder so the menu keeps reading “Tools”
        self.tools_menu.set("Tools")

    def _on_save_select(self, choice: str) -> None:
        """Direct handler for the toolbar Save-menu."""
        self._handle_save_option(choice)
        # Reset placeholder so the menu keeps reading “Save”
        self.save_menu.set("Save")

    def _place_holo_arrows(self) -> None:
        """Ensure arrows are gridded in row-4 if they were removed."""
        self.left_arrow_holo.grid(row=4, column=0, sticky="w",
                                  padx=20, pady=5)
        self.right_arrow_holo.grid(row=4, column=1, sticky="e",
                                   padx=20, pady=5)

    def show_holo_arrows(self) -> None:
        """Show the navigation arrows when >1 hologram is loaded."""
        self._place_holo_arrows()

    def hide_holo_arrows(self) -> None:
        """Hide the navigation arrows."""
        self.left_arrow_holo.grid_remove()
        self.right_arrow_holo.grid_remove()

    def _show_ft_mode_menu(self):
        menu = tk.Menu(self, tearoff=0)
        for opt in ("With logarithmic scale", "Without logarithmic scale"):
            menu.add_radiobutton(label=opt,value=opt,variable=self.ft_mode_var,command=self._on_ft_mode_changed)
        menu.tk_popup(self.ft_mode_button.winfo_rootx(),self.ft_mode_button.winfo_rooty() + self.ft_mode_button.winfo_height())

    def get_load_menu_values(self):
        return ["Hologram", "Stack of holograms", "Sample"]

    def _on_load_select(self, choice: str) -> None:
        """Handle selections from the Load drop-down."""
        if choice == "Hologram":
            self.load_hologram()
        elif choice == "Stack of holograms":
            self.load_images()
        elif choice == "Sample":
            self.load_image_generic()
        # Keep the placeholder text fixed to “Load”
        self.load_menu.set("Load")

    def _refresh_all_ft_views(self) -> None:
        """
        Rebuild every cached FT view (numpy array + Tk thumbnail) to honour the
        *current* value stored in ``self.ft_mode_var`` (“With logarithmic… /
        Without logarithmic…”).  After regenerating the caches, the method also
        repaints the left viewer if it is showing a Fourier-Transform.
        """
        if not hasattr(self, "ft_frames"):
            self.ft_frames = []
        if not hasattr(self, "multi_ft_arrays"):
            self.multi_ft_arrays = []
        # MULTI-HOLOGRAM path
        if getattr(self, "multi_holo_arrays", None):
            n = len(self.multi_holo_arrays)
            # resize holders
            self.ft_frames = (self.ft_frames + [None] * n)[:n]
            self.multi_ft_arrays = (self.multi_ft_arrays + [None] * n)[:n]
            for i, holo in enumerate(self.multi_holo_arrays):
                tk_ft, ft_disp = self._create_ft_frame(holo)
                self.ft_frames[i] = tk_ft
                self.multi_ft_arrays[i] = ft_disp  # keep arrays in sync for saving
            # If FT currently shown, repaint active index
            if self.holo_view_var.get() == "Fourier Transform":
                idx = getattr(self, "current_left_index", 0)
                self.captured_title_label.configure(text=f"Fourier Transform")
                self.captured_label.configure(image=self.ft_frames[idx])
                self.captured_label.image = self.ft_frames[idx]
                self.current_ft_array = self.multi_ft_arrays[idx]
            return
        # SINGLE-HOLOGRAM path
        if getattr(self, "arr_hologram", None) is None:
            return
        tk_ft, ft_disp = self._create_ft_frame(self.arr_hologram)
        self.current_ft_array = ft_disp
        if len(self.multi_ft_arrays) == 0:
            self.multi_ft_arrays.append(ft_disp)
        else:
            self.multi_ft_arrays[0] = ft_disp
        if self.holo_view_var.get() == "Fourier Transform":
            self.captured_title_label.configure(text=f"Fourier Transform")
            self.captured_label.configure(image=tk_ft)
            self.captured_label.image = tk_ft

    def _on_ft_mode_changed(self):
        if hasattr(self, "_last_ft_display"):
            try:
                delattr(self, "_last_ft_display")
            except Exception:
                pass
        self._refresh_all_ft_views()

    def _show_amp_mode_menu(self):
        menu = tk.Menu(self, tearoff=0)
        opts = ["Amplitude", "Intensities"]
        for opt in opts:
            menu.add_radiobutton(
                label=opt, value=opt,
                variable=self.amp_mode_var,
                command=self._on_amp_mode_changed
            )
        menu.tk_popup(self.amp_mode_button.winfo_rootx(),
                      self.amp_mode_button.winfo_rooty() + self.amp_mode_button.winfo_height())

    def _on_amp_mode_changed(self):
        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.update_right_view()

    def _init_ft_defaults(self):
        if not hasattr(self, "ft_mode_var"):
            self.ft_mode_var = tk.StringVar(value="With logarithmic scale")
        if not hasattr(self, "amp_mode_var"):
            self.amp_mode_var = tk.StringVar(value="Amplitude")

    def _is_log_scale_selected(self) -> bool:
        try:
            return str(self.ft_mode_var.get()) == "With logarithmic scale"
        except Exception:
            # Safe default
            return True

    def _generate_ft_display(self, holo_array: np.ndarray, log_scale: bool = True) -> np.ndarray:
        """
        Compute centered 2D-FFT magnitude (uint8) with optional log compression.
        """
        if holo_array is None or holo_array.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        f = np.fft.fftshift(np.fft.fft2(holo_array.astype(np.float32)))
        mag = np.abs(f)
        if log_scale:
            mag = np.log1p(mag)  # dynamic-range compression
        mag = mag / (mag.max() + 1e-12)
        return (mag * 255.0).astype(np.uint8)

    def _generate_intensity_display(self, amp_array_8bit: np.ndarray) -> np.ndarray:
        amp_f = amp_array_8bit.astype(np.float32) / 255.0
        intens = amp_f ** 2
        intens = intens / (intens.max() + 1e-9) * 255.0
        return intens.astype(np.uint8)

    def _preserve_aspect_ratio_right(self, pil_image: Image.Image) -> ImageTk.PhotoImage:
        max_w, max_h = self.viewbox_width, self.viewbox_height
        orig_w, orig_h = pil_image.size

        if orig_w <= max_w and orig_h <= max_h:
            resized = pil_image
        else:
            ratio_w = max_w / float(orig_w)
            ratio_h = max_h / float(orig_h)
            scale_factor = min(ratio_w, ratio_h)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def previous_hologram_view(self):
        if not self.hologram_frames:
            return
        self.current_left_index = (self.current_left_index - 1) % len(self.hologram_frames)

        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Hologram")
        else:  # Fourier Transform
            tk_ft, ft_disp = self._create_ft_frame(self.multi_holo_arrays[self.current_left_index])
            self.ft_frames[self.current_left_index] = tk_ft
            self.captured_label.configure(image=tk_ft)
            self.captured_label.image = tk_ft
            self.current_ft_array = ft_disp
            self.captured_title_label.configure(text="Fourier Transform")

        # restore filter sliders for this index
        tGUI.load_ui_from_filter_state(self, dimension=0, index=self.current_left_index)

    def next_hologram_view(self):
        if not self.hologram_frames:
            return
        self.current_left_index = (self.current_left_index + 1) % len(self.hologram_frames)

        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Hologram")
        else:  # Fourier Transform
            tk_ft, ft_disp = self._create_ft_frame(self.multi_holo_arrays[self.current_left_index])
            self.ft_frames[self.current_left_index] = tk_ft
            self.captured_label.configure(image=tk_ft)
            self.captured_label.image = tk_ft
            self.current_ft_array = ft_disp
            self.captured_title_label.configure(text="Fourier Transform")

        tGUI.load_ui_from_filter_state(self, dimension=0, index=self.current_left_index)

    def update_left_view(self):

        choice = self.holo_view_var.get()

        # No multi-hologram list yet → single array path
        if not hasattr(self, "multi_holo_arrays") or len(self.multi_holo_arrays) == 0:
            if choice == "Hologram":
                self.captured_title_label.configure(text="Hologram")
                self.captured_label.configure(image=self.img_hologram)
                self.current_holo_array = self.arr_hologram
            else:  # Fourier Transform
                tk_ft, ft_disp = self._create_ft_frame(self.arr_hologram)
                self.captured_title_label.configure(
                    text=f"Fourier Transform ({'log' if self._is_log_scale_selected() else 'linear'})")
                self.captured_label.configure(image=tk_ft)
                self.captured_label.image = tk_ft  # keep ref
                self.current_ft_array = ft_disp
            return

        # Multiple holograms loaded
        idx = getattr(self, "current_left_index", 0)
        if choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=self.hologram_frames[idx])
            self.captured_label.image = self.hologram_frames[idx]
            self.current_holo_array = self.multi_holo_arrays[idx]
        else:  # Fourier Transform
            tk_ft, ft_disp = self._create_ft_frame(self.multi_holo_arrays[idx])
            self.captured_title_label.configure(
                text=f"Fourier Transform")
            self.captured_label.configure(image=tk_ft)
            self.captured_label.image = tk_ft
            self.current_ft_array = ft_disp

        # Keep all original arrow / UI-state logic
        tGUI.load_ui_from_filter_state(self, dimension=0, index=self.current_left_index)
        if len(self.hologram_frames) > 1:
            self.show_holo_arrows()
        else:
            self.hide_holo_arrows()

        if self.holo_view_var.get() == "Fourier Transform":
            self._activate_ft_coordinate_display()
        else:
            self._deactivate_ft_coordinate_display()

    # ------------------- RIGHT-VIEW UPDATE (MOD) ---------------------- #
    def update_right_view(self):
        choice = self.recon_view_var.get()
        if choice == "Phase Reconstruction ":
            # Unchanged original phase branch
            if hasattr(self, 'phase_frames') and self.phase_frames:
                idx = getattr(self, 'current_phase_index', 0)
                self.processed_label.configure(image=self.phase_frames[idx])
                self.processed_label.image = self.phase_frames[idx]
            else:
                self.processed_label.configure(image=self.img_black)
            self.dimensions_var.set(2)
            if hasattr(self, 'current_phase_index'):
                tGUI.load_ui_from_filter_state(self, dimension=2, index=self.current_phase_index)

        else:  # Amplitude
            if hasattr(self, 'amplitude_arrays') and self.amplitude_arrays:
                idx = getattr(self, 'current_amp_index', 0)
                if self.amp_mode_var.get() == "Amplitude":
                    self.processed_label.configure(image=self.amplitude_frames[idx])
                    self.processed_label.image = self.amplitude_frames[idx]
                else:  # “Intensities”
                    amp8 = self.amplitude_arrays[idx]
                    disp_img = self._generate_intensity_display(amp8)
                    tk_amp = self._preserve_aspect_ratio_right(Image.fromarray(disp_img))
                    self.processed_label.configure(image=tk_amp)
                    self.processed_label.image = tk_amp
            else:
                self.processed_label.configure(image=self.img_black)

            self.dimensions_var.set(1)
            if hasattr(self, 'current_amp_index'):
                tGUI.load_ui_from_filter_state(self, dimension=2, index=self.current_amp_index)

        self._update_distance_label()

    def zoom_holo_view(self, *args, **kwargs):
        tGUI.zoom_holo_view(self, *args, **kwargs)

    def zoom_recon_view(self, *args, **kwargs):
        tGUI.zoom_recon_view(self, *args, **kwargs)

    def _open_zoom_view(self, *args, **kwargs):
        tGUI._open_zoom_view(self, *args, **kwargs)

    def _refresh_zoom_view(self, *args, **kwargs):
        tGUI._refresh_zoom_view(self, *args, **kwargs)

    def _get_current_array(self, target: str) -> np.ndarray | None:
        if target == "Hologram":
            return getattr(self, "current_holo_array", None)
        elif target == "Fourier Transform":
            return getattr(self, "current_ft_array", None)
        elif target == "Amplitude":
            idx = getattr(self, "current_amp_index", 0)
            arrs = getattr(self, "amplitude_arrays", [])
            if isinstance(arrs, list) and len(arrs) > idx:
                return arrs[idx]
            return None
        elif target == "Phase":
            idx = getattr(self, "current_phase_index", 0)
            arrs = getattr(self, "phase_arrays", [])
            if isinstance(arrs, list) and len(arrs) > idx:
                return arrs[idx]
            return None
        return None

    def _update_distance_label(self):
        # Decide which reconstruction view is active
        current_mode = self.recon_view_var.get()

        # Figure out the index for amplitude vs phase
        if current_mode == "Amplitude Reconstruction ":
            dim = 1
            idx = getattr(self, 'current_amp_index', 0)
        else:  # "Phase Reconstruction "
            dim = 2
            idx = getattr(self, 'current_phase_index', 0)

        multi_distances = hasattr(self, 'propagation_distances') and len(self.propagation_distances) > 1

        if multi_distances and idx < len(self.propagation_distances):
            # Valid distance for this index
            dist_um = self.propagation_distances[idx]
            dist_str = self._convert_distance_for_display(dist_um)

            if current_mode == "Amplitude Reconstruction ":
                new_title = f"Amplitude Image. Distance: {dist_str}"
            else:  # Phase
                new_title = f"Phase Image. Distance: {dist_str}"

            self.processed_title_label.configure(text=new_title)
        else:
            # Not numerical propagation or only 1 image => revert to normal titles
            if current_mode == "Amplitude Reconstruction ":
                self.processed_title_label.configure(text="Amplitude Reconstruction ")
            else:
                self.processed_title_label.configure(text="Phase Reconstruction ")

        # Hide the old distance_label_recon entirely:
        if hasattr(self, 'distance_label_recon'):
            self.distance_label_recon.configure(text="")
            self.distance_label_recon.grid_remove()

    def _convert_distance_for_display(self, dist_um: float, unit: str | None = None,
                                      include_magnification=False) -> str:
        unit = unit or getattr(self, "last_distance_unit", "µm")

        # Get active widgets based on context
        source = getattr(self, "compensation_source", "pc")
        if source == "pc":
            widgets = self.propagate_widgets_pc
        elif source == "ps":
            widgets = self.propagate_widgets_ps
        else:
            widgets = {}

        # Magnification adjustment
        if include_magnification:
            try:
                mag_str = widgets.get("propagation_magnification", None)
                if mag_str:
                    M = float(mag_str.get().replace("x", "").strip())
                    if M > 0:
                        dist_um /= (M ** 2)
            except Exception:
                pass

        # Unit conversion
        if unit == "µm":
            val = dist_um
        elif unit == "nm":
            val = dist_um * 1e3
        elif unit == "mm":
            val = dist_um / 1e3
        elif unit == "cm":
            val = dist_um / 1e4
        elif unit == "m":
            val = dist_um / 1e6
        elif unit == "in":
            val = dist_um / 25_400.0
        else:
            unit = "µm"
            val = dist_um

        return f"{val:.2f} {unit}"

    # Load / Save
    def show_load_options(self):
        if hasattr(self, 'load_options_menu') and self.load_options_menu.winfo_ismapped():
            self.load_options_menu.grid_forget()
            return
        self.load_options_menu = ctk.CTkOptionMenu(
            self.buttons_frame,
            values=["Hologram,Stack of holograms,Sample"],
            command=self.choose_load_option,
            width=270
        )
        self.load_options_menu.grid(row=0, column=0, padx=4, pady=5, sticky='w')

    def choose_load_option(self, selected_option):
        if selected_option == "Hologram":
            self.load_hologram()
        elif selected_option == "":
            self.xxx

    def _preserve_aspect_ratio(self, pil_image: Image.Image, max_width: int, max_height: int) -> ImageTk.PhotoImage:
        """
        Scales 'pil_image' down (never up) to fit within (max_width x max_height),
        preserving the original aspect ratio. Returns the resulting PhotoImage.
        """
        original_w, original_h = pil_image.size

        # If smaller or equal, no upscaling (unless you want to allow it).
        if original_w <= max_width and original_h <= max_height:
            resized = pil_image
        else:
            # Shrink to keep the aspect ratio correct
            ratio_w = max_width / float(original_w)
            ratio_h = max_height / float(original_h)
            scale_factor = min(ratio_w, ratio_h)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(resized)

    def _create_ft_frame(self, holo_array: np.ndarray):
        """
        Generate both the 8-bit display array *and* the PhotoImage that
        fits in the left view-box.  It always obeys the setting in
        `self.ft_mode_var` (“With logarithmic scale” / “Without …”).
        """
        ft_u8 = self._generate_ft_display(holo_array, log_scale=self._is_log_scale_selected())
        tk_img = self._preserve_aspect_ratio(
            Image.fromarray(ft_u8),
            self.viewbox_width, self.viewbox_height
        )
        return tk_img, ft_u8

    # ────────────────────────────────────────────────────────────
    # Load hologram for phase compensation
    # ────────────────────────────────────────────────────────────
    def load_hologram(self):
        """Re-implemented so the stored FT is *not* log‐scaled up-front."""
        # housekeeping
        self.multi_holo_arrays, self.original_multi_holo_arrays = [], []
        self.hologram_frames, self.ft_frames = [], []
        self.multi_ft_arrays, self.filter_states_dim0 = [], []
        self.current_left_index = 0

        holo_path = filedialog.askopenfilename(title='Select single hologram')
        if not holo_path:
            messagebox.showinfo(
                "Information",
                "No hologram selected."
            )
            return

        arr_gray = np.array(Image.open(holo_path).convert('L'))
        self.multi_holo_arrays.append(arr_gray)
        self.original_multi_holo_arrays.append(arr_gray.copy())
        self.filter_states_dim0.append(tGUI.default_filter_state())

        # Build FT frame according to current mode
        tk_ft, ft_disp = self._create_ft_frame(arr_gray)
        self.ft_frames = [tk_ft]
        self.multi_ft_arrays = [ft_disp]

        # Build hologram frame
        tk_holo = self._preserve_aspect_ratio(
            Image.fromarray(arr_gray),
            self.viewbox_width, self.viewbox_height
        )
        self.hologram_frames = [tk_holo]

        # show hologram by default
        self.arr_hologram = arr_gray
        self.current_holo_array = arr_gray
        self.current_ft_array = ft_disp
        self.captured_label.configure(image=tk_holo)
        self.captured_title_label.configure(text="Hologram")
        self.hide_holo_arrows()
        print(f"Hologram loaded from: {holo_path}")

    # ──────────────────────────────
    # Load images for phase shifting method
    # ──────────────────────────────
    def load_images(self):
        """Load one or more hologram images and prepare their corresponding frames."""
        # Open file dialog to select
        file_paths = filedialog.askopenfilenames(title='Select hologram(s)')
        if not file_paths:
            messagebox.showinfo(
                "Information",
                "No holograms selected."
            )
            return

        # Reset containers and indices
        self.multi_holo_arrays = []
        self.original_multi_holo_arrays = []
        self.hologram_frames = []
        self.ft_frames = []
        self.multi_ft_arrays = []
        self.filter_states_dim0 = []
        self.current_left_index = 0

        max_w, max_h = self.viewbox_width, self.viewbox_height

        # Process each selected image
        for fpath in file_paths:
            arr_gray = np.array(Image.open(fpath).convert('L'))

            # Store hologram and its default filter state
            self.multi_holo_arrays.append(arr_gray)
            self.original_multi_holo_arrays.append(arr_gray.copy())
            self.filter_states_dim0.append(tGUI.default_filter_state())

            # Create hologram frame
            tk_holo = self._preserve_aspect_ratio(Image.fromarray(arr_gray), max_w, max_h)
            self.hologram_frames.append(tk_holo)

            # Create Fourier Transform frame
            tk_ft, ft_disp = self._create_ft_frame(arr_gray)
            self.ft_frames.append(tk_ft)
            self.multi_ft_arrays.append(ft_disp)

        # Display the first loaded image
        self.arr_hologram = self.multi_holo_arrays[0]
        self.current_holo_array = self.arr_hologram
        self.current_ft_array = self.multi_ft_arrays[0]

        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=self.hologram_frames[0])
            self.captured_title_label.configure(text="Hologram")
        else:
            self.captured_label.configure(image=self.ft_frames[0])
            self.captured_title_label.configure(text="Fourier Transform")

        # Show or hide navigation arrows
        if len(self.hologram_frames) > 1:
            self.show_holo_arrows()
        else:
            self.hide_holo_arrows()

        # Store for phase-shifting module
        self.phase_shift_imgs = self.multi_holo_arrays

        # (Optional) Create right-panel amplitude/phase frames
        self.amplitude_frames = []
        self.phase_frames = []

        for amp, phs in zip(getattr(self, "amplitude_arrays", []), getattr(self, "phase_arrays", [])):
            amp_pil = Image.fromarray(amp, mode='L')
            phs_pil = Image.fromarray(phs, mode='L')
            self.amplitude_frames.append(self._preserve_aspect_ratio_right(amp_pil))
            self.phase_frames.append(self._preserve_aspect_ratio_right(phs_pil))

        print(f"Loaded {len(file_paths)} hologram(s).")

    # ────────────────────────────────────────────────────────────
    # Load generic image for numerical propagation
    # ────────────────────────────────────────────────────────────
    def load_image_generic(self):
        file_path = filedialog.askopenfilename(title="Select image")
        if not file_path:
            messagebox.showinfo(
                "Information",
                "No Image selected."
            )
            return

        img = Image.open(file_path).convert("L")
        img_array = np.array(img)

        self.coherent_input_image = img_array

        # Check if the user has selected the "Coherent Image" option
        if self.np_source_var.get() == 1:
            self._load_and_display_coherent_image()
            return  # ← Very important: avoid continuing with the 'Digital Hologram' workflow

        # If it's not a coherent image, continue with the standard hologram workflow
        self.multi_holo_arrays = [img_array]
        self.original_multi_holo_arrays = [img_array.copy()]
        tk_img = self._preserve_aspect_ratio(img, self.viewbox_width, self.viewbox_height)
        tk_ft, ft_disp = self._create_ft_frame(img_array)

        self.hologram_frames = [tk_img]
        self.multi_ft_arrays = [ft_disp]
        self.ft_frames = [tk_ft]
        self.current_left_index = 0
        self.arr_hologram = img_array
        self.current_ft_array = ft_disp

        self.captured_label.configure(image=tk_img)
        self.captured_label.image = tk_img
        self.captured_title_label.configure(text="Digital Hologram")

        if self.holo_view_var.get() == "Fourier Transform":
            self.captured_label.configure(image=tk_ft)

        self.hide_holo_arrows()

    def show_save_options(self):
        """
        Now it offers "Save FT", "Save Phase", and "Save Amplitude".
        If you click "Save FT", we actually store the Fourier transform images
        (not the hologram).
        """
        # If user re-clicks while open, just hide it
        if hasattr(self, 'save_options_menu') and self.save_options_menu.winfo_ismapped():
            self.save_options_menu.grid_forget()
            return

        self.save_options_menu = ctk.CTkOptionMenu(
            self.buttons_frame,
            values=["Save FT", "Save Phase", "Save Amplitude"],
            command=lambda option: self._handle_save_option(option),
            width=270
        )
        self.save_options_menu.set("Save")
        self.save_options_menu.grid(row=0, column=2, padx=4, pady=5, sticky='w')

    def ask_filename(self, option, default_name=""):
        def on_submit():
            self.filename = entry.get()
            popup.destroy()
            self.save_images(option, self.filename)

        popup = tk.Toplevel(self)
        popup.title("Enter filename")
        popup.geometry("600x300")

        label = tk.Label(popup, text="Enter filename:", font=("Helvetica", 14))
        label.pack(pady=20)

        entry = tk.Entry(popup, font=("Helvetica", 14), width=40)
        entry.insert(0, default_name)
        entry.pack(pady=20)

        submit_button = tk.Button(popup, text="Save", font=("Helvetica", 14), command=on_submit)
        submit_button.pack(pady=20)

        popup.transient(self)
        popup.grab_set()
        self.wait_window(popup)

    def _handle_save_option(self, option):
        if hasattr(self, 'save_options_menu'):
            try:
                self.save_options_menu.grid_forget()
            except Exception:
                pass

        if option == "Save FT":
            self.save_ft_images()
        elif option == "Save Phase":
            self.save_phase_images()
        elif option == "Save Amplitude":
            self.save_amplitude_images()

    def _normalize_for_save(self, array_in):
        """
        Ensures we only apply (val + pi) / (2*pi) * 255 once.
        If the array is already in [0..255], we skip the formula.
        Otherwise we assume it's a 'raw' phase in [-pi..+pi] (or something similar),
        and do: (value + pi)/(2*pi)*255, clipped to [0..255].
        """
        arr = array_in.astype(np.float32)
        min_val = arr.min()
        max_val = arr.max()

        # if it's already in [0..255], we do nothing:
        if min_val >= 0.0 and max_val <= 255.0:
            return arr.astype(np.uint8)

        # Otherwise we do the phase-like normalization:
        arr = (arr + np.pi) / (2.0 * np.pi)
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr * 255.0
        return arr.astype(np.uint8)

    def _save_single_array(self, arr8: np.ndarray,
                           dialog_title: str = "Save",
                           default_name: str = "") -> None:

        filetypes = [("MATLAB files", "*.mat"),
                     ("PNG files", "*.png"),
                     ("BMP files", "*.bmp"),
                     ("JPEG files", "*.jpg"),
                     ("TIFF files", "*.tif"),
                     ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            title=dialog_title,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=filetypes
        )
        if not path:
            print("Save cancelled.")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == ".mat":
            from scipy.io import savemat
            savemat(path, {"data": arr8})
            print(f"Saved MAT file: {path}")
        else:
            Image.fromarray(arr8).save(path)
            print(f"Saved image: {path}")

    def save_ft_images(self):
        if not hasattr(self, 'multi_ft_arrays') or len(self.multi_ft_arrays) == 0:
            print("No FT images to save.")
            return

        count = len(self.multi_ft_arrays)
        if len(self.multi_ft_arrays) == 1:
            arr8 = self._normalize_for_save(self.multi_ft_arrays[0])
            self._save_single_array(arr8, dialog_title="Save Fourier Transform",
                                    default_name="FT")
            return
        else:
            zip_path = filedialog.asksaveasfilename(
                title="Save multiple FT as ZIP",
                defaultextension=".zip",
                filetypes=[("Zip archive", "*.zip"), ("All files", "*.*")]
            )
            if not zip_path:
                print("Canceled.")
                return

            extension_win = tk.Toplevel(self)
            extension_win.title("Choose image format for FT inside ZIP")
            extension_win.geometry("400x200")
            lab = tk.Label(extension_win, text="Pick format (png, bmp, jpg, etc.):")
            lab.pack(pady=10)
            fmt_var = tk.StringVar(value="png")
            fmt_entry = tk.Entry(extension_win, textvariable=fmt_var, width=10, font=("Helvetica", 14))
            fmt_entry.pack(pady=5)

            def confirm_fmt():
                extension_win.destroy()

            btn = tk.Button(extension_win, text="OK", command=confirm_fmt)
            btn.pack(pady=10)
            extension_win.transient(self)
            extension_win.grab_set()
            extension_win.wait_window(extension_win)

            chosen_fmt = fmt_var.get().lower().replace(".", "")

            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i, arr in enumerate(self.multi_ft_arrays):
                    arr_norm = self._normalize_for_save(arr)
                    file_in_zip = f"FT_{i:03d}.{chosen_fmt}"
                    buf = io.BytesIO()
                    Image.fromarray(arr_norm).save(buf, format=chosen_fmt.upper())
                    zf.writestr(file_in_zip, buf.getvalue())

    def save_phase_images(self):
        if not self.phase_arrays:
            messagebox.showinfo(
                "Information",
                "No phase image to save."
            )
            return

        arr8 = self._normalize_for_save(self.phase_arrays[0])
        self._save_single_array(arr8, dialog_title="Save Phase",
                                default_name="Phase")

    def save_amplitude_images(self):
        if not self.amplitude_arrays:
            messagebox.showinfo(
                "Information",
                "No amplitude image to save."
            )
            return

        arr8 = self._normalize_for_save(self.amplitude_arrays[0])
        self._save_single_array(arr8, dialog_title="Save Amplitude",
                                default_name="Amplitude")

    def reset_reconstruction_data(self):
        self.amplitude_arrays.clear()
        self.phase_arrays.clear()
        self.amplitude_frames.clear()
        self.phase_frames.clear()
        self.original_amplitude_arrays.clear()
        self.original_phase_arrays.clear()

        # Wipe dimension=1 and dimension=2 filter states
        self.filter_states_dim1.clear()
        self.filter_states_dim2.clear()

        self.last_filter_settings = None

    def _init_colormap_settings(self):
        """Centralise all colour-map related state."""
        # Allowed names shown to the user
        self.available_colormaps = [
            "Original", "Viridis", "Plasma", "Inferno",
            "Magma", "Cividis", "Hot", "Cool", "Wistia"
        ]
        # Tk variables that will hold the current choice
        self.colormap_amp_var = tk.StringVar(self, value="Original")
        self.colormap_phase_var = tk.StringVar(self, value="Original")

    def on_filters_dimensions_change(self, *args):
        """
        NEW method. Automatically toggles the main viewing window’s radio buttons
        whenever the user changes the ‘Hologram / Amplitude image / Phase image’
        selection in the Filters panel.
        """
        selected_value = self.filters_dimensions_var.get()
        if selected_value == 0:
            # Hologram => set the left radio to "Hologram"
            self.holo_view_var.set("Hologram")
            self.update_left_view()
        elif selected_value == 1:
            # Amplitude => set the right radio to "Amplitude Reconstruction "
            self.recon_view_var.set("Amplitude Reconstruction ")
            self.update_right_view()
        elif selected_value == 2:
            # Phase => set the right radio to "Phase Reconstruction "
            self.recon_view_var.set("Phase Reconstruction ")
            self.update_right_view()
        # No extra else needed, just keep it at 3 values.

    def update_qpi_placeholder(self) -> None:
        """
        Enable or disable input fields based on the selected QPI mode:
        - If mode is 2 (Thickness): enable thickness input, disable index fields.
        - Otherwise (Index mode): enable index fields, disable thickness input.
        """
        mode = self.option_meas_var.get()

    def _micron_axes(self, ax, width_px, height_px, μm_per_px):
        if μm_per_px > 0:
            x_px = np.linspace(0, width_px, 5)
            y_px = np.linspace(0, height_px, 5)

            ax.set_xticks(x_px)
            ax.set_xticklabels([f"{x * μm_per_px:.1f}" for x in x_px])
            ax.set_yticks(y_px)
            ax.set_yticklabels([f"{y * μm_per_px:.1f}" for y in y_px])

            ax.set_xlim(0, width_px)
            ax.set_ylim(height_px, 0)

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
        else:
            ax.axis("off")

    def show_dataframe_in_table(self, df, title="QPI Results"):
        """
        Displays a pandas DataFrame in a new Toplevel window using pandastable,
        without blocking the main Tkinter loop.
        """
        if df.empty:
            print("No data to display in the table.")
            return

        # Create a Toplevel so it runs inside the main app
        table_win = tk.Toplevel(self)
        table_win.title(title)
        table_win.geometry("800x400")

        # Create a frame for the pandastable
        frame = tk.Frame(table_win)
        frame.pack(fill="both", expand=True)

        pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
        pt.show()

    def _add_processing_nav(self, parent: ctk.CTkFrame, current: str) -> None:
        width = PARAMETER_FRAME_WIDTH
        nav_wrap = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0, width=width)
        nav_wrap.grid(row=0, column=0, sticky="ew", pady=(30, 10))
        nav_wrap.grid_columnconfigure(0, weight=1)
        nav_wrap.configure(width=width)

        nav_bar = ctk.CTkFrame(
            nav_wrap,
            fg_color=("gray90", "gray25"),
            corner_radius=0,
            height=38,
            width=width
        )
        nav_bar.pack(fill="x", expand=True)
        nav_bar.configure(width=width)
        nav_bar.grid_columnconfigure((0, 1, 2), weight=1, uniform="nav")

        def _make_btn(text: str, target: str, col: int) -> None:
            btn = ctk.CTkButton(
                nav_bar, text=text,
                command=lambda dest=target: self.change_menu_to(dest),
                height=38,
                fg_color=("gray80", "gray20"),
                hover_color=("gray70", "gray15"),
                corner_radius=0,
                width=int(width // 3),
                font=ctk.CTkFont(size=12),
                text_color="black"
            )
            if target == current:
                btn.configure(state="disabled", fg_color=("gray60", "gray10"))
            btn.grid(row=0, column=col, sticky="nsew", padx=(1 if col else 0, 1))

        _make_btn("Phase\nCompensation", "phase_compensation", 0)
        _make_btn("Phase\nShifting", "phase_shifting", 1)
        _make_btn("Numerical\nPropagation", "numerical_propagation", 2)

    # --------------------------------------------------------------------------
    # --------------------- Phase-Shifting--------------------------------------
    # --------------------------------------------------------------------------
    def init_phase_shifting_frame(self):
        """
        Initializes the user interface for phase-shifting parameters and method selection.
        """
        # MAIN CONTAINER
        # Create the outer frame for the phase-shifting section
        self.phase_shifting_frame = ctk.CTkFrame(self, corner_radius=8)
        self.phase_shifting_frame.grid_propagate(False)

        # Create a container frame to hold scrollable content
        self.param_container = ctk.CTkFrame(self.phase_shifting_frame, corner_radius=8, width=420)
        self.param_container.grid_propagate(False)
        self.param_container.pack(fill="both", expand=True)

        # Add a vertical scrollbar to the container
        self.param_scrollbar = ctk.CTkScrollbar(self.param_container, orientation='vertical')
        self.param_scrollbar.grid(row=0, column=0, sticky='ns')

        # Create a canvas for scrollable content inside the container
        self.param_canvas = ctk.CTkCanvas(self.param_container, width=PARAMETER_FRAME_WIDTH)
        self.param_canvas.grid(row=0, column=1, sticky='nsew')

        # Configure the container grid layout
        self.param_container.grid_rowconfigure(0, weight=1)
        self.param_container.grid_columnconfigure(1, weight=1)

        # Link scrollbar and canvas
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        self.param_scrollbar.configure(command=self.param_canvas.yview)

        # Create an inner frame inside the canvas to place widgets
        self.parameters_inner_frame = ctk.CTkFrame(self.param_canvas)
        self.param_canvas.create_window((0, 0), window=self.parameters_inner_frame, anchor='nw')

        # Navigation strip
        self._add_processing_nav(self.parameters_inner_frame, current="phase_shifting")

        # Shifting method selection
        self.shifting_method_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT * 2
        )
        self.shifting_method_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.shifting_method_frame.grid_propagate(False)

        # Label for method selection section
        self.method_label = ctk.CTkLabel(
            self.shifting_method_frame,
            text='Choose a Phase Shifting Method',
            font=ctk.CTkFont(weight="bold")
        )
        self.method_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # # List to keep track of dynamically added widgets
        self.VShif = tk.IntVar(value=0)

        # Define phase-shifting methods as (label, value)
        shifting_options = [
            ("5 Frames", 0),
            ("4 Frames", 1),
            ("3 Frames", 2),
            ("Quadrature Method", 3),
            ("Blind 3 Raw Frames", 4),
            ("Blind 2 Raw Frames", 5)
        ]

        # Configure columns for better layout
        self.shifting_method_frame.columnconfigure(0, weight=1)
        self.shifting_method_frame.columnconfigure(1, weight=1)

        # Create radio buttons for each shifting method
        self.radio_5frames = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[0][0],
            variable=self.VShif,
            value=shifting_options[0][1],
            command=self.update_shifting_params
        )
        self.radio_5frames.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.radio_4frames = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[1][0],
            variable=self.VShif,
            value=shifting_options[1][1],
            command=self.update_shifting_params
        )
        self.radio_4frames.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.radio_3frames = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[2][0],
            variable=self.VShif,
            value=shifting_options[2][1],
            command=self.update_shifting_params
        )
        self.radio_3frames.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        self.radio_quad = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[3][0],
            variable=self.VShif,
            value=shifting_options[3][1],
            command=self.update_shifting_params
        )
        self.radio_quad.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        self.radio_blind3 = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[4][0],
            variable=self.VShif,
            value=shifting_options[4][1],
            command=self.update_shifting_params
        )
        self.radio_blind3.grid(row=3, column=0, padx=5, pady=5, sticky='w')

        self.radio_blind2 = ctk.CTkRadioButton(
            self.shifting_method_frame,
            text=shifting_options[5][0],
            variable=self.VShif,
            value=shifting_options[5][1],
            command=self.update_shifting_params
        )
        self.radio_blind2.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        # Parameters frame
        # Frame to hold input parameters for phase-shifting reconstruction
        self.params_shifting_frame = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT * 2.2
        )
        self.params_shifting_frame.grid(row=3, column=0, sticky='ew', pady=2)
        self.params_shifting_frame.grid_propagate(False)

        # Configure columns for a 3-column layout
        for c in range(3):
            self.params_shifting_frame.columnconfigure(c, weight=1)

        # Title for reconstruction parameters
        ctk.CTkLabel(
            self.params_shifting_frame,
            text="Loading Reconstruction Parameters",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=5, pady=(5, 2), sticky="w")

        # Unit options for dropdown
        units = ["µm", "nm", "mm", "cm", "m", "in"]

        # Dictionary to store the entry widgets
        self.param_entries_ps = {}

        # Create each parameter using the modular function from functions.py
        fGUI.create_param_with_arrow(
            parent=self.params_shifting_frame,
            row=1, col=0,
            label_text=f"Wavelength ({self.wavelength_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_ps,
            entry_key="wavelength",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_shifting_frame,
            row=1, col=1,
            label_text=f"Pitch X ({self.pitch_x_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_ps,
            entry_key="pitch_x",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_shifting_frame,
            row=1, col=2,
            label_text=f"Pitch Y ({self.pitch_y_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_ps,
            entry_key="pitch_y",
            unit_update_callback=self._set_unit_in_label
        )

        self.wave_entry_ps = self.param_entries_ps["wavelength"]
        self.pitchx_entry_ps = self.param_entries_ps["pitch_x"]
        self.pitchy_entry_ps = self.param_entries_ps["pitch_y"]

        # Add the "Reconstruction" button that calls run_phase_shifting_method
        self.compensate_button_ps = ctk.CTkButton(
            self.params_shifting_frame,
            text="Reconstruction",
            command=self.run_phase_shifting_method,
            width=100,
        )
        self.compensate_button_ps.grid(row=3, column=0, padx=10, columnspan=3, pady=(10, 10), sticky='w')

        # Propagation panel
        self.ps_propagation_panel = ctk.CTkFrame(
            self.parameters_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT * 4
        )
        self.ps_propagation_panel.grid(row=5, column=0, sticky='ew', pady=1.6)
        self.ps_propagation_panel.grid_propagate(True)

        for col in range(3):
            self.ps_propagation_panel.columnconfigure(col, weight=1)

        # Title label
        ctk.CTkLabel(
            self.ps_propagation_panel,
            text="Propagation Options",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="w")

        # Create propagation panel using reusable function
        self.propagate_widgets_ps = fGUI.create_propagate_panel(
            parent=self.ps_propagation_panel,
            attr_prefix="propagation",
            on_slider_change=self._on_slider_moved
        )

        # Make these widgets the current active ones for processing
        self.active_propagate_widgets = self.propagate_widgets_ps
        self.compensation_source = "ps"

        self.propagate_widgets_ps["propagation_apply_button"].configure(
            command=lambda: self._apply_propagation(request_magnification=True)
        )

        # Update scroll region
        self.parameters_inner_frame.update_idletasks()
        self.param_canvas.config(scrollregion=self.param_canvas.bbox("all"))

    def run_phase_shifting_method(self):
        self.compensation_source = "ps"
        method_idx = self.VShif.get()

        if not hasattr(self, 'phase_shift_imgs') or len(self.phase_shift_imgs) == 0:
            messagebox.showinfo(
                "Information",
                "No holograms are currently loaded. Please load valid holograms before applying phase shifting."
            )
            return

        try:
            w_val = self.get_value_in_micrometers(self.wave_entry_ps.get(), self.wavelength_unit)
        except:
            messagebox.showinfo(
                "Information",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them beore proceeding."
            )
        try:
            px_val = self.get_value_in_micrometers(self.pitchx_entry_ps.get(), self.pitch_x_unit)
        except:
            messagebox.showinfo(
                "Information",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them beore proceeding."
            )
        try:
            py_val = self.get_value_in_micrometers(self.pitchy_entry_ps.get(), self.pitch_y_unit)
        except:
            messagebox.showinfo(
                "Information",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them beore proceeding."
            )

        self._cache_compensation_params(w_val, px_val, py_val)
        self.reset_reconstruction_data()

        # Run whichever method is selected:
        comp_output = None
        if method_idx == 0:  # 5 frames
            if len(self.phase_shift_imgs) != 5:
                messagebox.showinfo(
                    "Information",
                    "Error: 5 frames method requires exactly 5 holograms."
                )
                return
            comp_output = PS5(self.phase_shift_imgs[0], self.phase_shift_imgs[1],
                              self.phase_shift_imgs[2], self.phase_shift_imgs[3],
                              self.phase_shift_imgs[4])
        elif method_idx == 1:  # 4 frames
            if len(self.phase_shift_imgs) != 4:
                messagebox.showinfo(
                    "Information",
                    "Error: 4 frames method requires exactly 4 holograms."
                )
                return
            comp_output = PS4(self.phase_shift_imgs[0], self.phase_shift_imgs[1],
                              self.phase_shift_imgs[2], self.phase_shift_imgs[3])
        elif method_idx == 2:  # 3 frames
            if len(self.phase_shift_imgs) != 3:
                messagebox.showinfo(
                    "Information",
                    "Error: 3 frames method requires exactly 3 holograms."
                )
                return
            comp_output = PS3(self.phase_shift_imgs[0], self.phase_shift_imgs[1], self.phase_shift_imgs[2])
        elif method_idx == 3:  # Quadrature (SOSR)
            if len(self.phase_shift_imgs) != 4:
                messagebox.showinfo(
                    "Information",
                    "Error: Quadrature method requires exactly 4 holograms."
                )
                return
            upper = True
            s_val = 1
            steps_val = 4
            comp_output = SOSR(self.phase_shift_imgs[0], self.phase_shift_imgs[1],
                               self.phase_shift_imgs[2], self.phase_shift_imgs[3],
                               upper, w_val, px_val, py_val, s=s_val, steps=steps_val)
        elif method_idx == 4:  # Blind 3 Raw Frames (BPS3)
            if len(self.phase_shift_imgs) != 3:
                messagebox.showinfo(
                    "Information",
                    "Error: Blind 3 Raw Frames requires exactly 3 holograms."
                )
                return
            comp_output = BPS3(self.phase_shift_imgs[0], self.phase_shift_imgs[1],
                               self.phase_shift_imgs[2], w_val, px_val, py_val)
        elif method_idx == 5:  # Blind 2 Raw Frames (BPS2)
            if len(self.phase_shift_imgs) != 2:
                messagebox.showinfo(
                    "Information",
                    "Error: Blind 2 Raw Frames requires exactly 2 holograms."
                )
                return
            comp_output = BPS2(self.phase_shift_imgs[0], self.phase_shift_imgs[1],
                               w_val, px_val, py_val)
        else:
            print("No valid method selected.")
            messagebox.showinfo(
                "Information",
                "No valid method selected."
            )
            return

        if comp_output is None:
            print("Phase-shifting returned None or encountered an error.")
            return

        self.compensated_field_complex = comp_output.copy()
        self.complex_fields = [comp_output]
        self.current_amp_index = 0
        self.current_phase_index = 0
        self.original_complex_fields = [comp_output.copy()]
        self.complex_fields = [comp_output.copy()]

        # Compute amplitude & phase
        amp = np.abs(comp_output)
        raw_phase = np.angle(comp_output)

        amp_norm = (amp - amp.min()) / (amp.max() - amp.min() + 1e-9) * 255
        amp_norm = amp_norm.astype(np.uint8)

        phase_0to1 = (raw_phase + np.pi) / (2 * np.pi + 1e-9)
        phase_0to1 = np.clip(phase_0to1, 0, 1)
        phase_8bit = (phase_0to1 * 255).astype(np.uint8)

        # Build images for the right side
        amp_pil = Image.fromarray(amp_norm, mode='L')
        phs_pil = Image.fromarray(phase_8bit, mode='L')

        # Preserve aspect ratio for the right frame
        tk_amp = self._preserve_aspect_ratio_right(amp_pil)
        tk_phs = self._preserve_aspect_ratio_right(phs_pil)

        # Store arrays in self.amplitude_arrays / self.phase_arrays
        self.amplitude_arrays = [amp_norm]
        self.phase_arrays = [phase_8bit]

        # Also store original copies for filtering
        self.original_amplitude_arrays = [amp_norm.copy()]
        self.original_phase_arrays = [phase_8bit.copy()]

        self.filter_states_dim1 = [tGUI.default_filter_state()]
        self.filter_states_dim2 = [tGUI.default_filter_state()]

        self.amplitude_frames = [tk_amp]
        self.phase_frames = [tk_phs]
        self.current_amp_index = 0
        self.current_phase_index = 0

        # Show the new Phase
        self.processed_label.configure(image=tk_phs)
        self.processed_title_label.configure(text="Phase Reconstruction ")

        # Force the right radio button to match the displayed Phase
        self.recon_view_var.set("Phase Reconstruction ")
        self.update_right_view()

        print("Phase-shifting complete. Right-frame images scaled properly.")

    def update_shifting_params(self):
        for w in self.widget_refs:
            w.grid_forget()
            w.destroy()
        self.widget_refs.clear()

        metodo = self.VShif.get()

        label_msg = ctk.CTkLabel(self.shifting_method_frame, text="", width=200)
        label_msg.grid(row=0, column=1, pady=5, sticky='w')
        self.widget_refs.append(label_msg)

        if metodo == 0:
            label_msg.configure(text="Upload 5 images")
        elif metodo == 1:
            label_msg.configure(text="Upload 4 images")
        elif metodo == 2:
            label_msg.configure(text="Upload 3 images")
        elif metodo == 3:
            label_msg.configure(text="Upload 4 images")
        elif metodo == 4:
            label_msg.configure(text="Upload 3 images")
        elif metodo == 5:
            label_msg.configure(text="Upload 2 images")
        else:
            label_msg.configure(text="Select a method:")

    def _set_unit_in_label(self, lbl, unit):
        text_now = lbl.cget("text")
        base = text_now.split("(")[0].strip()
        lbl.configure(text=f"{base} ({unit})")
        if "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pitch X" in base:
            self.pitch_x_unit = unit
        elif "Pitch Y" in base:
            self.pitch_y_unit = unit
        elif "Distance" in base:
            self.distance_unit = unit

    # Phase Compensation
    def init_phase_compensation_frame(self):
        # Create the main frame for the phase compensation panel
        self.phase_compensation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.phase_compensation_frame.grid_propagate(False)

        # Container that holds the scrollable canvas and scrollbar
        self.pc_container = ctk.CTkFrame(self.phase_compensation_frame, corner_radius=8, width=420)
        self.pc_container.grid_propagate(False)
        self.pc_container.pack(fill="both", expand=True)

        # Vertical scrollbar for the parameter panel
        self.pc_scrollbar = ctk.CTkScrollbar(self.pc_container, orientation='vertical')
        self.pc_scrollbar.grid(row=0, column=0, sticky='ns')

        # Canvas for scrollable content
        self.pc_canvas = ctk.CTkCanvas(self.pc_container, width=PARAMETER_FRAME_WIDTH)
        self.pc_canvas.grid(row=0, column=1, sticky='nsew')

        # Configure scrolling behavior
        self.pc_container.grid_rowconfigure(0, weight=1)
        self.pc_container.grid_columnconfigure(1, weight=1)
        self.pc_canvas.configure(yscrollcommand=self.pc_scrollbar.set)
        self.pc_scrollbar.configure(command=self.pc_canvas.yview)

        # Inner frame where all interactive widgets will be placed
        self.phase_compensation_inner_frame = ctk.CTkFrame(self.pc_canvas)
        self.pc_canvas.create_window((0, 0), window=self.phase_compensation_inner_frame, anchor='nw')

        # Add a navigation bar specific to this processing module
        self._add_processing_nav(self.phase_compensation_inner_frame, current="phase_compensation")

        # Compensation Method Selection
        self.pc_method_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT * 1.3
        )
        self.pc_method_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.pc_method_frame.grid_propagate(False)

        for col in range(2):
            self.pc_method_frame.columnconfigure(col, weight=1)

        # Label center
        self.pc_method_label = ctk.CTkLabel(
            self.pc_method_frame,
            text='Choose a Compensation Method',
            font=ctk.CTkFont(weight="bold")
        )
        self.pc_method_label.grid(row=0, column=0, columnspan=2, padx=5, pady=(5, 10), sticky='w')

        # variable to save the selection
        self.pc_method_var = ctk.IntVar(value=0)

        # First row
        self.radio_as_pc = ctk.CTkRadioButton(
            self.pc_method_frame, text='Semi-Heuristic',
            variable=self.pc_method_var, value=0
        )
        self.radio_as_pc.grid(row=1, column=0, padx=10, pady=2, sticky='ew')

        self.radio_fr_pc = ctk.CTkRadioButton(
            self.pc_method_frame, text='Tu-DHM',
            variable=self.pc_method_var, value=1
        )
        self.radio_fr_pc.grid(row=1, column=1, padx=10, pady=2, sticky='ew')

        # Second Row
        self.radio_bl_pc = ctk.CTkRadioButton(
            self.pc_method_frame, text='No Telecentric',
            variable=self.pc_method_var, value=2
        )
        self.radio_bl_pc.grid(row=2, column=0, padx=10, pady=2, sticky='ew')

        self.radio_vl_pc = ctk.CTkRadioButton(
            self.pc_method_frame, text='Vortex Legendre',
            variable=self.pc_method_var, value=3
        )
        self.radio_vl_pc.grid(row=2, column=1, padx=10, pady=2, sticky='ew')

        # Parameters Input Section
        self.params_pc_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
        )
        self.params_pc_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.params_pc_frame.grid_propagate(True)
        for col in range(3):
            self.params_pc_frame.columnconfigure(col, weight=1)

        # Title label for the reconstruction parameters section
        ctk.CTkLabel(
            self.params_pc_frame,
            text="Loading Reconstruction Parameters",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=5, pady=(5, 2), sticky="w")

        # Define the unit options for the dropdowns
        units = ["µm", "nm", "mm", "cm", "m", "in"]

        # Dictionary to store the input fields for external access
        self.param_entries_pc = {}

        # Create each parameter field using a reusable function
        fGUI.create_param_with_arrow(
            parent=self.params_pc_frame,
            row=1, col=0,
            label_text=f"Wavelength ({self.wavelength_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_pc,
            entry_key="wavelength",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_pc_frame,
            row=1, col=1,
            label_text=f"Pitch X ({self.pitch_x_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_pc,
            entry_key="pitch_x",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_pc_frame,
            row=1, col=2,
            label_text=f"Pitch Y ({self.pitch_y_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_pc,
            entry_key="pitch_y",
            unit_update_callback=self._set_unit_in_label
        )

        self.wave_label_pc_entry = self.param_entries_pc["wavelength"]
        self.pitchx_label_pc_entry = self.param_entries_pc["pitch_x"]
        self.pitchy_label_pc_entry = self.param_entries_pc["pitch_y"]

        # Compensation Filter Options
        self.filter_pc_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=int(PARAMETER_FRAME_HEIGHT * 2),
        )
        self.filter_pc_frame.grid(row=4, column=0, sticky="ew", pady=1.6)
        self.filter_pc_frame.grid_propagate(True)
        for c in range(4):
            self.filter_pc_frame.columnconfigure(c, weight=1)

        # Title label for filter configuration
        ctk.CTkLabel(
            self.filter_pc_frame,
            text="Compensation Filter Options",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=5, pady=(10, 10), sticky="w")

        # Shared control variables for filter selection
        self.spatial_section_var = tk.IntVar(value=0)
        self.spatial_auto_var_pc = tk.StringVar(value="Circular")
        self.spatial_manual_var_pc = tk.StringVar(value="None")
        self.spatial_filter_var_pc = tk.StringVar(value="Circular")

        # Radio button for selecting automatic filter mode
        self.radio_auto_section = ctk.CTkRadioButton(
            self.filter_pc_frame, text="Automatic",
            variable=self.spatial_section_var, value=0,
            command=self._on_filter_section_changed
        )
        self.radio_auto_section.grid(row=1, column=0, padx=(5, 2), pady=5, sticky="w")

        # Dropdown for automatic filter shapes
        self.auto_menu_pc = ctk.CTkOptionMenu(
            self.filter_pc_frame,
            values=["Circular", "Rectangle"],
            variable=self.spatial_auto_var_pc,
            width=150
        )
        self.auto_menu_pc.grid(row=2, column=0, padx=(2, 10), pady=5, sticky="ew")

        # Radio button for selecting manual filter mode
        self.radio_manual_section = ctk.CTkRadioButton(
            self.filter_pc_frame, text="Manual",
            variable=self.spatial_section_var, value=1,
            command=self._on_filter_section_changed
        )
        self.radio_manual_section.grid(row=1, column=1, padx=(10, 2), pady=5, sticky="w")

        # Dropdown for manual filter types and configurations
        self.manual_menu_pc = ctk.CTkOptionMenu(
            self.filter_pc_frame,
            values=[
                "None",
                "Circular Coor.",
                "Circular Man.",
                "Rectangle Coor.",
                "Rectangle Man.",
                "NonTele. Coor.",
                "NonTele. Man.",
            ],
            variable=self.spatial_manual_var_pc,
            width=150
        )
        self.manual_menu_pc.grid(row=2, column=1, padx=(2, 5), pady=5, sticky="ew")

        # Button to execute the phase compensation process
        self.pc_compensate_button = ctk.CTkButton(
            self.filter_pc_frame,
            width=100,
            text='Compensate',
            command=self.run_phase_compensation
        )
        self.pc_compensate_button.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="w")

        # Propagation Frame
        self.propagation_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=int(PARAMETER_FRAME_HEIGHT * 4)
        )
        self.propagation_frame.grid(row=5, column=0, sticky="ew", pady=1.6)
        self.propagation_frame.grid_propagate(True)

        for col in range(3):
            self.propagation_frame.columnconfigure(col, weight=1)

        # Title label
        ctk.CTkLabel(
            self.propagation_frame,
            text="Propagation Options",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="w")

        # Create propagation panel using reusable function
        self.propagate_widgets_pc = fGUI.create_propagate_panel(
            parent=self.propagation_frame,
            attr_prefix="propagation",
            on_slider_change=self._on_slider_moved
        )

        self.propagate_widgets_pc["propagation_apply_button"].configure(
            command=lambda: self._apply_propagation(request_magnification=True)
        )

        # widgets activation
        self.active_propagate_widgets = self.propagate_widgets_pc
        self.compensation_source = "pc"

        # Trigger UI updates based on the selected section (auto/manual)
        self._on_filter_section_changed()

        # Update canvas scroll region based on current content
        self.phase_compensation_inner_frame.update_idletasks()
        self.pc_canvas.config(scrollregion=self.pc_canvas.bbox("all"))

    def _on_filter_section_changed(self, *_):
        """
        Callback function triggered when the spatial filter section selection changes.

        This method enables the corresponding dropdown menu (Automatic or Manual)
        based on the user's selection, and disables the inactive one to prevent conflicting input.

        Parameters:
        *_ : Accepts any additional arguments (e.g., from event bindings), but they are unused.
        """
        # Check if the current selection is 'Automatic' (value = 0)
        auto = (self.spatial_section_var.get() == 0)

        # Enable the automatic filter menu if 'Automatic' is selected; otherwise, disable it
        self.auto_menu_pc.configure(state="normal" if auto else "disabled")

        # Enable the manual filter menu if 'Manual' is selected; otherwise, disable it
        self.manual_menu_pc.configure(state="normal" if not auto else "disabled")

    # Function to display a pop-up window for entering circle coordinates
    def _prompt_circle_coordinates(self):
        """Return (cx, cy, r) or None."""

        # Create dialog window
        dlg = tk.Toplevel(self)
        dlg.title("Manual Circular Filter – Enter Coordinates")
        dlg.configure(bg="#f0f0f0")
        dlg.grab_set()

        entries = {}
        ok = {"val": None}

        labels = ["Radius (px)", "Centre X (px)", "Centre Y (px)"]

        # Row 0: Labels
        for i, lbl in enumerate(labels):
            tk.Label(dlg, text=lbl, font=("Helvetica", 12), bg="#f0f0f0") \
                .grid(row=0, column=i, padx=10, pady=(10, 4))

        # Row 1: Entry fields
        radius_entry = tk.Entry(dlg, width=6)
        radius_entry.grid(row=1, column=0, padx=10, pady=(0, 10))
        entries["Radius (px)"] = radius_entry

        cx_entry = tk.Entry(dlg, width=6)
        cx_entry.grid(row=1, column=1, padx=10, pady=(0, 10))
        entries["Centre X (px)"] = cx_entry

        cy_entry = tk.Entry(dlg, width=6)
        cy_entry.grid(row=1, column=2, padx=10, pady=(0, 10))
        entries["Centre Y (px)"] = cy_entry

        # Define OK button behavior
        def _accept():
            try:
                r = int(entries["Radius (px)"].get())
                cx = int(entries["Centre X (px)"].get())
                cy = int(entries["Centre Y (px)"].get())
                if r <= 0:
                    raise ValueError
                ok["val"] = (cx, cy, r)
                dlg.destroy()
            except ValueError:
                tk.messagebox.showerror("Input", "All fields must be positive integers.")

        # Create a frame to hold the buttons and center it
        button_frame = tk.Frame(dlg, bg="#f0f0f0")
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 12))

        # Buttons inside the centered frame
        ok_button = tk.Button(button_frame, text="OK", command=_accept,
                              bg="#1e90ff", fg="white", activebackground="#1c86ee", width=10)
        cancel_button = tk.Button(button_frame, text="Cancel", command=dlg.destroy, width=10)

        ok_button.pack(side="left", padx=10)
        cancel_button.pack(side="left", padx=10)

        dlg.wait_window()
        return ok["val"]

    # Function to display a pop-up window for entering rectangle coordinates
    def _prompt_rectangle_coordinates(self, title="Manual Rectangle Filter – Enter Coordinates"):
        """Return (x1, y1, x2, y2) or None."""

        # Create modal dialog window
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg="#f0f0f0")
        dlg.grab_set()

        # Define labels
        labels = {
            "Top-left X (px)": (0, 0),
            "Top-left Y (px)": (0, 1),
            "Bottom-right X (px)": (1, 0),
            "Bottom-right Y (px)": (1, 1)
        }

        entries = {}

        # Add labels and entry fields
        for text, (row, col) in labels.items():
            tk.Label(dlg, text=text, font=("Helvetica", 12), bg="#f0f0f0") \
                .grid(row=row * 2, column=col, padx=10, pady=(10, 4), sticky="e")
            entry = tk.Entry(dlg, width=6)
            entry.grid(row=row * 2 + 1, column=col, padx=10, pady=(0, 10))
            entries[text] = entry

        ok = {"val": None}

        def _accept():
            try:
                x1 = int(entries["Top-left X (px)"].get())
                y1 = int(entries["Top-left Y (px)"].get())
                x2 = int(entries["Bottom-right X (px)"].get())
                y2 = int(entries["Bottom-right Y (px)"].get())
                if x2 <= x1 or y2 <= y1:
                    raise ValueError
                ok["val"] = (x1, y1, x2, y2)
                dlg.destroy()
            except ValueError:
                tk.messagebox.showerror("Input", "Coordinates must be integers and define a valid rectangle.")

        # Centered button frame
        button_frame = tk.Frame(dlg, bg="#f0f0f0")
        button_frame.grid(row=4, column=0, columnspan=2, pady=(0, 12))

        # Buttons with equal width
        button_width = 10
        ok_button = tk.Button(button_frame, text="OK", command=_accept,
                              bg="#1e90ff", fg="white", activebackground="#1c86ee", width=button_width)
        cancel_button = tk.Button(button_frame, text="Cancel", command=dlg.destroy, width=button_width)

        ok_button.pack(side="left", padx=10)
        cancel_button.pack(side="left", padx=10)

        dlg.wait_window()
        return ok["val"]

    def _value_from_um(self, value_um: float, unit: str) -> float:
        """Inverse of *_um_from_value_and_unit*."""
        u = unit.lower()
        if u == "µm":
            return value_um
        if u == "nm":
            return value_um * 1e3
        if u == "mm":
            return value_um / 1e3
        if u == "cm":
            return value_um / 1e4
        if u == "m":
            return value_um / 1e6
        if u == "in":
            return value_um / 25_400.0
        return value_um

    def _on_slider_moved(self, val):
        if self.active_propagate_widgets["propagation_mode_var"].get() != "sweep":
            return

        try:
            distance_um = float(val)
        except ValueError:
            print("Invalid slider value")
            return

        if not hasattr(self, "compensated_field_complex") or self.compensated_field_complex is None:
            print("No compensated field available.")
            return

        self._propagate_current_field(distance_um)

    def _apply_propagation(self, request_magnification):
        """
        Handle Apply button click for different propagation modes.
        """
        widgets = self.active_propagate_widgets
        mode = widgets["propagation_mode_var"].get()
        unit = widgets["propagation_unit_var"].get()
        unit_scales = {"µm": 1.0, "mm": 1000.0, "cm": 10000.0}

        # Read lateral magnification
        if request_magnification:
            try:
                mag_entry = widgets["propagation_magnification"]
                M = float(mag_entry.get().replace("x", "").strip())
                if M <= 0:
                    messagebox.showinfo(
                        "Information",
                        "Please enter a valid magnification value."
                    )
                    return
                self.magnification = M
            except Exception:
                messagebox.showinfo(
                    "Information",
                    "Please enter a valid magnification value."
                )
                return
            M = self.magnification
        else:
            self.magnification = 1.0
            M = self.magnification

        # Scale factor for the magnification
        scale_img = M ** 2

        # Mode sweep
        if mode == "sweep":
            try:
                min_val = float(widgets["propagation_min_entry"].get())
                max_val = float(widgets["propagation_max_entry"].get())
                scale = unit_scales[unit]

                # unit at microns and image plane
                min_val_um = min_val * scale * scale_img
                max_val_um = max_val * scale * scale_img
            except ValueError:
                messagebox.showinfo(
                    "Information",
                    "Please enter a valid min or max value."
                )
                return

            if min_val_um >= max_val_um:
                messagebox.showinfo(
                    "Information",
                    "Minimum must be less than maximum."
                )
                return

            slider = widgets["propagation_slider"]
            slider.configure(from_=min_val_um, to=max_val_um)
            slider.set(min_val_um)

            self._propagate_current_field(min_val_um)

        # Mode fixed
        elif mode == "fixed":
            try:
                distance_val = float(widgets["propagation_fixed_distance"].get())
                scale = unit_scales[unit]
                distance_um = distance_val * scale * scale_img
            except ValueError:
                messagebox.showinfo(
                    "Information",
                    "Please enter a valid numeric fixed distance."
                )
                return

            if self.compensated_field_complex is None:
                messagebox.showinfo(
                    "Information",
                    "No field available for propagation."
                )
                return

            self._propagate_current_field(distance_um)

        # mode auto
        elif mode == "auto":
            img_to_show = np.abs(self.compensated_field_complex)
            roi_coords = {}

            def onselect(eclick, erelease):
                roi_coords['x1'], roi_coords['y1'] = int(eclick.xdata), int(eclick.ydata)
                roi_coords['x2'], roi_coords['y2'] = int(erelease.xdata), int(erelease.ydata)
                plt.close()

            fig, ax = plt.subplots()
            ax.imshow(img_to_show, cmap='gray')
            ax.set_title("Select ROI with mouse")
            selector = RectangleSelector(ax, onselect,
                                         useblit=True,
                                         button=[1],
                                         minspanx=5, minspany=5,
                                         interactive=True)
            plt.show()

            if not roi_coords:
                messagebox.showinfo(
                    "Information",
                    "No ROI selected. Autofocus cancelled.."
                )
                return

            roi = roi_coords['x1'], roi_coords['y1'], roi_coords['x2'], roi_coords['y2']

            self._show_autofocus_progress()

            def run_autofocus_in_thread():
                try:
                    z_min = float(widgets["propagation_min_entry"].get())
                    z_max = float(widgets["propagation_max_entry"].get())
                    scale = unit_scales[unit]
                    z_min_um = z_min * scale * scale_img
                    z_max_um = z_max * scale * scale_img

                    metric_name = widgets["propagation_metric_var"].get()
                    if metric_name == "Normalized Variance":
                        metric_fn = pyDHM.metric_nv
                    elif metric_name == "Tenengrad":
                        metric_fn = pyDHM.metric_tenv
                    else:
                        print("Unknown metric selected.")
                        self._close_autofocus_progress()
                        return

                    plot_metric = widgets["propagation_plot_metric_var"].get()

                    z_best, z_vals, scores = pyDHM.autofocus_field(
                        field=self.compensated_field_complex,
                        z_range=(z_min_um, z_max_um),
                        wavelength=self.wavelength,
                        dx=self.dx,
                        dy=self.dy,
                        steps=20,
                        metric_fn=metric_fn,
                        progress_callback=self._update_autofocus_progress,
                        plot_results=False,
                        roi=roi
                    )

                    z_vals_obj = [z / (M ** 2) for z in z_vals]
                    z_best_obj = z_best / (M ** 2)

                    def plot_in_main_thread():
                        try:
                            metric_name = metric_fn.__name__
                            plt.figure(figsize=(10, 6))
                            plt.plot(z_vals_obj, scores, 'b-', linewidth=2, label=metric_name)
                            plt.axvline(z_best_obj, color='r', linestyle='--',
                                        label=f'Optimal Distance = {z_best_obj:.2f} µm')
                            plt.xlabel('Propagation Distance (µm)')
                            plt.ylabel('Sharpness Metric')
                            plt.title(f'Focus Curve – {metric_name}')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            print("[Plot Error]", e)

                    if plot_metric:
                        self.after(0, plot_in_main_thread)

                    self.after(0, self._on_autofocus_finished, z_best)

                except Exception as e:
                    print("Autofocus failed:", e)
                    messagebox.showinfo(
                        "Information",
                        "No field available for autofocusing."
                    )
                    self.after(0, self._close_autofocus_progress)

            import threading
            threading.Thread(target=run_autofocus_in_thread, daemon=True).start()

    def _propagate_current_field(self, distance_um: float):
        """
        Propagate current complex field to the given distance.
        """
        if not hasattr(self, "compensated_field_complex") or self.compensated_field_complex is None:
            messagebox.showinfo(
                "Information",
                "No compensated field available."
            )
            return

        comp_field = self.compensated_field_complex

        # Load correct λ, dx, dy from GUI depending on the mode
        self._update_physical_params()

        if abs(self.wavelength) < 1e-12 or abs(self.dx) < 1e-12 or abs(self.dy) < 1e-12:
            print("Invalid physical parameters: check wavelength or pixel size.")
            return

        field_out = pyDHM.angularSpectrum(comp_field, distance_um, self.wavelength, self.dx, self.dy)
        amp = np.abs(field_out)
        amp8 = ((amp - amp.min()) / (np.ptp(amp) + 1e-9) * 255).astype(np.uint8)

        phs = np.angle(field_out)
        phs8 = (((phs + np.pi) / (2. * np.pi)) * 255).astype(np.uint8)

        # Store results
        self.amplitude_arrays[0] = amp8
        self.phase_arrays[0] = phs8
        self.original_amplitude_arrays[0] = amp8.copy()
        self.original_phase_arrays[0] = phs8.copy()

        tk_amp = self._preserve_aspect_ratio_right(Image.fromarray(amp8))
        tk_phs = self._preserve_aspect_ratio_right(Image.fromarray(phs8))

        self.amplitude_frames = [tk_amp] if not hasattr(self, "amplitude_frames") or len(
            self.amplitude_frames) == 0 else self.amplitude_frames
        self.phase_frames = [tk_phs] if not hasattr(self, "phase_frames") or len(
            self.phase_frames) == 0 else self.phase_frames

        self.amplitude_frames[0] = tk_amp
        self.phase_frames[0] = tk_phs

        dist_txt = self._convert_distance_for_display(distance_um, include_magnification=True)
        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.processed_label.configure(image=tk_amp)
            self.processed_label.image = tk_amp
            self.processed_title_label.configure(text=f"Amplitude Image – {dist_txt}")
        else:
            self.processed_label.configure(image=tk_phs)
            self.processed_label.image = tk_phs
            self.processed_title_label.configure(text=f"Phase Image – {dist_txt}")

    def _show_autofocus_progress(self):
        self.progress_window = ctk.CTkToplevel(self)
        self.progress_window.title("Auto-Focus in Progress")
        self.progress_window.geometry("300x100")
        self.progress_window.resizable(False, False)

        self.progress_window.transient(self)
        self.progress_window.grab_set()
        self.progress_window.lift()
        self.progress_window.focus_force()

        ctk.CTkLabel(self.progress_window, text="Propagating... Please wait").pack(pady=10)
        self.progress_bar = ctk.CTkProgressBar(self.progress_window, width=220)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        self.progress_bar.update()

    def _update_autofocus_progress(self, current, total):
        progress = current / total
        if hasattr(self, "progress_bar"):
            self.progress_bar.set(progress)
            self.progress_bar.update()

    def _close_autofocus_progress(self):
        if hasattr(self, "progress_window"):
            self.progress_window.destroy()
            del self.progress_window

    def _on_autofocus_finished(self, z_best):
        self._close_autofocus_progress()

        if z_best is None:
            print("AutoFocus did not return a valid result.")
            return

        print(f"Auto-focus result: {z_best:.2f} µm")

        widgets = self.active_propagate_widgets
        slider = widgets["propagation_slider"]

        try:
            min_val = float(widgets["propagation_min_entry"].get())
            max_val = float(widgets["propagation_max_entry"].get())
            slider.configure(from_=min_val, to=max_val)
        except Exception:
            pass

        slider.set(z_best)

        # Propagation in the object plane
        try:
            M = float(widgets["propagation_magnification"].get().replace("x", "").strip())
            if M > 0:
                z_display = z_best / (M ** 2)
            else:
                z_display = z_best
        except:
            z_display = z_best

        # update distance in slide label
        unit = widgets["propagation_unit_var"].get()
        widgets["propagation_current_label"].configure(
            text=f"{z_display:.2f} {unit}"
        )

        self._propagate_current_field(z_best)

    def _update_physical_params(self):
        """Read and update wavelength, dx, dy from the appropriate panel."""
        source = getattr(self, "compensation_source", "pc")

        try:
            if source == "ps":
                self.wavelength = self.get_value_in_micrometers(self.wave_entry_ps.get(), self.wavelength_unit)
                self.dx = self.get_value_in_micrometers(self.pitchx_entry_ps.get(), self.pitch_x_unit)
                self.dy = self.get_value_in_micrometers(self.pitchy_entry_ps.get(), self.pitch_y_unit)
            elif source == "pc":
                self.wavelength = self.get_value_in_micrometers(self.wave_label_pc_entry.get(), self.wavelength_unit)
                self.dx = self.get_value_in_micrometers(self.pitchx_label_pc_entry.get(), self.pitch_x_unit)
                self.dy = self.get_value_in_micrometers(self.pitchy_label_pc_entry.get(), self.pitch_y_unit)
            elif source == "np":
                self.wavelength = self.get_value_in_micrometers(self.wave_label_np_entry.get(), self.wavelength_unit)
                self.dx = self.get_value_in_micrometers(self.pitchx_label_np_entry.get(), self.pitch_x_unit)
                self.dy = self.get_value_in_micrometers(self.pitchy_label_np_entry.get(), self.pitch_y_unit)
        except Exception:
            print("Error reading physical parameters.")

    def activate_phase_compensation(self):
        self.phase_compensation_frame.tkraise()
        self.active_propagate_widgets = self.propagate_widgets_pc
        self.compensation_source = "pc"

    def activate_phase_shifting(self):
        self.phase_shifting_frame.tkraise()
        self.active_propagate_widgets = self.propagate_widgets_ps
        self.compensation_source = "ps"

    def activate_numerical_propagation(self):
        self.numerical_propagation_frame.tkraise()
        self.active_propagate_widgets = self.propagate_widgets_np
        self.compensation_source = "np"

    def _cache_compensation_params(self, wavelength_um: float, dx_um: float, dy_um: float) -> None:
        self.wavelength = wavelength_um
        self.pitch_x = dx_um
        self.pitch_y = dy_um

    def run_phase_compensation(self):
        self.compensation_source = "pc"

        if (
                not hasattr(self, "arr_hologram") or
                self.arr_hologram is None or
                not isinstance(self.arr_hologram, np.ndarray) or
                self.arr_hologram.size == 0 or
                np.all(self.arr_hologram == 0)
        ):
            messagebox.showinfo(
                "Warning",
                "No hologram is currently loaded. Please load a valid hologram before applying phase compensation."
            )
            return

        try:
            self.wavelength = self.get_value_in_micrometers(self.wave_label_pc_entry.get(), self.wavelength_unit)
        except:
            self.wavelength = 0.0
        try:
            self.dx = self.get_value_in_micrometers(self.pitchx_label_pc_entry.get(), self.pitch_x_unit)
        except:
            self.dx = 0.0
        try:
            self.dy = self.get_value_in_micrometers(self.pitchy_label_pc_entry.get(), self.pitch_y_unit)
        except:
            self.dy = 0.0

        # Check if any of the parameters are zero
        if self.wavelength == 0.0 or self.dx == 0.0 or self.dy == 0.0:
            messagebox.showwarning(
                "Warning",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them before proceeding."
            )
            return

        w_val = self.wavelength
        px_val = self.dx
        py_val = self.dy

        self._cache_compensation_params(w_val, px_val, py_val)

        # Spatial-filter choice
        if self.spatial_section_var.get() == 0:
            selected_filter = self.spatial_auto_var_pc.get().strip()
        else:
            selected_filter = self.spatial_manual_var_pc.get().strip()
            if selected_filter == "None":
                tk.messagebox.showwarning("Filter", "Choose a manual filter type or switch to Automatic.")
                return

        # Ask for coordinates when required
        self.manual_coords = None
        if selected_filter == "Circular Coor.":
            self.manual_coords = self._prompt_circle_coordinates()
            if self.manual_coords is None:
                messagebox.showinfo(
                    "Information",
                    "User cancelled circular coordinate input."
                )
                return

        elif selected_filter == "Rectangle Coor.":
            self.manual_coords = self._prompt_rectangle_coordinates()
            if self.manual_coords is None:
                messagebox.showinfo(
                    "Information",
                    "User cancelled rectangle coordinate input."
                )
                return

        elif selected_filter == "Non Telecentric Coordinates":
            self.manual_coords = self._prompt_rectangle_coordinates(title="Non-telecentric filter")
            if self.manual_coords is None:
                messagebox.showinfo(
                    "Information",
                    "UUser cancelled non-telecentric coordinate input."
                )
                return

        # canonicalise legacy names so the rest of the pipeline keeps working
        alias = {
            "Circular Manual": "Manual Circular",
            "Rectangle Manual": "Manual Rectangle",
            "Non Telecentric Manual": "No Telecentric",
        }
        selected_filter = alias.get(selected_filter, selected_filter)

        # make the (internal-use) variable reflect the final choice
        self.spatial_filter_var_pc.set(selected_filter)

        # Clear old reconstructions:
        self.reset_reconstruction_data()

        method = self.pc_method_var.get()
        if method == 0:
            comp_output = pyDHM.ERS(inp=self.arr_hologram, wavelength=self.wavelength, dx=self.dx,
                                    dy=self.dy, s=5, step=0.2, filter_type=self.spatial_filter_var_pc.get(),
                                    manual_coords=self.manual_coords,
                                    filtering_function=self.custom_filtering_function)
        elif method == 1:
            comp_output = pyDHM.CFS(inp=self.arr_hologram, wavelength=self.wavelength, dx=self.dx,
                                    dy=self.dy, filter_type=self.spatial_filter_var_pc.get(),
                                    manual_coords=self.manual_coords,
                                    spatial_filtering_fn=self.custom_filtering_function)
        elif method == 2:
            chosen_filter = self.spatial_filter_var_pc.get()
            comp_output = self.CNT(self.arr_hologram, self.wavelength, self.dx, self.dy, chosen_filter)
        elif method == 3:
            comp_output = pyDHM.vortexLegendre(inp=self.arr_hologram, wavelength=self.wavelength, dx=self.dx,
                                               dy=self.dy, limit=256 / 2, filter_type=self.spatial_filter_var_pc.get(),
                                               manual_coords=self.manual_coords,
                                               spatial_filtering_fn=self.custom_filtering_function)
        else:
            print("Unknown phase compensation method.")
            return

        if comp_output is None:
            print("Phase compensation returned None.")
            return

        # Store the compensated complex field for reuse
        self.compensated_field_complex = comp_output.copy()
        self.original_complex_fields = [comp_output.copy()]
        self.complex_fields = [comp_output.copy()]

        if not hasattr(self, "complex_fields"):
            self.complex_fields = []
        self.complex_fields.append(comp_output)

        # Compute amplitude and phase
        amp = np.abs(comp_output)
        raw_phase = np.angle(comp_output)

        amp_norm = (amp - amp.min()) / (amp.max() - amp.min() + 1e-9) * 255
        amp_norm = amp_norm.astype(np.uint8)

        phase_0to1 = (raw_phase + np.pi) / (2 * np.pi + 1e-9)
        phase_0to1 = np.clip(phase_0to1, 0, 1)
        phase_8bit = (phase_0to1 * 255).astype(np.uint8)

        # Create PIL images
        amp_pil = Image.fromarray(amp_norm, mode='L')
        phs_pil = Image.fromarray(phase_8bit, mode='L')

        # Preserve aspect ratio for the right frame
        tk_amp = self._preserve_aspect_ratio_right(amp_pil)
        tk_phs = self._preserve_aspect_ratio_right(phs_pil)

        # Store arrays in self.amplitude_arrays / self.phase_arrays
        self.amplitude_arrays = [amp_norm]
        self.phase_arrays = [phase_8bit]

        # Also store original copies for filtering
        self.original_amplitude_arrays = [amp_norm.copy()]
        self.original_phase_arrays = [phase_8bit.copy()]

        self.filter_states_dim1 = [tGUI.default_filter_state()]
        self.filter_states_dim2 = [tGUI.default_filter_state()]

        self.last_filter_settings = None
        self.amplitude_frames = [tk_amp]
        self.phase_frames = [tk_phs]
        self.current_amp_index = 0
        self.current_phase_index = 0

        # Display the new Phase
        self.processed_label.configure(image=tk_phs)
        self.processed_title_label.configure(text="Phase Reconstruction ")

        # Force the right radio button to match the displayed Phase
        self.recon_view_var.set("Phase Reconstruction ")
        self.update_right_view()

    def CNT(self, inp, wavelength, dx, dy, spatialFilter=None):
        # Unids to microns
        wavelength_m = wavelength * 1e-6
        dx_m = dx * 1e-6
        dy_m = dy * 1e-6
        k = (2 * np.pi) / wavelength_m

        inp = np.array(inp)
        M, N = inp.shape
        x = np.arange(0, N, 1)
        y = np.arange(0, M, 1)
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

        # Manuel spatial filter
        if spatialFilter == "Non Telecentric Manual":
            Xcenter, Ycenter, holo_filter, ROI_array = self.spatialFilterinCNT(inp, M, N)
        elif spatialFilter == "Non Telecentric Coordinates":
            # Use the mask already created via spatialFilteringCF upstream
            print("CNT: using supplied non-telecentric coordinates.")
        else:
            print("CNT: select a non-telecentric option in the GUI.")
            return None
        print("CNT: Filtrado finalizado.")

        ThetaXM = math.asin((N / 2 - Xcenter) * wavelength_m / (M * dx_m))
        ThetaYM = math.asin((M / 2 - Ycenter) * wavelength_m / (N * dy_m))
        reference = np.exp(1j * k * ((math.sin(ThetaXM) * X * dx_m) + (math.sin(ThetaYM) * Y * dy_m)))
        comp_phase = holo_filter * reference
        phase_c = np.angle(comp_phase)

        minVal = phase_c.min()
        maxVal = phase_c.max()
        if abs(maxVal - minVal) < 1e-9:
            print("CNT: Fase degenerada. No se puede compensar.")
            return comp_phase

        phase_norm = (phase_c - minVal) / (maxVal - minVal)
        binary_phase = (phase_norm > 0.2)

        m = abs(ROI_array[2] - ROI_array[0])
        n = abs(ROI_array[3] - ROI_array[1])
        Cx = (M * dx_m) ** 2 / (wavelength_m * m) if m else 1
        Cy = (N * dy_m) ** 2 / (wavelength_m * n) if n else 1
        cur = (Cx + Cy) / 2

        print("CNT: Necesitas el centro de la fase circular en la imagen binarizada.")
        print("Lo normal es implementarlo con input() o un click manual, Code1 pide input.")
        p = (M / 2)
        q = (N / 2)
        f = ((M / 2) - p) / 2
        g = ((N / 2) - q) / 2

        print("CNT: Inicia búsqueda gruesa ...")
        cont = 0
        sum_max = 0
        s = 100
        step = 50
        perc = 0.4
        arrayCurv = np.arange(cur - (cur * perc), cur + (cur * perc), (cur * perc) / 6.0)
        arrayXc = np.arange(f - s, f + s, step)
        arrayYc = np.arange(g - s, g + s, step)
        for ctemp in arrayCurv:
            for ftemp in arrayXc:
                for gtemp in arrayYc:
                    cont += 1
                    phi_sph = ((X - ftemp) ** 2 * (dx_m ** 2) / ctemp) + ((Y - gtemp) ** 2 * (dy_m ** 2) / ctemp)
                    phi_sph = np.exp(-1j * (np.pi * phi_sph / wavelength_m))
                    phaseComp = np.angle(comp_phase * phi_sph)
                    pmin = phaseComp.min()
                    pmax = phaseComp.max()
                    if abs(pmax - pmin) < 1e-9:
                        continue
                    ph_sca = (phaseComp - pmin) / (pmax - pmin)
                    bin_sc = (ph_sca > 0.2)
                    ssum = np.sum(bin_sc)
                    if ssum > sum_max:
                        sum_max = ssum
                        f_out = ftemp
                        g_out = gtemp
                        cur_out = ctemp

        cont = 0
        sum_max = 0
        s = 10
        step = 2
        perc = 0.1
        arrayXc = np.arange(f_out - s, f_out + s, step)
        arrayYc = np.arange(g_out - s, g_out + s, step)
        arrayCurv = np.arange(cur_out - (cur_out * perc), cur_out + (cur_out * perc), 0.01)

        for ctemp in arrayCurv:
            for ftemp in arrayXc:
                for gtemp in arrayYc:
                    cont += 1
                    phi_sph = ((X - ftemp) ** 2 * (dx_m ** 2) / ctemp) + ((Y - gtemp) ** 2 * (dy_m ** 2) / ctemp)
                    phi_sph = np.exp(-1j * (np.pi * phi_sph / wavelength_m))
                    phaseComp = np.angle(comp_phase * phi_sph)
                    pmin = phaseComp.min()
                    pmax = phaseComp.max()
                    if abs(pmax - pmin) < 1e-9:
                        continue
                    ph_sca = (phaseComp - pmin) / (pmax - pmin)
                    bin_sc = (ph_sca > 0.2)
                    ssum = np.sum(bin_sc)
                    if ssum > sum_max:
                        sum_max = ssum
                        f_out = ftemp
                        g_out = gtemp
                        cur_out = ctemp

        # Initial phase
        phi_sph = ((X - f_out) ** 2 * (dx_m ** 2) / cur_out) + ((Y - g_out) ** 2 * (dy_m ** 2) / cur_out)
        phi_sph = np.exp(-1j * (np.pi * phi_sph / wavelength_m))
        phaseCompensate = comp_phase * phi_sph

        print("CNT: Fase compensada completada.")
        return phaseCompensate

    def spatialFilterinCNT(self, inp, M, N):
        """
        Interactive spatial filtering with cv2.selectROI for Non-Telecentric.
        Returns (Xcenter, Ycenter, holo_filter, ROI_array).
        """
        ROI_array = np.zeros(4)
        holoFT = np.float32(inp)
        fft_holo = cv2.dft(holoFT, flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_holo = np.fft.fftshift(fft_holo)
        fft_holo_image = 20 * np.log(cv2.magnitude(fft_holo[:, :, 0], fft_holo[:, :, 1]))
        mini = np.amin(np.abs(fft_holo_image))
        maxi = np.amax(np.abs(fft_holo_image))
        fft_holo_image = cv2.convertScaleAbs(fft_holo_image,
                                             alpha=255.0 / (maxi - mini), beta=-mini * 255.0 / (maxi - mini))

        ROI = cv2.selectROI("Seleccione ROI - No Telecentric", fft_holo_image, fromCenter=True)
        cv2.destroyWindow("Seleccione ROI - No Telecentric")

        x1_ROI = int(ROI[0])
        y1_ROI = int(ROI[1])
        x2_ROI = int(ROI[0] + ROI[2])
        y2_ROI = int(ROI[1] + ROI[3])

        ROI_array[0] = y1_ROI
        ROI_array[1] = x1_ROI
        ROI_array[2] = y2_ROI
        ROI_array[3] = x2_ROI

        # Center
        Ycenter = (y1_ROI + y2_ROI) / 2.0
        Xcenter = (x1_ROI + x2_ROI) / 2.0

        holo_filterFT = np.zeros((M, N, 2), np.float32)
        holo_filterFT[y1_ROI:y2_ROI, x1_ROI:x2_ROI] = 1.0
        holo_filterFT = holo_filterFT * fft_holo
        holo_filterFT = np.fft.ifftshift(holo_filterFT)
        holo_filter_spatial = cv2.idft(holo_filterFT, flags=cv2.DFT_INVERSE)
        real_part = holo_filter_spatial[:, :, 0]
        imag_part = holo_filter_spatial[:, :, 1]
        holo_filter = real_part + 1j * imag_part

        return Xcenter, Ycenter, holo_filter, ROI_array

    def regime(self, field):
        holoFT = np.float32(field)
        fft_holo = cv2.dft(holoFT, flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_holo = np.fft.fftshift(fft_holo)
        fft_holo_image = 20 * np.log(cv2.magnitude(fft_holo[:, :, 0], fft_holo[:, :, 1]))
        minVal = np.amin(np.abs(fft_holo_image))
        maxVal = np.amax(np.abs(fft_holo_image))
        fft_holo_image = cv2.convertScaleAbs(fft_holo_image, alpha=255.0 / (maxVal - minVal),
                                             beta=-minVal * 255.0 / (maxVal - minVal))

        ret, thresh = cv2.threshold(fft_holo_image, 200, 255, cv2.THRESH_BINARY)
        thresh_rize = cv2.resize(thresh, (1024, 1024))

        contours, hierarchy = cv2.findContours(image=thresh_rize, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        fft_holo_image = cv2.resize(thresh, (1024, 1024))
        image_copy = fft_holo_image.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        orders = len(contours)
        return orders, thresh

    def custom_filtering_function(self, field, height, width, filter_type, manual_coords=None):
        # Define optional callbacks
        if filter_type == "Circular Man.":
            draw_circle = lambda: draw_manual_circle(arr_hologram=field)
        else:
            draw_circle = None

        if filter_type == "Rectangle Man.":
            draw_rect = lambda: draw_manual_rectangle(arr_hologram=field)
        else:
            draw_rect = None

        # Pass the popup function when using coordinate-based filtering
        if filter_type in ["Rectangle Coor.", "Non Telecentric Coordinates"]:
            prompt_rect = self._prompt_rectangle_coordinates
        else:
            prompt_rect = None

        if filter_type == "Circular Coor.":
            prompt_circ = self._prompt_circle_coordinates
        else:
            prompt_circ = None

        return spatialFilteringCF(
            field=field,
            height=height,
            width=width,
            filter_type=filter_type,
            manual_coords=manual_coords,
            draw_manual_rectangle=draw_rect,
            draw_manual_circle=draw_circle,
            prompt_rectangle_coords=prompt_rect,
            prompt_circle_coords=prompt_circ,
            show_ft_and_filter=True
        )

    # Numerical Propagation
    def init_numerical_propagation_frame(self) -> None:
        self.numerical_propagation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.numerical_propagation_frame.grid_propagate(False)

        self.numprop_container = ctk.CTkFrame(self.numerical_propagation_frame, corner_radius=8, width=420)
        self.numprop_container.grid_propagate(False)
        self.numprop_container.pack(fill="both", expand=True)

        self.numprop_scrollbar = ctk.CTkScrollbar(self.numprop_container, orientation='vertical')
        self.numprop_scrollbar.grid(row=0, column=0, sticky='ns')

        self.numprop_canvas = ctk.CTkCanvas(self.numprop_container, width=PARAMETER_FRAME_WIDTH)
        self.numprop_canvas.grid(row=0, column=1, sticky='nsew')

        self.numprop_container.grid_rowconfigure(0, weight=1)
        self.numprop_container.grid_columnconfigure(1, weight=1)

        self.numprop_canvas.configure(yscrollcommand=self.numprop_scrollbar.set)
        self.numprop_scrollbar.configure(command=self.numprop_canvas.yview)

        self.numerical_propagation_inner_frame = ctk.CTkFrame(self.numprop_canvas)
        self.numprop_canvas.create_window((0, 0), window=self.numerical_propagation_inner_frame, anchor='nw')

        # Navigation strip (replaces old title label)
        self._add_processing_nav(self.numerical_propagation_inner_frame, current="numerical_propagation")

        # Field source frame
        self.field_src_frame = ctk.CTkFrame(
            self.numerical_propagation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.field_src_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.field_src_frame.grid_propagate(False)

        # Title aligned to the left as before
        ctk.CTkLabel(
            self.field_src_frame,
            text="Field to propagate",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        # Radio buttons aligned horizontally, and vertically across frames
        self.np_source_var = tk.IntVar(value=0)
        self.np_source_var.trace_add("write", self.on_np_source_changed)

        # Left option (column 0)
        rb_holo = ctk.CTkRadioButton(
            self.field_src_frame, text="Digital Hologram",
            variable=self.np_source_var, value=0
        )
        rb_holo.grid(row=1, column=0, sticky='w', padx=50, pady=4)

        # Right option (column 1)
        rb_coh = ctk.CTkRadioButton(
            self.field_src_frame, text="Coherent Image",
            variable=self.np_source_var, value=1
        )
        rb_coh.grid(row=1, column=1, sticky='w', padx=50, pady=4)

        # Propagation method frame
        self.prop_method_frame = ctk.CTkFrame(
            self.numerical_propagation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT
        )
        self.prop_method_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.prop_method_frame.grid_propagate(False)

        # Title aligned left as before
        ctk.CTkLabel(
            self.prop_method_frame,
            text='Choose a Propagation Method',
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        # Same columns as above: column 0 and 1
        self.prop_method_var = tk.IntVar(value=0)
        self.prop_method_var.trace_add("write", self.update_propagation_params)

        # Left option aligned with "Digital Hologram"
        ctk.CTkRadioButton(
            self.prop_method_frame, text='Angular Spectrum',
            variable=self.prop_method_var, value=0
        ).grid(row=1, column=0, sticky='w', padx=50, pady=4)

        # Right option aligned with "Coherent Image"
        ctk.CTkRadioButton(
            self.prop_method_frame, text='Fresnel',
            variable=self.prop_method_var, value=1
        ).grid(row=1, column=1, sticky='w', padx=50, pady=4)

        # PARAMETERS
        self.params_np_frame = ctk.CTkFrame(
            self.numerical_propagation_inner_frame,
            width=PARAMETER_FRAME_WIDTH * 2,
        )
        self.params_np_frame.grid(row=3, column=0, sticky='ew', pady=2)
        self.params_np_frame.grid_propagate(True)
        for col in range(3):
            self.params_np_frame.columnconfigure(col, weight=1)

        # Title label for the reconstruction parameters section
        ctk.CTkLabel(
            self.params_np_frame,
            text="Loading Physical Parameters",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=5, pady=(5, 2), sticky="w")

        # Define the unit options for the dropdowns
        units = ["µm", "nm", "mm", "cm", "m", "in"]

        # Dictionary to store the input fields for external access
        self.param_entries_np = {}

        # Create each parameter field using a reusable function
        fGUI.create_param_with_arrow(
            parent=self.params_np_frame,
            row=1, col=0,
            label_text=f"Wavelength ({self.wavelength_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_np,
            entry_key="wavelength",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_np_frame,
            row=1, col=1,
            label_text=f"Pitch X ({self.pitch_x_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_np,
            entry_key="pitch_x",
            unit_update_callback=self._set_unit_in_label
        )

        fGUI.create_param_with_arrow(
            parent=self.params_np_frame,
            row=1, col=2,
            label_text=f"Pitch Y ({self.pitch_y_unit})",
            unit_list=units,
            entry_name_dict=self.param_entries_np,
            entry_key="pitch_y",
            unit_update_callback=self._set_unit_in_label
        )

        self.wave_label_np_entry = self.param_entries_np["wavelength"]
        self.pitchx_label_np_entry = self.param_entries_np["pitch_x"]
        self.pitchy_label_np_entry = self.param_entries_np["pitch_y"]

        # Add the "Apply" button that calls run_numerical_propagation
        self.apply_button_np = ctk.CTkButton(
            self.params_np_frame,
            text="Apply",
            command=self.run_numerical_propagation,
            width=100,
        )
        self.apply_button_np.grid(row=3, column=0, padx=10, columnspan=3, pady=(10, 10), sticky='w')

        # PROPAGATION PANEL
        self.np_propagation_panel = ctk.CTkFrame(
            self.numerical_propagation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=PARAMETER_FRAME_HEIGHT * 4
        )
        self.np_propagation_panel.grid(row=4, column=0, sticky='ew', pady=2)
        self.np_propagation_panel.grid_propagate(True)

        for col in range(3):
            self.np_propagation_panel.columnconfigure(col, weight=1)

        # Title label
        ctk.CTkLabel(
            self.np_propagation_panel,
            text="Propagation Options",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=5, pady=(5, 2), sticky="w")

        # Create propagation panel using reusable function
        self.propagate_widgets_np = fGUI.create_propagate_panel(
            parent=self.np_propagation_panel,
            attr_prefix="propagation",
            on_slider_change=self._on_slider_moved
        )

        # Make these widgets the current active ones for processing
        self.active_propagate_widgets = self.propagate_widgets_np
        self.compensation_source = "np"

        self.propagate_widgets_np["propagation_apply_button"].configure(
            command=lambda: self._apply_propagation(request_magnification=False)
        )

        # final canvas refresh
        self.numerical_propagation_inner_frame.update_idletasks()
        self.numprop_canvas.config(scrollregion=self.numprop_canvas.bbox("all"))

    # Apply button in Numerical Propagation
    def run_numerical_propagation(self):
        source = self.np_source_var.get()

        if source == 0:
            print("[DEBUG] Working with digital hologram")

            if (
                    not hasattr(self, "arr_hologram") or
                    self.arr_hologram is None or
                    not isinstance(self.arr_hologram, np.ndarray) or
                    self.arr_hologram.size == 0 or
                    np.all(self.arr_hologram == 0)
            ):
                messagebox.showinfo(
                    "Warning",
                    "No hologram is currently loaded. Please load a valid hologram before applying numerical propagation."
                )
                return

            self._apply_propagation()

        elif source == 1:
            print("[DEBUG] Working with coherent image")

            if (
                    not hasattr(self, "coherent_input_image") or
                    self.coherent_input_image is None or
                    not isinstance(self.coherent_input_image, np.ndarray) or
                    self.coherent_input_image.size == 0 or
                    np.all(self.coherent_input_image == 0)
            ):
                messagebox.showinfo(
                    "Warning",
                    "No coherent image is currently loaded. Please load a valid image before applying numerical propagation."
                )
                return

            self._load_and_display_coherent_image()
            self._apply_propagation(request_magnification=True)

        try:
            w_val = self.get_value_in_micrometers(self.wave_label_np_entry.get(), self.wavelength_unit)
        except:
            w_val = 0.0

        try:
            px_val = self.get_value_in_micrometers(self.pitchx_label_np_entry.get(), self.pitch_x_unit)
        except:
            px_val = 0.0

        try:
            py_val = self.get_value_in_micrometers(self.pitchy_label_np_entry.get(), self.pitch_y_unit)
        except:
            py_val = 0.0

        # Check if the parameter have beem defined
        if w_val == 0.0 or px_val == 0.0 or py_val == 0.0:
            messagebox.showwarning(
                "Warning",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them before proceeding."
            )
            return

    # Check the radio button options for Numerical Propagation
    def on_np_source_changed(self, *_):
        selected = self.np_source_var.get()

        if selected == 1:
            if hasattr(self, "coherent_input_image") and self.coherent_input_image is not None:
                print("[DEBUG] Coherent Image already loaded – loading and displaying...")
            else:
                print("[DEBUG] No coherent image loaded yet.")

    # Load and Display for coherent Image
    def _load_and_display_coherent_image(self):
        """Load a coherent image, convert it to complex field, and update both GUI panels."""
        field = pyDHM.convert_loaded_image_to_field(self)

        if field is None:
            print("No coherent image loaded or conversion failed.")
            return

        self.coherent_field_complex = field  # Optional: keep for reuse
        self.compensated_field_complex = field

        img_uint8 = np.asarray(self.coherent_input_image)

        # Left panel: original image + FT
        tk_img = self._preserve_aspect_ratio(
            Image.fromarray(img_uint8),
            self.viewbox_width, self.viewbox_height
        )
        tk_ft, ft_disp = self._create_ft_frame(img_uint8)

        self.multi_holo_arrays = [img_uint8]
        self.original_multi_holo_arrays = [img_uint8.copy()]
        self.hologram_frames = [tk_img]
        self.multi_ft_arrays = [ft_disp]
        self.ft_frames = [tk_ft]
        self.current_left_index = 0
        self.arr_hologram = img_uint8
        self.current_ft_array = ft_disp

        self.captured_label.configure(image=tk_img)
        self.captured_label.image = tk_img
        self.captured_title_label.configure(text="Coherent image")

        if self.holo_view_var.get() == "Fourier Transform":
            self.captured_label.configure(image=tk_ft)

        self.hide_holo_arrows()

        # Right panel: amplitude
        amp = np.abs(field)
        amp8 = ((amp - amp.min()) / (np.ptp(amp) + 1e-9) * 255).astype(np.uint8)
        phs = np.angle(field)
        phs8 = (((phs + np.pi) / (2 * np.pi)) * 255).astype(np.uint8)

        self.amplitude_arrays = [amp8]
        self.phase_arrays = [phs8]
        self.original_amplitude_arrays = [amp8.copy()]
        self.original_phase_arrays = [phs8.copy()]

        tk_amp = self._preserve_aspect_ratio_right(Image.fromarray(amp8))
        if not hasattr(self, "amplitude_frames"):
            self.amplitude_frames = []

        if len(self.amplitude_frames) == 0:
            self.amplitude_frames.append(tk_amp)
        else:
            self.amplitude_frames[0] = tk_amp

        self.processed_label.configure(image=tk_amp)
        self.processed_label.image = tk_amp
        self.processed_title_label.configure(text="Amplitude – Coherent Image")

    # propagation parameters
    def update_propagation_params(self, *args):
        """Handle changes in propagation method selection and update the parameter section."""
        metodo = self.prop_method_var.get()
        print(f"[DEBUG] Propagation method changed to: {metodo}")

        # Common "Propagate" button
        self.propagate_button = ctk.CTkButton(
            self.params_np_frame,
            width=PARAMETER_BUTTON_WIDTH,
            text='Propagate',
            command=self.run_propagation
        )
        self.propagate_button.grid(row=6, column=1, sticky='ew', padx=5, pady=(10, 10))

    def get_value_in_micrometers(self, value, unit):
        conversion_factors = {
            "Micrometers": 1,
            "µm": 1,
            "nm": 1e-3,
            "mm": 1e3,
            "cm": 1e4,
            "m": 1e6,
            "in": 2.54e4
        }

        val_str = value.strip().replace(',', '.')
        try:
            float_val = float(val_str)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to float.")

        factor = conversion_factors.get(unit, 1)
        return float_val * factor

    # ─────────────────────────────────────────────
    # Menu navigation buttons"
    # ─────────────────────────────────────────────
    def change_menu_to(self, name: str):
        if name == 'home':
            self.navigation_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.navigation_frame.grid_forget()

        # Phase Shifting
        if name == "phase_shifting":
            self.phase_shifting_frame.grid(row=0, column=0, sticky="nsew", padx=5)
            self.activate_phase_shifting()
        else:
            self.phase_shifting_frame.grid_forget()

        # Numerical Propagation
        if name == 'numerical_propagation':
            self.numerical_propagation_frame.grid(row=0, column=0, sticky='nsew', padx=5)
            self.activate_numerical_propagation()
        else:
            self.numerical_propagation_frame.grid_forget()

        # Phase Compensation
        if name == "phase_compensation":
            self.phase_compensation_frame.grid(row=0, column=0, sticky="nsew", padx=5)
            self.activate_phase_compensation()
        else:
            self.phase_compensation_frame.grid_forget()

        # bio
        if name == 'bio':
            self.bio_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.bio_frame.grid_forget()

        # Filters
        if name == 'filters':
            self.filters_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.filters_frame.grid_forget()

        # Speckle
        if name == 'speckle':
            self.speckles_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.speckles_frame.grid_forget()

    def _sync_canvas_and_frame_bg(self):
        mode = ctk.get_appearance_mode()
        color = "gray15" if mode == "Dark" else "gray85"

        # Canvases
        for canvas_attr in [
            "filters_canvas", "tools_canvas", "param_canvas",
            "numprop_canvas", "pc_canvas"
        ]:
            canvas = getattr(self, canvas_attr, None)
            if canvas is not None:
                canvas.configure(background=color)

        # Frames
        for frame_attr in [
            "filters_frame", "filters_container", "filters_inner_frame",
            "bio_frame", "tools_container", "tools_inner_frame",
            "parameters_inner_frame", "phase_shifting_frame", "param_container",
            "numerical_propagation_frame", "numprop_container", "numerical_propagation_inner_frame",
            "phase_compensation_frame", "pc_container", "phase_compensation_inner_frame",
            "viewing_frame", "navigation_frame",
            "two_views_frame"
        ]:
            frame = getattr(self, frame_attr, None)
            if frame is not None:
                frame.configure(fg_color=color)

    def after_idle_setup(self):
        self._sync_canvas_and_frame_bg()

    def change_appearance_mode_event(self, new_appearance_mode):
        if new_appearance_mode == "Main Menu":
            self.open_main_menu()
        else:
            ctk.set_appearance_mode(new_appearance_mode)
            self._sync_canvas_and_frame_bg()

    def open_main_menu(self):
        self.destroy()
        # replace 'main_menu' with the actual module name where MainMenu lives
        main_mod = import_module("Main_")
        reload(main_mod)
        MainMenu = getattr(main_mod, "MainMenu")
        MainMenu().mainloop()

    def release(self):
        os.system("taskkill /f /im python.exe")


if __name__ == '__main__':
    app = App()
    app.mainloop()
    app.release()
    app.release()
