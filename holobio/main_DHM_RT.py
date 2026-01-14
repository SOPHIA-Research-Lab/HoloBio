
import zipfile, io
import customtkinter as ctk
from .parallel_rc import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import math
import cv2, os, time, tkinter as tk
from importlib import import_module, reload
from . import functions_GUI as fGUI
import threading, queue
from pandastable.core import Table
import pandas as pd
import re
from .track_particles_kalman import track_particles_kalman as track


class App(ctk.CTk):

    _PREFERRED_CAM_KEYWORDS = [
        "imaging",
        "the imaging source",
        "ic capture",
        "dfk", "dmk", "dff"
    ]
    _FALLBACK_MIN_WIDTH = 960

    def __init__(self):
        self._configure_ffmpeg_single_thread()
        ctk.set_appearance_mode("Light")
        super().__init__()
        self.title('HoloBio: DHM - Real Time')
        self.attributes('-fullscreen', False)
        self.state('normal')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE) / 1.8

        # Parameters
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

        # Arrays
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
        self.record_frame = None

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

        self.current_left_index = 0

        self.original_hologram = None
        self.phase_shift_imgs = []
        self.amplitude_arrays = []
        self.phase_arrays = []
        self.amplitude_frames = []
        self.phase_frames = []
        self.original_amplitude_arrays = []
        self.original_phase_arrays = []

        self.multi_ft_arrays = []
        self.multi_holo_arrays = []
        self.original_multi_holo_arrays = []
        self.hologram_frames = []
        self.ft_frames = []

        # Keep track of last applied filter settings
        self.last_filter_settings = None
        self.speckle_kernel_var = tk.IntVar(self, value=5)

        self.filter_states_dim0 = []
        self.filter_states_dim1 = []
        self.filter_states_dim2 = []

        # Live-preview / sequence-recording flags
        self.preview_active = False
        self.sequence_recording = False
        self.seq_save_root = ""
        self.seq_frame_counter = 0
        self.last_preview_gray = None
        self.last_preview_ft = None
        self.ft_display_filtered = False
        self.video_playing = None
        self.is_video_preview = None
        self.source_mode = None
        self.is_playing = False

        self.viewbox_width = 600
        self.viewbox_height = 450

        self.init_phase_compensation_frame()

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid_rowconfigure(0, weight=0)
        self.viewing_frame.grid_rowconfigure(1, weight=1)
        self.viewing_frame.grid_columnconfigure(0, weight=1)

        # Build toolbar and panels
        fGUI.build_toolbar(self)
        fGUI.build_two_views_panel(self)

        self.tools_menu.configure(state="disabled")
        self.update_idletasks()
        self.phase_compensation_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        self.viewing_frame.grid(row=0, column=1, sticky="nsew")

        self._sync_canvas_and_frame_bg()
        self._init_async_engine()
        self._stop_compensation = threading.Event()

    def _init_async_engine(self) -> None:
        """Prepare a queue and stop-flag for the background worker."""
        self._comp_queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop_compensation = threading.Event()

    def get_load_menu_values(self) -> list[str]:
        """Order now fixed exactly as requested."""
        return ["Init Camera", "Load Video"]

    def _on_load_select(self, choice: str) -> None:
        """Dispatch the two options from the Load menu."""
        self._reset_source()

        if choice == "Init Camera":
            self.source_mode = "camera"
            self._init_camera()
            if self.cap and self.cap.isOpened():
                self.start_preview_stream()

        elif choice == "Load Video":
            self.source_mode = "video"
            self.load_video()

        self.load_menu.set("Load")
        self.after(200, lambda: self.load_menu.set("Load"))

    def _reset_source(self) -> None:
        """Stops any running camera/video and clears cache/memory."""
        self.preview_active = False
        self.video_playing = False
        self.realtime_active = False

        # Cancel any video loop
        if hasattr(self, "_video_loop_id"):
            self.after_cancel(self._video_loop_id)
            del self._video_loop_id

        # Cancel background thread
        if hasattr(self, "play_thread") and self.play_thread is not None:
            if self.play_thread.is_alive():
                self.play_thread.join(timeout=1.0)
            self.play_thread = None

        # Cancel compensation worker
        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()

        # Release any video/camera
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

        # Clean frame buffers
        self.hologram_frames.clear()
        self.ft_frames.clear()
        self.multi_holo_arrays.clear()
        self.multi_ft_arrays.clear()
        self.phase_arrays.clear()
        self.amplitude_arrays.clear()
        self.current_holo_array = None
        self.current_ft_array = None
        self.current_phase_array = None
        self.current_amplitude_array = None

    def pause_visualization(self) -> None:
        """Pauses the current preview (camera or video) *without* closing it."""
        self.preview_active  = False
        if hasattr(self, "video_playing"):
            self.video_playing = False

        # Stop any ongoing compensation thread
        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()

    def resume_video_preview(self) -> None:
        """Resumes the video-preview loop after a pause."""
        if not getattr(self, "cap", None):
            return
        self.video_playing = True
        self._play_video_preview()

    def _on_tools_select(self, *_):
        """Tools menu is intentionally disabled – nothing happens."""
        tk.messagebox.showinfo("Tools", "Feature unavailable in this build.")

    def _on_save_select(self, option: str) -> None:
        """Hook for the ‘Save’ dropdown in the toolbar."""
        self._handle_save_option(option)
        self.save_menu.set("Save")

    def _on_theme_select(self, theme: str) -> None:
        """Light / Dark selector from the toolbar."""
        self.change_appearance_mode_event(theme)

    # Minimal video loader
    def _open_video_file(self, path: str) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None

        # Force single-thread decoding (avoids ‘async_lock’ assert)
        prop_threads = getattr(cv2, "CAP_PROP_THREADS", 59)
        cap.set(prop_threads, 1)
        return cap

    def load_video(self) -> None:
        """Opens video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        if hasattr(self, "cap") and self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Video Error", "Could not open the selected video.")
            return

        # Store source FPS for later use when saving
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self.source_fps <= 0 or self.source_fps > 240:  # fallback if invalid
            self.source_fps = 30.0

        self.is_video_preview = True
        self.preview_active = False
        self.video_playing = False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read the first frame and show it
        ok, frame = self.cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Create thumbnail for hologram viewer
            holo_tk = self._preserve_aspect_ratio(
                Image.fromarray(gray), self.viewbox_width, self.viewbox_height)

            # Update the arrays and frames for navigation
            self.hologram_frames = [holo_tk]
            self.multi_holo_arrays = [gray]
            self.current_holo_array = gray
            self.current_left_index = 0

            # Show in hologram viewer
            if self.holo_view_var.get() == "Hologram":
                self.captured_label.configure(image=holo_tk)
                self.captured_label.image = holo_tk
                self.captured_title_label.configure(text="Hologram")

    def _show_image(self, img: np.ndarray) -> None:
        """Displays a grayscale image in the amplitude view pane."""
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_normalized.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        if hasattr(self, "amplitude_view"):
            self.amplitude_view.configure(image=img_tk)
            self.amplitude_view.image = img_tk

    def _handle_video_end(self) -> None:
        self.video_playing   = False
        self.preview_active  = False
        self.is_video_preview = False
        self.stop_compensation()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _play_video_preview(self) -> None:
        if not getattr(self, "video_playing", False):
            return

        ok, frm = self.cap.read()
        if not ok:
            self._handle_video_end()
            return

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gray.astype(np.float32))))

        # Apply logarithmic scaling based on ft_mode_var
        if hasattr(self, 'ft_mode_var') and self.ft_mode_var.get() == "With logarithmic scale":
            ft_log = (np.log1p(np.abs(ft)) / np.log1p(np.abs(ft)).max() * 255).astype(np.uint8)
        else:
            ft_log = (np.abs(ft) / np.abs(ft).max() * 255).astype(np.uint8)

        # Always update array
        self.current_holo_array = gray
        self.current_ft_array = ft_log

        # Prepare unfiltered FT data for _refresh_ft_display
        self.current_ft_unfiltered_array = ft_log
        self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
            Image.fromarray(ft_log), self.viewbox_width, self.viewbox_height
        )

        holo_tk = self._preserve_aspect_ratio(Image.fromarray(gray),
                                              self.viewbox_width, self.viewbox_height)

        self.hologram_frames = [holo_tk]
        self.ft_frames = [self.current_ft_unfiltered_tk]
        self.multi_holo_arrays = [gray]
        self.multi_ft_arrays = [ft_log]
        self.current_left_index = 0

        # Always check the selection fo the radio button
        current_choice = self.holo_view_var.get()
        if current_choice == "Hologram":
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk
        elif current_choice == "Fourier Transform":
            # Use _refresh_ft_display to respect the filtered/unfiltered selection
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        self.after(40, self._play_video_preview)

    def _comp_worker_loop(self, source: str) -> None:
        while not self._stop_compensation.is_set():
            ok, frm = self.cap.read()
            if not ok:
                if source == "camera":
                    break
                else:
                    # notify GUI thread to perform clean-up
                    self._stop_compensation.set()
                    self.after(0, self._handle_video_end)
                    break

            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            data = self._compute_comp_arrays(gray,
                                             first=(not self.first_frame_done))
            self.first_frame_done = True

            try:
                self._comp_queue.put_nowait(data)
            except queue.Full:
                try: self._comp_queue.get_nowait()
                except queue.Empty: pass
                self._comp_queue.put_nowait(data)

    def _compute_comp_arrays(self, gray: np.ndarray, *, first: bool) -> dict:
        """
        Return a dict with uint8 arrays:
        {'holo', 'ft', 'amp', 'phase'}
        """
        if first:
            h, w = gray.shape
            ftype = self.selected_filter_type
            ft_filt, fy, fx = self.spatialFilteringCF(
                gray, h, w, filter_type=ftype)
            self.fx, self.fy = fx[0], fy[0]
        else:
            h, w = gray.shape
            ft_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gray)))
            radius = 0.08 * min(h, w)

            ftype = self.selected_filter_type
            if ftype == "Rectangular":
                mask = self.rectangularMask(h, w, radius, self.fy, self.fx)
            else:
                # Default to circular
                yy, xx = np.ogrid[:h, :w]
                mask = ((yy - self.fy) ** 2 + (xx - self.fx) ** 2) <= radius ** 2

            ft_filt = ft_raw * mask

        # reconstruction  (same maths as before, no ImageTk)
        M, N = gray.shape[1], gray.shape[0]
        theta_x = np.arcsin((M/2 - self.fx) * self.lambda_um / (M * self.dx_um))
        theta_y = np.arcsin((N/2 - self.fy) * self.lambda_um / (N * self.dy_um))
        if not hasattr(self, "m_mesh"):
            Y, X = np.meshgrid(np.arange(N) - N/2,
                               np.arange(M) - M/2, indexing="ij")
            self.m_mesh, self.n_mesh = X, Y
        carrier = np.exp(1j * self.k * (
                   np.sin(theta_x) * self.m_mesh * self.dx_um +
                   np.sin(theta_y) * self.n_mesh * self.dy_um))

        field = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft_filt))) * carrier
        amp_u8 = cv2.normalize(np.abs(field), None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        phase_u8 = (((np.angle(field) + np.pi)/(2*np.pi))*255).astype(np.uint8)

        ft_log = (np.log1p(np.abs(ft_filt)) /
                  np.log1p(np.abs(ft_filt)).max() * 255).astype(np.uint8)
        holo_u8 = gray.copy()

        return {"holo": holo_u8, "ft": ft_log,
                "amp": amp_u8,  "phase": phase_u8}

    def stop_compensation(self) -> None:
        """Request the background worker to finish and clear queue."""
        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()
        if hasattr(self, "_comp_queue"):
            while not self._comp_queue.empty():
                try: self._comp_queue.get_nowait()
                except queue.Empty: break

    def _configure_ffmpeg_single_thread(self) -> None:
        """
        Prevent libavcodec’s multi-thread decoder from crashing Python
        with ‘Assertion fctx->async_lock failed …pthread_frame.c:173’.
        Must be called *before* the first VideoCapture().
        """
        # OpenCV honours this env-var since 4.8.0
        if "OPENCV_VIDEOIO_FFMPEG_DECODER_N_THREADS" not in os.environ:
            os.environ["OPENCV_VIDEOIO_FFMPEG_DECODER_N_THREADS"] = "1"

    def _poll_comp_queue(self) -> None:
        """
        Pull one processed packet from the worker queue and refresh the UI.
        This is the ONLY place we append frames to the recording buffers
        when compensation runs in the background thread.

        Expected packet layout: dict with uint8 arrays:
          {"holo": HxW, "ft": HxW, "amp": HxW, "phase": HxW}
        """
        # Try to get data from the queue; if empty, reschedule and exit
        try:
            data = self._comp_queue.get_nowait()
        except queue.Empty:
            if not self._stop_compensation.is_set():
                self.after(10, self._poll_comp_queue)
            return

        # Cache numpy arrays (used for zoom, save, etc.)
        self.current_holo_array = data["holo"]
        self.current_ft_array = data["ft"]  # default for zoom/save
        self.current_amplitude_array = data["amp"]
        self.current_phase_array = data["phase"]

        # Build small Tk images for the viewers (lightweight thumbnails)
        holo_tk = self._preserve_aspect_ratio(
            Image.fromarray(data["holo"]),
            self.viewbox_width, self.viewbox_height
        )
        ft_tk = self._preserve_aspect_ratio(
            Image.fromarray(data["ft"]),
            self.viewbox_width, self.viewbox_height
        )
        amp_tk = self._preserve_aspect_ratio_right(
            Image.fromarray(data["amp"])
        )
        pha_tk = self._preserve_aspect_ratio_right(
            Image.fromarray(data["phase"])
        )

        # Keep lists in sync so navigation & "Save ..." always find data
        self.hologram_frames.append(holo_tk)
        self.ft_frames.append(ft_tk)
        self.multi_holo_arrays.append(data["holo"])
        self.multi_ft_arrays.append(data["ft"])

        self.amplitude_frames.append(amp_tk)
        self.phase_frames.append(pha_tk)
        self.amplitude_arrays.append(data["amp"])
        self.phase_arrays.append(data["phase"])

        # Prepare filtered/unfiltered FT for left viewer toggle
        self.current_ft_filtered_tk = ft_tk
        self.current_ft_filtered_array = data["ft"]

        # "unfiltered" = recompute from the current hologram (display-only)
        ft_unfiltered = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(data["holo"].astype(np.float32)))
        )
        if hasattr(self, 'ft_mode_var') and self.ft_mode_var.get() == "With logarithmic scale":
            ft_unf_disp = (np.log1p(np.abs(ft_unfiltered)) /
                           np.log1p(np.abs(ft_unfiltered)).max() * 255).astype(np.uint8)
        else:
            abs_ft = np.abs(ft_unfiltered)
            abs_max = abs_ft.max() if abs_ft.max() > 0 else 1.0
            ft_unf_disp = (abs_ft / abs_max * 255).astype(np.uint8)

        self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
            Image.fromarray(ft_unf_disp),
            self.viewbox_width, self.viewbox_height
        )
        self.current_ft_unfiltered_array = ft_unf_disp

        # Update LEFT viewer (respect Hologram/FT + filtered/unfiltered choice)
        left_choice = self.holo_view_var.get()
        if left_choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk
        elif left_choice == "Fourier Transform":
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        # Update RIGHT viewer (Amplitude / Phase)
        right_choice = self.recon_view_var.get()
        if right_choice == "Amplitude Reconstruction ":
            self.processed_label.configure(image=amp_tk)
            self.processed_label.image = amp_tk
        else:
            self.processed_label.configure(image=pha_tk)
            self.processed_label.image = pha_tk

        # While recording, append the correct buffer HERE (worker path)
        if getattr(self, "is_recording", False):
            if self.target_to_record == "Amplitude":
                self.buff_amp.append(data["amp"].copy())
            elif self.target_to_record == "Phase":
                self.buff_phase.append(data["phase"].copy())
            elif self.target_to_record == "Hologram":
                self.buff_holo.append(data["holo"].copy())

        # Schedule next poll if the worker is still active
        if not self._stop_compensation.is_set():
            self.after(10, self._poll_comp_queue)

    def start_compensation(self) -> None:
        """Launches SHPC compensation in a background thread."""
        # parameters
        lam, dx, dy = self._get_pc_parameter_values()
        self.lambda_um, self.dx_um, self.dy_um = lam, dx, dy
        self.k = 2 * math.pi / self.lambda_um
        self.selected_filter_type = self.spatial_filter_var_pc.get().strip()

        # stop any previous compensation first
        self.stop_compensation()

        if getattr(self, "is_video_preview", False):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            src = "video"
        else:
            if not self._ensure_camera():
                tk.messagebox.showerror("Camera", "Camera unavailable.")
                return
            src = "camera"

        # reset flags
        self._stop_compensation.clear()
        self.first_frame_done = False
        self.is_playing = True
        self.play_button.configure(text="⏸ Pause")

        #if src == "camera":
        #    self._update_realtime()
        #    print("inside real time")

        self._comp_thread = threading.Thread(
            target=self._comp_worker_loop, args=(src,), daemon=True
        )
        self._comp_thread.start()
        self.after(20, self._poll_comp_queue)

    def _play_video_frame(self) -> None:
        if not getattr(self, "video_playing", False):
            return

        ok, frm = self.cap.read()
        if not ok:
            self._handle_video_end()
            return

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        print("DEBUG: INSIDE VIDEO FRAME")
        if not self.first_frame_done:
            self._process_first_frame(gray)
        else:
            self._process_next_frame(gray)

        self.after(40, self._play_video_frame)

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
        opts = ["With logarithmic scale", "Without logarithmic scale"]
        for opt in opts:
            menu.add_radiobutton(
                label=opt, value=opt,
                variable=self.ft_mode_var,
                command=self._on_ft_mode_changed
            )
        menu.tk_popup(self.ft_mode_button.winfo_rootx(),
                      self.ft_mode_button.winfo_rooty() + self.ft_mode_button.winfo_height())

    def update_left_view_video(self):
        """Update left view specifically for video mode"""
        choice = self.holo_view_var.get()

        # Check if we have current hologram data
        if hasattr(self, 'current_holo_array') and self.current_holo_array is not None:

            if choice == "Hologram":
                # Show hologram
                self.captured_title_label.configure(text="Hologram")
                holo_tk = self._preserve_aspect_ratio(
                    Image.fromarray(self.current_holo_array),
                    self.viewbox_width, self.viewbox_height
                )
                self.captured_label.configure(image=holo_tk)
                self.captured_label.image = holo_tk

            elif choice == "Fourier Transform":
                # Use the existing _refresh_ft_display logic for consistency
                # But first ensure we have the basic FT data for video
                if not hasattr(self, 'current_ft_array') or self.current_ft_array is None:
                    # Calculate unfiltered FFT from current hologram
                    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.current_holo_array.astype(np.float32))))

                    # Apply logarithmic scaling based on ft_mode_var
                    if hasattr(self, 'ft_mode_var') and self.ft_mode_var.get() == "With logarithmic scale":
                        self.current_ft_array = (np.log1p(np.abs(ft)) / np.log1p(np.abs(ft)).max() * 255).astype(
                            np.uint8)
                    else:
                        self.current_ft_array = (np.abs(ft) / np.abs(ft).max() * 255).astype(np.uint8)

                    # Create the PhotoImage for unfiltered FT
                    self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
                        Image.fromarray(self.current_ft_array),
                        self.viewbox_width, self.viewbox_height
                    )
                    self.current_ft_unfiltered_array = self.current_ft_array

                # Now use the existing _refresh_ft_display function
                self.captured_title_label.configure(text="Fourier Transform")
                self._refresh_ft_display()

        else:
            # Fallback if no current arrays are available
            if choice == "Hologram":
                self.captured_title_label.configure(text="Hologram")
                self.captured_label.configure(image=self.img_hologram)
            else:
                self.captured_title_label.configure(text="Fourier Transform")
                self.captured_label.configure(image=self.img_ft)

    def _on_ft_mode_changed(self):
        if self.holo_view_var.get() == "Fourier Transform":
            if hasattr(self, "comp_source"):
                if self.comp_source == "camera":
                    self.update_left_view_camera()
                elif self.comp_source == "video":
                    self.update_left_view_video()
                else:
                    self.update_left_view()
            else:
                self.update_left_view()

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

    def start_preview_stream(self) -> None:
        """Begin grabbing frames and showing Hologram + FT only."""
        if not self._ensure_camera():
            tk.messagebox.showerror("Camera error", "No active camera was found.")
            return
        if self.preview_active:
            return
        self.preview_active = True
        self._update_preview()

    def _update_preview(self) -> None:
        """Internal loop: grab → show Hologram & FT → (optionally) save."""
        if not self.preview_active:
            return

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.after(30, self._update_preview)
            return

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if getattr(self, "is_recording", False) and self.target_to_record == "Hologram":
            self.buff_holo.append(gray.copy())

        self.last_preview_gray = gray

        # Fourier transform (for display only)
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gray.astype(np.float32))))
        self.last_preview_ft = ft
        ft_log = (np.log1p(np.abs(ft)) /
                  np.log1p(np.abs(ft)).max() * 255).astype(np.uint8)

        # Tk thumbnails
        holo_tk = self._preserve_aspect_ratio(
            Image.fromarray(gray), self.viewbox_width, self.viewbox_height)
        ft_tk = self._preserve_aspect_ratio(
            Image.fromarray(ft_log), self.viewbox_width, self.viewbox_height)

        self.hologram_frames = [holo_tk]
        self.ft_frames = [ft_tk]
        self.multi_holo_arrays = [gray]
        self.multi_ft_arrays = [ft_log]
        self.current_holo_array = gray
        self.current_ft_array = ft_log
        self.current_left_index = 0

        # Refresh left viewer
        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=holo_tk)
        else:
            self.captured_label.configure(image=ft_tk)

        # Save to disk if sequence capture is active
        if self.sequence_recording:
            self._save_sequence_frame(gray)

        # Schedule next frame
        self.after(20, self._update_preview)

    '''
    def stop_preview_stream(self) -> None:
        """Stops the live hologram/FT preview."""
        self.preview_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[Preview] Stream stopped.")

    def stop_realtime_stream(self):
        if not self.realtime_active:
            print("[Realtime] No active stream to stop.")
        return

        self.realtime_active = False
        self.first_frame_done = False
        self.cap.release()
        self.cap = None
        print("[Realtime] Realtime stream stopped.")
    '''

    '''
    def start_sequence_recording(self) -> None:
        """Ask for a parent folder and start saving incoming frames."""
        if self.sequence_recording:
            tk.messagebox.showinfo("Sequence", "Sequence already in progress.")
            return

        root_dir = filedialog.askdirectory(title="Choose folder for sequence")
        if not root_dir:
            return

        for sub in ("hologram", "amplitude", "phase"):
            os.makedirs(os.path.join(root_dir, sub), exist_ok=True)

        self.seq_save_root = root_dir
        self.seq_frame_counter = 0
        self.sequence_recording = True
        tk.messagebox.showinfo("Sequence", "Recording started.")
    '''

    '''
    def stop_sequence_recording(self) -> None:
        """Stop writing new frames to disk."""
        if not self.sequence_recording:
            return
        self.sequence_recording = False
        tk.messagebox.showinfo("Sequence", "Recording stopped.")
    '''

    def _save_sequence_frame(
        self,
        holo_arr: np.ndarray,
        amp_arr:  np.ndarray | None = None,
        phase_arr:np.ndarray | None = None
    ) -> None:
        """Internal: write the given arrays as PNGs inside their folders."""
        if not self.sequence_recording or not self.seq_save_root:
            return

        idx = self.seq_frame_counter
        self.seq_frame_counter += 1

        cv2.imwrite(
            os.path.join(self.seq_save_root, "hologram",
                         f"holo_{idx:06d}.png"), holo_arr)

        if amp_arr is not None:
            cv2.imwrite(
                os.path.join(self.seq_save_root, "amplitude",
                             f"amp_{idx:06d}.png"), amp_arr)

        if phase_arr is not None:
            cv2.imwrite(
                os.path.join(self.seq_save_root, "phase",
                             f"phase_{idx:06d}.png"), phase_arr)

    def _ensure_camera(self) -> bool:
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            return True

        return getattr(self, "cap", None) is not None and self.cap.isOpened()

    def _find_available_cameras(self, max_index: int = 10) -> list[int]:
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()
            if ok:
                available.append(idx)
        return available

    # Pick the first index whose “description” or resolution screams TIS
    def _pick_preferred_camera(self, indices: list[int]) -> int | None:
        def _descr(idx: int) -> str:
            tmp = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            desc = ""
            try:
                # OpenCV ≥ 4.6 exposes the device string here (prop-ID 268).
                desc = str(tmp.get(cv2.CAP_PROP_DEVICE_DESCRIPTION))
            except Exception:
                pass
            finally:
                tmp.release()
            return desc.lower()

        # Keyword match (Imaging Source usually self-identifies)
        for idx in indices:
            d = _descr(idx)
            if any(kw in d for kw in self._PREFERRED_CAM_KEYWORDS):
                return idx

        # Otherwise grab the first *external* camera with “biggish” frames
        for idx in indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            ok, frm = cap.read()
            cap.release()
            if ok and frm.shape[1] >= self._FALLBACK_MIN_WIDTH and idx != 0:
                return idx

        # Nothing fancy? Fine, just give me the first that works
        return indices[0] if indices else None

    #  Initialise camera
    def _init_camera(self) -> cv2.VideoCapture | None:
        self.cap = None
        self.selected_camera_index = None
        self.realtime_active = False

        # Search for available devices
        avail = self._find_available_cameras()
        if not avail:
            self._show_camera_error_once("No camera detected – realtime disabled.")
            return None

        # Choose preferred camera
        preferred = self._pick_preferred_camera(avail)
        if preferred is None:
            self._show_camera_error_once("No suitable camera found – realtime disabled.")
            return None

        # Try to open the camera with DirectShow and then MSMF if it fails
        cap = cv2.VideoCapture(preferred, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(preferred, cv2.CAP_MSMF)
        if not cap.isOpened():
            print(f"[Camera] Could not open camera index {preferred}.")
            self._show_camera_error_once(f"Could not open camera index {preferred}.")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)

        # Test if the camera delivers a frame
        ok, first = cap.read()
        if not ok:
            cap.release()
            print("[Camera] Camera opened but delivers no frames.")
            self._show_camera_error_once("Camera opened but delivers no frames.")
            return None

        # Everything OK, save camera
        self.cap = cap
        self.selected_camera_index = preferred
        self.first_frame_done = False
        self.video_buffer_rec = []
        self.video_buffer_raw = []
        self.start_time_fps = time.time()
        self.frame_counter_fps = 0

        if not getattr(self, "_camera_success_shown", False):
            messagebox.showinfo(
                "Information",
                f"Using device index {preferred} – resolution {first.shape[1]}×{first.shape[0]}"
            )
            self._camera_success_shown = True

        return self.cap

    def start_preview_stream(self) -> None:
        if not self._ensure_camera():
            messagebox.showerror("Camera error", "No active camera was found.")
            return

        self.preview_active = True
        self._update_preview()

    # Ensure_camera para debugging
    def _ensure_camera(self) -> bool:
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            return True

        return False

    def _show_camera_error_once(self, message: str) -> None:
        """Shows a single message box for camera errors."""
        try:
            if not getattr(self, "_camera_error_shown", False):
                messagebox.showinfo("Camera Info", message)
                self._camera_error_shown = True
        except Exception:
            print(f"[Camera] {message}")

    def _make_ctk_image(
        self,
        pil_img: Image.Image,
        max_size: tuple[int, int] | None = None
    ) -> ctk.CTkImage:
        """
        Returns a `customtkinter.CTkImage` scaled down (never up) so that it
        fits inside *max_size* (width, height) while keeping aspect-ratio.
        If *max_size* is None the PIL image is converted unchanged.
        """
        if max_size is not None:
            max_w, max_h = max_size
            w, h = pil_img.size
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.LANCZOS
                )
        return ctk.CTkImage(light_image=pil_img, size=pil_img.size)

    def _preserve_aspect_ratio(self, pil_image: Image.Image, max_width: int, max_height: int) -> ImageTk.PhotoImage:
        """
        Resize 'pil_image' to fit within (max_width x max_height),
        preserving aspect ratio. The image is scaled down (never up).
        Returns a PhotoImage that fits the space with possible black borders.
        """
        original_w, original_h = pil_image.size

        scale = min(max_width / original_w, max_height / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas_mode = pil_image.mode if pil_image.mode in ["RGB", "L"] else "RGB"
        canvas = Image.new(canvas_mode, (max_width, max_height), color=0 if canvas_mode == "L" else (0, 0, 0))

        offset_x = (max_width - new_w) // 2
        offset_y = (max_height - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return ImageTk.PhotoImage(canvas)

    def _preserve_aspect_ratio_right(self, pil_image: Image.Image) -> ImageTk.PhotoImage:
        max_width, max_height = self.viewbox_width, self.viewbox_height
        original_w, original_h = pil_image.size

        #  If smaller or equal, no upscaling (unless you want to allow it).
        if original_w <= max_width and original_h <= max_height:
            resized = pil_image
        else:
            # We shrink to keep the aspect ratio correct
            ratio_w = max_width / float(original_w)
            ratio_h = max_height / float(original_h)
            scale_factor = min(ratio_w, ratio_h)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(resized)

    def previous_hologram_view(self):
        # Store current UI filter settings for the current hologram/FT index
        if self.holo_view_var.get() in ["Hologram", "Fourier Transform"]:
            self._store_current_ui_filter_state(dimension=0, index=self.current_left_index)

        # Proceed with the usual logic to change index
        if not hasattr(self, 'hologram_frames') or not self.hologram_frames:
            print("No multiple holograms to navigate.")
            return

        # Decrement index
        self.current_left_index = (self.current_left_index - 1) % len(self.hologram_frames)

        # Update the displayed image
        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Hologram")
        else:
            self.captured_label.configure(image=self.ft_frames[self.current_left_index])
            self.current_ft_array = self.multi_ft_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Fourier Transform")

        # Load the filter settings for the newly selected index
        if self.holo_view_var.get() in ["Hologram", "Fourier Transform"]:
            self._load_ui_from_filter_state(dimension=0, index=self.current_left_index)

    def next_hologram_view(self):
        # Store current UI filter settings for the current hologram/FT index
        if self.holo_view_var.get() in ["Hologram", "Fourier Transform"]:
            self._store_current_ui_filter_state(dimension=0, index=self.current_left_index)

        # Proceed with the usual logic to change index
        if not hasattr(self, 'hologram_frames') or not self.hologram_frames:
            print("No multiple holograms to navigate.")
            return

        # Increment index
        self.current_left_index = (self.current_left_index + 1) % len(self.hologram_frames)

        # Update the displayed image
        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Hologram")
        else:
            self.captured_label.configure(image=self.ft_frames[self.current_left_index])
            self.current_ft_array = self.multi_ft_arrays[self.current_left_index]
            self.captured_title_label.configure(text="Fourier Transform")

        # Load the filter settings for the newly selected index
        if self.holo_view_var.get() in ["Hologram", "Fourier Transform"]:
            self._load_ui_from_filter_state(dimension=0, index=self.current_left_index)

    def update_left_view(self, *, reload_ui: bool = True):
        """Handle left view changes for both camera and video modes"""

        # If we are in video mode and have frames available
        if hasattr(self, 'source_mode') and self.source_mode == "video":
            self.update_left_view_video()

        # If we are in camera mode, view updates automatically in _update_preview
        elif hasattr(self, 'source_mode') and self.source_mode == "camera":
            # Do nothing, _update_preview handles this automatically
            pass

        else:
            # Default mode (static images) - original logic
            self.update_left_view_static(reload_ui=reload_ui)

    def update_left_view_static(self, *, reload_ui: bool = True):
        """Update left view for static images (original logic)"""
        choice = self.holo_view_var.get()

        # Fallback when no images are loaded
        if not hasattr(self, 'hologram_frames') or len(self.hologram_frames) == 0:
            if choice == "Hologram":
                self.captured_title_label.configure(text="Hologram")
                self.captured_label.configure(image=self.img_hologram)
                self.current_holo_array = self.arr_hologram
            else:
                self.captured_title_label.configure(text="Fourier Transform")
                self.captured_label.configure(image=self.img_ft)
                self.current_ft_array = self.arr_ft
            return

        # Normal navigation
        if choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.captured_label.image = self.hologram_frames[self.current_left_index]
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
        else:
            self.captured_title_label.configure(text="Fourier Transform")
            self.captured_label.configure(image=self.ft_frames[self.current_left_index])
            self.captured_label.image = self.ft_frames[self.current_left_index]
            self.current_ft_array = self.multi_ft_arrays[self.current_left_index]

        if choice == "Fourier Transform":
            self._refresh_ft_display()

    def update_right_view(self, *, reload_ui: bool = True):
        """Refresh the right viewer (Phase / Amplitude).  See note on reload_ui
        in update_left_view().
        """
        choice = self.recon_view_var.get()

        if choice == "Phase Reconstruction":
            idx = getattr(self, 'current_phase_index', 0)
            frame_list = getattr(self, 'phase_frames', [])
            array_list = getattr(self, 'phase_arrays', [])
            if idx < len(frame_list):
                self.processed_label.configure(image=frame_list[idx])
                self.processed_label.image = frame_list[idx]
                self.current_phase_array = array_list[idx]
        else:
            idx = getattr(self, 'current_amp_index', 0)
            frame_list = getattr(self, 'amplitude_frames', [])
            array_list = getattr(self, 'amplitude_arrays', [])
            if idx < len(frame_list):
                self.processed_label.configure(image=frame_list[idx])
                self.processed_label.image = frame_list[idx]
                self.current_amplitude_array = array_list[idx]

    '''
    def show_options(self):
        if hasattr(self, 'Options_menu') and self.Options_menu.winfo_ismapped():
            self.Options_menu.grid_forget()
            return

        self.Options_menu = ctk.CTkOptionMenu(
        self.buttons_frame,
        values=["QPI", "Filters"],
        command=self.choose_option,
        width=270
        )
        self.Options_menu.grid(row=0, column=1, padx=4, pady=5, sticky='w')
    '''

    def choose_option(self, selected_option):
        if selected_option == "QPI":
            self.change_menu_to('QPI')
        elif selected_option == "Filters":
            self.change_menu_to('filters')

    def show_reconstruction_arrows(self):
        self.left_arrow_recon.grid(row=4, column=0, padx=(30, 5), pady=5, sticky='w')
        self.right_arrow_recon.grid(row=4, column=1, padx=(5, 30), pady=5, sticky='e')

    def _get_current_array(self, target: str) -> np.ndarray | None:
        """Return the ndarray that corresponds to *target*."""
        if target == "Hologram": return getattr(self, "current_holo_array", None)
        elif target == "Fourier Transform": return getattr(self, "current_ft_array", None)
        elif target == "Amplitude": return getattr(self, "current_amplitude_array", None)
        elif target == "Phase": return getattr(self, "current_phase_array",     None)
        return None

    def _start_live_zoom(self,
                         target_type: str,
                         roi: tuple[int, int, int, int],
                         scale: int = 2,
                         refresh_ms: int = 200) -> None:

        # Kill any previous live‑zoom
        if getattr(self, "live_zoom_active", False):
            self.live_zoom_active = False
            if getattr(self, "live_zoom_window", None):
                self.live_zoom_window.destroy()

        # Cache parameters
        self.live_zoom_target = target_type
        self.live_zoom_roi = roi
        self.live_zoom_scale = scale
        self.zoom_refresh_ms = refresh_ms

        # Build the window
        self.live_zoom_window   = tk.Toplevel(self)
        self.live_zoom_window.title(f"Live Zoom – {target_type}")
        self.live_zoom_label    = tk.Label(self.live_zoom_window)
        self.live_zoom_label.pack()
        self.live_zoom_active   = True

        def _on_close() -> None:
            self.live_zoom_active = False
            self.live_zoom_window.destroy()
            self.live_zoom_window = None
        self.live_zoom_window.protocol("WM_DELETE_WINDOW", _on_close)

        self._update_live_zoom()

    def _update_live_zoom(self) -> None:
        """Internal: refresh the cropped view and re‑schedule itself."""
        if not getattr(self, "live_zoom_active", False):
            return

        arr = self._get_current_array(self.live_zoom_target)
        if arr is None:
            self.after(self.zoom_refresh_ms, self._update_live_zoom)
            return

        x1, y1, x2, y2 = self.live_zoom_roi
        h, w = arr.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            crop = arr
        else:
            crop = arr[y1:y2, x1:x2]

        pil = Image.fromarray(crop)
        pil = pil.resize((crop.shape[1]*self.live_zoom_scale,
                           crop.shape[0]*self.live_zoom_scale),
                          Image.Resampling.LANCZOS)
        tkim = ImageTk.PhotoImage(pil)
        self.live_zoom_label.configure(image=tkim)
        self.live_zoom_label.image = tkim

        self.after(self.zoom_refresh_ms, self._update_live_zoom)

    def _open_zoom_view(self, target_type: str) -> None:
        if getattr(self, "_zoom_win", None):
            try:
                self._zoom_win.destroy()
            except tk.TclError:
                pass

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = int(sw * 0.7), int(sh * 0.9)
        px, py = (sw - ww) // 2, (sh - wh) // 2

        self._zoom_win = tk.Toplevel(self)
        self._zoom_win.title(f"Zoom – {target_type}")
        self._zoom_win.geometry(f"{ww}x{wh}+{px}+{py}")
        self._zoom_win.minsize(400, 300)

        self._zoom_canvas = tk.Canvas(self._zoom_win, highlightthickness=0, bd=0)
        self._zoom_canvas.pack(fill="both", expand=True)

        self._zoom_target = target_type
        self._zoom_roi = None
        self._zoom_start_pt = None
        self._zoom_rect_id = None
        self._zoom_img_id = None
        self._zoom_live = True

        # Helpers
        def _canvas_to_img(xc: int, yc: int) -> tuple[int, int]:
            """
            Converts canvas coordinates (pixels in the window)
            to **original image** coordinates, taking into account
            the ROI already applied (if any).
            """
            arr = self._get_current_array(self._zoom_target)
            if arr is None:
                return 0, 0
            full_h, full_w = arr.shape[:2]

            if self._zoom_roi is None:
                base_x0, base_y0, base_x1, base_y1 = 0, 0, full_w, full_h
            else:
                base_x0, base_y0, base_x1, base_y1 = self._zoom_roi

            view_w = base_x1 - base_x0
            view_h = base_y1 - base_y0
            win_w = max(self._zoom_canvas.winfo_width(),  1)
            win_h = max(self._zoom_canvas.winfo_height(), 1)
            scale_x = view_w / win_w
            scale_y = view_h / win_h

            ix = base_x0 + int(xc * scale_x)
            iy = base_y0 + int(yc * scale_y)
            return ix, iy

        # Bindings
        def _on_press(event):
            self._zoom_start_pt = (event.x, event.y)
            if self._zoom_rect_id:
                self._zoom_canvas.delete(self._zoom_rect_id)
                self._zoom_rect_id = None

        def _on_drag(event):
            if not self._zoom_start_pt:
                return
            if self._zoom_rect_id:
                self._zoom_canvas.coords(self._zoom_rect_id,
                                         self._zoom_start_pt[0], self._zoom_start_pt[1],
                                         event.x, event.y)
            else:
                self._zoom_rect_id = self._zoom_canvas.create_rectangle(
                    self._zoom_start_pt[0], self._zoom_start_pt[1],
                    event.x, event.y, outline="red", width=2)

        def _on_release(event):
            if not self._zoom_start_pt:
                return
            x0c, y0c = self._zoom_start_pt
            x1c, y1c = event.x, event.y
            self._zoom_start_pt = None

            if abs(x1c - x0c) < 4 or abs(y1c - y0c) < 4:
                if self._zoom_rect_id:
                    self._zoom_canvas.delete(self._zoom_rect_id)
                    self._zoom_rect_id = None
                return

            ix0, iy0 = _canvas_to_img(min(x0c, x1c), min(y0c, y1c))
            ix1, iy1 = _canvas_to_img(max(x0c, x1c), max(y0c, y1c))

            if ix1 - ix0 >= 2 and iy1 - iy0 >= 2:
                self._zoom_roi = (ix0, iy0, ix1, iy1)

            if self._zoom_rect_id:
                self._zoom_canvas.delete(self._zoom_rect_id)
                self._zoom_rect_id = None

        def _on_clear_roi(event):
            self._zoom_roi = None

        self._zoom_canvas.bind("<ButtonPress-1>",   _on_press)
        self._zoom_canvas.bind("<B1-Motion>",       _on_drag)
        self._zoom_canvas.bind("<ButtonRelease-1>", _on_release)
        self._zoom_canvas.bind("<ButtonPress-3>",   _on_clear_roi)

        def _on_close():
            self._zoom_live = False
            self._zoom_win.destroy()
            self._zoom_win = None
        self._zoom_win.protocol("WM_DELETE_WINDOW", _on_close)

        self._refresh_zoom_view()

    def _refresh_zoom_view(self, refresh_ms: int = 100) -> None:
        if not getattr(self, "_zoom_live", False):
            return

        arr = self._get_current_array(self._zoom_target)
        if arr is None:
            self.after(refresh_ms, self._refresh_zoom_view)
            return

        if self._zoom_roi:
            x0, y0, x1, y1 = self._zoom_roi
            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            arr_view = arr[y0:y1, x0:x1]
        else:
            arr_view = arr

        win_w = max(self._zoom_canvas.winfo_width(),  1)
        win_h = max(self._zoom_canvas.winfo_height(), 1)
        pil = Image.fromarray(arr_view).resize((win_w, win_h),
                                                 Image.Resampling.NEAREST)
        tkim = ImageTk.PhotoImage(pil)

        if self._zoom_img_id is None:
            self._zoom_img_id = self._zoom_canvas.create_image(0, 0, anchor="nw",
                                                               image=tkim)
        else:
            self._zoom_canvas.itemconfig(self._zoom_img_id, image=tkim)

        self._zoom_canvas.image = tkim   # evita GC
        self.after(refresh_ms, self._refresh_zoom_view)

    def _on_zoom_wheel(self, event):
        delta = event.delta if hasattr(event, "delta") else (120 if event.num == 4 else -120)
        step  = 1.1 if delta > 0 else 0.9
        self._zoom_scale = max(self._zoom_min_scale,
                               min(self._zoom_max_scale, self._zoom_scale * step))

    # Press → start pan
    def _on_zoom_press(self, event):
        self._zoom_pan_start = (event.x, event.y)

    # Drag → pan
    def _on_zoom_drag(self, event):
        if self._zoom_pan_start is None:
            return
        dx = event.x - self._zoom_pan_start[0]
        dy = event.y - self._zoom_pan_start[1]
        self._zoom_off_x = max(0, self._zoom_off_x - dx)
        self._zoom_off_y = max(0, self._zoom_off_y - dy)
        self._zoom_pan_start = (event.x, event.y)

    def zoom_holo_view(self):
        """Called by the 🔍 button in the left viewer."""
        choice = self.holo_view_var.get()
        self._open_zoom_view(choice)

    def zoom_recon_view(self):
        """Called by the 🔍 button in the right viewer."""
        choice = self.recon_view_var.get()
        target = "Phase" if choice.startswith("Phase") else "Amplitude"
        self._open_zoom_view(target)

    def previous_recon_view(self):
        """
        Same as before, calling _update_distance_label() at the end.
        """
        current_mode = self.recon_view_var.get()

        if current_mode == "Amplitude Reconstruction ":
            if hasattr(self, 'current_amp_index'):
                self._store_current_ui_filter_state(dimension=1, index=self.current_amp_index)
        else:
            if hasattr(self, 'current_phase_index'):
                self._store_current_ui_filter_state(dimension=2, index=self.current_phase_index)

        if current_mode == "Phase Reconstruction ":
            if not hasattr(self, 'phase_frames') or len(self.phase_frames) == 0:
                print("No phase frames to show.")
                return
            self.current_phase_index = (self.current_phase_index - 1) % len(self.phase_frames)
            self.processed_label.configure(image=self.phase_frames[self.current_phase_index])
            self._load_ui_from_filter_state(dimension=2, index=self.current_phase_index)
        else:
            if not hasattr(self, 'amplitude_frames') or len(self.amplitude_frames) == 0:
                print("No amplitude frames to show.")
                return
            self.current_amp_index = (self.current_amp_index - 1) % len(self.amplitude_frames)
            self.processed_label.configure(image=self.amplitude_frames[self.current_amp_index])
            self._load_ui_from_filter_state(dimension=1, index=self.current_amp_index)

        self._update_distance_label()

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

        # If you prefer to hide the old distance_label_recon entirely:
        if hasattr(self, 'distance_label_recon'):
            self.distance_label_recon.configure(text="")
            self.distance_label_recon.grid_remove()

    def _convert_distance_for_display(self, dist_um):
        unit = self.unit_var.get()  # e.g. "mm", "µm", "nm", etc.
        if unit == "µm":
            val = dist_um
        elif unit == "nm":
            val = dist_um * 1000.0
        elif unit == "mm":
            val = dist_um / 1000.0
        elif unit == "cm":
            val = dist_um / 10000.0
        elif unit == "m":
            val = dist_um / 1e6
        elif unit == "in":
            val = dist_um / 25400.0
        else:
            # fallback
            unit = "µm"
            val = dist_um

        return f"{val:.2f} {unit}"

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
        """
        Decides which set of images to store.
        "Save FT" =>store the Fourier transforms.
        "Save Phase" => store phase image.
        "Save Amplitude" => store amplitude images.
        """
        # Hide the dropdown
        if hasattr(self, "save_options_menu") and self.save_options_menu.winfo_exists():
            self.save_menu.grid_forget()

        if option == "Save FT":
            self.save_hologram_images()
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

    # Save FT
    def save_hologram_images(self):
        """
         Saves the currently displayed Fourier Transform image as a file.
         """
        if not hasattr(self, "current_ft_array"):
            messagebox.showerror("No image", "No Fourier Transform image available to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Fourier Transform",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("BMP files", "*.bmp"),
                       ("JPEG files", "*.jpg"),
                       ("All files", "*.*")]
        )
        if not save_path:
            return

        arr_norm = self._normalize_for_save(self.current_ft_array)
        img = Image.fromarray(arr_norm)
        img.save(save_path)

    # Save Phase Images
    def save_phase_images(self):
        if not hasattr(self, "current_phase_array"):
            messagebox.showerror("No image", "No phase image available to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Phase",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp"),
                       ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not save_path:
            return

        arr_norm = self._normalize_for_save(self.current_phase_array)
        img = Image.fromarray(arr_norm)
        img.save(save_path)

    # Save Amplitude Images
    def save_amplitude_images(self):
        if not hasattr(self, "current_amplitude_array"):
            messagebox.showerror("No image", "No amplitude image available to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Amplitude",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp"),
                       ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not save_path:
            return

        arr_norm = self._normalize_for_save(self.current_amplitude_array)
        img = Image.fromarray(arr_norm)
        img.save(save_path)

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

    def _get_pc_parameter_values(self):
        try:
            lam_um = self.get_value_in_micrometers(
                self.wave_label_pc_entry.get(), self.wavelength_unit)
        except Exception:
            lam_um = 0.0

        try:
            pitch_x = self.get_value_in_micrometers(
                self.pitchx_label_pc_entry.get(), self.pitch_x_unit)
        except Exception:
            pitch_x = 0.0

        try:
            pitch_y = self.get_value_in_micrometers(
                self.pitchy_label_pc_entry.get(), self.pitch_y_unit)
        except Exception:
            pitch_y = 0.0

            # Verificación
        if lam_um == 0.0 or pitch_x == 0.0 or pitch_y == 0.0:
            messagebox.showwarning(
                "Warning",
                "Reconstruction parameters (wavelength and pixel size) cannot be zero. Please verify them before proceeding."
            )
            return None, None, None

        return lam_um, pitch_x, pitch_y

    def _prefill_record_buffer(self) -> None:
        """
        Copies the frame(s) already displayed in the GUI into the
        corresponding recording buffer.  That way you can press
        Start ▸ Stop immediately and still get a video.
        """
        tgt = self.target_to_record

        if tgt == "Phase":
            # grab every phase frame available (or at least the current one)
            if self.phase_arrays:
                self.buff_phase.extend([frm.copy() for frm in self.phase_arrays])
            elif self.current_phase_array is not None:
                self.buff_phase.append(self.current_phase_array.copy())

        elif tgt == "Amplitude":
            if self.amplitude_arrays:
                self.buff_amp.extend([frm.copy() for frm in self.amplitude_arrays])
            elif self.current_amplitude_array is not None:
                self.buff_amp.append(self.current_amplitude_array.copy())

        else:  # "Hologram"
            if self.multi_holo_arrays:
                self.buff_holo.extend([frm.copy() for frm in self.multi_holo_arrays])
            elif self.current_holo_array is not None:
                self.buff_holo.append(self.current_holo_array.copy())

    def start_record(self):
        if not hasattr(self, "is_recording"):
            self.is_recording = False
            self.buff_phase = []
            self.buff_amp = []
            self.buff_holo = []

        if self.is_recording:
            return

        # Target to capture
        self.target_to_record = self.record_var.get()

        # Wipe previous run
        self.buff_phase.clear()
        self.buff_amp.clear()
        self.buff_holo.clear()

        # Pre-fill with the frames already on screen
        self._prefill_record_buffer()

        # Flag + UI feedback
        self.is_recording = True
        self.record_indicator.grid()
        tk.messagebox.showinfo(
            "Record",
            (f"Recording {self.target_to_record}. "
             "Press Stop whenever you’re ready.")
        )

    def stop_recording(self):
        if not getattr(self, "is_recording", False):
            return
        self.is_recording = False
        self.record_indicator.grid_remove()

        # Pick the correct buffer
        buf = (self.buff_phase if self.target_to_record == "Phase"
               else self.buff_amp if self.target_to_record == "Amplitude"
               else self.buff_holo)
        if not buf:
            tk.messagebox.showwarning("Record", "Nothing captured yet.")
            return

        # Ask user where to save
        path = filedialog.asksaveasfilename(
            title="Save recorded video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
        if not path:
            tk.messagebox.showinfo("Record", "Save cancelled.")
            return

        # Build VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.lower().endswith(".mp4")
                                          else "XVID"))
        h, w = buf[0].shape[:2]
        fps_to_use = getattr(self, "source_fps", 15.0)
        vw = cv2.VideoWriter(path, fourcc, fps_to_use, (w, h), isColor=True)

        for f in buf:
            # ensure 3-channel for codecs that insist on colour
            if f.ndim == 2:
                f_col = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            else:
                f_col = f
            vw.write(f_col)
        vw.release()

    def init_phase_compensation_frame(self):
        # Main container
        self.phase_compensation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.phase_compensation_frame.grid_propagate(False)

        self.pc_container = ctk.CTkFrame(self.phase_compensation_frame, corner_radius=8, width=420)
        self.pc_container.grid_propagate(False)
        self.pc_container.pack(fill="both", expand=True)

        # Scrollbar + canvas
        self.pc_scrollbar = ctk.CTkScrollbar(self.pc_container, orientation='vertical')
        self.pc_scrollbar.grid(row=0, column=0, sticky='ns')

        mode = ctk.get_appearance_mode()
        fg = self.phase_compensation_frame.cget("fg_color")
        bg_col = fg[1] if isinstance(fg, (tuple, list)) and mode == "Dark" else fg[0] if isinstance(fg, (
        tuple, list)) else fg

        self.pc_canvas = ctk.CTkCanvas(
            self.pc_container,
            width=PARAMETER_FRAME_WIDTH,
            highlightthickness=0,
            bd=0,
            background=bg_col
        )
        self.pc_canvas.grid(row=0, column=1, sticky='nsew')

        self.pc_container.grid_rowconfigure(0, weight=1)
        self.pc_container.grid_columnconfigure(1, weight=1)

        self.pc_canvas.configure(yscrollcommand=self.pc_scrollbar.set)
        self.pc_scrollbar.configure(command=self.pc_canvas.yview)

        self.phase_compensation_inner_frame = ctk.CTkFrame(self.pc_canvas)
        self.pc_canvas.create_window((0, 0), window=self.phase_compensation_inner_frame, anchor='nw')

        # Title
        self.main_title_pc = ctk.CTkLabel(
            self.phase_compensation_inner_frame,
            text='Real-time Compensation',
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.main_title_pc.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

        # Parameters panel
        self.params_pc_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=400,
            height=110
        )
        self.params_pc_frame.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        self.params_pc_frame.grid_propagate(False)
        for col in range(3):
            self.params_pc_frame.columnconfigure(col, weight=1)

        self.update_compensation_params()

        # Compensation + FT visualization panel
        self.filter_pc_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=400,
            height=110
        )
        self.filter_pc_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.filter_pc_frame.grid_propagate(False)
        for col in (0, 1):
            self.filter_pc_frame.columnconfigure(col, weight=1)

        # Panel title
        self.filter_label_pc = ctk.CTkLabel(
            self.filter_pc_frame,
            text="FT Visualization Options",
            font=ctk.CTkFont(weight="bold")
        )
        self.filter_label_pc.grid(row=0, column=0, columnspan=2, padx=5, pady=(5, 2), sticky="w")

        # Geometry filter selector
        geometry_label = ctk.CTkLabel(
            self.filter_pc_frame,
            text="Spatial Filter Geometry:"
        )
        geometry_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.spatial_filter_var_pc = ctk.StringVar(value="Circular")
        self.filter_menu_pc = ctk.CTkOptionMenu(
            self.filter_pc_frame,
            values=["Circular", "Rectangular"],
            variable=self.spatial_filter_var_pc
        )
        self.filter_menu_pc.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # FT visualization options
        self.ft_display_var = tk.StringVar(value="unfiltered")
        ctk.CTkRadioButton(
            self.filter_pc_frame, text="Show FT Filtered",
            variable=self.ft_display_var, value="filtered",
            command=self.show_ft_filtered
        ).grid(row=2, column=0, padx=5, pady=(5, 0), sticky="w")

        ctk.CTkRadioButton(
            self.filter_pc_frame, text="Show FT Unfiltered",
            variable=self.ft_display_var, value="unfiltered",
            command=self.show_ft_unfiltered
        ).grid(row=2, column=1, padx=5, pady=(5, 0), sticky="w")

        # Compensation controls panel
        self.compensate_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=80
        )
        self.compensate_frame.grid(row=3, column=0, sticky="ew", pady=(2, 6))
        self.compensate_frame.grid_propagate(False)
        for col in (0, 1):
            self.compensate_frame.columnconfigure(col, weight=1)

        ctk.CTkLabel(self.compensate_frame,
                     text="Compensation Controls",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w"
        )

        self.compensate_button = ctk.CTkButton(
            self.compensate_frame, text="⚙ Compensate", width=120,
            command=self.start_compensation
        )
        self.compensate_button.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        self.playstop_frame = ctk.CTkFrame(self.compensate_frame, fg_color="transparent")
        self.playstop_frame.grid(row=1, column=1, sticky="e", padx=10, pady=(0, 10))

        self.play_button = ctk.CTkButton(
            self.playstop_frame, text="▶ Play", width=80, command=self._on_play
        )
        self.play_button.pack(side="left", padx=(0, 5))

        self.stop_button = ctk.CTkButton(
            self.playstop_frame, text="⏹ Stop", width=80, command=self._on_stop
        )
        self.stop_button.pack(side="left")

        # Record panel
        self.record_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=80
        )
        self.record_frame.grid(row=4, column=0, sticky="ew", pady=(2, 6))
        self.record_frame.grid_propagate(False)

        # IMPORTANT: add an extra column (4) for the REC indicator
        for col in (0, 1, 2, 3, 4):
            self.record_frame.columnconfigure(col, weight=1)

        ctk.CTkLabel(self.record_frame,
                     text="Record Options",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=5, padx=10, pady=(5, 5), sticky="w"
        )

        ctk.CTkLabel(self.record_frame, text="Record").grid(
            row=1, column=0, padx=(10, 5), pady=(3, 5), sticky="w"
        )

        self.record_var = ctk.StringVar(value="Phase")
        ctk.CTkOptionMenu(
            self.record_frame,
            values=["Phase", "Amplitude", "Hologram"],
            variable=self.record_var,
            width=120
        ).grid(row=1, column=1, padx=(0, 5), pady=(3, 5), sticky="w")

        ctk.CTkButton(
            self.record_frame, text="Start", width=70,
            command=self.start_record
        ).grid(row=1, column=2, padx=(0, 5), pady=(3, 5), sticky="ew")

        ctk.CTkButton(
            self.record_frame, text="Stop", width=70,
            command=self.stop_recording
        ).grid(row=1, column=3, padx=(0, 10), pady=(3, 5), sticky="ew")

        # NEW: Create the "● REC" indicator label and keep it hidden until recording starts
        self.record_indicator = ctk.CTkLabel(
            self.record_frame,
            text="● REC",
            text_color="red",
            font=ctk.CTkFont(weight="bold")
        )
        # Place it in the new column 4, aligned to the right
        self.record_indicator.grid(row=1, column=4, padx=(0, 10), pady=(3, 5), sticky="e")
        # Hide by default; start_record() will call grid() to show it
        self.record_indicator.grid_remove()

        # Particle Tracking Panel
        self.particle_tracking_frame = ctk.CTkFrame(
            self.phase_compensation_inner_frame,
            width=PARAMETER_FRAME_WIDTH,
            height=400
        )
        self.particle_tracking_frame.grid(row=5, column=0, sticky="ew", pady=(2, 6))
        self.particle_tracking_frame.grid_propagate(False)
        for col in range(6):
            self.particle_tracking_frame.columnconfigure(col, weight=0)

        ctk.CTkLabel(
            self.particle_tracking_frame,
            text="Particle Tracking",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        # Filter method + Color filter row
        self.filterrow_frame = ctk.CTkFrame(self.particle_tracking_frame, fg_color="transparent")
        self.filterrow_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.filterrow_frame, text="Filter method:").pack(side="left", padx=(0, 5))
        self.filter_method_var = ctk.StringVar(value="Gaussian Filter")
        self.filter_method_menu = ctk.CTkOptionMenu(
            self.filterrow_frame,
            values=["Gaussian Filter", "Bilateral Filter"],
            variable=self.filter_method_var,
            width=140
        )
        self.filter_method_menu.pack(side="left", padx=(0, 20))

        self.use_color_filter_var = tk.BooleanVar(value=True)
        self.color_filter_checkbox = ctk.CTkCheckBox(
            self.filterrow_frame,
            text="Color filtering",
            variable=self.use_color_filter_var,
            onvalue=True,
            offvalue=False
        )
        self.color_filter_checkbox.pack(side="left", padx=(0, 5))

        # Area and blob settings
        self.minmax_frame = ctk.CTkFrame(self.particle_tracking_frame, fg_color="transparent")
        self.minmax_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.minmax_frame, text="Min Area.").pack(side="left", padx=(0, 3))
        self.min_area_entry = ctk.CTkEntry(self.minmax_frame, width=50)
        self.min_area_entry.insert(0, "110")
        self.min_area_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(self.minmax_frame, text="Max Area.").pack(side="left", padx=(0, 3))
        self.max_area_entry = ctk.CTkEntry(self.minmax_frame, width=50)
        self.max_area_entry.insert(0, "500")
        self.max_area_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(self.minmax_frame, text="Blob Color").pack(side="left", padx=(0, 3))
        self.blob_color_entry = ctk.CTkEntry(self.minmax_frame, width=50)
        self.blob_color_entry.insert(0, "255")
        self.blob_color_entry.pack(side="left", padx=(0, 5))

        # Kalman filter parameters
        self.kalman_frame = ctk.CTkFrame(self.particle_tracking_frame, fg_color="transparent")
        self.kalman_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.kalman_frame, text="Kalman P:").pack(side="left", padx=(0, 3))
        self.kalman_p_entry = ctk.CTkEntry(self.kalman_frame, width=50)
        self.kalman_p_entry.insert(0, "100")
        self.kalman_p_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(self.kalman_frame, text="Kalman Q:").pack(side="left", padx=(0, 3))
        self.kalman_q_entry = ctk.CTkEntry(self.kalman_frame, width=50)
        self.kalman_q_entry.insert(0, "0.01")
        self.kalman_q_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(self.kalman_frame, text="Kalman R:").pack(side="left", padx=(0, 3))
        self.kalman_r_entry = ctk.CTkEntry(self.kalman_frame, width=50)
        self.kalman_r_entry.insert(0, "1")
        self.kalman_r_entry.pack(side="left", padx=(0, 5))

        #World coordinates label
        self.coord_frame = ctk.CTkFrame(self.particle_tracking_frame, fg_color="transparent")
        self.coord_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.world_coordinates = tk.BooleanVar(value=True)
        self.world_coordinates_checkbox = ctk.CTkCheckBox(
            self.coord_frame,
            text="World Coordinates",
            variable=self.world_coordinates,
            onvalue=True,
            offvalue=False
        )
        self.world_coordinates_checkbox.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(self.coord_frame, text="Magnification (M):").pack(side="left", padx=(0, 3))
        self.magnification = ctk.CTkEntry(self.coord_frame, width=50)
        self.magnification.insert(0, "40")
        self.magnification.pack(side="left", padx=(0, 10))

        # Tracking button
        self.tracking_button = ctk.CTkButton(
            self.particle_tracking_frame,
            text="Tracking",
            width=120,
            command=self.run_tracking
        )
        self.tracking_button.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        # Final canvas update
        self.phase_compensation_inner_frame.update_idletasks()
        self.pc_canvas.config(scrollregion=self.pc_canvas.bbox("all"))

        # (Second update is fine; some UIs prefer a second pass)
        self.phase_compensation_inner_frame.update_idletasks()
        self.pc_canvas.config(scrollregion=self.pc_canvas.bbox("all"))

    def run_tracking(self):

        if not hasattr(self, "cap") or self.cap is None:
            tk.messagebox.showwarning("No Video", "Please load a video first.")
            return

        # read parameters form GUI
        try:
            min_area = int(self.min_area_entry.get())
            max_area = int(self.max_area_entry.get())
            blob_color = int(self.blob_color_entry.get())
            use_color_filter = self.use_color_filter_var.get()
            filter_method = self.filter_method_var.get()
            kalman_p = float(self.kalman_p_entry.get())
            kalman_q = float(self.kalman_q_entry.get())
            kalman_r = float(self.kalman_r_entry.get())
            use_world_coords = self.world_coordinates.get()
            magnification = float(self.magnification.get())
            pitch_x_str = self.pitchx_label_pc_entry.get().strip()
            pitch_y_str = self.pitchy_label_pc_entry.get().strip()

            if use_world_coords:
                if not pitch_x_str or not pitch_y_str:
                    tk.messagebox.showwarning(
                        "Missing Pitch Values",
                        "Please enter both Pitch X and Pitch Y values before running tracking."
                    )
                    return

            pitch_x = float(pitch_x_str) if pitch_x_str else 1.0
            pitch_y = float(pitch_y_str) if pitch_y_str else 1.0

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Call tracking function Kalman
            trajectories, detected_positions, df_full, df_positions_vector = track(
                cap=self.cap,
                min_area=min_area,
                max_area=max_area,
                blob_color=blob_color,
                kalman_p=kalman_p,
                kalman_q=kalman_q,
                kalman_r=kalman_r,
                filter_method=filter_method,
                enable_color_filter=use_color_filter,
                use_world_coords=use_world_coords,
                magnification=magnification,
                pitch_x=pitch_x,
                pitch_y=pitch_y
            )

            self.df_positions_vector = df_positions_vector
            self.show_dataframe_in_table(self.df_positions_vector, title="Positions vector")

            print("Tracking completed. Total trajectories:", len(trajectories))

        except Exception as e:
            print("Error during tracking:", e)
            import traceback
            traceback.print_exc()

    def show_dataframe_in_table(self, df, title="Coordinates"):
        df = df.copy()
        if df.columns[0].lower().startswith("frame"):
            df.rename(columns={df.columns[0]: "frame"}, inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["frame"] + [f"P{pid}_{coord}"
                                    for coord, pid in df.columns[1:]]
        else:
            new_cols = []
            for col in df.columns:
                if col != "frame":
                    match = re.match(r".*?(\d+).*?([xy])", col, re.IGNORECASE)
                    if match:
                        pid, coord = match.groups()
                        new_cols.append(f"P{pid}_{coord}")
                    else:
                        new_cols.append(col)
                else:
                    new_cols.append(col)
            df.columns = new_cols

        table_win = tk.Toplevel(self)
        table_win.title(title)
        table_win.geometry("800x400")

        frame = tk.Frame(table_win)
        frame.pack(fill="both", expand=True)

        pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
        pt.show()

    def _on_play(self):
        """Handles Play button."""
        self.stop_compensation()

        if self.is_playing:
            # Action Pause
            self.is_playing = False
            self.play_button.configure(text="▶ Play")

            if self.source_mode == "video":
                self.video_playing = False
            elif self.source_mode == "camera":
                self.preview_active = False
        else:
            # Action Play
            self.is_playing = True
            self.play_button.configure(text="⏸ Pause")

            if self.source_mode == "video":
                self.resume_video_preview()
            elif self.source_mode == "camera":
                self.start_preview_stream()

    def _on_stop(self):
        """Handles Stop button."""
        self.preview_active = False
        self.video_playing = False
        self.realtime_active = False
        self.is_playing = False
        self.play_button.configure(text="▶ Play")
        self.stop_compensation()

        # If mode is camera / video
        if self.source_mode == "camera":
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
                self.cap = None
        elif self.source_mode == "video":
            if hasattr(self, "_video_loop_id"):
                self.after_cancel(self._video_loop_id)
                del self._video_loop_id
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
                self.cap = None

    def _sync_canvas_and_frame_bg(self):
        mode = ctk.get_appearance_mode()
        color = "gray15" if mode == "Dark" else "gray85"

        # Update all CTkCanvas backgrounds
        for canvas_attr in [
            "filters_canvas", "pc_canvas", "QPI_canvas"
            ]:
            canvas = getattr(self, canvas_attr, None)
        if canvas is not None:
            canvas.configure(background=color)

        # Update all CTkFrame fg_color backgrounds
        for frame_attr in [
            "filters_frame", "filters_container", "filters_inner_frame",
            "phase_compensation_frame", "pc_container", "phase_compensation_inner_frame",
            "QPI_frame", "QPI_container", "QPI_inner_frame",
            "viewing_frame"
            ]:
            frame = getattr(self, frame_attr, None)
        if frame is not None:
            frame.configure(fg_color=color)

    def after_idle_setup(self):
        self._sync_canvas_and_frame_bg()

    def change_appearance_mode_event(self, new_appearance_mode):
     if new_appearance_mode == "🏠 Main Menu":
         self.open_main_menu()
     else:
         ctk.set_appearance_mode(new_appearance_mode)
         self._sync_canvas_and_frame_bg()

    def open_main_menu(self):
        self.destroy()
        # Replace 'main_menu' with the actual module name where MainMenu lives
        main_mod = import_module("Main_")
        reload(main_mod)

        MainMenu = getattr(main_mod, "MainMenu")
        MainMenu().mainloop()

    def _hide_parameters_nav_button(self) -> None:
        if hasattr(self, "param_button"):
            self.param_button.destroy()
        self.change_menu_to("parameters")

    def _refresh_ft_display(self):
        if self.holo_view_var.get() != "Fourier Transform":
            return

        show_filtered = self.ft_display_var.get() == "filtered"
        img_tk, arr = None, None

        if show_filtered and hasattr(self, "current_ft_filtered_tk"):
            img_tk = self.current_ft_filtered_tk
            arr = getattr(self, "current_ft_filtered_array", None)

        elif not show_filtered and hasattr(self, "current_ft_unfiltered_tk"):
            img_tk = self.current_ft_unfiltered_tk
            arr = getattr(self, "current_ft_unfiltered_array", None)

        elif hasattr(self, "ft_frames") and self.ft_frames:
            img_tk = self.ft_frames[self.current_left_index]
            arr = (self.multi_ft_arrays[self.current_left_index]
                      if self.multi_ft_arrays else None)

        if img_tk is None:
            return

        self.captured_label.configure(image=img_tk)
        self.captured_label.image = img_tk
        if arr is not None:
            self.current_ft_array = arr

    def show_ft_filtered(self):
        """Callback for the “Show FT filtered” radio button."""
        # keep StringVar & boolean flag in sync
        self.ft_display_var.set("filtered")
        self._refresh_ft_display()

    def show_ft_unfiltered(self):
        """Callback for the “Show FT unfiltered” radio button."""
        self.ft_display_var.set("unfiltered")
        self._refresh_ft_display()

    def update_compensation_params(self, *_) -> None:
        """Regenerates the wavelength / pitch entries (no Apply button here)."""
        for w in self.params_pc_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.params_pc_frame, text="Parameters",font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=5, pady=5, sticky="w")

        # Wavelength
        self.wave_label_pc = ctk.CTkLabel(
            self.params_pc_frame, text=f"Wavelength ({self.wavelength_unit})")
        self.wave_label_pc.grid(row=1, column=0, padx=5, sticky="w")
        self._create_param_with_arrow_pc(2, 0, self.wave_label_pc, 'wave_label_pc_entry')

        # Pitch X
        self.pitchx_label_pc = ctk.CTkLabel(
            self.params_pc_frame, text=f"Pitch X ({self.pitch_x_unit})")
        self.pitchx_label_pc.grid(row=1, column=1, padx=5, sticky="w")
        self._create_param_with_arrow_pc(2, 1, self.pitchx_label_pc, 'pitchx_label_pc_entry')

        # Pitch Y
        self.pitchy_label_pc = ctk.CTkLabel(
            self.params_pc_frame, text=f"Pitch Y ({self.pitch_y_unit})")
        self.pitchy_label_pc.grid(row=1, column=2, padx=5, sticky="w")
        self._create_param_with_arrow_pc(2, 2, self.pitchy_label_pc, 'pitchy_label_pc_entry')

    def _create_param_with_arrow_pc(self, row, col, label_widget, entry_name):
        container = ctk.CTkFrame(self.params_pc_frame, fg_color="transparent")
        container.grid(row=row, column=col, padx=5, pady=5, sticky='w')

        entry = ctk.CTkEntry(container, width=70, placeholder_text='0.0')
        entry.grid(row=0, column=0, sticky='w')
        setattr(self, entry_name, entry)

        arrow_btn = ctk.CTkButton(container, width=30, text='▼')
        arrow_btn.grid(row=0, column=1, sticky='e')

        def on_arrow_click_pc(event=None):
            menu = tk.Menu(self, tearoff=0, font=("Helvetica", 14))
            for unit in ["µm", "nm", "mm", "cm", "m", "in"]:
                menu.add_command(
                    label=unit,
                    command=lambda u=unit: self._set_unit_in_label(label_widget, u)
                )
            menu.post(arrow_btn.winfo_rootx(), arrow_btn.winfo_rooty() + arrow_btn.winfo_height())

        arrow_btn.bind("<Button-1>", on_arrow_click_pc)

    def get_value_in_micrometers(self, value: str, unit: str) -> float:
        # Normalise decimal separator
        clean = value.strip().replace(",", ".")
        if clean == "":
            return 0.0
        try:
            v = float(clean)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to float.")

        factors = {
            "µm": 1.0,  "Micrometers": 1.0,
            "nm": 1e-3, "Nanometers": 1e-3,
            "mm": 1e3,  "Millimeters": 1e3,
            "cm": 1e4,  "Centimeters": 1e4,
            "m":  1e6,  "Meters": 1e6,
            "in": 2.54e4, "Inches": 2.54e4
        }
        return v * factors.get(unit, 1.0)

    def _set_unit_in_label(self, lbl, unit):
        base = lbl.cget("text").split("(")[0].strip()
        lbl.configure(text=f"{base} ({unit})")

        if "Wavelength" in base:
            self.wavelength_unit = unit
        elif "Pitch X" in base:
            self.pitch_x_unit = unit
        elif "Pitch Y" in base:
            self.pitch_y_unit = unit
        elif "Distance" in base:
            self.distance_unit = unit

    def tiro(self, holo, fx_0, fy_0, fx_tmp, fy_tmp,
             lamb, M, N, dx, dy, k, m, n):
        """Replica of the reference ‘tiro’ routine."""
        theta_x = math.asin((fx_0 - fx_tmp) * lamb / (M * dx))
        theta_y = math.asin((fy_0 - fy_tmp) * lamb / (N * dy))

        # Carrier compensation
        phase_carr = np.exp(1j * k * ((math.sin(theta_x) * m * dx) + (math.sin(theta_y) * n * dy)))
        holo = holo * phase_carr

        # Binarise the resulting phase
        phase = np.angle(holo, deg=False)
        phase_norm = (phase - phase.min()) / (np.ptp(phase) + 1e-12)
        phase_bin = np.where(phase_norm > 0.2, 1, 0)

        return phase_bin.sum(), phase_carr

    def _search_fx_fy(self, holo, fx0, fy0,Fox, Foy, lamb, M, N, dx, dy, k, m, n, G_initial):
        paso = 0.2
        G_temp = G_initial
        suma_maxima = 0
        fx, fy = fx0, fy0

        fin = 0
        while fin == 0:
            x_max_out, y_max_out = fx, fy
            frec_x = np.arange(fx - paso*G_temp, fx + paso*G_temp, paso)
            frec_y = np.arange(fy - paso*G_temp, fy + paso*G_temp, paso)

            for fy_tmp in frec_y:
                for fx_tmp in frec_x:
                    score, _ = self.tiro(holo, Fox, Foy, fx_tmp, fy_tmp,
                                         lamb, M, N, dx, dy, k, m, n)
                    if score > suma_maxima:
                        suma_maxima = score
                        x_max_out = fx_tmp
                        y_max_out = fy_tmp

            G_temp -= 1
            if x_max_out == fx and y_max_out == fy:
                fin = 1
            fx, fy = x_max_out, y_max_out

        return fx, fy

    def spatialFilteringCF(self, field, height, width, filter_type: str = "Circular", show_ft_and_filter: bool = False):
        """
        Apply spatial filtering in frequency domain for holographic reconstruction.

        Args:
            field: Input hologram field
            height: Image height
            width: Image width
            filter_type: "Circular" or "Rectangular" filtering
            show_ft_and_filter: Show OpenCV windows for debugging

        Returns:
            tuple: (filtered_ft, fy_array, fx_array)
        """
        ft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
        magnitude = np.abs(ft_shift)
        # Keep the magnitude so the pop-up can display it
        self.arr_ft = magnitude
        fy0, fx0 = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        holo = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft_shift)))

        Y, X = np.meshgrid(np.arange(height) - height / 2,
                           np.arange(width) - width / 2,
                           indexing="ij")
        self.m_mesh, self.n_mesh = X, Y
        self.k = 2 * math.pi / self.lambda_um

        fx, fy = self._search_fx_fy(
            holo, fx0, fy0,
            width / 2, height / 2,
            self.lambda_um, width, height,
            self.dx_um, self.dy_um,
            self.k, X, Y,
            G_initial=3)
        self.fx, self.fy = fx, fy

        # Mask & filtering
        cy, cx = height // 2, width // 2
        rr = int(min(height, width) * 0.30)
        yy, xx = np.ogrid[:height, :width]
        ft_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 > rr ** 2) & (yy < cy)
        ft_shift = ft_shift * ft_mask

        # Locate brightest peak AFTER the DC mask
        mag_masked = np.abs(ft_shift)
        fy_peak, fx_peak = np.unravel_index(
            np.argmax(mag_masked), mag_masked.shape)

        # Fallback if the mask wiped everything
        if mag_masked[fy_peak, fx_peak] == 0:
            fy_peak, fx_peak = int(fy), int(fx)
        fy, fx = fy_peak, fx_peak

        # Radius for ROI (same for both circular and rectangular)
        d = np.hypot(fy - cy, fx - cx)
        radius = d / 3 if d > 1e-9 else max(rr, 10)

        # Create mask based on filter type
        if filter_type == "Circular":
            mask = self.circularMask(height, width, radius, fy, fx).astype(np.uint8)
        elif filter_type == "Rectangular":
            mask = self.rectangularMask(height, width, radius, fy, fx).astype(np.uint8)
        else:
            # Default to circular
            mask = self.circularMask(height, width, radius, fy, fx).astype(np.uint8)

        filtered_ft = ft_shift * mask

        # Thumbnails for the GUI
        log_unf = (np.log1p(np.abs(ft_shift)) /
                   np.log1p(np.abs(ft_shift)).max() * 255).astype(np.uint8)
        log_fil = (np.log1p(np.abs(filtered_ft)) /
                   np.log1p(np.abs(filtered_ft)).max() * 255).astype(np.uint8)

        self.current_ft_unfiltered_array = log_unf
        self.current_ft_filtered_array = log_fil

        pil_unf = Image.fromarray(log_unf)
        pil_fil = Image.fromarray(log_fil)

        self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
            pil_unf, self.viewbox_width, self.viewbox_height)
        self.current_ft_filtered_tk = self._preserve_aspect_ratio(
            pil_fil, self.viewbox_width, self.viewbox_height)

        if show_ft_and_filter:
            cv2.imshow("FT – unfiltered", log_unf)
            cv2.imshow("FT – filtered", log_fil)
            cv2.waitKey(1)

        # Return the *updated* carrier coordinates
        return filtered_ft, np.array([fy]), np.array([fx])

    # Rectangular Mask
    def rectangularMask(self, height: int, width: int, radius: float, centY: int, centX: int) -> np.ndarray:
        """
        Create a rectangular mask with center at (centY, centX) and size based on radius.
        The rectangle will have width = 2*radius and height = 2*radius (square).
        """
        # Convert center coordinates to integers
        centY = int(round(centY))
        centX = int(round(centX))
        half_size = int(round(radius))

        # Calculate rectangle boundaries
        half_size = int(radius)

        y1 = max(0, centY - half_size)
        y2 = min(height, centY + half_size)
        x1 = max(0, centX - half_size)
        x2 = min(width, centX + half_size)

        # Create mask
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True

        return mask

    # Circular Mask
    def circularMask(self, height: int, width: int, radius: float, centY: int, centX: int) -> np.ndarray:
        Y, X = np.ogrid[:height, :width]
        return ((Y - centY) ** 2 + (X - centX) ** 2) <= radius ** 2

    def _process_next_frame(self, frame: np.ndarray) -> None:
        """All subsequent frames → SHPC refinement."""
        h, w = frame.shape
        ft_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame)))

        ftype = self.spatial_filter_var_pc.get().strip()
        radius = 0.08 * min(h, w)
        if ftype == "Rectangular":
            mask = self.rectangularMask(h, w, radius, self.fy, self.fx)
        else:  # default or "Circular"
            mask = self.circularMask(h, w, radius, self.fy, self.fx)

        ft_filt = ft_raw * mask

        # Refine carrier
        holo = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft_filt)))
        self.fx, self.fy = self._search_fx_fy(
            holo, self.fx, self.fy, w / 2, h / 2,
            self.lambda_um, w, h,
            self.dx_um, self.dy_um,
            self.k, self.m_mesh, self.n_mesh, G_initial=1
        )

        self._reconstruct_and_update_views(frame, ft_filt)

        # Refresh FT thumbnails
        log_unf = (np.log1p(np.abs(ft_raw)) /
                   np.log1p(np.abs(ft_raw)).max() * 255).astype(np.uint8)
        log_fil = (np.log1p(np.abs(ft_filt)) /
                   np.log1p(np.abs(ft_filt)).max() * 255).astype(np.uint8)

        self.current_ft_unfiltered_array = log_unf
        self.current_ft_filtered_array = log_fil
        self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
            Image.fromarray(log_unf), self.viewbox_width, self.viewbox_height)
        self.current_ft_filtered_tk = self._preserve_aspect_ratio(
            Image.fromarray(log_fil), self.viewbox_width, self.viewbox_height)
        self._refresh_ft_display()

    def _process_first_frame(self, frame: np.ndarray) -> None:
        """First frame → SHPC only (Vortex removed)."""
        h, w = frame.shape
        ftype = getattr(self, "selected_filter_type", "Circular")
        filtered_ft, fy_arr, fx_arr = self.spatialFilteringCF(
            frame, h, w, filter_type=ftype
        )
        self.fx, self.fy = fx_arr[0], fy_arr[0]
        self._reconstruct_and_update_views(frame, filtered_ft)
        self.first_frame_done = True

    def _reconstruct_and_update_views(
        self,
        hologram_gray: np.ndarray,
        filtered_ft:   np.ndarray
    ) -> None:
        """
        Reconstruct amplitude/phase for *one* frame, update caches,
        push the right thumbnails to the two viewers, and (if active)
        append the frame to the current recording buffer.
        """
        # Basic geometry & cached carrier
        M, N = hologram_gray.shape[1], hologram_gray.shape[0]
        fx = self.fx[0] if isinstance(self.fx, np.ndarray) else self.fx
        fy = self.fy[0] if isinstance(self.fy, np.ndarray) else self.fy

        theta_x = np.arcsin((M/2 - fx) * self.lambda_um / (M * self.dx_um))
        theta_y = np.arcsin((N/2 - fy) * self.lambda_um / (N * self.dy_um))

        carrier = np.exp(1j * self.k * (
                  np.sin(theta_x) * self.m_mesh * self.dx_um +
                  np.sin(theta_y) * self.n_mesh * self.dy_um))

        field = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(filtered_ft))) * carrier
        amplitude_raw = np.abs(field)
        phase_raw = np.angle(field)

        # 0-255 thumbnails
        holo_u8 = cv2.normalize(hologram_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        amp_u8 = cv2.normalize(amplitude_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        phase_u8 = (((phase_raw + np.pi) / (2*np.pi)) * 255).astype(np.uint8)

        # FT quick-looks
        ft_unf = np.log1p(np.abs(np.fft.fftshift(
                   np.fft.fft2(np.fft.fftshift(hologram_gray)))))
        ft_fil = np.log1p(np.abs(filtered_ft))
        ft_unf = cv2.normalize(ft_unf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ft_fil = cv2.normalize(ft_fil, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Record if requested
        if getattr(self, "is_recording", False):
            if self.target_to_record == "Amplitude":
                self.buff_amp.append(amp_u8.copy())
            elif self.target_to_record == "Phase":
                self.buff_phase.append(phase_u8.copy())
            elif self.target_to_record == "Hologram":
                self.buff_holo.append(holo_u8.copy())

        # Cache arrays for zoom / save
        self.current_holo_array = holo_u8
        self.current_amplitude_array = amp_u8
        self.current_phase_array = phase_u8
        self.current_ft_unfiltered_array = ft_unf
        self.current_ft_filtered_array = ft_fil

        # Build CTkImages (Hi-DPI friendly)
        tk_holo = self._preserve_aspect_ratio(Image.fromarray(holo_u8),
                                               self.viewbox_width, self.viewbox_height)
        tk_ft_un = self._preserve_aspect_ratio(Image.fromarray(ft_unf),
                                               self.viewbox_width, self.viewbox_height)
        tk_ft_fi = self._preserve_aspect_ratio(Image.fromarray(ft_fil),
                                               self.viewbox_width, self.viewbox_height)
        tk_amp = self._preserve_aspect_ratio_right(Image.fromarray(amp_u8))
        tk_phase = self._preserve_aspect_ratio_right(Image.fromarray(phase_u8))

        self.current_ft_unfiltered_tk = tk_ft_un
        self.current_ft_filtered_tk = tk_ft_fi

        # Single-frame buffers for navigation
        self.hologram_frames = [tk_holo]
        self.ft_frames = [tk_ft_un]
        self.multi_holo_arrays = [holo_u8]
        self.multi_ft_arrays = [ft_unf]
        self.amplitude_frames = [tk_amp]
        self.phase_frames = [tk_phase]
        self.amplitude_arrays = [amp_u8]
        self.phase_arrays = [phase_u8]

        # Push to viewers
        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=tk_holo)
        else:
            self._refresh_ft_display()

        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.processed_label.configure(image=tk_amp)
        else:
            self.processed_label.configure(image=tk_phase)

        # ensure zoom gets correct FT
        self.current_ft_array = (self.current_ft_filtered_array
                                 if self.ft_display_filtered
                                 else self.current_ft_unfiltered_array)

    def _update_realtime(self) -> None:
        if not self._ensure_camera():
            print("[Camera] Lost connection – realtime loop stopped.")
            self.realtime_active = False
            return

        ok, frame = self.cap.read()
        if not ok:
            self.after(30, self._update_realtime)
            return

        # Fast detection YUYV
        if frame.ndim == 3 and frame.shape[2] == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUY2)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dark Frame
        if gray.mean() < 5:
            print("[WARN] Frame negro – skip")
            self.after(20, self._update_realtime)
            return

        # Frames one or two
        if not self.first_frame_done:
            self._process_first_frame(gray)
        else:
            self._process_next_frame(gray)

        # call next frame
        self.after(20, self._update_realtime)

        # parameters
        try:
            l_txt = self.wave_label_pc_entry.get()
            self.lambda_um = self.get_value_in_micrometers(l_txt, self.wavelength_unit)
            if self.lambda_um == 0:
                raise ValueError("Wavelength is empty.")
        except ValueError as e:
            tk.messagebox.showwarning("Parameters", f"Bad parameters: {e}")
            return

        try:
            dx_txt = self.pitchx_label_pc_entry.get()
            self.dx_um = self.get_value_in_micrometers(dx_txt, self.pitch_x_unit)
            dy_txt = self.pitchy_label_pc_entry.get()
            self.dy_um = self.get_value_in_micrometers(dy_txt, self.pitch_y_unit)
            if self.dx_um == 0 or self.dy_um == 0:
                raise ValueError("Pixel pitch is empty.")
        except ValueError as e:
            tk.messagebox.showwarning("Parameters", f"Bad parameters: {e}")
            return

        # filter
        self.selected_filter_type = self.spatial_filter_var_pc.get().strip()
        self.wavelength = self.lambda_um
        self.dxy = (self.dx_um + self.dy_um) / 2.0
        self.k = 2 * math.pi / self.lambda_um

    def stop_realtime_stream(self) -> None:
        """Bind to any Stop/Close button."""
        self.realtime_active = False

    # Slimmed-down menu switcher
    def change_menu_to(self, name: str) -> None:
        """
        Now there is only *one* auxiliary frame (‘phase_compensation’).
        Any other request just hides everything and shows that one.
        """
        name = "phase_compensation" if name in ("home", "parameters") else name

        # Hide everything first
        for f in ("phase_compensation_frame",):
            fr = getattr(self, f, None)
            if fr is not None:
                fr.grid_forget()

        if name == "phase_compensation":
            self.phase_compensation_frame.grid(row=0, column=0, sticky="nsew", padx=5)

    def release(self):
        os.system("taskkill /f /im python.exe")


if __name__=='__main__':
    app = App()
    app.mainloop()
    app.release()
