
import zipfile, io
import os
import sys
from pathlib import Path

import customtkinter as ctk

if __package__ is None or __package__ == "":
    _THIS_FILE = Path(__file__).resolve()
    _PACKAGE_DIR = _THIS_FILE.parent
    _PROJECT_ROOT = _PACKAGE_DIR.parent

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from holobio.parallel_rc import *
    from holobio import functions_GUI as fGUI
    from holobio.track_particles_kalman import track_particles_kalman as track
else:
    from .parallel_rc import *
    from . import functions_GUI as fGUI
    from .track_particles_kalman import track_particles_kalman as track

from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import math
import cv2, time, tkinter as tk
from importlib import import_module, reload
import threading, queue
from pandastable.core import Table
import pandas as pd
import re
from scipy.ndimage import median_filter

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import hilbert
from scipy.sparse.linalg import svds
from skimage.restoration import unwrap_phase

def reference_wave(fx_max, fy_max, m, n, _lambda, dx, k, fx_0, fy_0, M, N, dy=None):
    """
    Generates the reference wave for off-axis DHM phase compensation.

    Parameters
    ----------
    fx_max, fy_max : float
        Frequency coordinates of the +1 diffraction order peak.
    m, n : ndarray
        Spatial coordinate meshgrids (columns and rows respectively).
    _lambda : float
        Wavelength (µm).
    dx : float
        Pixel size in x (µm). Used for both axes if dy is not provided.
    k : float
        Wavenumber 2π/λ.
    fx_0, fy_0 : float
        Center of the frequency domain (M/2, N/2).
    M, N : int
        Width and height of the field.
    dy : float, optional
        Pixel size in y (µm). Defaults to dx if not provided (square pixels).
    """
    if dy is None:
        dy = dx
    arg_x = (fx_0 - fx_max) * _lambda / (M * dx)
    arg_y = (fy_0 - fy_max) * _lambda / (N * dy)
    theta_x = np.arcsin(arg_x)
    theta_y = np.arcsin(arg_y)
    ref_wave = np.exp(1j * k * (dx * np.sin(theta_x) * m + dy * np.sin(theta_y) * n))
    return ref_wave

def spatial_filter(holo, M, N, save='Yes', factor=2.0, rotate:bool=False):
    # Apply Fourier transform to the hologram
    ft_holo = fftshift(fft2(fftshift(holo)))
    ft_holo[:5, :5] = 0  # suppress low-frequency components at the origin

    # Create a mask to eliminate the central DC component
    mask1 = np.ones((N, M), dtype=np.float32)
    mask1[int(N / 2 - 20):int(N / 2 + 20), int(M / 2 - 20):int(M / 2 + 20)] = 0
    ft_holo_I = ft_holo * mask1

    # Remove specular reflection or bright central peak
    mask1 = np.ones((N, M), dtype=np.float32)
    mask1[0, 0] = 0
    ft_holo_I *= mask1

    region_interest = ft_holo_I;
    # Select region of interest: left half of the spectrum
    if rotate:
        region_interest[:, :int(M / 2)] = 0
    else:
        region_interest[:, int(M / 2):-1] = 0

    # Find the peak in the region of interest (corresponding to +1 diffraction order)
    max_value = np.max(np.abs(region_interest))
    max_pos = np.where(np.abs(region_interest) == max_value)
    fy_max = max_pos[0][0]  # vertical coordinate
    fx_max = max_pos[1][0]  # horizontal coordinate

    # Compute distance from center of ROI to peak (for circular mask)
    distance = np.sqrt((fx_max - M / 2) ** 2 + (fy_max - N / 2) ** 2)
    resc = distance / factor  # define mask radius relative to peak location

    # Create circular mask centered on peak location
    Y, X = np.meshgrid(np.arange(M), np.arange(N))
    cir_mask = np.sqrt((X - fy_max) ** 2 + (Y - fx_max) ** 2) <= resc
    cir_mask = cir_mask.astype(np.float32)

    # Apply circular mask to filter the +1 order
    ft_holo_filtered = ft_holo * cir_mask
    holo_filtered = fftshift(ifft2(ifftshift(ft_holo_filtered)))

    # Optional visualization
    if save == 'Yes':
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.log1p(np.abs(ft_holo) ** 2), cmap='gray')
        plt.title('FT Hologram')
        plt.axis('equal')

        plt.subplot(1, 3, 2)
        plt.imshow(cir_mask, cmap='gray')
        plt.title('Circular Filter')
        plt.axis('equal')

        plt.subplot(1, 3, 3)
        plt.imshow(np.log1p(np.abs(ft_holo_filtered) ** 2), cmap='gray')
        plt.title('FT Filtered Hologram')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    return ft_holo, holo_filtered, fx_max, fy_max, cir_mask

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def hilbert_transform_2d(c, hilbert_or_energy_operator=1):
    """
    hilbert_transform_2d(c, hilbert_or_energy_operator)

    Computes the 2D Hilbert Transform (Spiral Phase Transform) or Energy Operator.

    Parameters:
        c : 2D numpy array (real or complex)
            Input image or interferogram: c = b * cos(psi)
        hilbert_or_energy_operator : int
            If 1: computes i * exp(i * beta) * sin(psi)
            If 0: computes -b * exp(i * beta) * sin(psi)

    Returns:
        quadrature : 2D numpy array (complex)
            Quadrature signal (complex-valued)
    """
    NR, NC = c.shape
    u, v = np.meshgrid(np.arange(NC), np.arange(NR))
    u0 = NC // 2
    v0 = NR // 2

    u = u - u0
    v = v - v0

    # Avoid division by zero at the origin
    H = (u + 1j * v).astype(np.complex128)
    H /= (np.abs(H) + 1e-6)
    H[v0, u0] = 0

    C = fft2(c)

    if hilbert_or_energy_operator:
        CH = C * ifftshift(H)
    else:
        CH = C * ifftshift(1j * H)

    quadrature = np.conj(ifft2(CH))
    return quadrature


def vortex_compensation(field, fxOverMax, fyOverMax):
    cropVortex = 5  # Pixels for interpolation
    factorOverInterpolation = 55

    # Crop around the max frequency
    sd = field[
        int(fyOverMax - cropVortex) : int(fyOverMax + cropVortex),
        int(fxOverMax - cropVortex) : int(fxOverMax + cropVortex)
    ]

    # Hilbert transform
    sd_crop = hilbert_transform_2d(sd, hilbert_or_energy_operator=1)  # 2D Hilbert transform

    sz = np.abs(sd_crop).shape
    xg = np.arange(0, sz[0])
    yg = np.arange(0, sz[1])

    F_real = RegularGridInterpolator((xg, yg), np.real(sd_crop), bounds_error=False, fill_value=0)
    F_imag = RegularGridInterpolator((xg, yg), np.imag(sd_crop), bounds_error=False, fill_value=0)

    xq = np.arange(0, sz[0] - 1 / factorOverInterpolation + 1e-6, 1 / factorOverInterpolation)
    yq = np.arange(0, sz[1] - 1 / factorOverInterpolation + 1e-6, 1 / factorOverInterpolation)

    xv, yv = np.meshgrid(xq, yq, indexing='ij')
    pts = np.stack([xv.ravel(), yv.ravel()], axis=-1)

    vq = F_real(pts).reshape(xv.shape)
    vq2 = F_imag(pts).reshape(xv.shape)

    psi = np.angle(vq + 1j * vq2)

    n1, m1 = psi.shape
    Ml = np.zeros_like(psi)

    M1 = np.zeros_like(psi)
    M2 = np.zeros_like(psi)
    M3 = np.zeros_like(psi)
    M4 = np.zeros_like(psi)
    M5 = np.zeros_like(psi)
    M6 = np.zeros_like(psi)
    M7 = np.zeros_like(psi)
    M8 = np.zeros_like(psi)

    Y1 = np.arange(0, n1 - 2)
    Y2 = np.arange(1, n1 - 1)
    Y3 = np.arange(2, n1)
    X1 = np.arange(0, m1 - 2)
    X2 = np.arange(1, m1 - 1)
    X3 = np.arange(2, m1)

    M1[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X1)]
    M2[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X2)]
    M3[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X3)]
    M4[np.ix_(Y2, X2)] = psi[np.ix_(Y2, X3)]
    M5[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X3)]
    M6[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X2)]
    M7[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X1)]
    M8[np.ix_(Y2, X2)] = psi[np.ix_(Y2, X1)]

    D1 = wrap_to_pi(M2 - M1)
    D2 = wrap_to_pi(M3 - M2)
    D3 = wrap_to_pi(M4 - M3)
    D4 = wrap_to_pi(M5 - M4)
    D5 = wrap_to_pi(M6 - M5)
    D6 = wrap_to_pi(M7 - M6)
    D7 = wrap_to_pi(M8 - M7)
    D8 = wrap_to_pi(M1 - M8)

    Ml = (D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8) / (2 * np.pi)
    Ml = fftshift(Ml)
    Ml[70:, 70:] = 0
    Ml = ifftshift(Ml)

    linearIndex = np.argmin(Ml)
    yOverInterpolVortex, xOverInterpolVortex = np.unravel_index(linearIndex, Ml.shape)

    positions = []
    x_pos = (xOverInterpolVortex / factorOverInterpolation) + (fxOverMax - cropVortex)
    y_pos = (yOverInterpolVortex / factorOverInterpolation) + (fyOverMax - cropVortex)
    positions.append([x_pos, y_pos])

    return positions
    
def legendre_compensation(field_compensate, limit, RemovePiston=True, UsePCA=False):
    """
    Compensates the phase of a complex field using a fit with Legendre polynomials.

    Parameters:
    -----------
    field_compensate : np.ndarray
        Complex field to be corrected.
    limit : int
        Radius of the region to analyze around the center.
    RemovePiston : bool
        If True (default), removes the piston term by setting coefficient[0] = 0.
        If False, searches for the optimal piston value that minimizes phase variance.
    UsePCA : bool
        If True, uses SVD decomposition to extract the dominant wavefront.

    Returns:
    --------
    compensatedHologram : np.ndarray
        Phase-compensated complex field.
    Legendre_Coefficients : np.ndarray
        Coefficients of the Legendre polynomial fit.
    """

    # Centered Fourier transform
    fftField = fftshift(fft2(ifftshift(field_compensate)))

    A, B = fftField.shape
    center_A = int(round(A / 2))
    center_B = int(round(B / 2))

    start_A = int(center_A - limit)
    end_A = int(center_A + limit)
    start_B = int(center_B - limit)
    end_B = int(center_B + limit)

    fftField = fftField[start_A:end_A, start_B:end_B]
    square = ifftshift(ifft2(fftshift(fftField)))

    # Extract dominant wavefront
    if UsePCA:
        u, s, vt = svds(square, k=1, which='LM')
        dominant = u[:, :1] @ np.diag(s[:1]) @ vt[:1, :]
        dominant = unwrap_phase(np.angle(dominant))
    else:
        dominant = unwrap_phase(np.angle(square))

    # Normalized spatial grid
    gridSize = dominant.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)

    dA = (2 / gridSize) ** 2
    order = np.arange(1, 11)

    # Get orthonormal Legendre polynomial basis
    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))

    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA
    phaseVector = dominant.reshape(-1, 1)

    # Projection onto Legendre basis
    Legendre_Coefficients = np.sum(Legendres * phaseVector, axis=0) * dA

    if RemovePiston:
        # Zero out the piston coefficient and reconstruct the wavefront
        coeffs_used = Legendre_Coefficients.copy()
        coeffs_used[0] = 0.0
        coeffs_norm = coeffs_used / np.sqrt(Legendres_norm_const)
        wavefront = np.sum(coeffs_norm[:, np.newaxis] * Legendres.T, axis=0)
    else:
        # Search for the optimal piston value
        values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
        variances = []

        for val in values:
            temp_coeffs = Legendre_Coefficients.copy()
            temp_coeffs[0] = val
            coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
            wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
            temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
            variances.append(np.var(np.angle(temp_holo)))

        best = values[np.argmin(variances)]
        Legendre_Coefficients[0] = best
        coeffs_norm = Legendre_Coefficients / np.sqrt(Legendres_norm_const)
        wavefront = np.sum(coeffs_norm[:, np.newaxis] * Legendres.T, axis=0)

    # Final phase compensation
    wavefront = wavefront.reshape(ny, nx)
    compensatedHologram = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront)

    return compensatedHologram, Legendre_Coefficients


def square_legendre_fitting(order, X, Y):
    polynomials = []
    for i in order:
        if i == 1:
            polynomials.append(np.ones_like(X))
        elif i == 2:
            polynomials.append(X)
        elif i == 3:
            polynomials.append(Y)
        elif i == 4:
            polynomials.append((3 * X**2 - 1) / 2)
        elif i == 5:
            polynomials.append(X * Y)
        elif i == 6:
            polynomials.append((3 * Y**2 - 1) / 2)
        elif i == 7:
            polynomials.append((X * (5 * X**2 - 3)) / 2)
        elif i == 8:
            polynomials.append((Y * (3 * X**2 - 1)) / 2)
        elif i == 9:
            polynomials.append((X * (3 * Y**2 - 1)) / 2)
        elif i == 10:
            polynomials.append((Y * (5 * Y**2 - 3)) / 2)
        elif i == 11:
            polynomials.append((35 * X**4 - 30 * X**2 + 3) / 8)
        elif i == 12:
            polynomials.append((X * Y * (5 * X**2 - 3)) / 2)
        elif i == 13:
            polynomials.append(((3 * Y**2 - 1) * (3 * X**2 - 1)) / 4)
        elif i == 14:
            polynomials.append((X * Y * (5 * Y**2 - 3)) / 2)
        elif i == 15:
            polynomials.append((35 * Y**4 - 30 * Y**2 + 3) / 8)
    return np.stack(polynomials, axis=-1)
    
def fringes_normalization(hologram, R):
    M, N = hologram.shape
    u0 = N // 2
    v0 = M // 2

    u, v = np.meshgrid(np.arange(N), np.arange(M))
    u = u - u0
    v = v - v0

    H = 1 - np.exp(-(u ** 2 + v ** 2) / (2 * R ** 2))
    C = np.fft.fft2(hologram)
    CH = C * np.fft.ifftshift(H)
    ch = np.fft.ifft2(CH)

    ib = C * np.fft.ifftshift(np.exp(-(u ** 2 + v ** 2) / (2 * R ** 2)))
    background = np.real(np.fft.ifft2(ib))

    s = spiralTransform(ch)
    
    s = np.abs(s)

    fringeNorm = np.cos(np.arctan2(s, ch.real))

    modulation = np.abs(ch + 1j * s)


    return background, modulation, fringeNorm


def spiralTransform(c):
    """
    Computes the spiral phase transform of a complex-valued 2D array c.
    This corresponds to the quadrature component modulated by a spiral phase factor.

    Based on:
    Kieran G. Larkin, Donald J. Bone, and Michael A. Oldfield,
    "Natural demodulation of two-dimensional fringe patterns. I.
    General background of the spiral phase quadrature transform,"
    J. Opt. Soc. Am. A 18, 1862-1870 (2001)
    """

    try:
        TH = np.max(np.abs(c))
        if np.mean(np.real(c)) > 0.01 * TH:
            print("Warning: Input must be DC filtered")

        NR, NC = c.shape

        # Create normalized frequency coordinates in [-1, 1)
        x = np.linspace(-1, 1, NC, endpoint=False)
        y = np.linspace(-1, 1, NR, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Convert to polar coordinates
        Theta = np.arctan2(Y, X)

        # Spiral filter (vortex definition)
        H = np.exp(-1j * Theta)

        # Apply spiral filter in Fourier domain
        C = np.fft.fft2(c)
        CH = C * np.fft.ifftshift(H)

        # Inverse transform and apply complex conjugate (for coordinate system consistency)
        sd = np.conj(np.fft.ifft2(CH))


        return sd

    except Exception as e:
        raise e

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

        self._init_optimized_reconstruction_state()
        self._init_runtime_timing_state()

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

        self._build_fps_status_bar()

        self.tools_menu.configure(state="disabled")
        self.update_idletasks()
        self.phase_compensation_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        self.viewing_frame.grid(row=0, column=1, sticky="nsew")

        self._sync_canvas_and_frame_bg()
        self._init_async_engine()
        self._stop_compensation = threading.Event()


    def _init_async_engine(self) -> None:
        """Initialize the asynchronous acquisition and reconstruction engine."""
        self._comp_queue: queue.Queue = queue.Queue(maxsize=8)
        self._stop_compensation = threading.Event()
        self._latest_recon_lock = threading.Lock()
        self._latest_recon_frame = None
        self._latest_recon_index = -1
        self._processed_recon_index = -1
        self._capture_thread = None
        self._recon_thread = None
        self._vl_model = None
        self._vl_frame_counter = 0
        self._vl_last_valid_data = None
        self._vl_last_valid_field = None
        self._target_preview_fps = 30.0
        self._target_reconstruction_fps = 30.0


    def _init_optimized_reconstruction_state(self) -> None:
        """Initialize the operational state for the reconstruction protocol."""
        self.reconstruction_operation_mode = "acquisition"
        self.reconstruction_mode_var = ctk.StringVar(self, value="Acquisition")
        self.correction_mode_var = ctk.StringVar(self, value="Vortex Only")
        self.fast_spectral_crop_var = tk.BooleanVar(self, value=True)
        self._force_vl_model_refresh = True
        self._last_alignment_message_time = 0.0

    def _set_alignment_mode(self) -> None:
        """Activate alignment mode - carrier model is periodically refreshed."""
        self.reconstruction_operation_mode = "alignment"
        self.reconstruction_mode_var.set("Alignment")
        self._force_vl_model_refresh = True
        if hasattr(self, "reconstruction_mode_label"):
            self.reconstruction_mode_label.configure(text="Mode: Alignment")
        if hasattr(self, "vl_settings") and isinstance(self.vl_settings, dict):
            self.vl_settings.update(self._get_vortex_legendre_default_settings())

    def _set_acquisition_mode(self) -> None:
        """Activate acquisition mode - cached carrier model is reused."""
        self.reconstruction_operation_mode = "acquisition"
        self.reconstruction_mode_var.set("Acquisition")
        if hasattr(self, "reconstruction_mode_label"):
            self.reconstruction_mode_label.configure(text="Mode: Acquisition")
        if hasattr(self, "vl_settings") and isinstance(self.vl_settings, dict):
            self.vl_settings.update(self._get_vortex_legendre_default_settings())

    def _force_alignment_update(self) -> None:
        """Request a single model refresh without leaving acquisition mode."""
        self._force_vl_model_refresh = True
        if hasattr(self, "reconstruction_mode_label"):
            self.reconstruction_mode_label.configure(text=f"Mode: {self.reconstruction_mode_var.get()} | Align next frame")

    def _on_reconstruction_correction_changed(self, choice: str) -> None:
        """Update the phase-correction protocol and force a fresh carrier model."""
        if choice not in ("Vortex Only", "Vortex + Cached Legendre"):
            choice = "Vortex Only"
        self.correction_mode_var.set(choice)
        self._force_vl_model_refresh = True
        if hasattr(self, "vl_settings") and isinstance(self.vl_settings, dict):
            self.vl_settings.update(self._get_vortex_legendre_default_settings())
        if hasattr(self, "reconstruction_mode_label"):
            self.reconstruction_mode_label.configure(text=f"Mode: {self.reconstruction_mode_var.get()} | {choice}")

    def _on_spatial_filter_geometry_changed(self, choice: str) -> None:
        """Update the spatial mask geometry and force the carrier model to refresh."""
        if choice not in ("Circular", "Rectangular"):
            choice = "Circular"

        self.selected_filter_type = choice
        self._force_vl_model_refresh = True

        if hasattr(self, "vl_settings") and isinstance(self.vl_settings, dict):
            self.vl_settings.update(self._get_vortex_legendre_default_settings())
            self.vl_settings["filter_type"] = choice

        if getattr(self, "holo_view_var", None) is not None and self.holo_view_var.get() == "Fourier Transform":
            self._refresh_ft_display()

    def _init_runtime_timing_state(self) -> None:
        """
        Initialize frame-rate state variables for acquisition, reconstruction, display diagnostics, and repeatable video playback.
        """
        self.source_fps = 30.0
        self.source_frame_period_ms = 1000.0 / self.source_fps
        self.camera_target_fps = 30.0
        self.video_file_path = ""
        self.video_frame_count = 0
        self.video_loop_playback = True
        self._video_loop_id = None
        self._preview_loop_id = None
        self._comp_poll_loop_id = None
        self._holo_fps_count = 0
        self._recon_fps_count = 0
        self._holo_fps_window_t0 = time.perf_counter()
        self._recon_fps_window_t0 = time.perf_counter()
        self._holo_fps_value = 0.0
        self._recon_fps_value = 0.0

    def _build_fps_status_bar(self) -> None:
        """
        Construct the acquisition and reconstruction frame-rate indicators below the visualization panel.
        """
        if not hasattr(self, "viewing_frame"):
            return
        if hasattr(self, "fps_status_frame"):
            try:
                if self.fps_status_frame.winfo_exists():
                    return
            except Exception:
                pass
        self.viewing_frame.grid_rowconfigure(2, weight=0)
        self.fps_status_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.fps_status_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        self.fps_status_frame.grid_columnconfigure(0, weight=1)
        self.fps_status_frame.grid_columnconfigure(1, weight=1)
        self.hologram_fps_label = ctk.CTkLabel(self.fps_status_frame, text="Hologram acquisition FPS: 0.0 / 30.0", font=ctk.CTkFont(size=13, weight="bold"))
        self.hologram_fps_label.grid(row=0, column=0, sticky="w", padx=12, pady=6)
        self.reconstruction_fps_label = ctk.CTkLabel(self.fps_status_frame, text="Reconstruction display FPS: 0.0 / 30.0", font=ctk.CTkFont(size=13, weight="bold"))
        self.reconstruction_fps_label.grid(row=0, column=1, sticky="e", padx=12, pady=6)

    def _reset_fps_measurements(self) -> None:
        """
        Reset the sliding-window frame-rate estimators at the beginning of a new stream or reconstruction session.
        """
        now = time.perf_counter()
        self._holo_fps_count = 0
        self._recon_fps_count = 0
        self._holo_fps_window_t0 = now
        self._recon_fps_window_t0 = now
        self._holo_fps_value = 0.0
        self._recon_fps_value = 0.0
        self._update_fps_labels()

    def _set_source_fps(self, fps: float | None, fallback: float = 30.0) -> None:
        """
        Register the nominal source frame rate used to schedule acquisition and report measured display rates.
        """
        try:
            fps_value = float(fps) if fps is not None else float(fallback)
        except Exception:
            fps_value = float(fallback)
        if not np.isfinite(fps_value) or fps_value <= 0.0 or fps_value > 240.0:
            fps_value = float(fallback)
        self.source_fps = fps_value
        self.source_frame_period_ms = 1000.0 / self.source_fps
        self._target_preview_fps = self.source_fps
        self._target_reconstruction_fps = self.source_fps
        self._update_fps_labels()

    def _read_capture_fps(self, cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
        """
        Read OpenCV frame-rate metadata and replace missing or invalid values with a bounded fallback.
        """
        try:
            fps_value = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps_value = 0.0
        if not np.isfinite(fps_value) or fps_value <= 0.0 or fps_value > 240.0:
            fps_value = float(fallback)
        return fps_value

    def _source_delay_ms(self, t0: float) -> int:
        """
        Compute the GUI scheduling delay that preserves the nominal source period after current-frame overhead.
        """
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return max(1, int(round(self.source_frame_period_ms - elapsed_ms)))

    def _mark_hologram_frame_displayed(self) -> None:
        """
        Update the measured hologram display rate after one native hologram has reached the interface.
        """
        now = time.perf_counter()
        self._holo_fps_count += 1
        elapsed = now - self._holo_fps_window_t0
        if elapsed >= 0.5:
            self._holo_fps_value = self._holo_fps_count / elapsed
            self._holo_fps_count = 0
            self._holo_fps_window_t0 = now
            self._update_fps_labels()

    def _mark_reconstruction_frame_displayed(self) -> None:
        """
        Update the measured reconstruction display rate after one amplitude or phase frame has reached the interface.
        """
        now = time.perf_counter()
        self._recon_fps_count += 1
        elapsed = now - self._recon_fps_window_t0
        if elapsed >= 0.5:
            self._recon_fps_value = self._recon_fps_count / elapsed
            self._recon_fps_count = 0
            self._recon_fps_window_t0 = now
            self._update_fps_labels()

    def _update_fps_labels(self) -> None:
        """
        Refresh the acquisition and reconstruction frame-rate labels using measured and nominal source rates.
        """
        source_fps = float(getattr(self, "source_fps", 30.0))
        holo_fps = float(getattr(self, "_holo_fps_value", 0.0))
        recon_fps = float(getattr(self, "_recon_fps_value", 0.0))
        if hasattr(self, "hologram_fps_label"):
            try:
                self.hologram_fps_label.configure(text=f"Hologram acquisition FPS: {holo_fps:.1f} / {source_fps:.1f}")
            except Exception:
                pass
        if hasattr(self, "reconstruction_fps_label"):
            try:
                self.reconstruction_fps_label.configure(text=f"Reconstruction display FPS: {recon_fps:.1f} / {source_fps:.1f}")
            except Exception:
                pass

    def _safe_after_cancel(self, callback_name: str) -> None:
        """
        Cancel one Tkinter after callback only if the stored id is real and still cancellable.

        Tkinter raises ValueError when after_cancel receives None, an expired id, or a value
        that was already cancelled. Real-time video/camera code hits this easily because a
        scheduled callback can fire before the reset path tries to cancel it. The callback
        attribute is always reset to None so the next source load starts cleanly.
        """
        callback_id = getattr(self, callback_name, None)

        if callback_id:
            try:
                self.after_cancel(callback_id)
            except Exception:
                pass

        setattr(self, callback_name, None)

    def _cancel_timed_loops(self) -> None:
        """
        Safely cancel all GUI-scheduled acquisition/display loops.
        """
        for callback_name in ("_video_loop_id", "_preview_loop_id", "_comp_poll_loop_id"):
            self._safe_after_cancel(callback_name)

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
        """
        Stop the current source and clear cached frames before opening a new camera/video.

        The scheduled graphical callbacks are cancelled through a tolerant cancellation
        routine, which prevents stale or already executed Tkinter callback identifiers
        from aborting source reinitialization. The video path is also cleared here so a
        subsequent source selection cannot reopen an obsolete file.
        """
        self.preview_active = False
        self.video_playing = False
        self.realtime_active = False
        self.is_playing = False
        self.video_file_path = ""
        self.video_frame_count = 0

        self._cancel_timed_loops()

        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()

        for thread_name in ("_capture_thread", "_recon_thread", "_comp_thread", "play_thread"):
            th = getattr(self, thread_name, None)
            if th is not None and hasattr(th, "is_alive") and th.is_alive():
                try:
                    th.join(timeout=0.05)
                except RuntimeError:
                    pass
            setattr(self, thread_name, None)

        if hasattr(self, "cap") and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        for attr in ("hologram_frames", "ft_frames", "multi_holo_arrays", "multi_ft_arrays", "phase_arrays", "amplitude_arrays", "phase_frames", "amplitude_frames"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).clear()
                except Exception:
                    setattr(self, attr, [])

        self.current_holo_array = None
        self.current_ft_array = None
        self.current_phase_array = None
        self.current_amplitude_array = None
        self.current_ft_unfiltered_array = None
        self.current_ft_filtered_array = None

        self._set_source_fps(30.0)
        self._reset_fps_measurements()

    def pause_visualization(self) -> None:
        """Pauses the current preview (camera or video) *without* closing it."""
        self.preview_active  = False
        if hasattr(self, "video_playing"):
            self.video_playing = False

        # Stop any ongoing compensation thread
        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()

    def resume_video_preview(self) -> None:
        """
        Resume a loaded video preview and restart it from the first frame when the previous playback reached the end.
        """
        if not getattr(self, "cap", None):
            if not self._reopen_video_capture_from_path():
                return

        self._cancel_timed_loops()
        self.stop_compensation()
        self.preview_active = False
        self.video_playing = True
        self.is_playing = True

        if hasattr(self, "play_button"):
            try:
                self.play_button.configure(text="⏸ Pause")
            except Exception:
                pass

        self._reset_fps_measurements()
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

    def _frame_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert an acquired frame into a single-channel hologram."""
        if frame is None:
            raise ValueError("Input frame is None.")

        if frame.ndim == 2:
            return frame.copy()

        if frame.ndim == 3 and frame.shape[2] == 2:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUY2)

        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        raise ValueError(f"Unsupported frame format with shape {frame.shape}.")

    def _ft_uses_log_scale(self) -> bool:
        """Determine the current Fourier-display scaling mode."""
        if hasattr(self, "ft_mode_var"):
            return self.ft_mode_var.get() == "With logarithmic scale"
        return True

    def _fourier_display_from_hologram(
        self,
        hologram: np.ndarray,
        *,
        use_log: bool | None = None
    ) -> np.ndarray:
        """Compute the Fourier-transform display from the hologram."""
        if hologram is None:
            raise ValueError("Cannot compute Fourier transform from an empty hologram.")

        arr = np.asarray(hologram)

        if arr.ndim == 3:
            arr = self._frame_to_gray(arr)

        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr)))
        magnitude = np.abs(ft)

        if use_log is None:
            use_log = self._ft_uses_log_scale()

        if use_log:
            display = np.log1p(magnitude)
        else:
            display = magnitude

        max_value = float(np.max(display))
        if max_value <= 1e-12:
            return np.zeros(display.shape, dtype=np.uint8)

        display = display / max_value
        display = np.clip(display * 255.0, 0.0, 255.0)

        return display.astype(np.uint8)

    def _update_unfiltered_ft_from_hologram(self, hologram: np.ndarray | None = None) -> np.ndarray | None:
        """Update the unfiltered Fourier-transform cache from the current hologram."""
        if hologram is None:
            hologram = getattr(self, "current_holo_array", None)

        if hologram is None:
            return None

        ft_display = self._fourier_display_from_hologram(hologram)

        self.current_ft_unfiltered_array = ft_display
        self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
            Image.fromarray(ft_display),
            self.viewbox_width,
            self.viewbox_height
        )

        self.ft_frames = [self.current_ft_unfiltered_tk]
        self.multi_ft_arrays = [ft_display]

        if self.ft_display_var.get() == "unfiltered":
            self.current_ft_array = ft_display

        return ft_display

    def _update_filtered_ft_from_hologram(self, hologram: np.ndarray | None = None) -> np.ndarray | None:
        """Update the filtered Fourier-transform cache from the native hologram."""
        if hologram is None:
            hologram = getattr(self, "current_holo_array", None)
        if hologram is None:
            return None

        try:
            self._ensure_vortex_legendre_modules()
            settings = getattr(self, "vl_settings", None)
            if settings is None:
                settings = self._get_vortex_legendre_default_settings()
                self.vl_settings = settings

            sample, _crop_info = self._prepare_vl_processing_frame(hologram, max_side=None)
            filter_type = getattr(self, "selected_filter_type", settings.get("filter_type", "Circular"))
            if filter_type not in ("Circular", "Rectangular"):
                filter_type = "Circular"
            settings["filter_type"] = filter_type

            fft2 = self._vl_fft2
            fftshift = self._vl_fftshift
            ft_raw = fftshift(fft2(fftshift(sample.astype(np.float32, copy=False))))

            model = getattr(self, "_vl_model", None)
            model_is_usable = (
                model is not None
                and model.get("shape") == sample.shape
                and model.get("filter_type") == filter_type
            )

            if model_is_usable and model.get("fast_spectral_crop", False) and model.get("crop_model", None) is not None:
                mask_crop = model.get("mask_crop", None)
                crop_model = model.get("crop_model", None)
                if mask_crop is not None and crop_model is not None:
                    ft_filtered = self._vl_expand_crop_to_full_spectrum(ft_raw, crop_model, mask_crop)
                else:
                    ft_filtered = np.zeros_like(ft_raw)

            elif model_is_usable and model.get("mask", None) is not None:
                ft_filtered = ft_raw * model["mask"]

            else:
                ft_filtered, _ft_raw, _fx, _fy, _mask, _radius = self._spatial_filtering_cf_core(
                    sample,
                    sample.shape[0],
                    sample.shape[1],
                    filter_type=filter_type,
                )

            ft_display = self._vl_log_display(ft_filtered)

        except Exception as exc:
            print(f"[Vortex-Legendre] Filtered FT display fallback: {exc}")
            ft_display = self._fourier_display_from_hologram(hologram)

        self.current_ft_filtered_array = ft_display
        self.current_ft_filtered_tk = self._preserve_aspect_ratio(
            Image.fromarray(ft_display),
            self.viewbox_width,
            self.viewbox_height,
        )

        if self.ft_display_var.get() == "filtered":
            self.current_ft_array = ft_display

        return ft_display

    def _quick_ft_display(self, gray: np.ndarray, max_side: int | None = None) -> np.ndarray:
        """Compute a Fourier-transform display from the complete hologram."""
        return self._fourier_display_from_hologram(gray, use_log=True)

    def load_video(self) -> None:
        """
        Open a video file and initialize the hologram and Fourier caches.

        The first hologram frame is stored at its native resolution. The playback
        cursor is returned to the first frame after the preview snapshot so that
        pressing Play or Compensate starts from the beginning. The file path is
        retained to allow deterministic replay after end-of-file events.
        """
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not file_path:
            return

        if hasattr(self, "cap") and self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = self._open_video_file(file_path)
        if self.cap is None or not self.cap.isOpened():
            tk.messagebox.showerror("Video Error", "Could not open the selected video.")
            return

        self.video_file_path = file_path
        self.video_frame_count = int(max(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))
        self.video_loop_playback = True
        self._set_source_fps(self._read_capture_fps(self.cap, fallback=30.0), fallback=30.0)
        self._reset_fps_measurements()

        self.source_mode = "video"
        self.is_video_preview = True
        self.preview_active = False
        self.video_playing = False
        self.current_left_index = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ok, frame = self.cap.read()
        if not ok:
            tk.messagebox.showerror("Video Error", "Could not read the first frame.")
            return

        gray = self._frame_to_gray(frame)
        self.current_holo_array = gray
        self.last_preview_gray = gray

        holo_tk = self._preserve_aspect_ratio(Image.fromarray(gray), self.viewbox_width, self.viewbox_height)

        self.hologram_frames = [holo_tk]
        self.multi_holo_arrays = [gray]

        self._update_unfiltered_ft_from_hologram(gray)

        if self.holo_view_var.get() == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk

        elif self.holo_view_var.get() == "Fourier Transform":
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

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
        """Pause video stream after unrecoverable read failure."""
        self.video_playing = False
        self.preview_active = False
        self.is_playing = False

        if getattr(self, "video_file_path", ""):
            self.is_video_preview = True
            self._restart_video_capture_from_beginning()
        else:
            self.is_video_preview = False

        if hasattr(self, "play_button"):
            try:
                self.play_button.configure(text="▶ Play")
            except Exception:
                pass

        self._cancel_timed_loops()
        self.stop_compensation()

    def _reopen_video_capture_from_path(self) -> bool:
        """
        Recreate the OpenCV video capture from the stored file path.

        Some backends cannot seek reliably after an end-of-file event. Reopening
        the file provides a backend-independent fallback while preserving the
        native video frame rate used by the display scheduler.
        """
        path = getattr(self, "video_file_path", "")
        if not path:
            return False

        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = self._open_video_file(path)
        if self.cap is None or not self.cap.isOpened():
            self.cap = None
            return False

        self._set_source_fps(self._read_capture_fps(self.cap, fallback=getattr(self, "source_fps", 30.0)), fallback=getattr(self, "source_fps", 30.0))
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

        self.source_mode = "video"
        self.is_video_preview = True
        return True

    def _restart_video_capture_from_beginning(self) -> bool:
        """
        Return the active video capture to frame zero, reopening the file if the backend cannot seek after EOF.
        """
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True
            except Exception:
                pass

        return self._reopen_video_capture_from_path()

    def _read_video_frame_or_loop(self) -> tuple[bool, np.ndarray | None]:
        """Read one video frame and loop back to frame zero at end-of-file."""
        if not getattr(self, "cap", None):
            if not self._reopen_video_capture_from_path():
                return False, None

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if not getattr(self, "video_loop_playback", True):
            return False, None

        if not self._restart_video_capture_from_beginning():
            return False, None

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if self._reopen_video_capture_from_path():
            ok, frame = self.cap.read()
            if ok:
                return True, frame

        return False, None

    def _play_video_preview(self) -> None:
        """
        Preview a loaded video while preserving the native hologram spectrum and looping from the first frame at end-of-file.
        """
        if not getattr(self, "video_playing", False):
            return

        ok, frame = self._read_video_frame_or_loop()
        if not ok or frame is None:
            self._handle_video_end()
            return

        gray = self._frame_to_gray(frame)
        self.current_holo_array = gray
        self.last_preview_gray = gray
        self.current_left_index = 0

        holo_tk = self._preserve_aspect_ratio(Image.fromarray(gray), self.viewbox_width, self.viewbox_height)

        self.hologram_frames = [holo_tk]
        self.multi_holo_arrays = [gray]

        current_choice = self.holo_view_var.get()

        if current_choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk

        elif current_choice == "Fourier Transform":
            self._update_unfiltered_ft_from_hologram(gray)
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        self._mark_hologram_frame_displayed()
        delay_ms = int(1000.0 / max(float(getattr(self, "source_fps", 25.0)), 1.0))
        self._video_loop_id = self.after(max(delay_ms, 1), self._play_video_preview)

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
        Compute one optimized Vortex-Legendre reconstruction packet.

        The returned hologram is the original full frame. Only the numerical reconstruction field is cropped and resampled. Thus, the hologram viewer remains faithful to the acquisition stream, while the reconstruction viewer receives a computationally efficient phase and amplitude estimate.
        """
        try:
            settings = getattr(self, "vl_settings", None)
            if settings is None:
                settings = self._get_vortex_legendre_default_settings()
                self.vl_settings = settings

            data = self._compute_vortex_legendre_arrays(gray,settings=settings,first=first)
            self._vl_last_valid_data = data
            return data

        except Exception as exc:
            print("[Vortex-Legendre] Reconstruction failed:", exc)
            import traceback
            traceback.print_exc()

            if getattr(self, "_vl_last_valid_data", None) is not None:
                fallback = dict(self._vl_last_valid_data)
                fallback["holo"] = gray.copy()
                return fallback

            holo_u8 = self._normalize_to_uint8(gray)
            ft_u8 = self._quick_ft_display(gray,max_side=512)
            black = np.zeros_like(ft_u8,dtype=np.uint8)

            return {"holo": holo_u8, "ft": ft_u8, "ft_unfiltered": ft_u8, "amp": black, "phase": black}



    def _get_vortex_legendre_default_settings(self) -> dict:
        """Define numerical settings for the real-time reconstruction protocol."""
        filter_type = getattr(self, "selected_filter_type", "Circular")
        if filter_type not in ("Circular", "Rectangular"):
            filter_type = "Circular"

        operation_mode = getattr(self, "reconstruction_operation_mode", "acquisition")
        correction_mode = self.correction_mode_var.get() if hasattr(self, "correction_mode_var") else "Vortex Only"
        apply_legendre = correction_mode == "Vortex + Cached Legendre"
        model_update_interval = 8 if operation_mode == "alignment" else 0

        return {
            "factor": 5.0,
            "rotate": False,
            "filter_type": filter_type,
            "apply_legendre": apply_legendre,
            "remove_piston": True,
            "use_pca": False,
            "max_reconstruction_side": None,
            "model_update_interval": model_update_interval,
            "use_vortex_refinement": operation_mode == "alignment" or bool(getattr(self, "_force_vl_model_refresh", True)),
            "fast_spectral_crop": bool(self.fast_spectral_crop_var.get()) if hasattr(self, "fast_spectral_crop_var") else True,

            # Preserves the original reconstruction shape without applying a fixed size.
            # When True, the reconstructed object maintains the size of the processed square without additional changes.
            "preserve_reconstruction_shape": True, 
            "fixed_reconstruction_side": 768,

            "crop_radius_multiplier": 2.5,
            "minimum_crop_half_width": 48,
            "maximum_crop_side": 768,
            
            # Controls whether the filtered Fourier transform is included in reconstruction packets.
            # False: optimizes real-time performance by returning only amplitude and phase (no filtered FT).
            # True: would include filtered FT for diagnostic purposes, but reduces throughput significantly.
            "return_filtered_ft_in_recon_packets": False,
        }

    def _ensure_vortex_legendre_modules(self) -> None:
        """Initialize the Vortex-Legendre pipeline with local functions."""
        if getattr(self, "_vl_modules_ready", False):
            return

        # Use local functions directly
        class VL:
            reference_wave = staticmethod(reference_wave)
            spatial_filter = staticmethod(spatial_filter)
            vortex_compensation = staticmethod(vortex_compensation)
            legendre_compensation = staticmethod(legendre_compensation)

        self._VL = VL
        self._vl_fft2 = fft2
        self._vl_ifft2 = ifft2
        self._vl_fftshift = fftshift
        self._vl_ifftshift = ifftshift
        self._vl_median_filter = median_filter
        self._vl_modules_ready = True


    def _compute_vortex_legendre_arrays(self, gray: np.ndarray, *, settings: dict, first: bool) -> dict:
        """
        Reconstruct one holographic frame using a cached Vortex carrier model.

        The Fourier transform is computed from the native hologram. During
        alignment, the +1 order is detected and optionally refined with the
        vortex criterion. During acquisition, the cached fixed spectral crop and
        mask are reused so the high-throughput path remains: FFT, fixed-size
        cropped inverse FFT, and display conversion.
        """
        self._ensure_vortex_legendre_modules()

        holo_u8 = self._normalize_to_uint8(gray)
        sample, crop_info = self._prepare_vl_processing_frame(gray, max_side=settings.get("max_reconstruction_side", None))
        refresh_model = self._vl_should_refresh_model(sample, settings, first)
        return_ft_packet = bool(settings.get("return_filtered_ft_in_recon_packets", False))

        if refresh_model:
            obj_field, ft_filtered, ft_unfiltered = self._vl_build_model_and_object_field(
                sample,
                settings=settings,
                crop_info=crop_info,
                return_full_filtered_ft=return_ft_packet,
            )

            if settings.get("apply_legendre", False):
                recon_field, phase_offset = self._vl_estimate_legendre_phase_offset(
                    obj_field,
                    remove_piston=settings.get("remove_piston", True),
                    use_pca=settings.get("use_pca", False),
                )
                self._vl_model["phase_offset"] = phase_offset
                self._vl_model["phase_correction"] = np.exp(1j * phase_offset).astype(np.complex64, copy=False)
            else:
                recon_field = obj_field
                self._vl_model["phase_offset"] = None
                self._vl_model["phase_correction"] = None

            self._force_vl_model_refresh = False
        else:
            obj_field, ft_filtered, ft_unfiltered = self._vl_apply_cached_model(
                sample,
                return_full_filtered_ft=return_ft_packet,
            )
            phase_correction = None if self._vl_model is None else self._vl_model.get("phase_correction", None)

            if settings.get("apply_legendre", False) and phase_correction is not None and phase_correction.shape == obj_field.shape:
                recon_field = obj_field * phase_correction
            else:
                recon_field = obj_field

        self._vl_frame_counter += 1
        self._vl_last_valid_field = recon_field

        ft_u8 = self._vl_log_display(ft_filtered) if ft_filtered is not None else None
        ft_unfiltered_u8 = self._vl_log_display(ft_unfiltered) if return_ft_packet and ft_unfiltered is not None else None
        amp_u8 = self._normalize_to_uint8(np.abs(recon_field))
        phase_u8 = self._phase_to_uint8(np.angle(recon_field))

        return {
            "holo": holo_u8,
            "ft": ft_u8,
            "ft_unfiltered": ft_unfiltered_u8,
            "amp": amp_u8,
            "phase": phase_u8,
        }

    def _prepare_vl_processing_frame(self, image: np.ndarray, max_side: int | None = None) -> tuple[np.ndarray, dict]:
        """Prepare the numerical reconstruction field."""
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = self._frame_to_gray(arr)

        arr = arr.astype(np.float32, copy=False)
        h, w = arr.shape[:2]
        side = min(h, w)

        if side % 2 != 0:
            side -= 1

        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        square = arr[y0:y0 + side, x0:x0 + side]

        crop_info = {
            "x0": x0,
            "y0": y0,
            "original_side": side,
            "processing_side": side,
            "resize_scale": 1.0,
            "dx_eff": float(self.dx_um),
            "dy_eff": float(self.dy_um),
            "original_shape": image.shape,
        }

        return square, crop_info

    def _spatial_filtering_cf_core(self, field: np.ndarray, height: int, width: int, filter_type: str = "Circular") -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
        """
        Correct Fourier-order selection core.

        This now follows the same half-plane convention as vortexLegendre.spatial_filter():
        for rotate=False, the right half is suppressed and the +1 order is
        searched on the left side of the centered Fourier spectrum. The previous
        version searched the upper half-plane, which can select the wrong order
        or a reflection and produces noisy phase/amplitude reconstructions.
        """
        self._ensure_vortex_legendre_modules()
        VL = self._VL

        sample = np.asarray(field, dtype=np.float64)
        ft_raw, _holo_filtered, fx_peak, fy_peak, circular_mask = VL.spatial_filter(
            sample,
            width,
            height,
            save="No",
            factor=float(getattr(self, "vl_settings", {}).get("factor", 5.0)) if isinstance(getattr(self, "vl_settings", {}), dict) else 5.0,
            rotate=False,
        )

        cx = width / 2.0
        cy = height / 2.0
        distance = float(np.hypot(float(fx_peak) - cx, float(fy_peak) - cy))
        radius = distance / 5.0 if distance > 1e-9 else max(10.0, 0.08 * min(height, width))

        if filter_type == "Rectangular":
            order_mask = self.rectangularMask(height, width, radius, int(round(fy_peak)), int(round(fx_peak))).astype(np.float32)
        else:
            order_mask = circular_mask.astype(np.float32)

        filtered_ft = ft_raw * order_mask
        return filtered_ft, ft_raw, float(fx_peak), float(fy_peak), order_mask, float(radius)



    def _vl_should_refresh_model(self, sample: np.ndarray, settings: dict, first: bool) -> bool:
        """
        Decide whether the Fourier-order model must be recomputed.

        Acquisition mode refreshes only on the first frame, parameter changes,
        geometry changes, filter changes, or explicit user request. Alignment
        mode additionally refreshes at a controlled interval so carrier drift can
        be followed without forcing a heavy vortex search on every frame.
        """
        if first or self._vl_model is None:
            return True

        if bool(getattr(self, "_force_vl_model_refresh", False)):
            return True

        if self._vl_model.get("shape", None) != sample.shape:
            return True

        if self._vl_model.get("filter_type", None) != settings.get("filter_type", "Circular"):
            return True

        if bool(self._vl_model.get("preserve_reconstruction_shape", True)) != bool(settings.get("preserve_reconstruction_shape", True)):
            return True

        model_fixed_side = self._vl_model.get("fixed_reconstruction_side", None)
        settings_fixed_side = settings.get("fixed_reconstruction_side", None)
        if model_fixed_side != settings_fixed_side:
            return True

        if abs(float(self._vl_model.get("lambda_um", self.lambda_um)) - float(self.lambda_um)) > 1e-15:
            return True

        if abs(float(self._vl_model.get("dx_um", self.dx_um)) - float(self.dx_um)) > 1e-15:
            return True

        if abs(float(self._vl_model.get("dy_um", self.dy_um)) - float(self.dy_um)) > 1e-15:
            return True

        interval = int(settings.get("model_update_interval", 0) or 0)
        if interval > 0 and self._vl_frame_counter > 0 and self._vl_frame_counter % interval == 0:
            return True

        return False


    def _vl_get_reconstruction_side(self, settings: dict, sample_shape: tuple[int, int], radius: float | None = None) -> int:
        """
        Return the fixed square side used by the optimized Vortex reconstruction.

        DHM_RT_29 was fast because the inverse FFT was performed on a compact
        Fourier crop, but that crop changed size when the diffraction order moved.
        The full-size DHM_RT fix made the output stable by using the native square
        grid, but that also made the inverse FFT much heavier.

        This hybrid keeps a fixed fast grid: normally 768 x 768, automatically
        enlarged only if the detected Fourier mask would not fit. That preserves
        a stable square output without cutting the selected order.
        """
        sample_side = int(min(sample_shape[:2]))
        if sample_side % 2 != 0:
            sample_side -= 1
        sample_side = max(sample_side, 2)

        if bool(settings.get("preserve_reconstruction_shape", False)):
            return sample_side

        minimum_half_width = int(settings.get("minimum_crop_half_width", 48))
        crop_radius_multiplier = float(settings.get("crop_radius_multiplier", 2.5))

        required_half_width = minimum_half_width
        if radius is not None and np.isfinite(radius):
            required_half_width = max(required_half_width, int(np.ceil(float(radius) * crop_radius_multiplier)))

        required_side = max(2 * required_half_width, 2)

        requested = settings.get("fixed_reconstruction_side", settings.get("maximum_crop_side", 768))
        try:
            side = int(requested)
        except Exception:
            side = int(settings.get("maximum_crop_side", 768))

        if side <= 0:
            side = int(settings.get("maximum_crop_side", 768))

        # Never choose a side smaller than the mask support. If this happens,
        # speed is sacrificed only as much as needed to avoid cutting the FT order.
        side = max(side, required_side)
        side = min(side, sample_side)

        if side % 2 != 0:
            side -= 1

        return max(side, 2)

    def _vl_make_centered_order_crop(
        self,
        ft_raw: np.ndarray,
        fxm: float,
        fym: float,
        radius: float,
        filter_type: str,
        crop_side: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Build a fixed-size Fourier crop centered on the detected diffraction order.

        The crop is always crop_side x crop_side. If the requested crop extends
        outside the Fourier image, the missing region is zero-padded instead of
        shrinking the crop. This is the key fix: the inverse FFT always receives
        an array with the same shape, so amplitude and phase keep a stable size.
        """
        ft_raw = np.asarray(ft_raw)
        height, width = ft_raw.shape[:2]
        crop_side = int(crop_side)
        if crop_side % 2 != 0:
            crop_side -= 1
        crop_side = max(crop_side, 2)

        cy = int(round(float(fym)))
        cx = int(round(float(fxm)))
        half = crop_side // 2

        start_y = cy - half
        start_x = cx - half
        end_y = start_y + crop_side
        end_x = start_x + crop_side

        src_y1 = max(start_y, 0)
        src_y2 = min(end_y, height)
        src_x1 = max(start_x, 0)
        src_x2 = min(end_x, width)

        dst_y1 = src_y1 - start_y
        dst_y2 = dst_y1 + max(src_y2 - src_y1, 0)
        dst_x1 = src_x1 - start_x
        dst_x2 = dst_x1 + max(src_x2 - src_x1, 0)

        ft_crop = np.zeros((crop_side, crop_side), dtype=ft_raw.dtype)
        if src_y2 > src_y1 and src_x2 > src_x1:
            ft_crop[dst_y1:dst_y2, dst_x1:dst_x2] = ft_raw[src_y1:src_y2, src_x1:src_x2]

        yy = (start_y + np.arange(crop_side, dtype=np.float32))[:, None]
        xx = (start_x + np.arange(crop_side, dtype=np.float32))[None, :]

        if filter_type == "Rectangular":
            mask_crop = ((np.abs(yy - float(fym)) <= float(radius)) & (np.abs(xx - float(fxm)) <= float(radius))).astype(np.float32)
        else:
            mask_crop = (((yy - float(fym)) ** 2 + (xx - float(fxm)) ** 2) <= float(radius) ** 2).astype(np.float32)

        ft_crop = ft_crop * mask_crop
        crop_model = {
            "crop_side": crop_side,
            "start_y": int(start_y),
            "start_x": int(start_x),
            "source_bounds": (int(src_y1), int(src_y2), int(src_x1), int(src_x2)),
            "dest_bounds": (int(dst_y1), int(dst_y2), int(dst_x1), int(dst_x2)),
        }
        return ft_crop, mask_crop, crop_model

    def _vl_apply_crop_model_to_ft(self, ft_raw: np.ndarray, crop_model: dict, mask_crop: np.ndarray) -> np.ndarray:
        """
        Apply the cached fixed-size spectral crop model to a new Fourier frame.
        """
        crop_side = int(crop_model.get("crop_side", mask_crop.shape[0]))
        ft_crop = np.zeros((crop_side, crop_side), dtype=ft_raw.dtype)

        src_y1, src_y2, src_x1, src_x2 = crop_model["source_bounds"]
        dst_y1, dst_y2, dst_x1, dst_x2 = crop_model["dest_bounds"]

        height, width = ft_raw.shape[:2]
        src_y1 = max(0, min(int(src_y1), height))
        src_y2 = max(0, min(int(src_y2), height))
        src_x1 = max(0, min(int(src_x1), width))
        src_x2 = max(0, min(int(src_x2), width))

        if src_y2 > src_y1 and src_x2 > src_x1:
            ft_crop[dst_y1:dst_y2, dst_x1:dst_x2] = ft_raw[src_y1:src_y2, src_x1:src_x2]

        return ft_crop * mask_crop

    def _vl_expand_crop_to_full_spectrum(self, ft_raw: np.ndarray, crop_model: dict, mask_crop: np.ndarray) -> np.ndarray:
        """
        Place the cached crop mask back into the native Fourier grid for display.

        This keeps the left Fourier viewer full-size even when the reconstruction
        is computed from a centered crop.
        """
        ft_filtered_full = np.zeros_like(ft_raw)
        src_y1, src_y2, src_x1, src_x2 = crop_model["source_bounds"]
        dst_y1, dst_y2, dst_x1, dst_x2 = crop_model["dest_bounds"]

        height, width = ft_raw.shape[:2]
        src_y1 = max(0, min(int(src_y1), height))
        src_y2 = max(0, min(int(src_y2), height))
        src_x1 = max(0, min(int(src_x1), width))
        src_x2 = max(0, min(int(src_x2), width))

        if src_y2 > src_y1 and src_x2 > src_x1:
            local_mask = mask_crop[dst_y1:dst_y2, dst_x1:dst_x2]
            ft_filtered_full[src_y1:src_y2, src_x1:src_x2] = ft_raw[src_y1:src_y2, src_x1:src_x2] * local_mask

        return ft_filtered_full


    def _vl_build_model_and_object_field(
        self,
        sample: np.ndarray,
        *,
        settings: dict,
        crop_info: dict,
        return_full_filtered_ft: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Estimate and cache the off-axis carrier model.

        The model is built from the native Fourier lattice. A vortex-refined
        carrier coordinate is estimated during alignment, a fixed-size spectral
        support is extracted around that order, and the resulting crop model is
        reused during acquisition. This keeps the correct Fourier geometry while
        avoiding a full-frame inverse FFT in the high-throughput path.
        """
        VL = self._VL
        fft2 = self._vl_fft2
        ifft2 = self._vl_ifft2
        fftshift = self._vl_fftshift
        ifftshift = self._vl_ifftshift

        sample = np.asarray(sample, dtype=np.float32)
        N, M = sample.shape
        fx_0 = M / 2.0
        fy_0 = N / 2.0
        factor = float(settings.get("factor", 5.0))
        rotate = bool(settings.get("rotate", False))
        filter_type = settings.get("filter_type", "Circular")

        ft_raw, holo_filtered_full, fxm, fym, circular_mask = VL.spatial_filter(
            sample,
            M,
            N,
            save="No",
            factor=factor,
            rotate=rotate,
        )

        distance = float(np.hypot(float(fxm) - fx_0, float(fym) - fy_0))
        radius = distance / factor if distance > 1e-9 else max(10.0, 0.08 * min(N, M))

        if bool(settings.get("use_vortex_refinement", True)):
            crop_vortex = 6
            if crop_vortex <= fxm < M - crop_vortex and crop_vortex <= fym < N - crop_vortex:
                try:
                    logamp = 10.0 * np.log10(np.abs(fftshift(fft2(fftshift(holo_filtered_full))) + 1e-6) ** 2)
                    field_h = self._vl_median_filter(logamp, size=(1, 1), mode="reflect")
                    vortex_positions = VL.vortex_compensation(field_h, fxm, fym)
                    if vortex_positions:
                        fxm_refined, fym_refined = vortex_positions[0]
                        if np.isfinite(fxm_refined) and np.isfinite(fym_refined):
                            if abs(float(fxm_refined) - float(fxm)) <= max(radius, 8.0) and abs(float(fym_refined) - float(fym)) <= max(radius, 8.0):
                                fxm, fym = float(fxm_refined), float(fym_refined)
                except Exception as exc:
                    print(f"[Vortex] Refinement skipped: {exc}")

        if settings.get("fast_spectral_crop", True):
            crop_side = self._vl_get_reconstruction_side(settings, sample.shape, radius=radius)
            ft_crop, mask_crop, crop_model = self._vl_make_centered_order_crop(
                ft_raw,
                fxm,
                fym,
                radius,
                filter_type,
                crop_side,
            )
            obj_field = fftshift(ifft2(ifftshift(ft_crop))).astype(np.complex64, copy=False)
            ft_filtered_display = self._vl_expand_crop_to_full_spectrum(ft_raw, crop_model, mask_crop) if return_full_filtered_ft else None
            bbox = crop_model["source_bounds"]
            ref_wave = None
            mask = None
        else:
            crop_side = None
            crop_model = None
            if filter_type == "Rectangular":
                mask = self.rectangularMask(N, M, radius, int(round(fym)), int(round(fxm))).astype(np.float32)
            else:
                mask = circular_mask.astype(np.float32)

            ft_filtered_full = ft_raw * mask
            holo_filtered = fftshift(ifft2(ifftshift(ft_filtered_full)))
            m_grid, n_grid = np.meshgrid(np.arange(-M // 2, M // 2), np.arange(-N // 2, N // 2))
            ref_wave = VL.reference_wave(
                fxm,
                fym,
                m_grid,
                n_grid,
                self.lambda_um,
                float(crop_info["dx_eff"]),
                self.k,
                fx_0,
                fy_0,
                M,
                N,
                dy=float(crop_info["dy_eff"]),
            ).astype(np.complex64, copy=False)
            obj_field = (holo_filtered * ref_wave).astype(np.complex64, copy=False)
            ft_filtered_display = ft_filtered_full if return_full_filtered_ft else None
            bbox = None
            mask_crop = None

        self.fx = fxm
        self.fy = fym

        self._vl_model = {
            "shape": sample.shape,
            "mask": mask,
            "mask_crop": mask_crop,
            "bbox": bbox,
            "crop_model": crop_model,
            "crop_side": crop_side,
            "ref_wave": ref_wave,
            "fxm": float(fxm),
            "fym": float(fym),
            "filter_type": filter_type,
            "factor": factor,
            "radius": float(radius),
            "phase_offset": None,
            "phase_correction": None,
            "crop_info": dict(crop_info),
            "lambda_um": float(self.lambda_um),
            "dx_um": float(self.dx_um),
            "dy_um": float(self.dy_um),
            "fast_spectral_crop": bool(settings.get("fast_spectral_crop", True)),
            "preserve_reconstruction_shape": bool(settings.get("preserve_reconstruction_shape", False)),
            "fixed_reconstruction_side": settings.get("fixed_reconstruction_side", None),
        }

        return obj_field, ft_filtered_display, ft_raw


    def _vl_apply_cached_model(self, sample: np.ndarray, return_full_filtered_ft: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Reconstruct one frame using the cached Fourier mask and carrier model."""
        fft2 = self._vl_fft2
        ifft2 = self._vl_ifft2
        fftshift = self._vl_fftshift
        ifftshift = self._vl_ifftshift

        sample = np.asarray(sample, dtype=np.float32)
        ft_raw = fftshift(fft2(fftshift(sample)))

        if self._vl_model.get("fast_spectral_crop", True) and self._vl_model.get("crop_model", None) is not None:
            mask_crop = self._vl_model["mask_crop"]
            crop_model = self._vl_model["crop_model"]
            ft_crop = self._vl_apply_crop_model_to_ft(ft_raw, crop_model, mask_crop)
            obj_field = fftshift(ifft2(ifftshift(ft_crop))).astype(np.complex64, copy=False)
            ft_filtered_full = self._vl_expand_crop_to_full_spectrum(ft_raw, crop_model, mask_crop) if return_full_filtered_ft else None
            return obj_field, ft_filtered_full, ft_raw

        ft_filtered = ft_raw * self._vl_model["mask"]
        holo_filtered = fftshift(ifft2(ifftshift(ft_filtered)))
        obj_field = (holo_filtered * self._vl_model["ref_wave"]).astype(np.complex64, copy=False)

        return obj_field, ft_filtered if return_full_filtered_ft else None, ft_raw

    def _vl_estimate_legendre_phase_offset(self, field: np.ndarray, *, remove_piston: bool = True, use_pca: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate a low-order Legendre phase offset for cached reuse.

        The fit is deliberately restricted to the compact reconstruction field used for display. It is evaluated only during model refreshes, and acquisition frames reuse the corresponding complex correction instead of recomputing the polynomial or a complex exponential.
        """
        phase = np.angle(field).astype(np.float32, copy=False)

        try:
            phase_fit_source = unwrap_phase(phase).astype(np.float32, copy=False)
        except Exception:
            phase_fit_source = phase

        h, w = phase_fit_source.shape
        x = np.linspace(-1.0, 1.0, w, endpoint=True, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, h, endpoint=True, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")

        basis = [
            np.ones_like(X),
            X,
            Y,
            (3.0 * X ** 2 - 1.0) / 2.0,
            X * Y,
            (3.0 * Y ** 2 - 1.0) / 2.0,
        ]

        A = np.stack([b.reshape(-1) for b in basis], axis=1).astype(np.float32, copy=False)
        b = phase_fit_source.reshape(-1).astype(np.float32, copy=False)

        try:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception as exc:
            print(f"[Legendre] Fit skipped: {exc}")
            phase_offset = np.zeros_like(phase, dtype=np.float32)
            return field, phase_offset

        if remove_piston and coeffs.size > 0:
            coeffs[0] = 0.0

        wavefront = (A @ coeffs).reshape(h, w).astype(np.float32, copy=False)
        phase_offset = -wavefront
        compensated = (field * np.exp(1j * phase_offset)).astype(np.complex64, copy=False)

        return compensated, phase_offset

    def _crop_to_even_square(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Crops the input frame to a centered even square.
        """
        arr = image.astype(np.float64)
        h, w = arr.shape[:2]
        side = min(h, w)
        if side % 2 != 0:
            side -= 1
        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        square = arr[y0:y0 + side, x0:x0 + side]
        crop_info = {"x0": x0,"y0": y0,"side": side,"original_shape": image.shape,}
        return square, crop_info

    def _vl_extract_object_field(self,sample: np.ndarray,*,factor: float,rotate: bool,filter_type: str = "Circular") -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the Vortex-compensated object field.

        The +1 diffraction order is first located using vortexLegendre.spatial_filter().
        If filter_type is Circular, the original circular mask from vortexLegendre is used.
        If filter_type is Rectangular, a rectangular mask is rebuilt around the same
        detected diffraction order.
        """
        VL = self._VL
        fft2 = self._vl_fft2
        fftshift = self._vl_fftshift

        N, M = sample.shape

        fx_0 = M / 2.0
        fy_0 = N / 2.0

        m_grid, n_grid = np.meshgrid(
            np.arange(-M // 2, M // 2),
            np.arange(-N // 2, N // 2)
        )

        ft_raw, _holo_filtered_circular, fxm, fym, circular_mask = VL.spatial_filter(
            sample,
            M,
            N,
            save="No",
            factor=factor,
            rotate=rotate
        )

        distance = np.sqrt((fxm - fx_0) ** 2 + (fym - fy_0) ** 2)
        radius = distance / factor if distance > 1e-9 else max(10.0, min(N, M) * 0.08)

        if filter_type == "Rectangular":
            mask = self.rectangularMask(
                N,
                M,
                radius,
                int(round(fym)),
                int(round(fxm))
            ).astype(np.float32)
        else:
            mask = circular_mask.astype(np.float32)

        ft_filtered = ft_raw * mask
        holo_filtered = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(ft_filtered))
        )

        crop_vortex = 6
        if (
            fxm >= crop_vortex and fxm < M - crop_vortex and
            fym >= crop_vortex and fym < N - crop_vortex
        ):
            logamp = 10.0 * np.log10(
                np.abs(fftshift(fft2(fftshift(holo_filtered))) + 1e-6) ** 2
            )

            field_h = self._vl_median_filter(
                logamp,
                size=(1, 1),
                mode="reflect"
            )

            vortex_positions = VL.vortex_compensation(field_h, fxm, fym)
            if vortex_positions:
                fxm, fym = vortex_positions[0]

        ref_wave = VL.reference_wave(
            fxm,
            fym,
            m_grid,
            n_grid,
            self.lambda_um,
            self.dx_um,
            self.k,
            fx_0,
            fy_0,
            M,
            N,
            dy=self.dy_um
        )

        obj_field = ref_wave * holo_filtered

        return obj_field, ft_filtered
    
    def _vl_apply_legendre(self,field: np.ndarray,*,remove_piston: bool = True,use_pca: bool = True) -> np.ndarray:
        """
        Applies Legendre phase compensation to the Vortex-compensated field.
        """
        VL = self._VL
        n_rows, n_cols = field.shape
        limit = min(n_rows, n_cols) / 2.0
        phase_corrected, _coeffs = VL.legendre_compensation(field,limit,RemovePiston=remove_piston,UsePCA=use_pca)
        corrected_phase = np.angle(phase_corrected)
        if corrected_phase.shape != field.shape:
            corrected_phase = cv2.resize(corrected_phase.astype(np.float32),(field.shape[1], field.shape[0]),interpolation=cv2.INTER_LINEAR)
        compensated = np.abs(field) * np.exp(1j * corrected_phase)
        return compensated


    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """
        Linearly map a real-valued image to the 8-bit display range.

        Complex-valued arrays are converted to magnitude prior to normalization. Non-finite values are removed to guarantee stable display conversion.
        """
        arr = np.asarray(arr)

        if np.iscomplexobj(arr):
            arr = np.abs(arr)

        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr,nan=0.0,posinf=0.0,neginf=0.0)

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if abs(max_val - min_val) < 1e-12:
            return np.zeros(arr.shape,dtype=np.uint8)

        out = (arr - min_val) / (max_val - min_val)
        out = np.clip(out * 255.0,0.0,255.0)

        return out.astype(np.uint8)

    def _phase_to_uint8(self, phase: np.ndarray) -> np.ndarray:
        """
        Convert wrapped phase from radians to an 8-bit cyclic display interval.

        The input interval [-pi, pi] is mapped to [0, 255], preserving wrapped phase contrast for qualitative visualization.
        """
        phase = np.asarray(phase,dtype=np.float32)
        phase = np.nan_to_num(phase,nan=0.0,posinf=0.0,neginf=0.0)

        out = (phase + np.pi) / (2.0 * np.pi)
        out = np.clip(out * 255.0,0.0,255.0)

        return out.astype(np.uint8)

    def _vl_log_display(self, ft: np.ndarray) -> np.ndarray:
        """
        Convert a complex Fourier spectrum into an 8-bit logarithmic visualization.

        Logarithmic compression is used to retain visibility of weak diffraction components while avoiding saturation at the zero-order peak.
        """
        mag = np.log1p(np.abs(ft))
        return self._normalize_to_uint8(mag)


    def stop_compensation(self) -> None:
        """
        Stop acquisition and reconstruction and remove stale packets.

        The method does not block the graphical interface waiting for long thread termination. A short join is used only to reduce the probability of dangling workers while preserving interface responsiveness.
        """
        if hasattr(self, "_stop_compensation"):
            self._stop_compensation.set()

        if hasattr(self, "_safe_after_cancel"):
            self._safe_after_cancel("_comp_poll_loop_id")

        for thread_name in ("_capture_thread", "_recon_thread", "_comp_thread"):
            th = getattr(self, thread_name, None)
            if th is not None and th.is_alive():
                try:
                    th.join(timeout=0.05)
                except RuntimeError:
                    pass

        if hasattr(self, "_comp_queue"):
            while not self._comp_queue.empty():
                try:
                    self._comp_queue.get_nowait()
                except queue.Empty:
                    break

    def _comp_capture_loop(self, source: str) -> None:
        """
        Acquire frames for asynchronous preview and reconstruction.

        Video sources are treated as cyclic sequences: when the decoder reaches
        the final frame, the capture cursor is returned to frame zero and the
        acquisition index continues monotonically. This preserves reconstruction
        queue ordering while allowing continuous replay of finite videos.
        """
        frame_index = 0
        source_fps = float(getattr(self, "source_fps", 0.0) or 0.0)
        target_fps = source_fps if source_fps > 0.0 else float(getattr(self, "_target_preview_fps", 30.0))
        target_dt = 1.0 / max(target_fps, 1.0)

        while not self._stop_compensation.is_set():
            tic = time.perf_counter()

            if source == "video":
                ok, frame = self._read_video_frame_or_loop()
            else:
                ok, frame = self.cap.read()

            if not ok or frame is None:
                if source == "video":
                    self.after(0, self._handle_video_end)
                break

            gray = self._frame_to_gray(frame)

            packet = {"packet_type": "preview", "holo": gray, "frame_index": frame_index}

            if getattr(self, "holo_view_var", None) is not None and self.holo_view_var.get() == "Fourier Transform":
                packet["ft_unfiltered"] = self._fourier_display_from_hologram(gray)

            self._put_comp_packet(packet)

            with self._latest_recon_lock:
                self._latest_recon_frame = gray
                self._latest_recon_index = frame_index

            frame_index += 1
            elapsed = time.perf_counter() - tic
            sleep_time = target_dt - elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def _comp_reconstruction_loop(self) -> None:
        """Reconstruct the newest available frame and discard obsolete frames."""
        target_dt = 1.0 / max(float(getattr(self, "source_fps", getattr(self, "_target_reconstruction_fps", 30.0))), 1.0)

        while not self._stop_compensation.is_set():
            with self._latest_recon_lock:
                frame = self._latest_recon_frame
                frame_index = self._latest_recon_index

            if frame is None or frame_index <= self._processed_recon_index:
                time.sleep(0.003)
                continue

            tic = time.perf_counter()
            first = not self.first_frame_done

            try:
                data = self._compute_comp_arrays(frame,first=first)
                data["packet_type"] = "recon"
                data["frame_index"] = frame_index
                self._processed_recon_index = frame_index
                self.first_frame_done = True
                self._put_comp_packet(data)
            except Exception as exc:
                print("[Vortex-Legendre] Reconstruction loop failed:", exc)
                import traceback
                traceback.print_exc()

            elapsed = time.perf_counter() - tic
            sleep_time = target_dt - elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)


    def _put_comp_packet(self, packet: dict) -> None:
        """
        Insert a packet into the GUI queue using a drop-oldest policy.

        The queue is deliberately small to bound latency. When the graphical thread is temporarily busy, older packets are removed so that the next update corresponds to the most recent experimental state.
        """
        try:
            self._comp_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._comp_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._comp_queue.put_nowait(packet)
            except queue.Full:
                pass

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
        Refresh the graphical interface with the most recent preview and reconstruction packets.

        Multiple queued packets are collapsed into the newest preview and newest reconstruction packet. This reduces GUI workload and prevents old frames from being rendered after more recent data are already available.
        """
        latest_preview = None
        latest_recon = None
        packets_read = 0

        while packets_read < 20:
            try:
                data = self._comp_queue.get_nowait()
            except queue.Empty:
                break

            if data.get("packet_type") == "preview":
                latest_preview = data
            else:
                latest_recon = data

            packets_read += 1

        if latest_preview is None and latest_recon is None:
            if not self._stop_compensation.is_set():
                self._comp_poll_loop_id = self.after(10, self._poll_comp_queue)
            return

        if latest_preview is not None:
            self._apply_preview_packet(latest_preview)

        if latest_recon is not None:
            self._apply_reconstruction_packet(latest_recon)

        if not self._stop_compensation.is_set():
            self._comp_poll_loop_id = self.after(10, self._poll_comp_queue)

    def _apply_preview_packet(self, data: dict) -> None:
        """
        Apply a preview packet to the left viewer.

        The hologram cache is updated from the native frame. The unfiltered
        Fourier cache is either taken from the packet, when available, or
        recomputed from the native hologram. This prevents the Fourier viewer
        from displaying spectra associated with resized reconstruction fields.
        """
        holo = data["holo"]
        self.current_holo_array = holo
        self.last_preview_gray = holo

        holo_tk = self._preserve_aspect_ratio(
            Image.fromarray(holo),
            self.viewbox_width,
            self.viewbox_height
        )

        self.hologram_frames = [holo_tk]
        self.multi_holo_arrays = [holo]

        if "ft_unfiltered" in data:
            ft_unfiltered = data["ft_unfiltered"]
            self.current_ft_unfiltered_array = ft_unfiltered
            self.current_ft_unfiltered_tk = self._preserve_aspect_ratio(
                Image.fromarray(ft_unfiltered),
                self.viewbox_width,
                self.viewbox_height
            )
            self.ft_frames = [self.current_ft_unfiltered_tk]
            self.multi_ft_arrays = [ft_unfiltered]

            if self.ft_display_var.get() == "unfiltered":
                self.current_ft_array = ft_unfiltered

        elif self.holo_view_var.get() == "Fourier Transform":
            self._update_unfiltered_ft_from_hologram(holo)

        if self.holo_view_var.get() == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk

        elif self.holo_view_var.get() == "Fourier Transform":
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        if getattr(self, "is_recording", False) and self.target_to_record == "Hologram":
            self.buff_holo.append(holo.copy())

        self._mark_hologram_frame_displayed()


    def _apply_reconstruction_packet(self, data: dict) -> None:
        """
        Apply a reconstruction packet to the right viewer and update the filtered
        Fourier cache only when the reconstruction worker actually sent one.

        In the hybrid fast mode, the reconstruction thread skips full filtered-FT
        packet generation. If the user switches to the Fourier viewer, the full
        filtered FT is recomputed on demand from the current native hologram.
        """
        self.current_amplitude_array = data["amp"]
        self.current_phase_array = data["phase"]

        ft_arr = data.get("ft", None)
        if ft_arr is not None:
            self.current_ft_filtered_array = ft_arr
            self.current_reconstruction_ft_array = ft_arr

            ft_tk = self._preserve_aspect_ratio(
                Image.fromarray(ft_arr),
                self.viewbox_width,
                self.viewbox_height,
            )
            self.current_ft_filtered_tk = ft_tk
            self.current_reconstruction_ft_tk = ft_tk

        amp_tk = self._preserve_aspect_ratio_right(Image.fromarray(data["amp"]))
        pha_tk = self._preserve_aspect_ratio_right(Image.fromarray(data["phase"]))

        self.amplitude_frames = [amp_tk]
        self.phase_frames = [pha_tk]
        self.amplitude_arrays = [data["amp"]]
        self.phase_arrays = [data["phase"]]

        if self.holo_view_var.get() == "Fourier Transform":
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.processed_label.configure(image=amp_tk)
            self.processed_label.image = amp_tk
        else:
            self.processed_label.configure(image=pha_tk)
            self.processed_label.image = pha_tk

        if getattr(self, "is_recording", False):
            if self.target_to_record == "Amplitude":
                self.buff_amp.append(data["amp"].copy())
            elif self.target_to_record == "Phase":
                self.buff_phase.append(data["phase"].copy())

        self._mark_reconstruction_frame_displayed()

    def start_compensation(self) -> None:
        """Start the real-time reconstruction engine."""
        lam, dx, dy = self._get_pc_parameter_values()
        if lam is None or dx is None or dy is None:
            return

        self.lambda_um = float(lam)
        self.dx_um = float(dx)
        self.dy_um = float(dy)
        self.k = 2.0 * math.pi / self.lambda_um
        self.selected_filter_type = self.spatial_filter_var_pc.get().strip()
        self.vl_settings = self._get_vortex_legendre_default_settings()

        self.stop_compensation()
        self.preview_active = False
        self.video_playing = False

        if getattr(self, "is_video_preview", False):
            if not getattr(self, "cap", None):
                if not self._reopen_video_capture_from_path():
                    tk.messagebox.showerror("Video", "Video source unavailable.")
                    return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._set_source_fps(self._read_capture_fps(self.cap, fallback=self.source_fps), fallback=self.source_fps)
            src = "video"
        else:
            if not self._ensure_camera():
                tk.messagebox.showerror("Camera", "Camera unavailable.")
                return
            self._set_source_fps(self._read_capture_fps(self.cap, fallback=float(getattr(self, "camera_target_fps", 30.0))), fallback=float(getattr(self, "camera_target_fps", 30.0)))
            src = "camera"

        self._reset_fps_measurements()
        self._stop_compensation.clear()
        self.first_frame_done = False
        self.is_playing = True
        self.play_button.configure(text="⏸ Pause")

        self._vl_model = None
        self._vl_frame_counter = 0
        self._vl_last_valid_data = None
        self._vl_last_valid_field = None
        self._latest_recon_frame = None
        self._latest_recon_index = -1
        self._processed_recon_index = -1
        self._force_vl_model_refresh = True

        if hasattr(self, "reconstruction_mode_label"):
            self.reconstruction_mode_label.configure(text=f"Mode: {self.reconstruction_mode_var.get()} | {self.correction_mode_var.get()}")

        self._capture_thread = threading.Thread(target=self._comp_capture_loop, args=(src,), daemon=True)
        self._recon_thread = threading.Thread(target=self._comp_reconstruction_loop, args=(), daemon=True)

        self._capture_thread.start()
        self._recon_thread.start()
        self._comp_poll_loop_id = self.after(10, self._poll_comp_queue)

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
        """
        Update the left viewer during video mode.

        Switching to the Fourier view forces a fresh Fourier computation from
        the currently stored native hologram. This avoids stale spectra and
        prevents the viewer from reusing a Fourier image generated from a
        previous video frame, camera frame, or reconstruction crop.
        """
        choice = self.holo_view_var.get()

        if not hasattr(self, "current_holo_array") or self.current_holo_array is None:
            if choice == "Hologram":
                self.captured_title_label.configure(text="Hologram")
                self.captured_label.configure(image=self.img_hologram)
            else:
                self.captured_title_label.configure(text="Fourier Transform")
                self.captured_label.configure(image=self.img_ft)
            return

        if choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            holo_tk = self._preserve_aspect_ratio(
                Image.fromarray(self.current_holo_array),
                self.viewbox_width,
                self.viewbox_height
            )
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk

        elif choice == "Fourier Transform":
            self._update_unfiltered_ft_from_hologram(self.current_holo_array)
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

    def _on_ft_mode_changed(self):
        """
        Recompute the Fourier display when the visualization scaling changes.

        The recalculation is performed from the current native hologram instead
        of reusing the previous display array. This preserves the original
        spectral geometry while allowing the user to switch between logarithmic
        and linear magnitude visualization.
        """
        if self.holo_view_var.get() != "Fourier Transform":
            return

        if getattr(self, "current_holo_array", None) is not None:
            self._update_unfiltered_ft_from_hologram(self.current_holo_array)

        self._refresh_ft_display()

    def _show_amp_mode_menu(self):
        menu = tk.Menu(self, tearoff=0)
        opts = ["Amplitude"]
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
        self._reset_fps_measurements()
        self.preview_active = True
        self._update_preview()

    def _update_preview(self) -> None:
        """
        Preview the active camera without altering the hologram sampling grid.

        The hologram is stored exactly as delivered by the acquisition backend.
        When Fourier visualization is requested, the transform is computed from
        the complete frame rather than from a reduced preview image.
        """
        if not self.preview_active:
            return

        if not getattr(self, "cap", None):
            self.preview_active = False
            return

        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            self._preview_loop_id = self.after(self._source_delay_ms(t0), self._update_preview)
            return

        gray = self._frame_to_gray(frame)

        if getattr(self, "is_recording", False) and self.target_to_record == "Hologram":
            self.buff_holo.append(gray.copy())

        self.last_preview_gray = gray
        self.current_holo_array = gray
        self.current_left_index = 0

        holo_tk = self._preserve_aspect_ratio(
            Image.fromarray(gray),
            self.viewbox_width,
            self.viewbox_height
        )

        self.hologram_frames = [holo_tk]
        self.multi_holo_arrays = [gray]

        if self.holo_view_var.get() == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=holo_tk)
            self.captured_label.image = holo_tk

        else:
            self._update_unfiltered_ft_from_hologram(gray)
            self.captured_title_label.configure(text="Fourier Transform")
            self._refresh_ft_display()

        if self.sequence_recording:
            self._save_sequence_frame(gray)

        self._mark_hologram_frame_displayed()
        self._preview_loop_id = self.after(self._source_delay_ms(t0), self._update_preview)

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
        cap.set(cv2.CAP_PROP_FPS, float(getattr(self, "camera_target_fps", 30.0)))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Test if the camera delivers a frame
        ok, first = cap.read()
        if not ok:
            cap.release()
            print("[Camera] Camera opened but delivers no frames.")
            self._show_camera_error_once("Camera opened but delivers no frames.")
            return None

        # Everything OK, save camera
        self.cap = cap
        self.source_mode = "camera"
        self.selected_camera_index = preferred
        self.first_frame_done = False
        self.video_buffer_rec = []
        self.video_buffer_raw = []
        self.start_time_fps = time.time()
        self.frame_counter_fps = 0
        self._set_source_fps(self._read_capture_fps(cap, fallback=float(getattr(self, "camera_target_fps", 30.0))), fallback=float(getattr(self, "camera_target_fps", 30.0)))
        self._reset_fps_measurements()

        if not getattr(self, "_camera_success_shown", False):
            messagebox.showinfo(
                "Information",
                f"Using device index {preferred} – resolution {first.shape[1]}×{first.shape[0]} – nominal FPS {self.source_fps:.1f}"
            )
            self._camera_success_shown = True

        return self.cap

    def start_preview_stream(self) -> None:
        """
        Start camera preview using the nominal source period and reset display-rate measurements.
        """
        if not self._ensure_camera():
            messagebox.showerror("Camera error", "No active camera was found.")
            return
        if self.preview_active:
            return
        self._reset_fps_measurements()
        self.preview_active = True
        self.video_playing = False
        self.is_playing = True
        if hasattr(self, "play_button"):
            self.play_button.configure(text="⏸ Pause")
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
        """
        Display amplitude and phase in a fixed-size viewer canvas.

        The previous version returned the raw image when it was smaller than the
        panel, so a reconstruction computed from a smaller spectral crop looked
        physically smaller on screen. This wrapper always uses the same canvas
        size as the left viewer, preserving aspect ratio and centering the image.
        """
        return self._preserve_aspect_ratio(
            pil_image,
            self.viewbox_width,
            self.viewbox_height,
        )

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
        """
        Update the left viewer for static or already cached image data.

        The Fourier view is regenerated from the selected hologram array when
        possible. This maintains a strict relationship between the hologram
        cache and the displayed unfiltered spectrum.
        """
        choice = self.holo_view_var.get()

        if not hasattr(self, "hologram_frames") or len(self.hologram_frames) == 0:
            if choice == "Hologram":
                self.captured_title_label.configure(text="Hologram")
                self.captured_label.configure(image=self.img_hologram)
                self.current_holo_array = self.arr_hologram
            else:
                self.captured_title_label.configure(text="Fourier Transform")
                self.current_holo_array = self.arr_hologram
                self._update_unfiltered_ft_from_hologram(self.current_holo_array)
                self._refresh_ft_display()
            return

        if choice == "Hologram":
            self.captured_title_label.configure(text="Hologram")
            self.captured_label.configure(image=self.hologram_frames[self.current_left_index])
            self.captured_label.image = self.hologram_frames[self.current_left_index]
            self.current_holo_array = self.multi_holo_arrays[self.current_left_index]

        else:
            self.captured_title_label.configure(text="Fourier Transform")

            if self.multi_holo_arrays and self.current_left_index < len(self.multi_holo_arrays):
                self.current_holo_array = self.multi_holo_arrays[self.current_left_index]
                self._update_unfiltered_ft_from_hologram(self.current_holo_array)
            elif self.current_holo_array is not None:
                self._update_unfiltered_ft_from_hologram(self.current_holo_array)

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
            variable=self.spatial_filter_var_pc,
            command=self._on_spatial_filter_geometry_changed
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
        self.compensate_frame = ctk.CTkFrame(self.phase_compensation_inner_frame, width=PARAMETER_FRAME_WIDTH, height=175)
        self.compensate_frame.grid(row=3, column=0, sticky="ew", pady=(2, 6))
        self.compensate_frame.grid_propagate(False)
        for col in (0, 1, 2, 3):
            self.compensate_frame.columnconfigure(col, weight=1)

        ctk.CTkLabel(self.compensate_frame, text="Compensation Controls", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, padx=10, pady=(8, 4), sticky="w")

        self.compensate_button = ctk.CTkButton(self.compensate_frame, text="⚙ Compensate", width=120, command=self.start_compensation)
        self.compensate_button.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

        self.playstop_frame = ctk.CTkFrame(self.compensate_frame, fg_color="transparent")
        self.playstop_frame.grid(row=1, column=1, columnspan=3, sticky="e", padx=10, pady=(0, 6))

        self.play_button = ctk.CTkButton(self.playstop_frame, text="▶ Play", width=80, command=self._on_play)
        self.play_button.pack(side="left", padx=(0, 5))

        self.stop_button = ctk.CTkButton(self.playstop_frame, text="⏹ Stop", width=80, command=self._on_stop)
        self.stop_button.pack(side="left")

        self.reconstruction_mode_label = ctk.CTkLabel(self.compensate_frame, text=f"Mode: {self.reconstruction_mode_var.get()} | {self.correction_mode_var.get()}", font=ctk.CTkFont(size=12, weight="bold"))
        self.reconstruction_mode_label.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 4), sticky="w")

        self.alignment_button = ctk.CTkButton(self.compensate_frame, text="Alignment Mode", width=105, command=self._set_alignment_mode)
        self.alignment_button.grid(row=3, column=0, padx=(10, 4), pady=(0, 6), sticky="ew")

        self.acquisition_button = ctk.CTkButton(self.compensate_frame, text="Acquisition Mode", width=115, command=self._set_acquisition_mode)
        self.acquisition_button.grid(row=3, column=1, padx=4, pady=(0, 6), sticky="ew")

        self.align_once_button = ctk.CTkButton(self.compensate_frame, text="Align Once", width=90, command=self._force_alignment_update)
        self.align_once_button.grid(row=3, column=2, padx=4, pady=(0, 6), sticky="ew")

        self.correction_mode_menu = ctk.CTkOptionMenu(self.compensate_frame, values=["Vortex Only", "Vortex + Cached Legendre"], variable=self.correction_mode_var, command=self._on_reconstruction_correction_changed, width=160)
        self.correction_mode_menu.grid(row=4, column=0, columnspan=4, padx=10, pady=(0, 8), sticky="ew")

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
        """
        Stop acquisition, preview, and reconstruction, then release the active capture source.
        """
        self.preview_active = False
        self.video_playing = False
        self.realtime_active = False
        self.is_playing = False

        if hasattr(self, "play_button"):
            self.play_button.configure(text="▶ Play")

        self.stop_compensation()
        self._cancel_timed_loops()

        if hasattr(self, "cap") and self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.source_mode == "video":
            self.is_video_preview = bool(getattr(self, "video_file_path", ""))

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
        main_mod = import_module("holobio.Main_")
        reload(main_mod)

        MainMenu = getattr(main_mod, "MainMenu")
        MainMenu().mainloop()

    def _hide_parameters_nav_button(self) -> None:
        if hasattr(self, "param_button"):
            self.param_button.destroy()
        self.change_menu_to("parameters")


    def _refresh_ft_display(self):
        """
        Refresh the Fourier viewer using the selected spectral representation.

        The unfiltered and filtered Fourier views are tied to the full native hologram.
        Reconstruction-domain spectra are not used for the left Fourier viewer
        because cropped or resized numerical fields alter spectral geometry.
        """
        if self.holo_view_var.get() != "Fourier Transform":
            return

        show_filtered = self.ft_display_var.get() == "filtered"
        img_tk = None
        arr = None

        if show_filtered:
            if getattr(self, "current_holo_array", None) is not None:
                self._update_filtered_ft_from_hologram(self.current_holo_array)
            if hasattr(self, "current_ft_filtered_tk"):
                img_tk = self.current_ft_filtered_tk
                arr = getattr(self, "current_ft_filtered_array", None)

        else:
            if getattr(self, "current_holo_array", None) is not None:
                self._update_unfiltered_ft_from_hologram(self.current_holo_array)

            if hasattr(self, "current_ft_unfiltered_tk"):
                img_tk = self.current_ft_unfiltered_tk
                arr = getattr(self, "current_ft_unfiltered_array", None)

            elif hasattr(self, "ft_frames") and self.ft_frames:
                img_tk = self.ft_frames[self.current_left_index]
                arr = self.multi_ft_arrays[self.current_left_index] if self.multi_ft_arrays else None

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
        filtered_ft: np.ndarray
    ) -> None:
        """
        Reconstruct amplitude and phase while preserving the hologram FT cache.

        The amplitude and phase reconstructions are derived from the filtered
        diffraction order. The unfiltered Fourier display, however, is always
        computed from the full native hologram to avoid substituting a cropped,
        filtered, or reduced reconstruction spectrum into the left diagnostic
        viewer.
        """
        M, N = hologram_gray.shape[1], hologram_gray.shape[0]
        fx = self.fx[0] if isinstance(self.fx, np.ndarray) else self.fx
        fy = self.fy[0] if isinstance(self.fy, np.ndarray) else self.fy

        theta_x_arg = (M / 2 - fx) * self.lambda_um / (M * self.dx_um)
        theta_y_arg = (N / 2 - fy) * self.lambda_um / (N * self.dy_um)

        theta_x_arg = float(np.clip(theta_x_arg, -1.0, 1.0))
        theta_y_arg = float(np.clip(theta_y_arg, -1.0, 1.0))

        theta_x = np.arcsin(theta_x_arg)
        theta_y = np.arcsin(theta_y_arg)

        carrier = np.exp(
            1j * self.k * (
                np.sin(theta_x) * self.m_mesh * self.dx_um +
                np.sin(theta_y) * self.n_mesh * self.dy_um
            )
        )

        field = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(filtered_ft))) * carrier
        amplitude_raw = np.abs(field)
        phase_raw = np.angle(field)

        holo_u8 = self._normalize_to_uint8(hologram_gray)
        amp_u8 = self._normalize_to_uint8(amplitude_raw)
        phase_u8 = self._phase_to_uint8(phase_raw)

        ft_unf = self._fourier_display_from_hologram(hologram_gray)
        ft_fil = self._vl_log_display(filtered_ft)

        if getattr(self, "is_recording", False):
            if self.target_to_record == "Amplitude":
                self.buff_amp.append(amp_u8.copy())
            elif self.target_to_record == "Phase":
                self.buff_phase.append(phase_u8.copy())
            elif self.target_to_record == "Hologram":
                self.buff_holo.append(holo_u8.copy())

        self.current_holo_array = hologram_gray
        self.current_amplitude_array = amp_u8
        self.current_phase_array = phase_u8
        self.current_ft_unfiltered_array = ft_unf
        self.current_ft_filtered_array = ft_fil

        tk_holo = self._preserve_aspect_ratio(
            Image.fromarray(holo_u8),
            self.viewbox_width,
            self.viewbox_height
        )
        tk_ft_un = self._preserve_aspect_ratio(
            Image.fromarray(ft_unf),
            self.viewbox_width,
            self.viewbox_height
        )
        tk_ft_fi = self._preserve_aspect_ratio(
            Image.fromarray(ft_fil),
            self.viewbox_width,
            self.viewbox_height
        )
        tk_amp = self._preserve_aspect_ratio_right(Image.fromarray(amp_u8))
        tk_phase = self._preserve_aspect_ratio_right(Image.fromarray(phase_u8))

        self.current_ft_unfiltered_tk = tk_ft_un
        self.current_ft_filtered_tk = tk_ft_fi

        self.hologram_frames = [tk_holo]
        self.ft_frames = [tk_ft_un]
        self.multi_holo_arrays = [hologram_gray]
        self.multi_ft_arrays = [ft_unf]
        self.amplitude_frames = [tk_amp]
        self.phase_frames = [tk_phase]
        self.amplitude_arrays = [amp_u8]
        self.phase_arrays = [phase_u8]

        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=tk_holo)
            self.captured_label.image = tk_holo
        else:
            self._refresh_ft_display()

        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.processed_label.configure(image=tk_amp)
            self.processed_label.image = tk_amp
        else:
            self.processed_label.configure(image=tk_phase)
            self.processed_label.image = tk_phase

        self.current_ft_array = (
            self.current_ft_filtered_array
            if self.ft_display_var.get() == "filtered"
            else self.current_ft_unfiltered_array
        ) 
        
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
 
if __name__ == "__main__":
    app = App()
    app.mainloop()