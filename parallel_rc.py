import numpy as np
import time
from customtkinter import CTkImage
from multiprocessing import Queue
import cv2 as cv
from settings import *
from PIL import Image
import hashlib
from functools import lru_cache


def dlhm_rec(hologram, L, z, W_c, dx_out, wavelength):
    N, M = hologram.shape

    # The discretization on the camera is set as the same input's discretization
    P = M
    Q = N

    # Magnification factor
    Mag = L / z

    Mag_max = np.sqrt(W_c ** 2 / 2 + L ** 2) / z
    Dist_max = np.abs(Mag_max - Mag)
    # Apply distortion to the hologram
    camMat = np.array([[P, 0, P / 2], [0, Q, Q / 2], [0, 0, 1]])
    distCoeffs = np.array([Dist_max / (2 * Mag), 0, 0, 0, 0])  # Radial distortion parameters
    hologram = cv.undistort(hologram.astype(np.float32), camMat, distCoeffs)

    # Wave number
    k = 2 * np.pi / wavelength

    # Spatial coordinates in the camera's plane
    x = np.linspace(-W_c / 2, W_c / 2, P)
    y = np.linspace(-W_c / 2, W_c / 2, Q)

    # Spatial frequency coordinates at the sample's plane
    dfx = Mag / (dx_out * M)
    dfy = Mag / (dx_out * N)
    fx, fy = np.meshgrid(np.arange(-M / 2 * dfx, M / 2 * dfx, dfx),
                         np.arange(-N / 2 * dfy, N / 2 * dfy, dfy))

    # Propagation kernel for the Angular Spectrum Method (ASM)
    # E = np.exp(1j * (L - z) * np.sqrt(k ** 2 - 4 * np.pi ** 2 * (fx ** 2 + fy ** 2)))
    arg = k ** 2 - 4 * np.pi ** 2 * (fx ** 2 + fy ** 2)
    # for negative values, keep as complex instead of NaN
    E = np.exp(1j * (L - z) * np.sqrt(arg.astype(np.complex128)))
    # Compute hologram using inverse Fourier transform
    Uz = ifts(fts(hologram) * E)

    return np.abs(Uz), np.angle(np.conj(Uz))


def ifts(A):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(A)))


def fts(A):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A)))


def filtcosenoF(par, fi, num_fig):
    # Coordinates
    Xfc, Yfc = np.meshgrid(np.linspace(-fi / 2, fi / 2, fi), np.linspace(fi / 2, -fi / 2, fi))
    # Normalize coordinates [-π,π] and create horizontal and vertical filters
    FC1 = np.cos(Xfc * (np.pi / par) * (1 / Xfc.max())) ** 2
    FC2 = np.cos(Yfc * (np.pi / par) * (1 / Yfc.max())) ** 2
    # Intersection
    FC = (FC1 > 0) * (FC1) * (FC2 > 0) * (FC2)
    # Rescale
    FC = FC / FC.max()
    if num_fig != 0:
        fig = px.imshow(FC)
        fig.show()
    return FC


def prepairholoF(CH_m, xop, yop, Xp, Yp):
    # User function to prepare the hologram using nearest neihgboor interpolation strategy
    [row, a] = CH_m.shape
    # New coordinates measured in units of the -2*xop/row pixel size
    Xcoord = (Xp - xop) / (-2 * xop / row)
    Ycoord = (Yp - yop) / (-2 * xop / row)
    # Find lowest integer
    iXcoord = np.floor(Xcoord)
    iYcoord = np.floor(Ycoord)
    # Assure there isn't null pixel positions
    iXcoord[iXcoord == 0] = 1
    iYcoord[iYcoord == 0] = 1
    # Calculate the fractionating for interpolation
    x1frac = (iXcoord + 1.0) - Xcoord  # Upper value to integer
    x2frac = 1.0 - x1frac
    y1frac = (iYcoord + 1.0) - Ycoord  # Lower value to integer
    y2frac = 1.0 - y1frac
    x1y1 = x1frac * y1frac  # Corresponding pixel areas for each direction
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac
    # Pre allocate the prepared hologram
    CHp_m = np.zeros([row, row])
    # Prepare hologram (preparation - every pixel remapping)
    for it in range(0, row - 2):
        for jt in range(0, row - 2):
            CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] + (
                x1y1[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]), int(
                iXcoord[it, jt]) + 1] + (x2y1[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]) + 1, int(
                iXcoord[it, jt])] + (x1y2[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]) + 1, int(
                iXcoord[it, jt]) + 1] + (x2y2[it, jt]) * CH_m[it, jt]

    return CHp_m


def kreuzer3F(z, field, wavelength, pixel_pitch_in, pixel_pitch_out, L, FC):
    dx = pixel_pitch_in
    dX = pixel_pitch_out
    # Squared pixels
    deltaY = dX
    # Matrix size
    [row, a] = field.shape
    # Parameters
    k = 2 * np.pi / wavelength
    W = dx * row
    #  Matrix coordinates
    delta = np.linspace(1, row, num=row)
    [X, Y] = np.meshgrid(delta, delta)
    # Hologram origin coordinates
    xo = -W / 2
    yo = -W / 2
    # Prepared hologram, coordinates origin
    xop = xo * L / np.sqrt(L ** 2 + xo ** 2)
    yop = yo * L / np.sqrt(L ** 2 + yo ** 2)
    # Pixel size for the prepared hologram (squared)
    deltaxp = xop / (-row / 2)
    deltayp = deltaxp
    # Coordinates origin for the reconstruction plane
    Yo = -dX * row / 2
    Xo = -dX * row / 2
    Xp = (dx * (X - row / 2) * L / (np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)))
    Yp = (dx * (Y - row / 2) * L / (np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)))
    # Preparation of the hologram
    CHp_m = prepairholoF(field, xop, yop, Xp, Yp)
    # Multiply prepared hologram with propagation phase
    Rp = np.sqrt((L ** 2) - (deltaxp * X + xop) ** 2 - (deltayp * Y + yop) ** 2)
    r = np.sqrt((dX ** 2) * ((X - row / 2) ** 2 + (Y - row / 2) ** 2) + z ** 2)
    CHp_m = CHp_m * ((L / Rp) ** 4) * np.exp(-0.5 * 1j * k * (r ** 2 - 2 * z * L) * Rp / (L ** 2))
    # Padding constant value
    pad = int(row / 2)
    # Padding on the cosine rowlter
    FC = np.pad(FC, (int(pad), int(pad)))
    # Convolution operation
    # First transform
    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (
                2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + X ** 2 * deltaxp * dX + Y ** 2 * deltayp * deltaY))
    T1 = np.pad(T1, (int(pad), int(pad)))
    T1 = fts(T1 * FC)
    # Second transform
    T2 = np.exp(-1j * (k / (2 * L)) * ((X - row / 2) ** 2 * deltaxp * dX + (Y - row / 2) ** 2 * deltayp * deltaY))
    T2 = np.pad(T2, (int(pad), int(pad)))
    T2 = fts(T2 * FC)
    # Third transform
    K = ifts(T2 * T1)
    K = K[pad + 1:pad + row, pad + 1: pad + row]

    return K


def read(filename: str, path: str = '') -> np.ndarray:
    '''Reads image to double precision 2D array.'''
    if path != '':
        prefix = path + '\x5c'
    else:
        prefix = ''
    im = cv.imread(prefix + filename, cv.IMREAD_GRAYSCALE)  # you can pass multiple arguments in single line
    return im.astype(np.float64)


def normalize(x: np.ndarray, scale: float) -> np.ndarray:
    '''Normalize every value of an array to the 0-scale interval.'''
    x = x.astype(np.float64)
    min_val = np.min(x)
    x = x - min_val
    max_val = np.max(x) if np.max(x) != 0 else 1
    normalized_image = scale * x / max_val
    return normalized_image


# Function to propagate an optical field using the Angular Spectrum approach
def propagate(field, z, wavelength, dx, dy, scale_factor=1):
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')
    dfx = 1 / (dx * N)
    dfy = 1 / (dy * M)
    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
    kernel = np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j
    phase = np.exp(1j * z * scale_factor * 2 * np.pi * np.sqrt(kernel))
    tmp = field_spec * phase
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
    return out


def im2arr(path: str):
    '''Converts file image into numpy array.'''
    return np.asarray(Image.open(path).convert('L'))


def arr2im(array: np.ndarray):
    '''Converts numpy array into PhotoImage type'''
    return Image.fromarray(array.astype(np.uint8), 'L')


def create_image(img: Image.Image, width, height):
    '''Converts image into type usable by customtkinter'''
    return CTkImage(light_image=img, dark_image=img, size=(width, height))


def _to_float_image(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64, copy=False) if arr.dtype != np.float64 else arr


def gamma_filter(arr: np.ndarray, gamma: float) -> np.ndarray:
    x = _to_float_image(arr)
    g = max(float(gamma), 1e-8)
    y = np.power(x / (x.max() + 1e-9), 1.0 / g) * 255.0
    return np.clip(y, 0, 255)


def contrast_filter(arr: np.ndarray, contrast: float) -> np.ndarray:
    x = _to_float_image(arr)
    c = float(contrast)
    m = np.mean(x)
    y = (x - m) * c + m
    return np.clip(y, 0, 255)


def _ideal_mask(shape: tuple[int, int], cutoff: float, pass_type: str) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    radius = int(min(rows, cols) * float(cutoff) * 0.5)
    radius = max(radius, 0)
    Y, X = np.ogrid[:rows, :cols]
    dist_sq = (X - ccol) ** 2 + (Y - crow) ** 2
    if pass_type == 'high':
        mask = np.ones((rows, cols), np.uint8)
        if radius > 0:
            mask[dist_sq <= radius ** 2] = 0
    else:
        mask = np.zeros((rows, cols), np.uint8)
        if radius > 0:
            mask[dist_sq <= radius ** 2] = 1
    return mask


def highpass_filter(arr: np.ndarray, cutoff: float) -> np.ndarray:
    x = _to_float_image(arr)
    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    mask = _ideal_mask(x.shape, float(cutoff), pass_type='high')
    fshift = fshift * mask
    y = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return np.clip(y, 0, 255)


def lowpass_filter(arr: np.ndarray, cutoff: float) -> np.ndarray:
    x = _to_float_image(arr)
    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    mask = _ideal_mask(x.shape, float(cutoff), pass_type='low')
    fshift = fshift * mask
    y = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return np.clip(y, 0, 255)


def adaptative_eq_filter(arr: np.ndarray, _unused_param) -> np.ndarray:
    x = _to_float_image(arr)
    arr_min, arr_max = x.min(), x.max()
    rng = (arr_max - arr_min) + 1e-9
    scaled = (x - arr_min) / rng
    hist, bins = np.histogram(scaled.flatten(), 256, [0.0, 1.0])
    cdf = hist.cumsum().astype(np.float64)
    cdf /= (cdf[-1] + 1e-12)
    eq = np.interp(scaled.flatten(), bins[:-1], cdf)
    y = eq.reshape(x.shape) * (arr_max - arr_min) + arr_min
    return np.clip(y, 0, 255)


def capture(queue_manager: dict[dict[Queue, Queue], dict[Queue, Queue], dict[Queue, Queue]]):
    filter_dict = {'gamma': gamma_filter,
                   'contrast': contrast_filter,
                   'adaptative_eq': adaptative_eq_filter,
                   'highpass': highpass_filter,
                   'lowpass': lowpass_filter}

    input_dict = {'path': None,
                  'reference path': None,
                  'settings': None,
                  'filters': None,
                  'filter': None}

    output_dict = {'image': None,
                   'filtered': None,
                   'fps': None,
                   'size': None}

    # Initialize camera (0 by default most of the time means the integrated camera)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Verify that the camera opened correctly
    if not cap.isOpened():
        print("Is not posible to open the camara.")
        exit()

    # Sets the camera resolution to the closest chose in settings
    cap.set(cv.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)

    # Gets the actual resolution of the image
    width_ = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height_ = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print(f'Width: {width_}')
    print(f'Height: {height_}')

    while True:
        init_time = time.time()
        img = cv.cvtColor(cap.read()[1], cv.COLOR_BGR2GRAY)
        img = cv.flip(img, 1)
        filt_img = img

        height_, width_ = img.shape

        if not queue_manager['capture']['input'].empty():
            input = queue_manager['capture']['input'].get()

            for key in input_dict.keys():
                input_dict[key] = input[key]

        if input_dict['path']:
            img = im2arr(input_dict['path'])
            filt_img = img

            # Gets the actual resolution of the image
            height_, width_ = img.shape

        if input_dict['reference path']:
            ref = im2arr(input_dict['reference path'])
            if img.shape == ref.shape:
                img = img - ref
            else:
                print('Image sizes do not match')

            filt_img = img

        if input_dict['settings']:
            open_camera_settings(cap)

        if input_dict['filters']:
            filter_functions = input_dict['filters'][0]
            filter_params = input_dict['filters'][1]

            if input_dict['filter']:
                for filter, param, in zip(filter_functions, filter_params):
                    filt_img = filter_dict[filter](filt_img, param)

        filt_img = arr2im(filt_img.astype(np.uint8))
        filt_img = create_image(filt_img, width_, height_)

        end_time = time.time()

        elapsed_time = end_time - init_time
        fps = round(1 / elapsed_time, 1) if elapsed_time != 0 else 0

        if not queue_manager['capture']['output'].full():
            output_dict['image'] = img
            output_dict['filtered'] = filt_img
            output_dict['fps'] = fps
            output_dict['size'] = (width_, height_)

            queue_manager['capture']['output'].put(output_dict)


def open_camera_settings(cap):
    try:
        cap.set(cv.CAP_PROP_SETTINGS, 0)
    except:
        print('Cannot access camera settings.')


def _hash_array(arr: np.ndarray) -> str:
    """Fast, deterministic hash of a numpy array (content & shape)."""
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()


@lru_cache(maxsize=16)
def _precompute_kernel(shape: tuple[int, int], wavelength: float, dx: float, dy: float, scale: float) -> np.ndarray:
    M, N = shape
    x = np.arange(N) - N / 2
    y = np.arange(M) - M / 2
    X, Y = np.meshgrid(x, y, indexing="xy")
    dfx = 1.0 / (dx * N)
    dfy = 1.0 / (dy * M)
    return 2 * np.pi * np.sqrt((1.0 / wavelength) ** 2 - (X * dfx) ** 2 - (Y * dfy) ** 2) * scale


def _propagate_cached(field_spec: np.ndarray, z: float, wavelength: float, dx: float, dy: float,
                      scale: float) -> np.ndarray:
    kernel = _precompute_kernel(field_spec.shape, wavelength, dx, dy, scale)
    phase = np.exp(1j * z * kernel)  # e^{j·z·2π·scale·√(...)}
    tmp = field_spec * phase  # ∘ multiply in the spectrum
    tmp = np.fft.ifftshift(tmp)  # ∘ back to origin
    out = np.fft.ifft2(tmp)  # ∘ inverse FT
    out = np.fft.ifftshift(out)  # ∘ re-centre
    return out


def _compute_spectrum(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))


def reconstruct(queue_manager: dict[str, dict[str, Queue]]) -> None:
    last_holo_hash = None
    cached_spec = None
    cached_ft = None

    while True:
        inp = queue_manager["reconstruction"]["input"].get()

        holo_u8 = inp["image"].astype(np.float64)
        algorithm = inp["algorithm"]
        L, Z, r = inp["L"], inp["Z"], inp["r"]
        wl, dxy = inp["wavelength"], inp["dxy"]
        scale = inp["scale_factor"]
        deltaX = Z * dxy / L
        t0 = time.time()
        this_hash = _hash_array(holo_u8)
        # print(wl,dxy,L,Z)

        # build & cache spectrum only when the hologram changes

        if this_hash != last_holo_hash:
            cached_spec = _compute_spectrum(holo_u8)
            ft_cplx = _compute_spectrum(holo_u8)
            cached_ft = normalize(np.log1p(np.abs(ft_cplx)), 255).astype(np.uint8)
            last_holo_hash = this_hash
            M, N = cached_spec.shape

        # pick algorithm
        if algorithm == "AS":
            recon_c = propagate(holo_u8, r, wl, dxy, dxy, scale)
            amp_f = np.abs(recon_c)
            phase_f = np.angle(recon_c)


        elif algorithm == "KR":
            s = min(M, N)
            y0 = (M - s) // 2
            x0 = (N - s) // 2
            holo_sq = holo_u8[y0:y0 + s, x0:x0 + s]
            R0 = int(inp.get("cosine_period", 100))
            FC = filtcosenoF(R0, s, 0)
            deltaX = Z * dxy / L
            Uz = kreuzer3F(Z, holo_sq, wl, dxy, deltaX, L, FC)
            amp_f = np.abs(Uz)
            phase_f = np.angle(Uz)

        else:  # DLHM
            W_c = dxy * holo_u8.shape[1]
            amp_f, phase_f = dlhm_rec(holo_u8, L, Z, W_c, dxy, wl)

        # 8-bit views for the GUI
        amp_arr = normalize(amp_f, 255).astype(np.uint8)
        int_arr = normalize(amp_f ** 2, 255).astype(np.uint8)
        phase_arr = normalize((phase_f + np.pi) % (2 * np.pi) - np.pi, 255).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        packet = {
            "amp": amp_arr,
            "int": int_arr,
            "phase": phase_arr,
            "ft": cached_ft,
            "fps": round(fps, 1),
        }
        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(packet)


def reconstruct_pp(queue_manager: dict[str, dict[str, "Queue"]]) -> None:
    """
    Single-shot style worker for DLHM_PP.
    - Coalesces queued inputs (keeps only the latest).
    - Rebuilds the Fourier spectrum only when the hologram changes.
    - Skips heavy recomputation if (holo_hash + params) signature is unchanged.
    Emits 8-bit 'amp', 'int', 'phase', and cached log-FT as 'ft'.
    """
    last_holo_hash = None
    last_param_sig = None

    cached_spec = None
    cached_ft = None
    last_result = None

    M = N = None

    while True:
        #  Block for at least one input, then drain to keep only latest ---
        inp = queue_manager["reconstruction"]["input"].get()
        try:
            while True:
                # Keep replacing with the freshest packet if the UI spammed updates
                inp = queue_manager["reconstruction"]["input"].get_nowait()
        except Exception:
            pass

        t0 = time.time()

        # Extract inputs (keep names consistent with the existing RT worker) ---
        # NOTE: image arrives as uint8 (or similar); cast to float64 for math kernels.
        holo_u8 = inp["image"].astype(np.float64)
        algorithm = inp.get("algorithm", "AS")

        L = float(inp.get("L", 0.0))
        Z = float(inp.get("Z", 0.0))
        r = float(inp.get("r", 0.0))
        wl = float(inp.get("wavelength", 0.0))
        dxy = float(inp.get("dxy", 0.0))
        deltaX = Z * dxy / L
        scale = inp["scale_factor"]

        # Optional extras (ignored by kernels here but kept for completeness)
        R0 = int(inp.get("cosine_period", 100))

        # Cache & reuse spectrum only when the hologram actually changes ---
        this_hash = _hash_array(holo_u8)
        if this_hash != last_holo_hash:
            cached_spec = _compute_spectrum(holo_u8)  # complex spectrum for internal use if needed
            ft_cplx = _compute_spectrum(holo_u8)
            cached_ft = normalize(np.log1p(np.abs(ft_cplx)), 255).astype(np.uint8)
            last_holo_hash = this_hash
            M, N = cached_spec.shape

        # -Build a full parameter signature to skip redundant recomputes ---
        param_sig = (last_holo_hash, algorithm, L, Z, r, wl, dxy, scale, R0)
        if param_sig == last_param_sig and last_result is not None:
            # Nothing changed since last successful reconstruction → just re-emit
            if not queue_manager["reconstruction"]["output"].full():
                queue_manager["reconstruction"]["output"].put(last_result)
            continue

        # Run the selected reconstruction algorithm (heavy path) ---
        if algorithm == "AS":
            # Angular Spectrum
            recon_c = propagate(holo_u8, r, wl, dxy, dxy, scale)
            amp_f = np.abs(recon_c)
            phase_f = np.angle(recon_c)

        elif algorithm == "KR":
            # Kreuzer (crop to square, cosine filter in frequency)
            # Use image crop; many pipelines prefer spatial crop before KR kernel
            s = int(min(M or holo_u8.shape[0], N or holo_u8.shape[1]))
            y0 = (holo_u8.shape[0] - s) // 2
            x0 = (holo_u8.shape[1] - s) // 2
            holo_sq = holo_u8[y0:y0 + s, x0:x0 + s]

            FC = filtcosenoF(R0, s, 0)

            Uz = kreuzer3F(Z, holo_sq, wl, dxy, deltaX, L, FC)
            amp_f = np.abs(Uz)
            phase_f = np.angle(Uz)

        else:
            # DLHM direct reconstruction
            # W_c: camera width in micrometers (pitch * width)
            W_c = dxy * holo_u8.shape[1]
            amp_f, phase_f = dlhm_rec(holo_u8, L, Z, W_c, dxy, wl)

        # --- 6) GUI-friendly 8-bit buffers ---
        amp_arr = normalize(amp_f, 255).astype(np.uint8)
        int_arr = normalize(amp_f ** 2, 255).astype(np.uint8)
        phase_arr = normalize(((phase_f + np.pi) % (2 * np.pi)) - np.pi, 255).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        out_packet = {
            "amp": amp_arr,
            "int": int_arr,
            "phase": phase_arr,
            "ft": cached_ft,
            "fps": round(fps, 1),
        }

        # Publish & update caches
        last_param_sig = param_sig
        last_result = out_packet

        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(out_packet)