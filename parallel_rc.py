import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifft2
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import cv2 as cv
import time
from customtkinter import CTkImage
from multiprocessing import Queue
from skimage import exposure, filters
from settings import *
from PIL import Image
import hashlib
from functools import lru_cache

def read(filename:str, path:str = '') -> np.ndarray:
    '''Reads image to double precision 2D array.'''
    if path!='':
        prefix = path + '\x5c'
    else:
        prefix = ''
    im = cv2.imread(prefix + filename, cv2.IMREAD_GRAYSCALE) #you can pass multiple arguments in single line
    return im.astype(np.float64)



def propagate(field, z, wavelength, dx, dy, scale_factor=1):
    # Inputs:
    # field - complex field
    # wavelength - wavelength
    # z - propagation distance
    # dxy - sampling pitches
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

def ang_spectrum(field, z, wavelength, dx, dy):
 
    N, M = field.shape
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))
    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = fftshift(fft2(fftshift(field)))
    phase = np.exp(1j * z * 2 * np.pi * np.sqrt((1 / wavelength) ** 2 - (m * dfx) ** 2 - (n * dfy) ** 2))
    out = ifftshift(ifft2(ifftshift(field_spec * phase)))

    return out

def _fts(A):   return np.fft.fftshift(np.fft.fft2 (np.fft.ifftshift(A)))
def _ifts(A):  return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A)))

def dlhm_rec(hologram, L, z, W_c, dx_out, wavelength):
    """
    Digital Lensless Holographic Microscopy reconstruction.
    Devuelve (amplitud, fase) con la misma forma que el holograma.
    """
    # ------------------------------------------------------------------
    N, M = hologram.shape      # filas, columnas  (Q, P)
    Mag  = L / z               # factor de ampliación nominal

    # --- corrección de distorsión radial ------------------------------
    Mag_max   = np.sqrt(W_c**2 / 2 + L**2) / z
    dist_max  = abs(Mag_max - Mag)
    cam_mat   = np.array([[M, 0, M/2],
                          [0, N, N/2],
                          [0, 0,   1]], dtype=np.float32)
    dist_coef = np.array([dist_max / (2*Mag), 0, 0, 0, 0],
                         dtype=np.float32)

    holo_corr = cv.undistort(hologram.astype(np.float32),
                              cam_mat, dist_coef)

    # --- malla de frecuencias espaciales (¡tamaño exacto!) ------------
    dfx = Mag / (dx_out * M)
    dfy = Mag / (dx_out * N)

    ix = np.arange(-M//2, M//2) * dfx         # longitud M
    iy = np.arange(-N//2, N//2) * dfy         # longitud N
    fx, fy = np.meshgrid(ix, iy)              # N×M  —— sin “+1”

    # --- kernel ASM ---------------------------------------------------
    k  = 2*np.pi / wavelength
    E  = np.exp(1j*(L - z) *
                np.sqrt(np.maximum(k**2 - (2*np.pi*fx)**2 - (2*np.pi*fy)**2,
                                   0.0)))     # clip valores negativos

    # --- propagación --------------------------------------------------
    Uz = _ifts(_fts(holo_corr) * E)

    amp   = np.abs(Uz)
    phase = np.angle(np.conj(Uz))             # [-π, π]

    return amp, phase

def filtcosenoF(par: int, fi, num_fig: int) -> np.ndarray:
    """
    Filtro cosenoidal 2D cuadrado.
    - par: periodo (en píxeles) del coseno al que quieres dar paso
    - fi : tamaño del filtro. Puede ser:
           • int  -> lado del cuadrado
           • tuple/list/ndarray -> (W, H) y se toma min(W, H)
    - num_fig: si != 0 muestra la figura (opcional; aquí no se usa)
    """
    # Acepta fi como entero o (W,H)
    if isinstance(fi, (tuple, list, np.ndarray)):
        side = int(min(int(fi[0]), int(fi[1])))
    else:
        side = int(fi)

    # Coordenadas cuadradas
    Xfc, Yfc = np.meshgrid(
        np.linspace(-side/2, side/2, side),
        np.linspace( side/2, -side/2, side),
        indexing="xy"
    )

    # Filtros cosenoidales horizontales y verticales
    # Se normaliza por el máximo para mantener [0,1]
    FC1 = np.cos(Xfc * (np.pi / par) * (1.0 / np.max(np.abs(Xfc)))) ** 2
    FC2 = np.cos(Yfc * (np.pi / par) * (1.0 / np.max(np.abs(Yfc)))) ** 2

    FC = (FC1 > 0) * FC1 * (FC2 > 0) * FC2
    FC = FC / (FC.max() + 1e-12)

    return FC

def prepairholoF(CH_m: np.ndarray, xop: float, yop: float, Xp: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    """
    Remapeo geométrico (vecinos “área” al estilo de tu referencia).
    Idéntico a tu notebook, solo tipeado y con floats explícitos.
    """
    row, _ = CH_m.shape
    Xcoord = (Xp - xop) / (-2.0 * xop / row)
    Ycoord = (Yp - yop) / (-2.0 * xop / row)

    iXcoord = np.floor(Xcoord)
    iYcoord = np.floor(Ycoord)

    iXcoord[iXcoord == 0] = 1
    iYcoord[iYcoord == 0] = 1

    x1frac = (iXcoord + 1.0) - Xcoord
    x2frac = 1.0 - x1frac
    y1frac = (iYcoord + 1.0) - Ycoord
    y2frac = 1.0 - y1frac

    x1y1 = x1frac * y1frac
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac

    CHp_m = np.zeros((row, row), dtype=np.float64)
    for it in range(0, row - 2):
        for jt in range(0, row - 2):
            iy = int(iYcoord[it, jt]); ix = int(iXcoord[it, jt])
            CHp_m[iy, ix]         += x1y1[it, jt] * CH_m[it, jt]
            CHp_m[iy, ix + 1]     += x2y1[it, jt] * CH_m[it, jt]
            CHp_m[iy + 1, ix]     += x1y2[it, jt] * CH_m[it, jt]
            CHp_m[iy + 1, ix + 1] += x2y2[it, jt] * CH_m[it, jt]

    return CHp_m

def kreuzer3F(z: float,
              field: np.ndarray,
              wavelength: float,
              pixel_pitch_in: float,
              pixel_pitch_out: float,
              L: float,
              FC: np.ndarray) -> np.ndarray:
    """
    Reconstrucción de Kreuzer (idéntica en forma a tu referencia).
    - Recorta a cuadrado (center-crop) para evitar 1023×1024, etc.
    - Usa exactamente la misma secuencia de fases/convulsiones.
    """
    # --- recorte cuadrado (center-crop) ---
    h, w = field.shape
    row = int(min(h, w))
    y0 = (h - row) // 2
    x0 = (w - row) // 2
    CH_m = field[y0:y0 + row, x0:x0 + row].astype(np.float64, copy=False)

    dx = float(pixel_pitch_in)
    dX = float(pixel_pitch_out)
    deltaY = dX

    k = 2.0 * np.pi / float(wavelength)
    W = dx * row

    delta = np.linspace(1, row, num=row, dtype=np.float64)
    X, Y = np.meshgrid(delta, delta)

    xo = -W / 2.0
    yo = -W / 2.0

    xop = xo * L / np.sqrt(L**2 + xo**2)
    yop = yo * L / np.sqrt(L**2 + yo**2)

    deltaxp = xop / (-row / 2.0)
    deltayp = deltaxp

    Yo = -dX * row / 2.0
    Xo = -dX * row / 2.0

    Xp = (dx * (X - row / 2.0) * L /
          np.sqrt(L**2 + (dx**2)*(X - row/2.0)**2 + (dx**2)*(Y - row/2.0)**2))
    Yp = (dx * (Y - row / 2.0) * L /
          np.sqrt(L**2 + (dx**2)*(X - row/2.0)**2 + (dx**2)*(Y - row/2.0)**2))

    CHp_m = prepairholoF(CH_m, xop, yop, Xp, Yp)

    Rp = np.sqrt(L**2 - (deltaxp*X + xop)**2 - (deltayp*Y + yop)**2)
    r  = np.sqrt((dX**2)*((X - row/2.0)**2 + (Y - row/2.0)**2) + z**2)
    CHp_m = CHp_m * ((L / Rp)**4) * np.exp(-0.5j * k * (r**2 - 2.0*z*L) * Rp / (L**2))

    pad = int(row / 2)
    FC_pad = np.pad(FC, (pad, pad))

    T1 = CHp_m * np.exp((1j * k / (2.0 * L)) *
                        (2.0*Xo*X*deltaxp + 2.0*Yo*Y*deltayp + X**2*deltaxp*dX + Y**2*deltayp*deltaY))
    T1 = np.pad(T1, (pad, pad))
    T1 = _fts(T1 * FC_pad)

    T2 = np.exp(-1j * (k / (2.0 * L)) *
                ((X - row/2.0)**2*deltaxp*dX + (Y - row/2.0)**2*deltayp*deltaY))
    T2 = np.pad(T2, (pad, pad))
    T2 = _fts(T2 * FC_pad)

    K = _ifts(T2 * T1)
    K = K[pad + 1:pad + row, pad + 1: pad + row]
    return K


def normalize(x: np.ndarray, scale: float) -> np.ndarray:
    '''Normalize every value of an array to the 0-scale interval.'''
    x = x.astype(np.float64)
    min_val = np.min(x)
    x = x-min_val
    max_val = np.max(x) if np.max(x)!=0 else 1
    normalized_image = scale*x / max_val
    return normalized_image

def im2arr(path: str):
    '''Converts file image into numpy array.'''
    return np.asarray(Image.open(path).convert('L'))

def arr2im(array: np.ndarray):
    '''Converts numpy array into PhotoImage type'''
    return Image.fromarray(array.astype(np.uint8), 'L')

def create_image(img: Image.Image, width, height):
    '''Converts image into type usable by customtkinter'''
    return CTkImage(light_image=img, dark_image=img, size=(width, height))

def gamma_filter(arr, gamma):
    return np.uint8(np.clip(arr + gamma * 255, 0, 255))

def contrast_filter(arr, contrast):
    return np.uint8(np.clip(arr * contrast, 0, 255))

def adaptative_eq_filter(arr, _):
    arr = exposure.equalize_adapthist(normalize(arr, 1), clip_limit=DEFAULT_CLIP_LIMIT)
    return np.uint8(arr * 255)  # Convertir de 0-1 a 0-255

def highpass_filter(arr, freq):
    arr = filters.butterworth(normalize(arr, 1), freq, high_pass=True)
    return np.uint8(arr*255)

def lowpass_filter(arr, freq):
    arr = filters.butterworth(normalize(arr, 1), freq, high_pass=False)
    return np.uint8(arr*255)

def capture(queue_manager:dict[dict[Queue, Queue], dict[Queue, Queue], dict[Queue, Queue]]):
    filter_dict =  {'gamma':gamma_filter,
                    'contrast':contrast_filter,
                    'adaptative_eq':adaptative_eq_filter,
                    'highpass':highpass_filter,
                    'lowpass':lowpass_filter}
    
    input_dict = {'path':None,
                  'reference path':None,
                  'settings':None,
                  'filters':None,
                  'filter':None}
    
    output_dict = {'image':None,
                   'filtered':None,
                   'fps':None,
                   'size':None}

    # Initialize camera (0 by default most of the time means the integrated camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Verify that the camera opened correctly
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        exit()

    # Sets the camera resolution to the closest chose in settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)

    # Gets the actual resolution of the image
    width_  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_ = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f'Width: {width_}')
    print(f'Height: {height_}')

    while True:
        init_time = time.time()
        # Captura la imagen de la cámara
        img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        img = cv2.flip(img, 1)  # Voltea horizontalmente

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
                img = img-ref
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

        elapsed_time = end_time-init_time
        fps = round(1 / elapsed_time, 1) if elapsed_time!=0 else 0

        if not queue_manager['capture']['output'].full():
            
            output_dict['image']= img
            output_dict['filtered'] = filt_img
            output_dict['fps'] = fps
            output_dict['size'] = (width_, height_)

            queue_manager['capture']['output'].put(output_dict)

def open_camera_settings(cap):
    try:
        cap.set(cv2.CAP_PROP_SETTINGS, 0)
    except:
        print('Cannot access camera settings.')

def _hash_array(arr: np.ndarray) -> str:
    """Fast, deterministic hash of a numpy array (content & shape)."""
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()

@lru_cache(maxsize=16)
def _precompute_kernel(shape: tuple[int, int],
                       wavelength: float,
                       dx: float, dy: float,
                       scale: float) -> np.ndarray:
    """Return the z-independent part of the ASM kernel."""
    M, N = shape
    x = np.arange(N) - N / 2
    y = np.arange(M) - M / 2
    X, Y = np.meshgrid(x, y, indexing="xy")
    dfx = 1.0 / (dx * N)
    dfy = 1.0 / (dy * M)
    return 2 * np.pi * np.sqrt(
        (1.0 / wavelength) ** 2 - (X * dfx) ** 2 - (Y * dfy) ** 2
    ) * scale
 

def _propagate_cached(field_spec: np.ndarray,
                      z: float,
                      wavelength: float,
                      dx: float, dy: float,
                      scale: float) -> np.ndarray:
    """
    Identical maths to `propagate()` (the slow version) but re-uses
    the *field_spec* and a cached kernel.  Absolutely no wrap-around
    artefacts anymore.
    """
    kernel = _precompute_kernel(field_spec.shape,
                                wavelength, dx, dy, scale)

    phase = np.exp(1j * z * kernel)
    tmp = field_spec * phase
    tmp = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(tmp)
    out = np.fft.ifftshift(out)
    return out

def _compute_spectrum(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
def reconstruct(queue_manager: dict[str, dict[str, Queue]]) -> None:
    """
    Worker de reconstrucción:
      • Mapea amplitud/intensidad con escala fija (sin min–max por frame).
      • Para KR, devuelve **también** el campo complejo exacto (key "field")
        para que la UI no lo re-sintetice desde u8.
    """
    last_holo_hash = None
    cached_spec = None
    cached_ft = None

    while True:
        inp = queue_manager["reconstruction"]["input"].get()

        # Entradas
        holo_u8   = inp["image"].astype(np.float32)      # u8 ya restada referencia en la UI
        algorithm = inp["algorithm"]                      # "AS" | "KR" | "DLHM"
        L, Z, r   = inp["L"], inp["Z"], inp["r"]         # µm
        wl, dxy   = inp["wavelength"], inp["dxy"]        # µm
        scale     = inp.get("scale_factor", 1.0)
        cosine_p  = inp.get("cosine_period", DEFAULT_COSINE_PERIOD)

        t0 = time.time()

        # Escala fija 0..1
        holo01 = np.clip(holo_u8 / 255.0, 0.0, 1.0)

        # Cache del espectro para ASM/KR
        this_hash = _hash_array(holo_u8)
        if this_hash != last_holo_hash:
            field0 = np.sqrt(holo01)
            cached_spec = _compute_spectrum(field0)

            ft_cplx = _compute_spectrum(holo01)
            ft_mag  = np.log1p(np.abs(ft_cplx))
            cached_ft = (ft_mag / (ft_mag.max() + 1e-12) * 255.0).astype(np.uint8)

            last_holo_hash = this_hash

        # Selección de algoritmo
        recon_field = None

        if algorithm == "AS":
            recon_c = _propagate_cached(cached_spec, r, wl, dxy, dxy, scale)
            amp_f   = np.abs(recon_c)
            phase_f = np.angle(recon_c)
            recon_field = recon_c

        elif algorithm == "KR":
            # Δx_out (Kreuzer) = Z * dxy / L
            deltaX = Z * dxy / (L + 1e-12)

            # Filtro cosenoidal cuadrado del tamaño de la imagen cuadrada
            H, W = holo01.shape
            s = int(min(H, W))
            FC = filtcosenoF(cosine_p, s, 0)

            # Llamada con orden correcto: (z, field, λ, dx_in, dx_out, L, FC)
            K = kreuzer3F(Z, holo01, wl, dxy, deltaX, L, FC)

            amp_f   = np.abs(K)
            phase_f = np.angle(K)
            recon_field = K

        else:  # "DLHM"
            W_c = dxy * holo_u8.shape[1]
            amp_f, phase_f = dlhm_rec(holo01, L, Z, W_c, dxy, wl)
            # Si quisieras: recon_field = amp_f * np.exp(1j*phase_f)

        # Empaquetado 8-bit (sin min–max por frame)
        a = amp_f.astype(np.float32)
        p2, p98 = np.percentile(a, [2.0, 98.0])
        den = max(p98 - p2, 1e-9)
        a8  = np.clip((a - p2) / den, 0.0, 1.0)
        amp_arr = (a8 * 255.0).astype(np.uint8)

        i = (a8 * a8)
        int_arr = (i / (i.max() + 1e-9) * 255.0).astype(np.uint8)

        phase_wrapped = (phase_f + np.pi) % (2 * np.pi) - np.pi
        phase_arr = ((phase_wrapped + np.pi) / (2 * np.pi) * 255.0).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        packet = {
            "amp":   amp_arr,
            "int":   int_arr,
            "phase": phase_arr,
            "ft":    cached_ft,
            "fps":   round(fps, 1),
            "field": recon_field,   # <<--- CLAVE: la UI usará esto si existe
        }
        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(packet)
