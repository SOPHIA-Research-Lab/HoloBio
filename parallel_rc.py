import numpy as np
import time
from customtkinter import CTkImage
from multiprocessing import Queue
from kreuzer_functions import (kreuzer3F,filtcosenoF,dlhm_rec)
from skimage import exposure, filters
from settings import *
from _3DHR_Utilities import *
from PIL import Image
import hashlib
from functools import lru_cache


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
    last_holo_hash = None
    cached_spec = None
    cached_ft = None

    while True:
        inp = queue_manager["reconstruction"]["input"].get()

        holo_u8 = inp["image"].astype(np.float32)
        algorithm = inp["algorithm"]
        L, Z, r = inp["L"], inp["Z"], inp["r"]
        wl, dxy = inp["wavelength"], inp["dxy"]
        scale = inp["scale_factor"]

        t0 = time.time()
        this_hash = _hash_array(holo_u8)

        # Build & cache spectrum only when the hologram changes
        if this_hash != last_holo_hash:
            field = np.sqrt(normalize(holo_u8, 1))
            cached_spec = _compute_spectrum(field)          
            ft_cplx = _compute_spectrum(holo_u8)
            cached_ft = normalize(np.log1p(np.abs(ft_cplx)), 255).astype(np.uint8)
            last_holo_hash = this_hash

        # Pick algorithm
        if algorithm == "AS":
            recon_c = _propagate_cached(cached_spec, r, wl, dxy, dxy, scale)
            amp_f = np.abs(recon_c)
            phase_f = np.angle(recon_c)

        elif algorithm == "KR":
            FC = filtcosenoF(DEFAULT_COSINE_PERIOD,
                                 np.array((cached_spec.shape[1],
                                           cached_spec.shape[0])))
            deltaX = Z * dxy / L
            amp_f = kreuzer3F(np.sqrt(normalize(holo_u8, 1)),
                               Z, L, wl, dxy, deltaX, FC)
            phase_f = np.zeros_like(amp_f)

        else:
            W_c = dxy * holo_u8.shape[1]
            amp_f, phase_f = dlhm_rec(holo_u8, L, Z, W_c, dxy, wl)

        # 8-bit views for the GUI
        amp_arr = normalize(amp_f, 255).astype(np.uint8)
        int_arr = normalize(amp_f ** 2, 255).astype(np.uint8)
        phase_arr = normalize((phase_f + np.pi) % (2 * np.pi) - np.pi,
                              255).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        packet = {
            "amp":   amp_arr,
            "int":   int_arr,
            "phase": phase_arr,
            "ft":    cached_ft,
            "fps":   round(fps, 1),
        }
        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(packet)

