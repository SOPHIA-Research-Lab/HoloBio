import numpy as np
import time
from customtkinter import CTkImage
from multiprocessing import Queue
from kreuzer_functions import (kreuzer3F,filtcosenoF,dlhm_rec)
from skimage import exposure, filters
import cv2                            
from settings import *
from _3DHR_Utilities import *
import math as mt
from matplotlib import pyplot as plt
from PIL import Image
import glob
from datetime import datetime

def normalize(x: np.ndarray, scale: float) -> np.ndarray:
    '''Normalize every value of an array to the 0-scale interval.'''
    x = x.astype(np.float64)

    min_val = np.min(x)

    x = x-min_val

    max_val = np.max(x) if np.max(x)!=0 else 1

    normalized_image = scale*x / max_val

    return normalized_image


# Function to propagate an optical field using the Angular Spectrum approach
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

def reconstruct(queue_manager: dict[str, dict[str, Queue]]) -> None:
    while True:
        inp = queue_manager["reconstruction"]["input"].get()

        holo        = inp["image"].astype(np.float32)
        algorithm   = inp["algorithm"]
        L, Z, r     = inp["L"], inp["Z"], inp["r"]
        wl, dxy     = inp["wavelength"], inp["dxy"]
        scale       = inp["scale_factor"]

        t0 = time.time()

        # ➊ pick reconstruction algorithm (unchanged) -----------------
        if algorithm == "AS":
            field   = np.sqrt(normalize(holo, 1))
            recon_c = propagate(field, r, wl, dxy, dxy, scale)
            amp_f   = np.abs(recon_c)
            phase_f = np.angle(recon_c)

        elif algorithm == "KR":
            field   = np.sqrt(normalize(holo, 1))
            FC      = filtcosenoF(DEFAULT_COSINE_PERIOD,
                                  np.array((field.shape[1], field.shape[0])))
            deltaX  = Z * dxy / L
            recon_f = kreuzer3F(field, Z, L, wl, dxy, deltaX, FC)
            amp_f   = recon_f
            phase_f = np.zeros_like(amp_f)

        else:  # DLHM
            W_c   = dxy * holo.shape[1]
            amp_f, phase_f = dlhm_rec(holo, L, Z, W_c, dxy, wl)

        # ➋ brand-new: Fourier transform of the hologram --------------
        ft_cplx = np.fft.fftshift(np.fft.fft2(holo))
        ft_arr  = normalize(np.log1p(np.abs(ft_cplx)), 255).astype(np.uint8)

        # ➌ build 8-bit views -----------------------------------------
        amp_arr   = normalize(amp_f,        255).astype(np.uint8)
        int_arr   = normalize(amp_f ** 2,   255).astype(np.uint8)
        phase_arr = normalize((phase_f + np.pi) % (2*np.pi) - np.pi,
                              255).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        packet = {
            "amp":   amp_arr,
            "int":   int_arr,
            "phase": phase_arr,
            "ft":    ft_arr,      # <── sent to the GUI
            "fps":   round(fps, 1),
        }
        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(packet)
"""
def reconstruct(queue_manager: dict[str, dict[str, Queue]]) -> None:
    while True:
        inp = queue_manager["reconstruction"]["input"].get()

        holo        = inp["image"].astype(np.float32)
        algorithm   = inp["algorithm"]
        L, Z, r     = inp["L"], inp["Z"], inp["r"]
        wl, dxy     = inp["wavelength"], inp["dxy"]
        scale       = inp["scale_factor"]

        t0 = time.time()

        # ====================== choose algorithm ======================
        if algorithm == "AS":
            field   = np.sqrt(normalize(holo, 1))
            recon_c = propagate(field, r, wl, dxy, dxy, scale)
            amp_f   = np.abs(recon_c)
            phase_f = np.angle(recon_c)

        elif algorithm == "KR":
            field   = np.sqrt(normalize(holo, 1))
            FC      = filtcosenoF(DEFAULT_COSINE_PERIOD,
                                  np.array((field.shape[1], field.shape[0])))
            deltaX  = Z * dxy / L
            recon_f = kreuzer3F(field, Z, L, wl, dxy, deltaX, FC)
            amp_f   = recon_f                      # kreuzer3F returns |U|² normalised
            phase_f = np.zeros_like(amp_f)         # phase not retrieved here

        else:   # ------- DLHM -------------------------------------------------
            W_c   = dxy * holo.shape[1]            # sensor width
            amp_f, phase_f = dlhm_rec(holo, L, Z, W_c, dxy, wl)

        # ==================== generate three 8-bit views ======================
        amp_arr   = normalize(amp_f,        255).astype(np.uint8)
        int_arr   = normalize(amp_f**2,     255).astype(np.uint8)
        phase_arr = normalize((phase_f + np.pi) % (2*np.pi) - np.pi,
                              255).astype(np.uint8)

        fps = 1.0 / (time.time() - t0 + 1e-12)

        packet = {
            "amp":   amp_arr,
            "int":   int_arr,
            "phase": phase_arr,
            "ft":    ft_arr,
            "fps":   round(fps, 1),
        }
        if not queue_manager["reconstruction"]["output"].full():
            queue_manager["reconstruction"]["output"].put(packet)
"""

#DHM 

import numpy as np
import math as mt
from matplotlib import pyplot as plt
from PIL import Image
import glob
import cv2
from datetime import datetime
##Funciones hechas dentro de CUDA
#Funciones desde python
def hora_y_fecha():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

#Función de lectura de una imagen dada
def lectura(name_file):
    replica = Image.open("./" +
                         str(name_file)).convert('L')
    replica.save("./Imagenes/copia.png")
    return (replica)
def ajuste_tamano(archivo):
    N_, M_ = archivo.size
    N_ = N_ // 64
    M_ = M_ // 64
    replica = np.resize(archivo, (M_*64,N_*64))
    return (replica)

#Funcion para crear mascara circular
def crear_mascara_circular(shape, centro, radio):
    # Crear una imagen en blanco (negra) del tamaño especificado
    mascara = np.zeros(shape, dtype=np.uint8)
    
    # Dibujar el círculo en la máscara
    cv2.circle(mascara, centro, radio, (255, 255, 255), -1)
    
    return mascara

def ajuste_tamano1(archivo):
    N_, M_ = archivo.shape
    N_ = N_ // 64
    M_ = M_ // 64
    vector = archivo.flatten()
    replica = np.resize(archivo, (N_*64,M_*64))
    return (replica)

def lectura_continua(direccion):
    cv_img = []
    arepa=glob.glob(direccion)
    archivos = sorted(arepa, key=lambda x: x, reverse=True)
    for img in archivos:
        print(img)
        n = Image.open(img).convert('L')
        cv_img.append(n)
    return (cv_img)

# Función para el guardado de la imagen
def guardado(name_out, matriz):
    
    resultado = Image.fromarray(matriz)
    resultado = resultado.convert('RGB')
    resultado.save("./"+str(name_out))

# Función para graficar y ponerle nombre a los ejes
def mostrar(matriz, titulo="a", ejex="b", ejey="c"):
    plt.imshow(matriz, cmap='gray')
    plt.title(str(titulo))
    plt.xlabel(str(ejex))
    plt.ylabel(str(ejey))
    plt.show()

# Calculo de la magnitud, pero hagamos esto en cuda
def amplitud(matriz):
    amplitud = np.abs(matriz)
    return (amplitud)



def intensidad(matriz):
    intensidad = np.abs(matriz)
    intensidad = np.power(intensidad, 2)
    return (intensidad)


def fase(matriz):
    fase = np.angle(matriz, deg=False)
    return (fase)


def dual_img(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.show()


def dual_save(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.savefig('./Imagenes/guardado.png', dpi=1000)

def Espectro_angular(entrada,z,lamb,fx,fy,P,Q):
    result = np.exp2(1j*z*np.pi*np.sqrt(np.power(1/lamb, 2) -
              (np.power(fx*P, 2) + np.power(Q*fy, 2))))
    result = entrada*result
    return result

def tiro(holo,fx_0,fy_0,fx_tmp, fy_tmp,lamb,M,N,dx,dy,k,m,n):
    
    #Calculo de los angulos de inclinación

    theta_x=mt.asin((fx_0 - fx_tmp) * lamb /(M*dx))
    theta_y=mt.asin((fy_0 - fy_tmp) * lamb /(N*dy))

    #Creación de la fase asociada

    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    fase1=fase
    holo=holo*fase
    
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = (fase - min_val) / (max_val - min_val)
    threshold_value = 0.2
    fase = np.where(fase > threshold_value, 1, 0)
    value=np.sum(fase)
    return value, fase1

def normalizar(matriz):
    
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    matriz = 255*(matriz - min_val) / (max_val - min_val)
    return matriz