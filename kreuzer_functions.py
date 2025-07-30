import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifft2
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import cv2 as cv

#from _3DHR_Utilities import propagate

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

def ang_spectrum(field, z, wavelength, dx, dy):
    """
    Propagates a complex field using the angular spectrum method.

    Parameters:
    field (2D array): Complex field
    z (float): Propagation distance
    wavelength (float): Wavelength
    dx, dy (float): Sampling step sizes along x and y axes

    Returns:
    out (2D array): Field after propagation
    """
    N, M = field.shape
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))
    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = fftshift(fft2(fftshift(field)))
    phase = np.exp(1j * z * 2 * np.pi * np.sqrt((1 / wavelength) ** 2 - (m * dfx) ** 2 - (n * dfy) ** 2))
    out = ifftshift(ifft2(ifftshift(field_spec * phase)))

    return out


def filtcosenoF(par, fi:np.ndarray):
    """
    Creates a cosine filter.

    Parameters:
    par (int): Cosine period
    fi (int): Size of the filter

    Returns:
    FC (2D array): Cosine filter
    """
    Xfc, Yfc = np.meshgrid(np.linspace(-fi[0] // 2, fi[0] // 2, int(fi[0])), np.linspace(fi[1] // 2, -fi[1] // 2, int(fi[1])))
    FC1 = np.cos(Xfc * (np.pi / par) * (1 / np.max(np.abs(Xfc)))) ** 2
    FC2 = np.cos(Yfc * (np.pi / par) * (1 / np.max(np.abs(Yfc)))) ** 2
    FC = (FC1 > 0) * FC1 * (FC2 > 0) * FC2
    FC = FC / np.max(FC)

    return FC


def normalize(image):
    """
    Normalizes an image to the [0, 1] range.

    Parameters:
    image (2D array): Input image

    Returns:
    imageNormalized (2D array): Normalized image
    """
    imageNormalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return imageNormalized


def point_src(M, z, x0, y0, wavelength, dx):
    """
    Generates a point source at the coordinates (x0, y0) observed at a distance z.

    Parameters:
    M (int): Matrix size
    z (float): Propagation distance
    x0, y0 (float): Coordinates of the source center
    wavelength (float): Wavelength
    dx (float): Sampling step

    Returns:
    P (2D array): Field of the point source
    """
    N = M
    dy = dx
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))
    k = 2 * np.pi / wavelength
    r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)
    P = np.exp(1j * k * r) / r

    return P



def prepairholoF(CH_m, xop, yop, Xp, Yp):
    """
    Prepare the hologram using bilinear interpolation, preserving the complex data.

    Parameters:
    CH_m (2D array, complex): Original hologram matrix.
    xop, yop (float): Center coordinates of the hologram.
    Xp, Yp (2D arrays): New coordinates for interpolation.

    Returns:
    CHp_m (2D array, complex): Prepared hologram.
    """
    n_rows, n_cols = CH_m.shape
    Xcoord = (Xp - xop) / (-2 * xop / n_cols)
    Ycoord = (Yp - yop) / (-2 * yop / n_rows)
    
    # Ensure coordinates are within bounds
    Xcoord = np.clip(Xcoord, 0, n_cols - 2)
    Ycoord = np.clip(Ycoord, 0, n_rows - 2)
    
    # Calculate integer part (floored)
    iXcoord = np.floor(Xcoord).astype(int)
    iYcoord = np.floor(Ycoord).astype(int)
    
    # Calculate fractional part for interpolation
    x1frac = (iXcoord + 1.0) - Xcoord
    x2frac = 1.0 - x1frac
    y1frac = (iYcoord + 1.0) - Ycoord
    y2frac = 1.0 - y1frac
    
    # Bilinear interpolation coefficients
    x1y1 = x1frac * y1frac
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac
    
    # Initialize complex result matrix
    CHp_m = np.zeros((n_rows, n_cols), dtype=complex)
    
    # Apply bilinear interpolation
    for it in range(n_rows - 1):
        for jt in range(n_cols - 1):
            CHp_m[iYcoord[it, jt], iXcoord[it, jt]] += (x1y1[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt], iXcoord[it, jt] + 1] += (x2y1[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt]] += (x1y2[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt] + 1] += (x2y2[it, jt]) * CH_m[it, jt]

    return CHp_m


def kreuzer3F(hologram, z, L, wavelength, dx, deltaX, FC):
    """
    Reconstructs an in-line hologram using Kreuzer's method.

    Parameters:
    hologram (2D array): Hologram matrix
    z (float): Propagation distance
    L (float): Length parameter
    wavelength (float): Wavelength
    dx, deltaX (float): Pixel sizes at different stages
    FC (2D array): Cosine filter

    Returns:
    K (2D array): Reconstructed hologram
    """
    n_rows, n_cols = hologram.shape
    k = 2 * np.pi / wavelength
    W = dx * n_cols
    H = dx * n_rows

    deltaY = deltaX
    X, Y = np.meshgrid(np.arange(1, n_cols + 1), np.arange(1, n_rows + 1))

    xo = -W / 2
    yo = -H / 2
    xop = xo * L / np.sqrt(L ** 2 + xo ** 2)
    yop = yo * L / np.sqrt(L ** 2 + yo ** 2)

    deltaxp = xop / (-n_cols / 2)
    deltayp = yop / (-n_rows / 2)
    Xo = -deltaX * n_cols / 2
    Yo = -deltaY * n_rows / 2

    Xp = (dx * (X - n_cols / 2) * L) / np.sqrt(L ** 2 + (dx ** 2) * (X - n_cols / 2) ** 2 + (dx ** 2) * (Y - n_rows / 2) ** 2)
    Yp = (dx * (Y - n_rows / 2) * L) / np.sqrt(L ** 2 + (dx ** 2) * (Y - n_rows / 2) ** 2 + (dx ** 2) * (Y - n_rows / 2) ** 2)

    CHp_m = prepairholoF(hologram, xop, yop, Xp, Yp)



    Rp = np.sqrt(L ** 2 - (deltaxp * X + xop) ** 2 - (deltayp * Y + yop) ** 2)
    r = np.sqrt((deltaY**(2))*(deltaX ** (2)) * ((X - n_cols / 2) ** 2 + (Y - n_rows / 2) ** 2) + z ** 2)
    CHp_m *= ((L / Rp) ** 4) * np.exp(-0.5j * k * (r ** 2 - 2 * z * L) * Rp / (L ** 2))



    pad = n_cols // 2

    # Redimensionar FC si no coincide
    if FC.shape != CHp_m.shape:
        FC = np.pad(FC, ((pad, pad), (pad, pad)), mode='constant')


    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + (X ** 2) * deltaxp * deltaX + (Y ** 2) * deltayp * deltaY))

    # Asegurarse de que la matriz T1 también se redimensione correctamente
    if T1.shape != FC.shape:
        T1 = np.pad(T1, ((pad, pad), (pad, pad)), mode='constant')

    K = propagate(T1, (L-z), wavelength, deltaX, deltaY)
    K = K[pad:pad + n_rows, pad:pad + n_rows]
    K = np.abs(K) ** 2
    K = normalize(K)

    return K


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
