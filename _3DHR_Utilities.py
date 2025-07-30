import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import cv2
from PIL import Image
from skimage.util import view_as_windows
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from skimage.restoration import unwrap_phase
from skimage.filters import threshold_otsu
from skimage import morphology
from kneed import KneeLocator

'''Semi-Heuristic Phase Compensation function

  Authors: Sofia Obando-Vasquez(1), Ana Doblas (2) and Carlos trujillo (1)
  Date: 08-04-2023

  Python Adaptation: Daniel Córdoba (1)

  (1) Universidad EAFIT, Medellin, Colombia
  (2) The University of Memphis, Memphis, TN, United States

  read function:
    Inputs:
      filename: name of the file of the hologram
      path: path of the file of the hologram

    Outputs:
      holo: ndarray with the information of the image.

  compensate function:
    Inputs:
      hologram: ndarray with the input hologram
      dx,dy: pixel pitch of the digital sensor in each direction
      lambda_: wavelenght of the register
      region: for the spatial, one must select in which region of the Fourier
      spectrum want to perform the spatial filter. Asigne a value of 1-4
      according to the cartessian regions of a plane.
      step: value of the step for each iteration of the search
      G: value of the depth of the search

    Outputs:
      phase: the phase reconstruction of the hologram

  save function:
    Inputs:
      hologram: ndarray of the compensated hologram
      outname: name of the output file
      path: path of the output file
      ext: extension of teh output file
      cmap: colormap object to be applied to the output image

    Outputs:
      None


'''

def read(filename:str, path:str = '') -> np.ndarray:
    '''Reads image to double precision 2D array.'''

    if path!='':
        prefix = path + '\x5c'
    else:
        prefix = ''

    im = cv2.imread(prefix + filename, cv2.IMREAD_GRAYSCALE) #you can pass multiple arguments in single line

    return im.astype(np.float64)



def filter_mask(holo:np.ndarray,
                N:int,
                M:int,
                region:int) -> tuple[np.ndarray, np.float64]:
    '''Applies broad filter to eliminate unwanted dc signal.

    returns the filtered ft of the hologram and the index of its max value.
    '''

    #Calculate the Fourier Transform of the hologram and shift the
    #zero-frequency component to the center
    ft_holo = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))


    filter_ = np.zeros((N, M))

    # Create a filter mask for the desired region
    if region==1:
        filter_[0:round(N/2-(N*0.1)),round(M/2+(M*0.1)):M] = 1; # 1st quadrant
    elif region==2:
        filter_[0:round(N/2-(N*0.1)),0:round(M/2-(M*0.1))] = 1; # 2nd quadrant
    elif region==3:
        filter_[round(N/2+(N*0.1)):N,0:round(M/2-(M*0.1))] = 1; # 3rd quadrant
    else:
        filter_[round(N/2+(N*0.1)):N,round(M/2+(M*0.1)):M] = 1; # 4th quadrant

    # Apply the filter to the Fourier Transform of the hologram
    ft_filtered_holo = ft_holo * filter_

    filtered_spect = np.log(np.abs(ft_filtered_holo)**2+1)
    # Find the maximum value in the filtered spectrum

    idx = np.argmax(filtered_spect, axis=None)
    return ft_filtered_holo, idx


def normalize(x: np.ndarray, scale: float) -> np.ndarray:
    '''Normalize every value of an array to the 0-scale interval.'''
    x = x.astype(np.float64)

    min_val = np.min(x)

    x = x-min_val

    max_val = np.max(x) if np.max(x)!=0 else 1

    normalized_image = scale*x / max_val

    return normalized_image


def metric( holo_rec:np.ndarray,
            fx_0:int,
            fy_0:int,
            fx_tmp:int,
            fy_tmp:int,
            lambda_:float,
            M:int,
            N:int,
            dx:float,
            dy:float,
            m:np.ndarray,
            n:np.ndarray,
            k:float) -> float:
    '''Function to reduce cluster in the compensation function.'''

    #Calculate the angles for the compensation wave
    theta_x = np.arcsin((fx_0 - fx_tmp) * lambda_ / (M * dx))
    theta_y = np.arcsin((fy_0 - fy_tmp) * lambda_ / (N * dy))

    #Calculate the reference wave
    ref = np.exp(1j * k * (np.sin(theta_x) * m * dx + np.sin(theta_y) * n * dy))

    #Apply the reference wave to the hologram reconstruction (elementwise)
    holo_rec2 = holo_rec * ref

    #Calculate the phase of the hologram reconstruction
    phase = np.angle(holo_rec2)

    #Normalize the phase
    phase = normalize(phase, 1)

    # Threshold the phase image (binarization)
    BW = np.where(phase>0.1, 1, 0)

    #Calculate the sum of all elements in the resulting binary image
    num = np.sum(BW)

    return num

def compensate(hologram:np.ndarray,
               dx:float, dy:float,
               lambda_:float,
               region:int,
               step:float = 0.5,
               depth:int = 3) -> np.ndarray:
    '''Filters and compensates the input hologram.'''
    N, M = np.shape(hologram)[:2]

    mm, nn = np.linspace(-M/2, M/2-1, M), np.linspace(-N/2, N/2-1, N)


    m, n = np.meshgrid(mm, nn)

    ft_filtered_holo, idx = filter_mask(hologram, N, M, region)

    # Define wavenumber
    k = 2 * np.pi / lambda_

    # Calculate the center frequencies for fx and fy
    fx_0 = M/2
    fy_0 = N/2

    # Get the maximum values of fx and fy
    fy_max, fx_max = np.unravel_index(idx, [N, M])


    # Define the step size for the search
    step = step

    # Initialize variables for the search
    j = 0

    # Calculate the Inverse Fourier Transform of the filtered hologram
    holo_rec = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft_filtered_holo)))

    # Define the search range (G)
    G = depth

    # Initialize flag for the search loop
    fin = 0

    # Set initial values for fx and fy
    fx = fx_max
    fy = fy_max

    # Initialize temporary search range
    G_temp = G

    # Loop to find the optimal fx and fy values
    while fin == 0:
        i = 0
        j = j + 1

        # Initialize the maximum sum (for thresholding)
        suma_maxima=0

        # Nested loops for searching in the range of fx and fy
        for fy_tmp in np.arange(fy-step*G_temp, fy+step*G_temp, step):
            for  fx_tmp in np.arange(fx-step*G_temp, fx+step*G_temp, step):

                i = i+1

                # Calculate the metric for the current fx and fy
                suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp,
                              lambda_, M, N , dx, dy, m, n, k)

                # Update maximum sum and corresponding fx and fy if
                # current sum is greater than the previous maximum
                if suma > suma_maxima:
                    x_max_out = fx_tmp
                    y_max_out = fy_tmp
                    suma_maxima = suma

        # Update the temporary search range
        G_temp = G_temp - 1

        # Check if the optimal values are found, set the flag to exit the loop
        if (x_max_out == fx) and (y_max_out == fy):
            fin = 1;


        # Update fx and fy for the next iteration
        fx = x_max_out
        fy = y_max_out


    # Calculate the angles for the compensation wave
    theta_x = np.arcsin((fx_0 - x_max_out) * lambda_ / (M * dx))
    theta_y = np.arcsin((fy_0 - y_max_out) * lambda_ / (N * dy))

    # Calculate the reference wave
    ref = np.exp(1j * k * (np.sin(theta_x) * m * dx + np.sin(theta_y) * n * dy))

    # Apply the reference wave to the hologram reconstruction
    holo_rec2 = holo_rec * ref


    return holo_rec2


def save(hologram:np.ndarray,
         outname:str, path:str = '',
         ext:str = 'bmp',
         cmap:str = 'gray',
         out_amp:bool = 'True') -> None:
    '''Function to save the hologram to a file'''
    CompA = np.abs(hologram)
    CompP = np.angle(hologram)

    # Normalize the phase and convert it to uint8
    CompP = normalize(CompP, 255).astype(np.uint8)

    # Normalize the amplitude and convert it to uint8
    CompA = normalize(CompA, 255).astype(np.uint8)

    # Save the phase image
    if path!='':
        prefix = path + '\x5c'
    else:
        prefix = ''


    plt.imsave(prefix + 'Phase-' + outname +  '.' + ext, CompP, cmap=cmap)

    # Save the phase image
    if out_amp:
        plt.imsave(prefix  + 'Amplitude-'+ outname + '.' + ext, CompA, cmap=cmap)

######################################################################################
'''The rest of this code containts utility functions from my previous project'''

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

# Measure of the local variance of the input image
def focus_variance(U: np.ndarray, S: int=3) -> np.ndarray:
    """Calculates the local variance of an array.

    Inputs:
        U: The input array.
        S: The size of the kernel to use for calculating the variance.

    Returns:
        variances: ndarray containing the local variance of the input array.
    """

    # Create a view of the array as a set of overlapping windows
    windows = view_as_windows(np.abs(U), window_shape=(S, S), step=1)

    # Calculate the variance of each window
    variances = np.var(windows, axis=(2, 3))

    return variances

# Measure of the local accutance of the input image
def focus_acutance(U:np.ndarray, sigma:int = 1) -> np.ndarray:
    """Calculates the local level of focus of a complex optical plane.

    Inputs:
        U: The complex optical plane.
        sigma: maximum standard deviation of the gradient field

    Outputs:
        acutance: map of acutance of the input array
    """

    acutance = sc.ndimage.gaussian_gradient_magnitude(np.abs(U), sigma=sigma)

    return acutance

# Metric of the mean variance of the input image
def metric_variance(U:np.ndarray) -> float:
  '''returns the variance of a complex array'''

  return np.var(np.abs(U))

# Metric of the mean accutance of the input image
def metric_acutance(U:np.ndarray, sigma: float) -> float:
  '''Returns the mean acutance of a complex array'''

  acutance = sc.ndimage.gaussian_gradient_magnitude(np.abs(U), sigma=sigma)

  return np.mean(acutance)


# Returns an array with the local focus image of the input at a number of distances
def focus3D(U:np.ndarray,
            range_:np.ndarray,
            lmbda:float,
            dx:float,
            dy:float,
            S:int,
            scale_factor:float=1,
            method:str = 'acutance') -> list[tuple]:
  """Calculates the local level of focus of a complex optical plane
  at different propagation distances.

  Inputs:
      U: The complex optical plane.
      dz: The propagation step size.
      lmbda: The wavelength of light.
      dx: The horizontal pitch of the input plane.
      dy: The vertical pitch of the input plane.
      S: The number of subsections to divide the plane into.
      n: The number of propagation steps.

  Outputs:
      focuss: list containing 1) acutance, 2) distance of propagation and
      3) complex field; at each step of propagation.
  """

  focuss = []

  for z in range_:
    U_prop = propagate(U, z, lmbda, dx, dy,
                       scale_factor=scale_factor)

    focuss.append((focus_acutance(U_prop, S), focus_variance(U_prop, S),
                          z,
                          U_prop))

  return focuss


# Propagates to a number of distances and returns the one with the most focus. 
# A second metric to be added
def prop_focus(U:np.ndarray, lambda_:float, dx:float,
               dy:float, min_z:float, max_z:float,
               steps:float, scale_factor:float = 1, sigma:float = 1,
               metric:str = 'variance') -> float:

    range_ = np.linspace(min_z, max_z, steps)
    range_ = np.round(range_, 4)


    metrics = []
    fields = []

    # For combined metric
    var = []
    acu = []

    for z in range_:
        G = propagate(U, z, lambda_, dx, dy, scale_factor)

        var.append(metric_variance(G))
        acu.append(metric_acutance(G, sigma))


        fields.append(G)


    var = np.array(var)
    acu = np.array(acu)


    if metric=='variance':
        metrics = var
    elif metric=='acutance':
        metrics = acu
    elif metric=='combined':
        metrics = normalize(var, 1)+normalize(acu, 1)

    #Variance is low for focused translucent samples and high for focused opaque samples
    min_focus = np.min(metrics)
    min_ind = np.argmin(metrics)
    return fields[min_ind], range_[min_ind], metrics


def propgif(U, dz, lambda_, dx, dy, S, n, scale_factor = 1):
    focall = focus3D(U, dz, lambda_, dx, dy, S, n, scale_factor=scale_factor)

    fig, ax = plt.subplots()

    def animate(i):
        im, z_, _ = focall[i]
        ax.clear()
        ax.imshow(im, cmap='gray')
        ax.set_title(f'Mapa de acutancia z={z_} um')

    anim = FuncAnimation(fig, animate, frames=len(focall), interval=100)

    anim.save('focus.gif', dpi=150, writer='pillow')
    #plt.show()

    fig, ax = plt.subplots()

    def animate(i):
        _, z_, im = focall[i]
        ax.clear()
        ax.imshow(np.abs(im), cmap='gray')
        ax.set_title(f'Amplitud z={z_} um')

    anim = FuncAnimation(fig, animate, frames=len(focall), interval=100)

    anim.save('absfield.gif', dpi=150, writer='pillow')
    #plt.show()


def prepare_sample(U):
    '''Prepares the image to be clusterized'''
    normal = normalize(unwrap_phase(np.angle(U)), 1)

    mean = np.mean(normal)
    imean = np.mean(np.ones(normal.shape) - normal)

    max = np.max(normal)
    imax = np.max(np.ones(normal.shape)-normal)

    sample = None

    # Decides to either invert the image or not based on which version's maximum value is 
    # the furthest from it's mean, which generally means that the version contains
    # the brightest points
    if (max-mean)<(imax-imean):
        sample = np.ones(normal.shape) - normal
    else:
        sample = normal

    
    # Get's rid of the portion of the image less bright than the mean of the whole image
    trunc = sample - np.ones(sample.shape)*np.mean(sample)

    filter = trunc < 0
    trunc[filter] = 0

    # Applies intelligent threshold
    threshold = threshold_otsu(trunc)
    th = trunc>threshold

    # Cleans the image from small dots that weren't catched by the threshold
    th_cleaned = morphology.remove_small_objects(th, min_size=100)
    th_cleaned = th_cleaned > 0 # Binarized again because it was misbehaving

    return th_cleaned


def cluster(U, max_clusters=50, manual_clusters = None, show_elbow_graph = False):
    '''Clusterizes the image'''
    
    #Calculate the phase of the hologram reconstruction
    BW = prepare_sample(U)

    #Converting the image to an array of points for kmeans to cluster
    vals = np.where(BW > [0])
    vals = np.column_stack(vals)

    #Determining the optimal number of clusters
    sum_squared_dist = []

    for k in range(1, max_clusters):

        km = KMeans(n_clusters=k, random_state=0)
        try:
            km = km.fit(vals)
            sum_squared_dist.append(km.inertia_)
        except:
            sum_squared_dist.append(10000)
        

    x = range(1, max_clusters)
    kn = KneeLocator(x, sum_squared_dist, curve='convex', direction='decreasing')

    if show_elbow_graph:

        font_size = 16
        tick_size = 16  

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.scatter(x, sum_squared_dist)
        ax.set_title('Inercia vs. K', fontsize=font_size)
        ax.set_xlabel('K', fontsize=font_size)
        ax.set_ylabel('Inercia', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)


    if manual_clusters == None:
        k = kn.knee
    else:
        k = manual_clusters

    #Calculate the clusters (n_clusters should ideally be kn.knee)
    clust = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(vals)

    # blobs = blob_doh(phase, threshold=0.002)
    # blobs = np.array(blobs)

    return BW, vals, clust

def window_extraction(image: np.ndarray, centers: list, window_size: tuple, rimpercent: tuple) -> list:
    '''Extracts a number of subimages from the input ignoring the rims.
    
    rimpercent should be between 0 and 1.
    '''
    sub_images = []

    for center in centers:
        # if (center[0]+window_size[0]/2)<window_size[0]*(1-rimpercent[0]) or (center[0]-window_size[0]/2)>window_size[0]*rimpercent[0]:
        #     if (center[1]+window_size[1]/2)<window_size[1]*(1-rimpercent[1]) or (center[1]-window_size[1]/2)>window_size[1]*rimpercent[1]:
        #         continue
        #     continue

        xleft, xright = max(int(center[0]-window_size[0]/2), 0), min(int(center[0]+window_size[0]/2), image.shape[1])
        yup, ydown = max(int(center[1]-window_size[0]/2), 0), min(int(center[1]+window_size[0]/2), image.shape[0])

        window = image[xleft:xright, yup:ydown]
        sub_images.append(window)

    return sub_images

def sphere_phase_shift(input_field, radius, pos, n, lambda_, dxy, scale_factor, n0=1):
    '''Produces a spherical phase shift in the input field
   
    The input field is affected by a pure-phase object in the shape of a sphere,
    locally inducing a phase shift without affecting its amplitude

    radius: real radius of the sphere in length units
    pos: real position of the sphere in the image in length units relative to the
    center of the image
    n: Refractive index of the sphere
    z: real distance from the sphere to the image in length units
    lambda_: wavelength of light in length units
    dx, dy: pixel pitches in length units
    scale_factor: scale factor induced by microscope magnification, affects z
    n0: refractive index of the liquid medium of the sample, set to air by default
    '''
    N, M = input_field.shape

    angle = np.zeros((M, N))

    radius = radius*scale_factor/dxy

    for x in range(N):
      for y in range(M):
      
        cx, cy = pos[0]*dxy+N//2, pos[1]*dxy+M//2

        if (x-cx)**2 + (y-cy)**2 >= radius**2:
          angle[y, x] = 0
        else:
          # The phase shift corresponding to a change in index of refraction is equal to 2pi(OPL)/lambda where
          # OPL=optical path length, and is equal to (n2-n1)*d where d is the distance, in this case
          # d corresponds to the width of a sphere in the direction of z. Naturally, the scale factor introduces
          # a difference between the real size of the object and its apparent size in the camera, changing the 
          # aparent optical path difference, which is corrected by dividing by the scale factor

          angle[y, x] = 2*np.pi*(n-n0)/lambda_ * (radius - np.sqrt((x-cx)**2 + (y-cy)**2))*dxy/scale_factor

    complex = np.abs(input_field)*np.exp(1j*(np.angle(input_field)+angle))

    return complex

def sphere_sample(input_field, radii, xys, zs, ns, lambda_, dxy, scale_factor, n0 = 1, final_z=None):
  '''Simulates a number of pure-phase spherical samples.

  The spheres can all have different 3D positions, radii and refraction indices

  Position format is (x, y, z), input field is assumed to be in z=0

  The final image of the sample will be propagated to a z position equal to final_prop, this position
  is equal to the position of the last sphere on the array by default

  There is not a restriction of order for the z positions of the spheres or the final propagation
  '''

  N = len(radii)

  for n in range(N):
    if n==0:
      input_field = propagate(input_field, zs[n], lambda_, dxy, dxy, scale_factor)
      input_field = sphere_phase_shift(input_field, radii[n], xys[n], ns[n], lambda_, dxy, scale_factor, n0)
    else:
      input_field = propagate(input_field, zs[n]-zs[n-1], lambda_, dxy, dxy, scale_factor)
      input_field = sphere_phase_shift(input_field, radii[n], xys[n], ns[n], lambda_, dxy, scale_factor, n0)     

  if final_z==None:
    return input_field
 
  input_field = propagate(input_field, final_z-zs[-1], lambda_, dxy, dxy, scale_factor)

  return input_field