
# import libraries
import numpy as np
import math
from math import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import scipy.optimize
from scipy import ndimage
from scipy.ndimage import median_filter
from skimage.restoration import unwrap_phase
from scipy.sparse.linalg import svds
from scipy.interpolate import RegularGridInterpolator
from tkinter import messagebox


# ─────────────────────────────────────────────
# Efficient ROI Search, from pyDHM library
# ─────────────────────────────────────────────
def ERS(inp, wavelength, dx, dy, s=5, step=0.2, filter_type="Circular", manual_coords=None,
        filtering_function=None):
    """
    Efficient ROI Search (ERS) for phase compensation in off-axis digital holograms.

    Parameters:
    -----------
    inp : ndarray
        Input hologram (intensity or complex field).
    wavelength : float
        Wavelength of the source (in meters).
    dx, dy : float
        Pixel size in the x and y directions (in meters).
    s : int, optional
        Initial search window size (default: 5).
    step : float, optional
        Refinement step for ROI search (default: 0.2).
    filter_type : str, optional
        Type of spatial filter to apply, e.g., "Circular", "Rectangular Man.", etc.
    manual_coords : tuple, optional
        Manually defined coordinates for custom filters (if applicable).
    filtering_function : callable, optional
        Spatial filtering function that must return (holo_filter, fy_max, fx_max).

    Returns:
    --------
    comp_phase : ndarray
        Complex field with compensated phase.
    """

    # Validate that step size is smaller than search window size
    if step >= s:
        raise ValueError('"step" must be smaller than "s" for ERS to work properly.')

    # Ensure a spatial filtering function is provided
    if filtering_function is None:
        raise ValueError("You must provide a spatial filtering function (filtering_function).")

    # Preprocessing: remove DC offset and ensure array type
    inp = inp - np.average(inp)
    inp = np.array(inp)
    M, N = inp.shape

    # Generate centered spatial coordinate mesh
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    # Apply spatial filtering to isolate the +1 diffraction order
    # print("ERS: Starting spatial filtering...")
    holo_filter, fy_max, fx_max = filtering_function(inp, M, N, filter_type, manual_coords)
    # print("ERS: Spatial filtering completed.")

    # Define the center of the spectrum
    fx_0, fy_0 = N / 2, M / 2

    # Initial coordinates from filtering
    fx_1 = float(fx_max[0])
    fy_1 = float(fy_max[0])

    # Wavenumber
    k = (2.0 * np.pi) / wavelength

    # Search initialization
    fin = 0
    fx = fx_1
    fy = fy_1
    G_temp = s

    x_max_out = fx
    y_max_out = fy

    # print("ERS: Searching for optimal ROI...")
    # Iterative refinement of the ROI center
    while fin == 0:
        sum_max = 0

        # Define search range around current fx, fy (scaled by 10 for subpixel resolution)
        arrayY = np.linspace((fy - step * G_temp) * 10, (fy + step * G_temp) * 10, int(10 * step))
        arrayX = np.linspace((fx - step * G_temp) * 10, (fx + step * G_temp) * 10, int(10 * step))

        for fx_temp in arrayX:
            for fy_temp in arrayY:
                fx_tmp = fx_temp / 10.0
                fy_tmp = fy_temp / 10.0

                # Compute angles based on frequency displacement
                theta_x = math.asin((fx_0 - fx_tmp) * wavelength / (N * dx))
                theta_y = math.asin((fy_0 - fy_tmp) * wavelength / (M * dy))

                # Generate reference wave for current angles
                ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dx) + (math.sin(theta_y) * Y * dy)))
                reconstruction = holo_filter * ref_wave

                # Normalize phase and threshold it to find information content
                phase = np.angle(reconstruction)
                minVal = np.amin(phase)
                maxVal = np.amax(phase)
                if abs(maxVal - minVal) < 1e-9:
                    continue
                phase_sca = (phase - minVal) / (maxVal - minVal)
                binary_phase = (phase_sca > 0.2)
                summ = np.sum(binary_phase)

                # Store coordinates with maximum binary content
                if summ > sum_max:
                    sum_max = summ
                    x_max_out = fx_tmp
                    y_max_out = fy_tmp

        # Reduce search window size
        G_temp -= 1

        # Stop condition: ROI center does not change significantly
        if abs(x_max_out - fx) < 1e-5 and abs(y_max_out - fy) < 1e-5:
            fin = 1
        fx = x_max_out
        fy = y_max_out

    # Final phase compensation using optimal ROI coordinates
    theta_x = math.asin((fx_0 - x_max_out) * wavelength / (N * dx))
    theta_y = math.asin((fy_0 - y_max_out) * wavelength / (M * dy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dx) + (math.sin(theta_y) * Y * dy)))
    comp_phase = holo_filter * ref_wave

    # print("ERS: Phase compensation completed.")
    return comp_phase


# ─────────────────────────────────────────────
# Cost Function Search to CFS-pyDHM library
# ─────────────────────────────────────────────
def costFunction(seeds, height, width, holo_filter, wavelength, dxy, Y, X, fy_0, fx_0, k):
    theta_x = math.asin((fx_0 - seeds[1]) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - seeds[0]) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * (
            (math.sin(theta_y) * Y * dxy) +
            (math.sin(theta_x) * X * dxy)
    ))
    phase = np.angle(holo_filter * ref_wave)

    # Thresholding
    binary_phase = (phase + math.pi) > 0.2
    sumIB = binary_phase.sum()

    # Cost function
    J = (height * width) - sumIB
    J += np.std(phase)

    return J


# ─────────────────────────────────────────────
# CFS Search, from pyDHM library
# ─────────────────────────────────────────────
def CFS(inp, wavelength, dx, dy, filter_type, manual_coords=None, spatial_filtering_fn=None):
    """
    CFS form pyDHM with cost-function improve version.

    Parameters:
    -----------
    inp : ndarray
        Input hologram (real or complex).
    wavelength : float
        Wavelength in micrometers.
    dx, dy : float
        Pixel pitch in micrometers.
    filter_type : str
        Type of spatial filter to apply.
    manual_coords : tuple or None
        Manual coordinates for filtering.
    spatial_filtering_fn : callable
        Function to apply spatial filtering. Must return (filtered_field, fy, fx).

    Returns:
    --------
    comp_phase : ndarray
        Phase-compensated complex field.
    """
    inp = np.asarray(inp, dtype=np.float32)
    height, width = inp.shape
    y = np.arange(height)
    x = np.arange(width)
    Y, X = np.meshgrid(y - height / 2, x - width / 2, indexing="ij")

    fy_0, fx_0 = height / 2, width / 2
    k = (2 * np.pi) / wavelength
    dxy = 0.5 * (dx + dy)

    if spatial_filtering_fn is None:
        raise ValueError("A spatial filtering function must be provided.")

    holo_filter, fy_seed, fx_seed = spatial_filtering_fn(
        inp, height, width, filter_type, manual_coords=manual_coords
    )
    if holo_filter is None:
        # print("Filtering was cancelled by the user.")
        return None

    fy_seed = float(fy_seed[0])
    fx_seed = float(fx_seed[0])

    # print("CFS (TU-DHM): cost-function optimisation …")
    step = 2
    bounds = ((fy_seed - step, fy_seed + step),
              (fx_seed - step, fx_seed + step))

    res = scipy.optimize.minimize(
        lambda seeds: costFunction(
            seeds, height, width, holo_filter, wavelength,
            dxy, Y, X, fy_0, fx_0, k
        ),
        x0=[fy_seed, fx_seed],
        method="TNC",
        bounds=bounds,
        tol=1e-3
    )
    fy_best, fx_best = res.x
    # print(f"   → optimal   fy={fy_best:.3f}   fx={fx_best:.3f}")

    theta_x = math.asin((fx_0 - fx_best) * wavelength / (width * dx))
    theta_y = math.asin((fy_0 - fy_best) * wavelength / (height * dy))
    ref_wave = np.exp(1j * k * ((np.sin(theta_y) * Y * dy) +
                                (np.sin(theta_x) * X * dx)))
    comp_phase = holo_filter * ref_wave
    # print("CFS (TU-DHM): phase compensation finished.")

    return comp_phase


# ─────────────────────────────────────────────
# Spatial filter
# ─────────────────────────────────────────────
def spatialFilteringCF(field, height, width, filter_type="Circular", manual_coords=None,
                       draw_manual_rectangle=None, draw_manual_circle=None,
                       prompt_circle_coords=None, prompt_rectangle_coords=None,
                       show_ft_and_filter=True):
    """
    Applies a spatial filter in the Fourier domain to isolate a diffraction order
    in an off-axis digital hologram.

    Parameters:
    -----------
    field : ndarray
        Input hologram (complex field or intensity).
    height, width : int
        Dimensions of the input hologram.
    filter_type : str
        Type of spatial filter to apply ("Circular", "Rectangular", "Manual Rectangular", etc.).
    manual_coords : tuple or None
        User-defined coordinates for manual filters (e.g., (x1, y1, x2, y2) or (cx, cy, r)).
    draw_manual_rectangle : callable or None
        Function that allows interactive manual rectangle selection.
    draw_manual_circle : callable or None
        Function that allows interactive manual circle selection.
    prompt_circle_coords : callable or None
        Function that returns predefined circle coordinates (cx, cy, radius).
    prompt_rectangle_coords : callable or None
        Function that returns predefined rectangle coordinates (x1, y1, x2, y2).
    show_ft_and_filter : bool
        Whether to visualize the filtered Fourier transform (default: True).

    Returns:
    --------
    holo_filter : ndarray
        The filtered hologram reconstructed in the spatial domain (complex field).
    fy_max, fx_max : array
        Coordinates of the selected diffraction order in the Fourier domain.
    """

    # Hologram Fourier Transform
    ft = np.fft.fftshift(np.fft.fft2(field))

    # Remueve el DC centrado correctamente
    center_x, center_y = width // 2, height // 2
    ft[center_y - 20:center_y + 20, center_x - 20:center_x + 20] = 0

    # Mask half
    mask = np.zeros_like(ft, dtype=bool)
    mask[:height // 2, :] = True

    # Find max value orden +1
    magnitude = np.abs(ft)
    peak_region = magnitude * mask
    fy_max, fx_max = np.unravel_index(np.argmax(peak_region), peak_region.shape)

    # Calcula radio basado en distancia al centro
    d = np.hypot(fy_max - height / 2, fx_max - width / 2)
    radius = max(1, d / 3)

    # Initialize an empty spatial filter mask
    mask = np.zeros((height, width), dtype=np.float32)

    # Apply circular mask centered at the peak
    if filter_type == "Circular":
        mask = circularMask(height, width, radius, fy_max, fx_max)

    # Apply rectangular mask centered at the peak
    elif filter_type == "Rectangle":
        top, bottom = fy_max - radius, fy_max + radius
        left, right = fx_max - radius, fx_max + radius
        mask[max(int(top), 0):min(int(bottom), height),
            max(int(left), 0):min(int(right), width)] = 1.0

    # Use manually selected rectangle from GUI or passed coordinates
    elif filter_type == "Rectangle Man.":
        if manual_coords is None and draw_manual_rectangle:
            manual_coords = draw_manual_rectangle()
        if manual_coords:
            x1, y1, x2, y2 = map(int, manual_coords)
            mask[y1:y2, x1:x2] = 1.0
            fy_max, fx_max = np.array([(y1 + y2) // 2]), np.array([(x1 + x2) // 2])
        else:
            mask = circularMask(height, width, radius, fy_max, fx_max, False)

    # Use manually drawn circular region
    elif filter_type == "Circular Man.":
        if manual_coords is None and draw_manual_circle:
            manual_coords = draw_manual_circle()
        if manual_coords:
            cx, cy, rad = manual_coords
            mask = circularMask(height, width, rad, cy, cx)
            fy_max, fx_max = np.array([int(cy)]), np.array([int(cx)])
        else:
            mask = circularMask(height, width, radius, fy_max, fx_max, False)

    # Use predefined circle coordinates
    elif filter_type == "Circular Coor.":
        if manual_coords is None and prompt_circle_coords:
            manual_coords = prompt_circle_coords()
        if manual_coords is None:
            messagebox.showinfo(
                "Information",
                "User cancelled circular coordinate entry. No filtering will be applied."
            )
            return None, None, None
        cx, cy, rad = manual_coords
        mask = circularMask(height, width, rad, cy, cx)
        fy_max, fx_max = np.array([int(cy)]), np.array([int(cx)])

    # Use predefined rectangle coordinates (telecentric or non-telecentric)
    elif filter_type in ["Rectangle Coor.", "Non Telecentric Coordinates"]:
        if manual_coords is None and prompt_rectangle_coords:
            manual_coords = prompt_rectangle_coords()
        if manual_coords is None:
            messagebox.showinfo(
                "Information",
                "User cancelled rectangle coordinate entry. No filtering applied."
            )
            return None, None, None
        x1, y1, x2, y2 = map(int, manual_coords)
        mask[y1:y2, x1:x2] = 1.0
        fy_max, fx_max = np.array([(y1 + y2) // 2]), np.array([(x1 + x2) // 2])

    # Apply the spatial filter in the Fourier domain
    filtered_spec = ft * mask

    # Optionally visualize the filtered Fourier transform
    if show_ft_and_filter:
        plt.figure("Fourier space after filtering")
        plt.imshow(np.log1p(np.abs(filtered_spec)), cmap="gray")
        plt.title(f"Filtered FT – {filter_type}")
        plt.axis("off")
        plt.show()

    # Reconstruct the filtered field by inverse FFT
    holo_filter = np.fft.ifft2(np.fft.ifftshift(filtered_spec))
    return holo_filter, np.array([fy_max]), np.array([fx_max])


# ─────────────────────────────────────────────
# Vortex + Legendre
# ─────────────────────────────────────────────
def vortexLegendre(inp, wavelength, dx, dy, limit, filter_type, manual_coords=None, spatial_filtering_fn=None):
    # Cut hologram (M x N) to (M x M)
    hologram = np.array(inp)
    if hologram.shape[1] != hologram.shape[0]:
        diff = hologram.shape[1] - hologram.shape[0]
        lim = diff // 2
        if diff % 2 == 0:
            hologram = hologram[:, lim:-lim]
        else:
            hologram = hologram[:, lim:-(lim + 1)]
    N, M = hologram.shape

    # load physical parameters
    k = 2 * pi / wavelength
    fx_0 = M / 2
    fy_0 = N / 2

    # Create coordinates
    m, n = np.meshgrid(
        np.arange(-M // 2, M // 2),
        np.arange(-N // 2, N // 2)
    )

    # Filter hologram
    if spatial_filtering_fn is None:
        raise ValueError("A spatial filtering function must be provided.")

    holo_filter, fy_max, fx_max = spatial_filtering_fn(
        hologram, N, M, filter_type, manual_coords=manual_coords
    )
    if holo_filter is None:
        print("")
        return None

    # Apply median filter
    medFilt = 3
    #ft_holo_filtered = fftshift(fft2(fftshift(holo_filter)))
    """ 
    ft_holo_filtered = np.fft.fftshift(np.fft.fft2(holo_filter))
    field_real = median_filter(np.real(ft_holo_filtered), size=(medFilt, medFilt), mode='reflect')
    field_imag = median_filter(np.imag(ft_holo_filtered), size=(medFilt, medFilt), mode='reflect')
    field_filtered = field_real + 1j * field_imag
    """
    # Filter hologram

    ft_holo_filtered = 10 * np.log10(np.abs((np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo_filter))))+1e-6) ** 2)
    field_filtered = median_filter(ft_holo_filtered, size=(medFilt, medFilt), mode='reflect')

    # Apply Vortex
    fx_max, fy_max = vortex_compensation(field_filtered, fx_max, fy_max)
    # print("Vortex Compensation (fx, fy):", fx_max, fy_max)

    # Reference wave
    refwa = reference_wave(fx_max, fy_max, m, n, wavelength, dx, dy, k, fx_0, fy_0, M, N)

    # Vortex compensation
    obj_complex_VL = refwa * holo_filter

    # ---------------------------------------------------------------------------------------------------------
    # ------------------------------ Legendre ------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    NoPistonCompensation = False
    UsePCA = True
    limit=256/2
    phase_corrected, legendre_coefficients = legendre_compensation(
        obj_complex_VL, limit, NoPistonCompensation, UsePCA)
    #imageShow(np.angle(phase_corrected), 'Phase + Vortex + Legendre')
    # print("Total Legendre coefficients:", len(legendre_coefficients))

    # 1. Normalized grid preparation
    gridSize = obj_complex_VL.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)
    dA = (2 / gridSize) ** 2

    # 2. Legendre basis (using orders 2 to 6, as in MATLAB)
    order = np.arange(2, 7)  # equivalent to 2:6
    polynomials = square_legendre_fitting(order, X, Y)

    ny, nx, n_polys = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_polys)

    # 3. Inner product for orthonormalization
    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))

    # 4. Normalization constants
    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA

    # 5. Projection (using coefficients 2 to 6)
    coeffs = legendre_coefficients[1:len(order) + 1] / np.sqrt(Legendres_norm_const[:len(order)])
    WavefrontReconstructed_Vect = np.sum(coeffs[np.newaxis, :] * Legendres[:, :len(order)], axis=1)

    # 6. Wavefront reconstruction
    WavefrontReconstructed = WavefrontReconstructed_Vect.reshape(ny, nx)

    # 7. Phase compensation
    compensatedHologram = np.abs(obj_complex_VL) * (
                np.exp(1j * np.angle(obj_complex_VL)) / np.exp(1j * WavefrontReconstructed))

    return compensatedHologram


# ─────────────────────────────────────────────
# circular mask for filters
# ─────────────────────────────────────────────
def circularMask(height, width, radius, centY, centX):
    """
    Generates a binary circular mask of given radius centered at (centX, centY).

    Parameters:
    -----------
    height : int
        Number of rows in the output mask (image height).
    width : int
        Number of columns in the output mask (image width).
    radius : float
        Radius of the circular mask (in pixels).
    centY : float
        Y-coordinate (row) of the circle center.
    centX : float
        X-coordinate (column) of the circle center.

    Returns:
    --------
    mask : ndarray
        A 2D binary array of shape (height, width) with ones inside the circle and zeros elsewhere.
    """

    # Create 2D coordinate grids using open mesh (saves memory)
    Y, X = np.ogrid[:height, :width]

    # Initialize the binary mask
    mask = np.zeros((height, width))

    # Compute circular region: pixels within given radius from (centX, centY)
    circle = np.sqrt((Y - centY) ** 2 + (X - centX) ** 2) <= radius

    # Set pixels inside the circle to 1
    mask[circle] = 1

    return mask


# ─────────────────────────────────────────────
# function for draw a circle to the spatial filter
# ─────────────────────────────────────────────
def draw_manual_circle(current_ft_array=None, arr_ft=None, arr_hologram=None):
    """
    Allows the user to manually define a circular region by clicking:
    first to set the center and second to define the radius.
    Returns (cx, cy, radius) in pixels, or None if cancelled or incomplete.

    Parameters:
    -----------
    current_ft_array : ndarray or None
        Preferred Fourier transform array to display, if already computed.
    arr_ft : ndarray or None
        Secondary option if current_ft_array is not provided.
    arr_hologram : ndarray or None
        Raw hologram used to compute the Fourier transform if none is precomputed.

    Returns:
    --------
    tuple or None
        (cx, cy, radius) as floats, or None if the user cancels or fails to complete selection.
    """

    # Determine which array to display
    if current_ft_array is not None:
        ft_to_show = current_ft_array
    elif arr_ft is not None and arr_ft.size:
        ft_to_show = arr_ft
    elif arr_hologram is not None and arr_hologram.size:
        spec = np.fft.fftshift(np.fft.fft2(arr_hologram))
        ft_to_show = np.abs(spec)
    else:
        messagebox.showinfo(
            "Information",
            "No Fourier transform available for manual filtering."
        )
        return None

    # Apply logarithmic scaling for better contrast
    ft_disp = np.log1p(ft_to_show)

    # Create interactive plot
    fig, ax = plt.subplots(num="Select centre, then radius")
    ax.imshow(ft_disp, cmap="gray")
    ax.set_title("Click centre, then click a point on the circumference")
    points = []

    def _onclick(event):
        if event.inaxes is not ax:
            return
        points.append((event.xdata, event.ydata))

        if len(points) == 1:
            # Mark center
            ax.scatter(*points[0], c='r')
            fig.canvas.draw()
        elif len(points) == 2:
            # Draw circle
            cx, cy = points[0]
            x2, y2 = points[1]
            r = np.hypot(x2 - cx, y2 - cy)
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, lw=2))
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    # Connect mouse click event
    cid = fig.canvas.mpl_connect("button_press_event", _onclick)
    plt.show()

    # Return results if valid
    if len(points) == 2:
        cx, cy = points[0]
        x2, y2 = points[1]
        return (cx, cy, np.hypot(x2 - cx, y2 - cy))

    messagebox.showinfo(
        "Information",
        "Circle not defined – falling back to default filter."
    )
    return None


# ─────────────────────────────────────────────
# function for draw a rectangle to the spatial filter
# ─────────────────────────────────────────────
def draw_manual_rectangle(current_ft_array=None, arr_ft=None, arr_hologram=None):
    """
    Allows the user to manually draw a rectangular region over the Fourier transform
    to select a region of interest (typically the +1 diffraction order in DHM).

    Parameters:
    -----------
    current_ft_array : ndarray or None
        Precomputed Fourier transform to display (optional).
    arr_ft : ndarray or None
        Alternative precomputed Fourier transform to use if current_ft_array is not provided.
    arr_hologram : ndarray or None
        Raw hologram from which to compute the Fourier transform if no FT is provided.

    Returns:
    --------
    tuple or None
        A 4-element tuple (x1, y1, x2, y2) with the rectangle's pixel coordinates,
        or None if the user cancels or no selection is made.
    """

    # Select the Fourier transform to display, preferring an explicitly passed array
    if current_ft_array is not None:
        ft_to_show = current_ft_array
    elif arr_ft is not None and arr_ft.size:
        ft_to_show = arr_ft
    elif arr_hologram is not None and arr_hologram.size:
        # Compute the Fourier transform of the hologram
        ft_to_show = np.abs(np.fft.fftshift(np.fft.fft2(arr_hologram)))
    else:
        messagebox.showinfo(
            "Information",
            "No Fourier transform available for manual filtering."
        )
        return None

    # Apply logarithmic scaling for better visualization contrast
    ft_disp = np.log1p(ft_to_show)

    # Create the interactive plot window
    fig, ax = plt.subplots(num="Draw a rectangle to select the +1 order")
    ax.imshow(ft_disp, cmap="gray", origin="upper")
    ax.set_title("Click-and-drag with the LEFT mouse button")

    coords = []  # Store the rectangle coordinates here

    # Callback function executed when the user completes the rectangle selection
    def _on_select(eclick, erelease):
        # Capture coordinates from mouse events
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        # Normalize to ensure (x1,y1) is top-left and (x2,y2) is bottom-right
        coords.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        plt.close(fig)  # Close the plot window after selection

    # Initialize the RectangleSelector widget for manual drawing
    selector = RectangleSelector(
        ax, _on_select,
        interactive=True,
        useblit=False,
        drag_from_anywhere=True,
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        button=[1]  # Use left mouse button
    )

    # Start the interactive plot (blocking until the window is closed)
    plt.show(block=True)

    # Return the selected coordinates if a rectangle was drawn, otherwise None
    return coords[0] if coords else None


# ─────────────────────────────────────────────
# Angular spectrum propagator
# ─────────────────────────────────────────────
def angularSpectrum(field, z, wavelength, dx, dy):
    '''
    # Function to diffract a complex field using the angular spectrum approximation
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx,dy - sampling pitches
    '''

    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    phase = np.exp(1j * z * 2 * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

    tmp = field_spec * phase

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out


# ─────────────────────────────────────────────
# Sharpness metric Tenengrand Variance
# ─────────────────────────────────────────────
def metric_tenv(amp):
    """
    Parameters:
    amp: 2D numpy array representing the amplitude of the optical field

    Returns:
    FTENV: Tenengrad variance value (higher values indicate better focus)
    """
    # Define Sobel operators
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)

    amp = amp.astype(np.float64)

    # Apply Sobel filters (equivalent to imfilter with 'replicate' and 'conv')
    Gx = ndimage.convolve(amp, sobel_x, mode='reflect')
    Gy = ndimage.convolve(amp, sobel_y, mode='reflect')

    # Calculate gradient magnitude squared
    G = Gx ** 2 + Gy ** 2

    # Calculate variance
    FTENV = np.var(G)

    return FTENV


# ─────────────────────────────────────────────
# Sharpness metric Normalized Variance
# ─────────────────────────────────────────────
def metric_nv(amplitude):
    """
    Normalized Variance (NV) as defined by:
    NV = sum((A - mean)^2) / mean

    Parameters:
    amplitude : 2D numpy array
        Amplitude or intensity image.

    Returns:
    nv_value : float
        Computed normalized variance.
    """
    amplitude = amplitude.astype(np.float64)
    mean_val = np.mean(amplitude)

    if mean_val == 0:
        return 0.0

    diff_squared = (amplitude - mean_val) ** 2
    nv_value = np.sum(diff_squared) / mean_val

    return nv_value


# ─────────────────────────────────────────────
# Autofocus block
# ─────────────────────────────────────────────
def autofocus_field(field, z_range, wavelength, dx, dy, steps=50, metric_fn=None,
                    progress_callback=None, plot_results=False, roi=None):
    """
    Performs autofocus by sweeping through a range of distances and selecting the one
    that yields the maximum sharpness according to the provided metric.

    Parameters:
        field : ndarray
            Complex field to be propagated.
        z_range : tuple(float, float)
            The (z_min, z_max) propagation range in micrometers.
        wavelength : float
            Wavelength in micrometers.
        dx, dy : float
            Pixel pitch in micrometers.
        steps : int
            Number of propagation steps to evaluate.
        metric_fn : callable
            Sharpness metric function (e.g., NV, Tenengrad).
        progress_callback : callable or None
            Optional function to report progress (e.g., to a progress bar).
        plot_results : bool
            Whether to plot the sharpness metric curve after autofocus.

    Returns:
        best_z : float - optimal propagation distance (µm)
        z_vals : list - propagation distances (µm)
        scores : list - sharpness scores per distance
    """
    if field is None:
        messagebox.showinfo(
            "Information",
            "Field is None."
        )
        return
        return None

    z_min, z_max = z_range
    if z_min >= z_max:
        messagebox.showinfo(
            "Invalid Range",
            "The specified range is invalid. Please double-check the minimum and maximum values."
        )
        return None

    z_vals = np.linspace(z_min, z_max, steps)
    scores = []

    best_z = z_vals[0]
    best_score = np.inf

    for i, z in enumerate(z_vals):
        propagated = angularSpectrum(field, z, wavelength, dx, dy)
        amplitude = np.abs(propagated)

        # ROI to evaluate metric
        x1, y1, x2, y2 = roi
        amplitude = amplitude[y1:y2, x1:x2]

        # Evaluate the sharpness metric
        score = metric_fn(amplitude) if metric_fn else 0.0
        scores.append(score)

        # Check the best score so far
        if score < best_score:
            best_score = score
            best_z = z

        # Report progress if callback is defined
        if progress_callback:
            progress_callback(i + 1, len(z_vals))

    return best_z, z_vals, scores


# ─────────────────────────────────────────────
# Coherent image to optical field
# ─────────────────────────────────────────────
def convert_loaded_image_to_field(field):
    """
    Converts a loaded coherent image into a complex optical field.

    This assumes the image was captured under coherent illumination (e.g., laser).
    The function constructs a complex field with the image intensity as amplitude
    and a constant zero phase (i.e., flat wavefront).

    Result is stored in: self.coherent_field_complex
    """
    if not hasattr(field, "coherent_input_image") or field.coherent_input_image is None:
        messagebox.showinfo(
            "Information",
            "No coherent image loaded."
        )
        return

    # Convert image to normalized float
    img_gray = np.asarray(field.coherent_input_image).astype(np.float32)
    img_gray = img_gray / 255.0  # Normalize to [0, 1]

    # Map grayscale values to phase range [-π, π]
    phase_map = 2 * np.pi * img_gray - np.pi

    # Generate complex field with zero phase (real amplitude only)
    # field.coherent_field_complex = img_gray * np.exp(1j * np.zeros_like(img_gray))
    field.coherent_field_complex = img_gray * np.exp(1j * phase_map)

    return field.coherent_field_complex


# ─────────────────────────────────────────────
# Functions to apply Vortex
# ─────────────────────────────────────────────
def hilbert_transform_2d(c, hilbert_or_energy_operator=True):
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
        cuadrature : 2D numpy array (complex)
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

    C = np.fft.fft2(c)

    if hilbert_or_energy_operator:
        CH = C * np.fft.ifftshift(H)
    else:
        CH = C * np.fft.ifftshift(1j * H)

    cuadrature = np.conj(np.fft.ifft2(CH))
    return cuadrature


def vortex_compensation(field, fxOverMax, fyOverMax):
    cropVortex = 5
    factorOverInterpolation = 55

    # Crop around the max frequency
    sd = field[
        int(fyOverMax - cropVortex) : int(fyOverMax + cropVortex),
        int(fxOverMax - cropVortex) : int(fxOverMax + cropVortex)
    ]

    # Hilbert transform
    sd_crop = hilbert_transform_2d(sd, hilbert_or_energy_operator=1)

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
    Ml = np.fft.fftshift(Ml)
    Ml[70:, 70:] = 0
    Ml = np.fft.ifftshift(Ml)

    linearIndex = np.argmin(Ml)
    yOverInterpolVortex, xOverInterpolVortex = np.unravel_index(linearIndex, Ml.shape)

    positions = []
    pos_x = (xOverInterpolVortex / factorOverInterpolation) + (fxOverMax - cropVortex)
    pos_y = (yOverInterpolVortex / factorOverInterpolation) + (fyOverMax - cropVortex)
    #positions.append([x_pos, y_pos])

    return pos_x, pos_y


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reference_wave(fx_max, fy_max, m, n, _lambda, dx, dy, k, fx_0, fy_0, M, N):
    arg_x = (fx_0 - fx_max) * _lambda / (M * dx)
    arg_y = (fy_0 - fy_max) * _lambda / (N * dy)
    theta_x = np.arcsin(arg_x)
    theta_y = np.arcsin(arg_y)
    ref_wave = np.exp(1j * k * (dx * np.sin(theta_x) * m + dy * np.sin(theta_y) * n))
    return ref_wave


def legendre_compensation(field_compensate, limit, NoPistonCompensation=True, UsePCA=False):
    """
    Compensates the phase of a complex field using a fit with Legendre polynomials.

    Parameters:
    -----------
    field_compensate : np.ndarray
        Complex field to be corrected.
    limit : int
        Radius of the region to analyze around the center.
    NoPistonCompensation : bool
        Indicates whether to skip compensation of the constant term (piston).
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
    fftField = np.fft.fftshift(np.fft.fft2(field_compensate))

    A, B = fftField.shape
    center_A = int(round(A / 2))
    center_B = int(round(B / 2))

    start_A = int(center_A - limit)
    end_A = int(center_A + limit)
    start_B = int(center_B - limit)
    end_B = int(center_B + limit)

    fftField = fftField[start_A:end_A, start_B:end_B]
    square = np.fft.ifft2(np.fft.ifftshift(fftField))

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

    if NoPistonCompensation:
        # Piston compensation disabled
        # Optionally skip the first coefficient in the reconstruction
        pass
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
