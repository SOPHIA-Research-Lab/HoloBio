import cv2
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage import measure, morphology
import pandas as pd
from tkinter import ttk
from tkinter import simpledialog, messagebox, filedialog
from tkinter import messagebox, Toplevel
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt


def create_binary_mask(image: np.ndarray, method: str = 'otsu', manual_threshold: float = 0.5) -> np.ndarray:
    """
    Create a binary mask from an image using the selected thresholding method.

    Parameters:
        image (np.ndarray): Input grayscale image.
        method (str): Thresholding method: 'otsu', 'manual', or 'adaptive'.
        manual_threshold (float): Manual threshold value in range [0.0, 1.0].

    Returns:
        np.ndarray: Binary mask (boolean array).
    """

    if image is None:
        raise ValueError("No image provided to create_binary_mask().")
        messagebox.showinfo(
            "Information",
            "The specified range is invalid. Please double-check the minimum and maximum values."
        )

    # Close any existing figures
    plt.close('all')

    if method == 'otsu':
        threshold_value, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)

    elif method == 'manual':
        #threshold_value = int(manual_threshold * 255)
        _, binary = cv2.threshold(image, manual_threshold, 255, cv2.THRESH_BINARY)
        threshold_value = manual_threshold
        binary = cv2.bitwise_not(binary)

    elif method == 'adaptive':
        threshold_value = None
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
        binary = cv2.bitwise_not(binary)

    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    if method != 'adaptive':
        plt.figure(figsize=(6, 4))
        plt.hist(image.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.axvline(threshold_value, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold = {threshold_value:.1f}')
        plt.axvspan(0, threshold_value, color='red', alpha=0.2, label='Below threshold')
        plt.axvspan(threshold_value, 255, color='green', alpha=0.2, label='Above threshold')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram with Threshold')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    # Convert to boolean mask
    binary_mask = binary > 0

    # Additional processing to separate touching samples using watershed
    if np.sum(binary_mask) > 0:
        binary_mask = separate_touching_samples(binary_mask)

    # Show result
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask - Is the sample white or black?")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)

    return binary_mask, threshold_value


def separate_touching_samples(binary_mask):
    """Separate touching samples using watershed segmentation"""
    # Calculate distance transform
    distance = distance_transform_edt(binary_mask)

    # Find local maxima
    from scipy import ndimage
    local_maxima = ndimage.maximum_filter(distance, size=20) == distance
    local_maxima = local_maxima & (distance > 5)  # Minimum distance threshold

    # Create markers for watershed
    markers = measure.label(local_maxima)

    if np.max(markers) > 1:
        print(f"Applying watershed separation: found {np.max(markers)} potential sample centers")

        # Apply watershed
        labels = watershed(-distance, markers, mask=binary_mask)

        # Convert back to binary (any labeled region becomes True)
        separated_mask = labels > 0

        # Compare results
        original_components = len(measure.regionprops(measure.label(binary_mask)))
        new_components = len(measure.regionprops(measure.label(separated_mask)))
        print(f"Watershed result: {original_components} → {new_components} components")

        return separated_mask
    else:
        print("No watershed separation needed (single component or no clear centers)")
        return binary_mask


def visualize_detection_step(labeled_image, all_regions, samples_circles):
    """Visualize the sample detection process (simplified - removed area histogram)"""
    # Close all existing figures
    plt.close('all')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Labeled regions (all)
    ax1.imshow(labeled_image, cmap='nipy_spectral')
    ax1.set_title(f'Connected Components ({len(all_regions)} found)')
    ax1.axis('off')

    # Detected samples with circles
    ax2.imshow(labeled_image, cmap='gray')
    for i, sam in enumerate(samples_circles):
        circle = plt.Circle((sam['center_x'], sam['center_y']),
                            sam['diameter'] / 2, fill=False, color='red', linewidth=2)
        ax2.add_patch(circle)
        ax2.plot(sam['center_x'], sam['center_y'], 'r+', markersize=8, markeredgewidth=2)
        ax2.text(sam['center_x'], sam['center_y'] - sam['diameter'] / 2 - 5,
                 f'{i}', color='yellow', fontsize=8, ha='center',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax2.set_title(f'Detected Samples ({len(samples_circles)} valid)')
    ax2.axis('off')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle('Samples Detection and Filtering', y=0.98, fontsize=14)

    def bring_to_front():
        fig.canvas.manager.window.lift()
        fig.canvas.manager.window.attributes('-topmost', 1)
        fig.canvas.manager.window.attributes('-topmost', 0)

    # Call lift after 100 ms to ensure the window is ready
    fig.canvas.manager.window.after(100, bring_to_front)

    plt.show()


def show_text_popup(title, text, geometry="400x200", parent=None):
    # Close any existing popup with the same title
    if parent is not None:
        for widget in parent.winfo_children():
            if hasattr(widget, 'title') and widget.title() == title:
                widget.destroy()


    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.geometry(geometry)
    popup.transient(parent)
    popup.lift()
    popup.attributes('-topmost', True)
    popup.after(200, lambda: popup.attributes('-topmost', False))
    popup.focus_force()

    # Frame
    frame = tk.Frame(popup)
    frame.pack(fill='both', expand=True)

    # Text with scroll
    text_widget = ScrolledText(frame, wrap='word', font=('Courier', 10), height=10)
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(fill='both', expand=True)

    # Button save
    def save_to_file():
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(text)

    tk.Button(popup, text="💾 Save as .txt", command=save_to_file).pack(pady=5)


def apply_count_particles(image: np.ndarray, method: str, threshold=None, min_area=100, max_area=10000, parent=None):
    """
    Applies thresholding to the image based on the selected method and parameters,
    and displays the resulting binary mask. Returns count information.
    """
    # Close all existing figures at the start
    plt.close('all')

    # Get processed data
    final_mask, samples_circles, threshold_value, sample_is_white = process_particles(
        image, method, threshold, min_area, max_area, parent=parent
    )

    # Show final mask
    plt.figure(figsize=(6, 6))
    plt.imshow(final_mask, cmap='gray')
    plt.title("Final Mask after Cleaning")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Summary
    output = []
    output.append("=== Particle Detection Summary ===")
    output.append(f"Thresholding method: {method}")
    output.append(
        f"Manual threshold: {threshold_value:.1f}" if method == 'manual'
        else f"Otsu threshold: {threshold_value:.1f}" if method == 'otsu'
        else "Adaptive method (no single threshold)"
    )
    output.append(f"Sample polarity: {'White' if sample_is_white else 'Black'}")
    output.append(f"Accepted particles (within area range): {len(samples_circles)}")

    summary_text = "\n".join(output)
    show_text_popup("Particle Detection Summary", summary_text, geometry="400x200")

    return final_mask, samples_circles


def apply_area_particles(image: np.ndarray, method: str, threshold=None, min_area=100, max_area=10000, μm_per_px=1.0, parent=None):
    """
    Performs particle detection and provides detailed area analysis including:
    - Histogram of particle areas
    - Excel-like table with particle details
    - Statistical summary of areas
    """
    # Close all existing figures at the start
    plt.close('all')

    # Get processed data
    final_mask, samples_circles, threshold_value, sample_is_white = process_particles(
        image, method, threshold, min_area, max_area, μm_per_px
    )

    if not samples_circles:
        print("No particles found for area analysis.")
        return final_mask, samples_circles

    # Extract areas for analysis
    areas = [particle['area'] for particle in samples_circles]
    areas_um2 = [a * (μm_per_px ** 2) for a in areas]

    # Create histogram
    fig = plt.figure(figsize=(10, 6))
    plt.hist(areas_um2, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Particle Area (µm²)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Particle Areas')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Bring figure to front
    def bring_to_front():
        fig.canvas.manager.window.lift()
        fig.canvas.manager.window.attributes('-topmost', 1)
        fig.canvas.manager.window.attributes('-topmost', 0)

    fig.canvas.manager.window.after(100, bring_to_front)

    plt.show()

    # Create Excel-like table
    create_particle_table(samples_circles, parent=parent)

    # Statistical analysis
    area_mean_um2 = np.mean(areas_um2)
    area_std_um2 = np.std(areas_um2)
    area_min_um2 = np.min(areas_um2)
    area_max_um2 = np.max(areas_um2)

    # Prepare detailed statistics
    stats_output = []
    stats_output.append("=== Area Analysis Summary ===")
    stats_output.append(f"Total particles analyzed: {len(samples_circles)}")
    stats_output.append(f"Average area: {area_mean_um2:.2f} ± {area_std_um2:.2f} µm²")
    stats_output.append(f"Area range: {area_min_um2:.2f} - {area_max_um2:.2f} µm²")
    stats_output.append("")
    stats_output.append("Individual particle areas:")

    for i, area_um2 in enumerate(areas_um2, 1):
        stats_output.append(f"  Particle {i:2d}: {area_um2:8.2f} µm²")

    stats_text = "\n".join(stats_output)
    show_text_popup("Area Analysis Results", stats_text, geometry="500x400")

    return final_mask, samples_circles


def process_particles(image: np.ndarray, method: str, threshold=None, min_area=100, max_area=10000, μm_per_px=1.0, parent=None):
    """
    Core particle processing function that handles thresholding, cleaning, and detection.
    Returns: final_mask, samples_circles, threshold_value, sample_is_white
    """
    # Create binary mask
    binary_mask, threshold_value = create_binary_mask(
        image=image,
        method=method,
        manual_threshold=threshold if threshold is not None else 0.5
    )

    # Ask for sample polarity
    answer = simpledialog.askstring(
        "Confirm Sample Polarity",
        "Does the sample appear white (w) or black (b)?\nPlease type: w or b",
        parent=parent
    )
    sample_is_white = True
    if answer:
        answer = answer.strip().lower()
        sample_is_white = (answer == 'w')
    print(f"User selected {'WHITE' if sample_is_white else 'BLACK'} sample.")

    # Clean samples outside regions
    if sample_is_white:
        print("Cleaning WHITE regions...")
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)
    else:
        print("Cleaning BLACK regions...")
        cleaned_mask = ~morphology.remove_small_objects(~binary_mask, min_size=min_area)

    final_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=50)

    if sample_is_white:
        mask_for_analysis = final_mask
    else:
        mask_for_analysis = ~final_mask

    # Label connected components
    labeled_image = measure.label(mask_for_analysis)
    regions = measure.regionprops(labeled_image)

    print(f"\nConnected component analysis:")
    print(f"  - Total components found: {len(regions)}")

    if len(regions) == 0:
        print("ERROR: No connected components found!")
        return final_mask, [], threshold_value, sample_is_white

    # Show area distribution of all components
    all_areas = [region.area for region in regions]
    print(f"  - Component areas: min={min(all_areas)}, max={max(all_areas)}, mean={np.mean(all_areas):.1f}")
    print(f"  - Area filter range: {min_area} - {max_area}")

    samples_circles = []

    for i, region in enumerate(regions):
        area = region.area

        # Filter by area
        if min_area <= area <= max_area:
            # Calculate circle parameters
            y_center, x_center = region.centroid

            # Estimate diameter from area (assuming circular samples)
            diameter_from_area = 2 * np.sqrt(area / np.pi)

            # Alternative: use equivalent diameter
            equivalent_diameter = region.equivalent_diameter

            # Use the larger of the two estimates for safety
            diameter = max(diameter_from_area, equivalent_diameter)

            samples_circles.append({
                'center_x': x_center,
                'center_y': y_center,
                'diameter': diameter,
                'diameter_um': diameter * μm_per_px,
                'area': area,
                'area_um2': area * (μm_per_px ** 2),
                'label': region.label,
                'bbox': region.bbox
            })

    print(f"\nFinal result: {len(samples_circles)} samples candidates accepted")

    if len(samples_circles) == 0:
        print("\nTROUBLESHOOTING SUGGESTIONS:")
        print(f"1. Adjust area limits:")
        print(f"   - Current min_area: {min_area}")
        print(f"   - Current max_area: {max_area}")
        print(f"   - Suggested min_area: {max(10, min(all_areas))}")
        print(f"   - Suggested max_area: {max(all_areas)}")

    # Visualize detection step (if function exists)
    try:
        visualize_detection_step(labeled_image, regions, samples_circles)
    except NameError:
        print("Visualization function not available")

    return final_mask, samples_circles, threshold_value, sample_is_white


def create_particle_table(samples_circles, parent=None):
    """
    Creates a professional data table showing particle details in a new window.
    Similar to the Line Profile Data Table style.
    """
    # Close any existing table window
    if parent is not None:
        for widget in parent.winfo_children():
            if hasattr(widget, 'title') and 'Particle Data Table' in widget.title():
                widget.destroy()

    # Create new window
    table_window = tk.Toplevel()
    table_window.title("🔬 Particle Data Table")
    table_window.geometry("650x500")
    table_window.configure(bg='#f0f0f0')
    table_window.lift()
    table_window.focus_force()

    # Create main frame
    main_frame = tk.Frame(table_window, bg='#f0f0f0')
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # Create toolbar frame
    toolbar_frame = tk.Frame(main_frame, bg='#e0e0e0', height=40)
    toolbar_frame.pack(fill='x', pady=(0, 5))
    toolbar_frame.pack_propagate(False)

    # Add toolbar buttons
    def export_to_csv():
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Particle Data"
            )
            if filename:
                # Create DataFrame
                data = []
                for i, particle in enumerate(samples_circles, 1):
                    data.append({
                        'Particle_ID': i,
                        'Center_X (px)': round(particle['center_x'], 2),
                        'Center_Y (px)': round(particle['center_y'], 2),
                        'Area (µm²)': round(particle.get('area_um2', 0.0), 2),
                        'Diameter (µm)': round(particle.get('diameter_um', 0.0), 2)
                    })

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                print(f"Data exported to {filename}")

                # Show success message
                success_popup = tk.Toplevel(table_window)
                success_popup.title("Export Successful")
                success_popup.geometry("300x100")
                success_popup.configure(bg='white')
                success_popup.lift()
                success_popup.focus_force()
                tk.Label(success_popup, text=f"Data exported successfully!\n{filename}",
                         bg='white', font=('Arial', 10)).pack(expand=True)
                tk.Button(success_popup, text="OK", command=success_popup.destroy).pack(pady=5)

        except Exception as e:
            print(f"Export failed: {e}")

    def copy_to_clipboard():
        try:
            # Create tab-separated text for clipboard
            header = "Particle ID\tCenter X (px)\tCenter Y (px)\tArea (µm²)\tDiameter (µm)\n"
            data_text = header

            for i, particle in enumerate(samples_circles, 1):
                area_um2 = particle.get('area_um2', 0.0)
                diameter_um = particle.get('diameter_um', 0.0)

                data_text += (f"{i}\t"f"{particle['center_x']:.2f}\t"f"{particle['center_y']:.2f}\t"f"{area_um2:.2f}\t"f"{diameter_um:.2f}\n")

            table_window.clipboard_clear()
            table_window.clipboard_append(data_text)
            print("Data copied to clipboard")

        except Exception as e:
            print(f"Copy failed: {e}")

    # Toolbar buttons
    tk.Button(toolbar_frame, text="💾 Save as CSV", command=export_to_csv,
              font=('Arial', 9), bg='white', relief='raised').pack(side='left', padx=5, pady=5)
    tk.Button(toolbar_frame, text="📋 Copy", command=copy_to_clipboard,
              font=('Arial', 9), bg='white', relief='raised').pack(side='left', padx=5, pady=5)

    # Info label
    info_label = tk.Label(toolbar_frame, text=f"{len(samples_circles)} rows × 5 columns",
                          font=('Arial', 9), bg='#e0e0e0', fg='#666666')
    info_label.pack(side='right', padx=10, pady=5)

    # Create table frame
    table_frame = tk.Frame(main_frame, bg='white', relief='sunken', bd=1)
    table_frame.pack(fill='both', expand=True)

    # Create treeview (table) with alternating row colors
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Treeview", background="white", foreground="black",
                    rowheight=25, fieldbackground="white")
    style.configure("Treeview.Heading", background="#4a90e2", foreground="white",
                    font=('Arial', 10, 'bold'))
    style.map('Treeview', background=[('selected', '#0078d4')])

    columns = ('ID', 'Center_X', 'Center_Y', 'Area_um2', 'Diameter_um')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

    # Define headings
    tree.heading('ID', text='Particle ID')
    tree.heading('Center_X', text='Center X (px)')
    tree.heading('Center_Y', text='Center Y (px)')
    tree.heading('Area_um2', text='Area (µm²)')
    tree.heading('Diameter_um', text='Diameter (µm)')

    # Configure column widths
    tree.column('ID', width=100, anchor='center')
    tree.column('Center_X', width=120, anchor='center')
    tree.column('Center_Y', width=120, anchor='center')
    tree.column('Area_um2', width=120, anchor='center')
    tree.column('Diameter_um', width=130, anchor='center')

    # Insert data with alternating colors
    for i, particle in enumerate(samples_circles, 1):
        # Add row with proper formatting
        tree.insert('', 'end', values=(
            i,
            f"{particle['center_x']:.2f}",
            f"{particle['center_y']:.2f}",
            f"{particle.get('area_um2', 0.0):.2f}",
            f"{particle.get('diameter_um', 0.0):.2f}"
        ), tags=('evenrow' if i % 2 == 0 else 'oddrow',))

    # Configure row colors
    tree.tag_configure('evenrow', background='#f8f9fa')
    tree.tag_configure('oddrow', background='white')

    # Add scrollbars
    v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
    h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview)
    tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # Pack table and scrollbars
    tree.grid(row=0, column=0, sticky='nsew')
    v_scrollbar.grid(row=0, column=1, sticky='ns')
    h_scrollbar.grid(row=1, column=0, sticky='ew')

    # Configure grid weights
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    # Add status bar
    status_frame = tk.Frame(main_frame, bg='#e0e0e0', height=25)
    status_frame.pack(fill='x', pady=(5, 0))
    status_frame.pack_propagate(False)

    status_label = tk.Label(status_frame, text=f"Ready - {len(samples_circles)} particles analyzed",
                            font=('Arial', 9), bg='#e0e0e0', anchor='w')
    status_label.pack(side='left', padx=10, pady=2)


def apply_thickness(image: np.ndarray,
                    method='otsu',
                    threshold=None,
                    min_area=100,
                    max_area=10000,
                    ind_sample=1.33,
                    ind_medium=1.00,
                    wavelength_um=None,
                    parent=None):
    """
    Calculates thickness from a grayscale phase image and also displays the delta-phase (Δφ) map.
    Assumes phase is encoded in 8-bit (0-255) and rescales to [-π, π].
    """

    # Mode must be PHASE
    if hasattr(parent, 'imageProcess_var') and parent.imageProcess_var.get() == 0:
        from tkinter import messagebox
        messagebox.showwarning(
            "Invalid Mode",
            "Thickness analysis is only valid for PHASE images.\nPlease select a phase image."
        )
        return

    # Validate wavelength
    if wavelength_um is None or wavelength_um <= 0:
        messagebox.showinfo(
            "Information",
            "Please enter a valid Wavelength (µm) before proceeding."
        )
        return

    # Validate refractive indices
    if abs(ind_sample - ind_medium) < 1e-9:
        messagebox.showinfo(
            "Information",
            "'Ind. Sample' must be different from 'Ind. Medium' to compute thickness."
        )
        return

    # Scale image [0, 255] → [-π, π]
    phase_image = (image.astype(np.float32) / 255.0) * (2 * np.pi) - np.pi

    # Binary mask
    binary_mask, _ = create_binary_mask(image=image, method=method, manual_threshold=threshold)

    # Polarity dialog
    answer = simpledialog.askstring("Sample Polarity", "Is the sample white (w) or black (b)?", parent=parent)
    if answer is None:
        print("User cancelled input.")
        return
    sample_is_white = answer.strip().lower() == 'w'

    # Background vs sample
    background_mask = ~binary_mask if sample_is_white else binary_mask
    sample_mask = binary_mask if sample_is_white else ~binary_mask

    # Δφ (relative to background)
    avg_background_phase = np.mean(phase_image[background_mask])
    delta_phi = np.abs(phase_image - avg_background_phase)

    # ---- NEW: visualize Δφ (delta phase) ----
    fig_delta = plt.figure(figsize=(6, 5))
    im1 = plt.imshow(delta_phi, cmap='inferno')
    plt.colorbar(im1, label='Δφ (rad)')
    plt.title("Delta Phase (Δφ) Map")
    plt.axis('off')
    plt.tight_layout()

    def _bring_to_front_delta():
        try:
            fig_delta.canvas.manager.window.lift()
            fig_delta.canvas.manager.window.attributes('-topmost', 1)
            fig_delta.canvas.manager.window.attributes('-topmost', 0)
        except Exception:
            pass

    # Call lift after 100 ms to ensure the window is ready
    fig_delta.canvas.manager.window.after(100, _bring_to_front_delta)

    # Thickness (uses wavelength from GUI)
    n_diff = ind_sample - ind_medium
    thickness_map = (delta_phi * wavelength_um) / (2 * np.pi * n_diff)
    thickness_map[~sample_mask] = 0

    # Thickness visualization
    fig_thick = plt.figure(figsize=(6, 5))
    im2 = plt.imshow(thickness_map, cmap='inferno')
    plt.colorbar(im2, label='Thickness (µm)')
    plt.title("Estimated Thickness Map")
    plt.axis('off')
    plt.tight_layout()

    def _bring_to_front_thick():
        try:
            fig_thick.canvas.manager.window.lift()
            fig_thick.canvas.manager.window.attributes('-topmost', 1)
            fig_thick.canvas.manager.window.attributes('-topmost', 0)
        except Exception:
            pass

    # Call lift after 100 ms to ensure the window is ready
    fig_thick.canvas.manager.window.after(100, _bring_to_front_thick)

    # Show both windows
    plt.show()

    return thickness_map


def automaticProfile(image: np.ndarray, method: str, threshold=None, min_area=100, max_area=10000, μm_per_px=1.0, parent=None):
    print("  ✔ Automatic Phase Profile")
    plt.close('all')

    final_mask, samples_circles, threshold_value, sample_is_white = process_particles(
        image, method, threshold, min_area, max_area, μm_per_px
    )

    if not samples_circles:
        messagebox.showinfo("Information", "No particles found for area analysis.")
        return []

    H, W = final_mask.shape
    profile_extension = 1.5
    mask_for_label = final_mask if sample_is_white else ~final_mask
    labeled_image = measure.label(mask_for_label, connectivity=2)

    def find_profile_endpoints(sample):
        center_x, center_y = sample['center_x'], sample['center_y']
        diameter = sample['diameter']
        label_excl = sample.get('label')
        min_extension = (diameter / 2) * 1.25
        max_extension = (diameter / 2) * profile_extension
        angles = np.linspace(0, np.pi, 36)

        for angle in angles:
            for extension in np.linspace(min_extension, max_extension, 8):
                dx = extension * np.cos(angle)
                dy = extension * np.sin(angle)
                x1 = center_x - dx; y1 = center_y - dy
                x2 = center_x + dx; y2 = center_y + dy

                if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
                    continue

                xi1 = int(np.clip(round(x1), 0, W - 1))
                yi1 = int(np.clip(round(y1), 0, H - 1))
                xi2 = int(np.clip(round(x2), 0, W - 1))
                yi2 = int(np.clip(round(y2), 0, H - 1))

                v1 = final_mask[yi1, xi1]
                v2 = final_mask[yi2, xi2]
                inside1 = v1 if sample_is_white else (not v1)
                inside2 = v2 if sample_is_white else (not v2)
                if inside1 or inside2:
                    continue

                npts = max(int(np.hypot(x2 - x1, y2 - y1)), 2)
                xs = np.linspace(x1, x2, npts).astype(int)
                ys = np.linspace(y1, y2, npts).astype(int)
                xs = np.clip(xs, 0, W - 1)
                ys = np.clip(ys, 0, H - 1)

                line_labels = labeled_image[ys, xs]
                labs = np.unique(line_labels)
                labs = labs[labs != 0]
                if label_excl is not None:
                    labs = labs[labs != label_excl]
                if len(labs) > 0:
                    continue

                return (x1, y1), (x2, y2), angle, extension
        return None

    samples_profiles = []
    valid_samples = []
    failed_samples = []

    for i, sample in enumerate(samples_circles):
        print(f"Processing Sample {i + 1}/{len(samples_circles)}...")
        res = find_profile_endpoints(sample)
        if res is None:
            print(f"Sample {i}: Could not find valid profile line - EXCLUDED from analysis")
            failed_samples.append(i)
            continue

        (x1, y1), (x2, y2), angle, extension = res
        num_points = max(int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)), 2)
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)
        x_coords = np.clip(x_coords, 0, W - 1)
        y_coords = np.clip(y_coords, 0, H - 1)

        phase_img = (image.astype(np.float32) / 255.0) * (2 * np.pi) - np.pi
        '''
        # map integer images to [0, 2π]
        if np.issubdtype(image.dtype, np.integer):
            maxv = np.iinfo(image.dtype).max
            phase_img = (phase_img / maxv) * (2 * np.pi)
        else:
            # heuristic: if dynamic range >> 2π, normalize to [0, 2π]
            if phase_img.max() > 2 * np.pi * 1.5:
                mmin, mmax = phase_img.min(), phase_img.max()
                phase_img = (phase_img - mmin) / max(mmax - mmin, 1e-9) * (2 * np.pi)
        '''
        # sample and wrap to [-π, π]
        phase_profile = phase_img[y_coords, x_coords]
        phase_profile = (phase_profile + np.pi) % (2 * np.pi) - np.pi

        profile_dict = {
            'sample_id': i,
            'center': (sample['center_x'], sample['center_y']),
            'diameter': sample['diameter'],
            'area': sample['area'],
            'angle': angle,
            'extension': extension,
            'endpoints': ((x1, y1), (x2, y2)),
            'coordinates': (x_coords, y_coords),
            'phase_profile': phase_profile
        }

        samples_profiles.append(profile_dict)
        valid_samples.append(i)
        # print(f"  Sample {i}: Valid profile extracted (angle: {angle:.2f} rad, extension: {extension:.1f} px)")
        # print(f"  Endpoints: ({x1:.1f},{y1:.1f}) to ({x2:.1f},{y2:.1f})")

    samples_circles = [samples_circles[i] for i in valid_samples]

    print("\n Profile extraction summary:")
    print(f"  Valid samples: {len(samples_profiles)}")
    print(f"  Excluded samples: {len(failed_samples)}")
    if failed_samples:
        print(f"  Excluded sample IDs: {failed_samples}")

    if len(samples_profiles) > 0:
        plt.figure(figsize=(7, 7))
        im = plt.imshow(phase_img, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        ax = plt.gca()
        for i, prof in enumerate(samples_profiles):
            (x1, y1), (x2, y2) = prof['endpoints']
            s = samples_circles[i]
            circ = plt.Circle((s['center_x'], s['center_y']), s['diameter'] / 2,
                              fill=False, color='white', linewidth=1)
            ax.add_patch(circ)
            plt.plot([x1, x2], [y1, y2], linewidth=2)
            plt.plot([x1, x2], [y1, y2], 'o')
        plt.title(f'Valid Profile Lines ({len(samples_profiles)})')
        plt.axis('off')
        plt.colorbar(im, label='Phase (rad)')
        plt.tight_layout()
        plt.show()

    # phase-shift summary popup
    results = []
    pct = 0.05
    for p in samples_profiles:
        data = p['phase_profile']
        max_val = np.max(data); min_val = np.min(data)
        max_thr = max_val - max_val * pct
        min_thr = min_val + abs(min_val) * pct
        max_mask = data >= max_thr
        min_mask = data <= min_thr
        max_avg = np.mean(data[max_mask]) if np.any(max_mask) else max_val
        min_avg = np.mean(data[min_mask]) if np.any(min_mask) else min_val
        delta = float(abs(max_avg - min_avg))
        results.append((p['sample_id'], delta))

    if parent is None:
        root = tk.Tk(); root.withdraw()
        win = Toplevel()
    else:
        win = Toplevel(parent)
    win.title("Phase Shift Results")
    win.geometry("500x600")
    txt = ScrolledText(win, wrap='word', font=('Courier', 10))
    lines = ["=== Phase Shift Summary ===", f"Total samples: {len(results)}", ""]
    for sid, dphi in results:
        lines.append(f"Sample {sid+1}: Δφ = {dphi:.3f} rad")
    txt.insert("1.0", "\n".join(lines))
    txt.configure(state="disabled")
    txt.pack(expand=True, fill="both")

    return samples_profiles
