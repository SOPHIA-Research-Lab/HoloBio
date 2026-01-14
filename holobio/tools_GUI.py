
# tools_GUI.py
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import Toplevel
from pandastable import Table
from . import tools_microstructure as tmic
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
import tkinter as tk


# Apply Dimension in Bio-Analysis Frame
def apply_dimensions(app):
    if not getattr(app, "phase_arrays", []):
        messagebox.showinfo(
            "Information",
            "No phase data available. Please reconstruct first."
        )
        return
    option = app.dimensions_var.get()

    try:
        px_size = float(app.pixel_size_entry.get().strip())
    except Exception:
        messagebox.showinfo(
            "Information",
            "Please enter a valid pixel size."
        )
    try:
        mag = float(app.magnification_entry.get().strip())
    except Exception:
        messagebox.showinfo(
            "Information",
            "Please enter a valid lateral magnification."
        )
    if abs(mag) < 1e-12:
        mag = 1.0

    microns_per_pixel = px_size if option == 0 else px_size / mag
    app.microns_per_pixel = microns_per_pixel

    if option == 0:
        if app.current_holo_array is not None and app.current_holo_array.size > 0:
            img_array = app.current_holo_array.copy()
        else:
            print("No hologram data available.")
            return
        title_str = "Dimensions - Hologram"
    elif option == 1:  # Amplitude
        idx = getattr(app, 'current_amp_index', 0)
        if idx >= len(app.amplitude_arrays):
            print("No amplitude array in memory.")
            return
        img_array = app.amplitude_arrays[idx]
        title_str = "Dimensions - Amplitude"
    else:  # Phase
        idx = getattr(app, 'current_phase_index', 0)
        if idx >= len(app.phase_arrays):
            print("No phase array in memory.")
            return
        img_array = app.phase_arrays[idx]
        title_str = "Dimensions - Phase"

    arr_min, arr_max = img_array.min(), img_array.max()
    disp = np.clip((img_array - arr_min) / (arr_max - arr_min + 1e-9) * 255, 0, 255).astype(np.uint8)

    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(disp, cmap="gray", interpolation="nearest")

    _micron_axes(ax, width_px=disp.shape[1], height_px=disp.shape[0], μm_per_px=microns_per_pixel)
    ax.set_title(title_str)

    busy = {"busy": False}
    bar_line = bar_text = None

    try:
        default_len_um = float(app.scalebar_length_entry.get())
    except Exception:
        default_len_um = 10.0

    bar_px = default_len_um / microns_per_pixel
    MARGIN = 15
    x_left, y_bar = MARGIN, disp.shape[0] - MARGIN
    bar_line, = ax.plot([x_left, x_left + bar_px], [y_bar, y_bar], color="white", linewidth=3, picker=True)
    bar_text = ax.text(x_left + bar_px / 2, y_bar - 6, f"{default_len_um:.1f} µm",
                       color="white", fontsize=12, fontweight="bold", ha="center", va="bottom", picker=True)

    wire_scalebar_interaction(fig, ax, bar_line, bar_text, microns_per_pixel, busy)

    state = {"start": None, "temp": None, "motion_cid": None}

    def _toolbar_busy():
        tb = getattr(fig.canvas.manager, "toolbar", None)
        return tb and tb.mode

    def _is_on_bar(ev):
        return (bar_line is not None and bar_line.contains(ev)[0]) or \
               (bar_text is not None and bar_text.contains(ev)[0])

    def _on_click(ev):
        if busy["busy"] or _toolbar_busy():
            return
        if ev.button != 1 or ev.inaxes != ax or _is_on_bar(ev):
            return

        if state["start"] is None:
            state["start"] = (ev.xdata, ev.ydata)
            temp, = ax.plot([ev.xdata, ev.xdata], [ev.ydata, ev.ydata],
                            color="red", linewidth=1, linestyle="--")
            state["temp"] = temp

            def _on_motion(mv):
                if state["start"] is None or mv.inaxes != ax:
                    return
                x0, y0 = state["start"]
                temp.set_data([x0, mv.xdata], [y0, mv.ydata])
                fig.canvas.draw_idle()

            state["motion_cid"] = fig.canvas.mpl_connect("motion_notify_event", _on_motion)

        else:
            x0, y0 = state["start"]
            x1, y1 = ev.xdata, ev.ydata

            if state["temp"]:
                state["temp"].remove()
            if state["motion_cid"]:
                fig.canvas.mpl_disconnect(state["motion_cid"])

            dist_um = np.hypot(x1 - x0, y1 - y0) * microns_per_pixel
            ax.plot([x0, x1], [y0, y1], color="red", linewidth=2)
            ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1) - 10,
                    f"{dist_um:.2f} µm",
                    color="red", fontsize=12, fontweight="bold",
                    ha="center", va="bottom")
            fig.canvas.draw_idle()

            state.update(start=None, temp=None, motion_cid=None)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.show(block=False)


# Scale bar options inside Dimension in Bio-Analysis Frame
def wire_scalebar_interaction(fig, ax, bar_line, bar_text, microns_per_pixel, busy_flag):
    MARGIN = 15
    current_len_um = [float(bar_text.get_text().split()[0])]
    dragging = {"active": False, "dx": 0.0, "dy": 0.0}

    def _update_bar(new_len_um, cx, cy):
        h, w = ax.get_images()[0].get_array().shape
        cx = np.clip(cx, MARGIN, w - MARGIN)
        cy = np.clip(cy, MARGIN, h - MARGIN)

        bar_px = new_len_um / microns_per_pixel
        if bar_px > w - 2 * MARGIN:
            bar_px = w - 2 * MARGIN
            new_len_um = bar_px * microns_per_pixel
        x0, x1 = cx - bar_px / 2, cx + bar_px / 2

        bar_line.set_data([x0, x1], [cy, cy])
        bar_text.set_position((cx, cy - 6))
        bar_text.set_text(f"{new_len_um:.1f} µm")
        current_len_um[0] = new_len_um
        fig.canvas.draw_idle()

    def _on_pick(event):
        artist = event.artist
        if artist not in (bar_line, bar_text):
            return
        m = event.mouseevent

        if m.dblclick:
            busy_flag["busy"] = True
            new_len = simpledialog.askfloat(
                "Scale‑bar length",
                "New length (µm):",
                initialvalue=current_len_um[0],
                minvalue=0.1,
            )
            busy_flag["busy"] = False
            if new_len is not None:
                cx, cy = bar_line.get_xydata().mean(axis=0)
                _update_bar(new_len, cx, cy)
            return

        if m.button == 3:
            colour = simpledialog.askstring(
                "Scale‑bar colour",
                "Choose colour (white / black / red):",
            )
            if colour and colour.strip().lower() in {"white", "black", "red"}:
                col = colour.strip().lower()
                bar_line.set_color(col)
                bar_text.set_color(col)
                fig.canvas.draw_idle()
            return

        if m.button == 1:
            busy_flag["busy"] = True
            cx, cy = bar_line.get_xydata().mean(axis=0)
            dragging.update(active=True, dx=m.xdata - cx, dy=m.ydata - cy)

    def _on_motion(event):
        if dragging["active"]:
            _update_bar(current_len_um[0], event.xdata - dragging["dx"], event.ydata - dragging["dy"])

    def _on_release(event):
        if dragging["active"]:
            dragging["active"] = False
            busy_flag["busy"] = False

    def _on_press_any(event):
        if event.inaxes != ax or event.button != 3:
            return
        if bar_line.contains(event)[0] or bar_text.contains(event)[0]:
            return
        _update_bar(current_len_um[0], event.xdata, event.ydata)

    fig.canvas.mpl_connect("pick_event", _on_pick)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("button_press_event", _on_press_any)


# Apply QPI in Bio-Analysis Frame
def apply_QPI(app):
    if not getattr(app, "phase_arrays", []):
        messagebox.showinfo(
            "Information",
            "No phase data available. Please reconstruct first."
        )
        return
    idx = getattr(app, "current_phase_index", 0)
    if idx >= len(app.phase_arrays):
        print("Phase index out of range.")
        return

    stored_phase_8bit = app.phase_arrays[idx].copy()
    real_phase = stored_phase_8bit.astype(np.float32) / 255.0 * (2 * np.pi)

    # Scale information
    try:
        px_size = float(app.pixel_size_entry.get())
    except Exception:
        messagebox.showinfo(
            "Information",
            "Please enter a valid Pixel Size."
        )
    try:
        mag = float(app.magnification_entry.get())
    except Exception:
        messagebox.showinfo(
            "Information",
            "Please enter a valid Lateral Magnification."
        )
    μm_per_px = px_size / mag if px_size > 0 and mag > 1e-6 else 0.0

    # GUI selections
    qpi_type = app.qpi_type_var.get()

    try:
        n_profiles = int(app.zones_get_qpi_entry.get())
    except Exception:
        messagebox.showinfo(
            "Information",
            "Please enter a valid number of zones."
        )
        return

    mode = app.option_meas_var.get()
    λ = app.wavelength

    if mode == "Thickness":
        try:
            n_s = float(app.ind_sample_entry.get())
            n_m = float(app.ind_medium_entry.get())
            print(n_s,n_m )
            n_rel_known = n_s / n_m
        except Exception:
            messagebox.showinfo(
                "Information",
                "Please enter a valid Refractive index [sample / medium] value."
            )
            return
        d_known = None
    else:
        try:
            d_known = float(app.thickness_entry.get())
        except Exception:
            messagebox.showerror("Thickness", "Invalid thickness value.")
            messagebox.showinfo(
                "Information",
                "Please enter a valid Thickness value."
            )
            return
        n_rel_known = None

    # Show image and collect profile clicks
    plt.ion()
    fig_phase, ax_phase = plt.subplots()
    ax_phase.imshow(stored_phase_8bit, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_phase.set_title("Click two points per profile (Esc to finish)")

    _micron_axes(ax_phase, width_px=stored_phase_8bit.shape[1],
                     height_px=stored_phase_8bit.shape[0], μm_per_px=μm_per_px)

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    clicks = []
    lines_drawn, circ_drawn = [], []
    picking_done = {"done": False}

    def _on_key(event):
        if event.key == "escape":
            picking_done["done"] = True

    def _on_click(event):
        if picking_done["done"] or event.inaxes != ax_phase or event.button != 1:
            return
        clicks.append((event.xdata, event.ydata))
        idx_pair = len(clicks) // 2 - 1
        colour = colours[idx_pair % len(colours)]
        if len(clicks) % 2 == 0:
            x1, y1 = clicks[-2]
            x2, y2 = clicks[-1]
            if qpi_type == 0:  # line
                line, = ax_phase.plot([x1, x2], [y1, y2], color=colour, linewidth=2)
                lines_drawn.append(line)
            else:  # circle
                r = np.hypot(x2 - x1, y2 - y1)
                circ = plt.Circle((x1, y1), r, fill=False, color=colour, linewidth=2)
                ax_phase.add_patch(circ)
                circ_drawn.append(circ)
            fig_phase.canvas.draw_idle()
        if len(clicks) // 2 == n_profiles:
            picking_done["done"] = True

    fig_phase.canvas.mpl_connect("key_press_event", _on_key)
    fig_phase.canvas.mpl_connect("button_press_event", _on_click)

    while not picking_done["done"]:
        plt.pause(0.05)

    if len(clicks) // 2 < n_profiles:
        print("Operation cancelled.")
        return

    # Profile interpolation
    H, W = real_phase.shape
    interp = RegularGridInterpolator((np.arange(H), np.arange(W)), real_phase,
                                     bounds_error=False, fill_value=np.nan)

    def phase_stats(profile):
        profile = np.asarray(profile).ravel()
        profile = profile[~np.isnan(profile)]
        if profile.size < 4:
            return np.nan, np.nan, np.nan
        srt = np.sort(profile)
        n5 = max(1, int(0.05 * srt.size))
        return srt[:n5].mean(), srt[-n5:].mean(), srt[-n5:].mean() - srt[:n5].mean()

    # Plotting results
    fig_prof, ax_prof = plt.subplots()
    ax_prof.set_title("Phase profiles")
    ax_prof.set_ylabel("Phase [rad]")
    ax_prof.set_xlabel("Distance [µm]" if μm_per_px > 0 else "Distance [pixels]")
    ax_prof.grid(True)

    rows = []
    lows, highs, dphis = [], [], []

    for i in range(n_profiles):
        x1, y1 = clicks[2 * i]
        x2, y2 = clicks[2 * i + 1]
        colour = colours[i % len(colours)]

        if qpi_type == 0:  # line
            L = int(np.hypot(x2 - x1, y2 - y1))
            t = np.linspace(0, 1, max(L, 2))
            xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        else:  # circle
            r = np.hypot(x2 - x1, y2 - y1)
            ang = np.linspace(0, 2 * np.pi, max(64, int(2 * np.pi * r)))
            xs, ys = x1 + r * np.cos(ang), y1 + r * np.sin(ang)

        prof = interp(np.vstack([ys, xs]).T)
        low, high, dphi = phase_stats(prof)
        lows.append(low)
        highs.append(high)
        dphis.append(dphi)

        dist = np.arange(len(prof))
        if μm_per_px > 0:
            dist = dist * μm_per_px
        ax_prof.plot(dist, prof, color=colour, label=f"Profile {i + 1}")

        if d_known is not None:
            n_rel = 2 * np.pi * d_known / (λ * dphi) if abs(dphi) > 1e-9 else np.nan
            rows.append({
                "Zone": i + 1,
                "φ_low": round(low, 2),
                "φ_high": round(high, 2),
                "Δφ [rad]": round(dphi, 2),
                "n_rel": round(n_rel, 2)
            })
        else:
            thickness = abs(dphi) * λ / (2 * np.pi * abs(n_s - n_m))
            rows.append({
                "Zone": i + 1,
                "φ_low": round(low, 2),
                "φ_high": round(high, 2),
                "Δφ [rad]": round(dphi, 2),
                "Thickness [µm]": round(thickness, 2)
            })

    ax_prof.legend()
    plt.show(block=False)

    if len(rows) > 1:
        rows.append({
            "Zone": "Mean±SD",
            "φ_low": round(np.mean(lows), 2),
            "φ_high": round(np.mean(highs), 2),
            "Δφ [rad]": f"{np.mean(dphis):.2f} ± {np.std(dphis):.2f}"
        })

    show_dataframe_in_table(app, pd.DataFrame(rows), "QPI results")


# Apply Microstructures Metrics in Bio-Analysis Frame
def apply_microstructure(app):
    if not getattr(app, "phase_arrays", []):
        messagebox.showinfo(
            "Information",
            "No phase data available. Please reconstruct first."
        )
        return
    # Check whether we are working with Amplitude or Phase
    mode = app.imageProcess_var.get()

    # Get microns_per_pixel
    μm_per_px = getattr(app, 'microns_per_pixel', None)
    if μm_per_px is None or abs(μm_per_px) < 1e-9:
        messagebox.showinfo(
            "Information",
            "Please enter a valid Pixel Size value."
        )

    # Get the image depending on the selected mode
    if mode == 0:  # Amplitude
        idx = getattr(app, 'current_amp_index', 0)
        if idx >= len(app.amplitude_arrays):
            messagebox.showinfo(
                "Information",
                "No amplitude image available."
            )
            return
        image = app.amplitude_arrays[idx]
    else:  # Phase
        idx = getattr(app, 'current_phase_index', 0)
        if idx >= len(app.phase_arrays):
            messagebox.showinfo(
                "Information",
                "No phase image available."
            )
            return
        image = app.phase_arrays[idx]

    thresh_method = app.thresh_method_var.get()

    # Get numerical parameters
    params = {}
    for key, entry in app.micro_entries.items():
        try:
            val = float(entry.get())
            params[key] = val
        except ValueError:
            print(f"[Warning] Invalid value for '{key}': {entry.get()}")
            params[key] = None

    if app.count_particles_var.get():
        tmic.apply_count_particles(
            image,
            method=thresh_method.lower(),
            threshold=params['Threshold'],
            min_area=params['Min Area'],
            max_area=params['Max Area'],
            parent=app
        )

    if app.particles_areas_var.get():
        tmic.apply_area_particles(
            image,
            method=thresh_method.lower(),
            threshold=params['Threshold'],
            min_area=params['Min Area'],
            max_area=params['Max Area'],
            μm_per_px=μm_per_px,
            parent=app
        )

    if app.automatic_profile_var.get():
        tmic.automaticProfile(
            image,
            method=thresh_method.lower(),
            threshold=params['Threshold'],
            min_area=params['Min Area'],
            max_area=params['Max Area'],
            μm_per_px=μm_per_px,
            parent=app)

    if app.profile_thickness_var.get():
         tmic.apply_thickness(
            image,
            method=thresh_method.lower(),
            threshold=params['Threshold'],
            min_area=params['Min Area'],
            max_area=params['Max Area'],
            ind_sample=params['Ind. Sample'],
            ind_medium=params['Ind. Medium'],
            parent=app
        )

    # If no options are selected
    if not any([
        app.count_particles_var.get(),
        app.particles_areas_var.get(),
        app.automatic_profile_var.get(),
        app.profile_thickness_var.get()
    ]):
        messagebox.showinfo(
            "Information",
            "No analysis options were selected."
        )


# Apply Filters
def apply_filters(self):
    sel = self.filters_dimensions_var.get()
    if sel == 0:
        if self.multi_holo_arrays:
            idx = self.current_left_index
        else:
            idx = 0
    elif sel == 1:
        idx = getattr(self, "current_amp_index", 0)
    elif sel == 2:
        idx = getattr(self, "current_phase_index", 0)
    else:
        idx = 0

    store_current_ui_filter_state(self, sel, idx)

    def any_on():
        return (self.gamma_checkbox_var.get() or self.contrast_checkbox_var.get() or
                self.highpass_checkbox_var.get() or self.lowpass_checkbox_var.get() or
                self.adaptative_eq_checkbox_var.get())

    if sel == 0:
        if not self.multi_holo_arrays:
            print("No hologram loaded."); return
        base = self.original_multi_holo_arrays[idx].astype(np.float32)
        filt = base if not any_on() else apply_all_filters_to_array(self, base)
        self.multi_holo_arrays[idx] = filt
        pil = Image.fromarray(np.clip(filt, 0, 255).astype(np.uint8))
        tkimg = self._preserve_aspect_ratio_right(pil)
        self.hologram_frames[idx] = tkimg
        if self.holo_view_var.get() == "Hologram":
            self.captured_label.configure(image=tkimg)

    elif sel == 1:
        if not self.amplitude_arrays:
            print("No amplitude data."); return
        base = self.original_amplitude_arrays[idx].astype(np.float32)
        filt = base if not any_on() else apply_all_filters_to_array(self, base)
        filt = np.clip(filt, 0, 255).astype(np.uint8)
        self.amplitude_arrays[idx] = filt
        pil = apply_matplotlib_colormap(self, filt, self.colormap_amp_var.get())
        tkimg = self._preserve_aspect_ratio_right(pil)
        self.amplitude_frames[idx] = tkimg
        if self.recon_view_var.get() == "Amplitude Reconstruction ":
            self.processed_label.configure(image=tkimg)

    else:
        if not self.phase_arrays:
            print("No phase data."); return
        base = self.original_phase_arrays[idx].astype(np.float32)
        filt = base if not any_on() else apply_all_filters_to_array(self, base)
        filt = np.clip(filt, 0, 255).astype(np.uint8)
        self.phase_arrays[idx] = filt
        pil = apply_matplotlib_colormap(self, filt, self.colormap_phase_var.get())
        tkimg = self._preserve_aspect_ratio_right(pil)
        self.phase_frames[idx] = tkimg
        if self.recon_view_var.get() == "Phase Reconstruction ":
            self.processed_label.configure(image=tkimg)

    update_colormap_display(self)


# Filters to arrays in Apply Filters
def apply_all_filters_to_array(self, arr):
    """
    Applies all enabled filters (Gamma, Contrast, High-pass, Low-pass, Adaptive Equalization)
    to the input image array, using the current GUI state for filter parameters.

    Parameters:
        arr (np.ndarray): Input image array (float32 or similar).

    Returns:
        np.ndarray: Filtered image array, clipped to [0, 255].
    """
    out = arr.copy()

    # Apply Gamma Correction
    if self.gamma_checkbox_var.get():
        gamma_val = max(float(self.gamma_slider.get()), 1e-8)
        normed = out / (out.max() + 1e-9)
        out = np.power(normed, 1.0 / gamma_val) * 255.0

    # Apply Contrast Adjustment
    if self.contrast_checkbox_var.get():
        cont_val = float(self.contrast_slider.get())
        mean_val = np.mean(out)
        out = (out - mean_val) * cont_val + mean_val

    # Apply High-Pass Filter in Frequency Domain
    if self.highpass_checkbox_var.get():
        cutoff = float(self.highpass_slider.get())
        f = np.fft.fft2(out)
        fshift = np.fft.fftshift(f)

        rows, cols = out.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(rows, cols) * cutoff * 0.5)

        # Create circular high-pass mask
        mask = np.ones((rows, cols), np.uint8)
        Y, X = np.ogrid[:rows, :cols]
        dist_sq = (X - ccol)**2 + (Y - crow)**2
        mask[dist_sq <= radius**2] = 0

        fshift *= mask
        out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))

    # Apply Low-Pass Filter in Frequency Domain
    if self.lowpass_checkbox_var.get():
        cutoff = float(self.lowpass_slider.get())
        f = np.fft.fft2(out)
        fshift = np.fft.fftshift(f)

        rows, cols = out.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(rows, cols) * cutoff * 0.5)

        # Create circular low-pass mask
        mask = np.zeros((rows, cols), np.uint8)
        Y, X = np.ogrid[:rows, :cols]
        dist_sq = (X - ccol)**2 + (Y - crow)**2
        mask[dist_sq <= radius**2] = 1

        fshift *= mask
        out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))

    # Apply Adaptive Histogram Equalization
    if self.adaptative_eq_checkbox_var.get():
        arr_min, arr_max = out.min(), out.max()
        scaled = (out - arr_min) / (arr_max - arr_min + 1e-9)

        hist, bins = np.histogram(scaled.flatten(), 256, [0, 1])
        cdf = hist.cumsum() / hist.sum()

        eq = np.interp(scaled.flatten(), bins[:-1], cdf)
        out = eq.reshape(out.shape) * (arr_max - arr_min) + arr_min

    # Final Clipping and Resize Check
    out = np.clip(out, 0, 255)

    if out.shape != arr.shape:
        # Resize to original dimensions if shape has changed
        out = cv2.resize(out, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return out


# Gamma in Apply Filters
def adjust_gamma(self, val):
    """
    Adjusts the gamma correction value for the currently selected image type (hologram, amplitude, or phase),
    but only if the manual gamma control is enabled for that type.
    """
    dim = self.filters_dimensions_var.get()

    if dim == 0:
        if self.manual_gamma_c_var.get():
            self.gamma_c = val
    elif dim == 1:
        if self.manual_gamma_a_var.get():
            self.gamma_a = val
    elif dim == 2:
        if self.manual_gamma_r_var.get():
            self.gamma_r = val



# Contrast in Apply Filters
def adjust_contrast(self, val):
    """
    Adjusts the contrast value for the currently selected image type,
    only if manual contrast adjustment is enabled.
    """
    dim = self.filters_dimensions_var.get()

    if dim == 0:
        if self.manual_contrast_c_var.get():
            self.contrast_c = val
    elif dim == 1:
        if self.manual_contrast_a_var.get():
            self.contrast_a = val
    elif dim == 2:
        if self.manual_contrast_r_var.get():
            self.contrast_r = val


# Highpass in Apply Filters
def adjust_highpass(self, val):
    """
    Adjusts the high-pass filter cutoff value for the selected image type,
    if manual high-pass control is enabled.
    """
    dim = self.filters_dimensions_var.get()

    if dim == 0:
        if self.manual_highpass_c_var.get():
            self.highpass_c = val
    elif dim == 1:
        if self.manual_highpass_a_var.get():
            self.highpass_a = val
    elif dim == 2:
        if self.manual_highpass_r_var.get():
            self.highpass_r = val


# Lowpass in Apply Filters
def adjust_lowpass(self, val):
    """
    Adjusts the low-pass filter cutoff value for the selected image type,
    if manual low-pass control is enabled.
    """
    dim = self.filters_dimensions_var.get()

    if dim == 0:
        if self.manual_lowpass_c_var.get():
            self.lowpass_c = val
    elif dim == 1:
        if self.manual_lowpass_a_var.get():
            self.lowpass_a = val
    elif dim == 2:
        if self.manual_lowpass_r_var.get():
            self.lowpass_r = val


# Adaptative in Apply Filters
def adjust_adaptative_eq(self):
    """
    Enables adaptive histogram equalization for the selected image type,
    but only if manual control is enabled for that filter.
    """
    dim = self.filters_dimensions_var.get()

    if dim == 0:
        if self.manual_adaptative_eq_c_var.get():
            self.adaptative_eq_c = True
    elif dim == 1:
        if self.manual_adaptative_eq_a_var.get():
            self.adaptative_eq_a = True
    elif dim == 2:
        if self.manual_adaptative_eq_r_var.get():
            self.adaptative_eq_r = True


# Default values for filters  in Apply Filters
def default_filter_state():
    return {
        "gamma_on": False,        "gamma_val": 0.0,
        "contrast_on": False,     "contrast_val": 1.0,
        "highpass_on": False,     "highpass_val": 0.0,
        "lowpass_on": False,      "lowpass_val": 0.0,
        "adapt_eq_on": False,
        "speckle_on": False,
        "speckle_method": 0,
        "speckle_param": 3
    }



# Store current values in Apply Filters
def store_current_ui_filter_state(self, dimension: int, index: int) -> None:
    """
    Stores the current state of the filter controls (checkboxes and sliders)
    into the appropriate list corresponding to the selected image type (dimension)
    and index. This allows the GUI to later restore the exact filter configuration.

    Parameters:
        dimension (int): 0 = Hologram, 1 = Amplitude, 2 = Phase
        index (int): Index of the image in the corresponding list
    """

    # Collect the current UI state for each filter
    st = {
        "gamma_on":     self.gamma_checkbox_var.get(),
        "gamma_val":    self.gamma_slider.get(),
        "contrast_on":  self.contrast_checkbox_var.get(),
        "contrast_val": self.contrast_slider.get(),
        "highpass_on":  self.highpass_checkbox_var.get(),
        "highpass_val": self.highpass_slider.get(),
        "lowpass_on":   self.lowpass_checkbox_var.get(),
        "lowpass_val":  self.lowpass_slider.get(),
        "adapt_eq_on":  self.adaptative_eq_checkbox_var.get(),
    }

    # Get the currently active speckle filter method and its parameter (if any)
    active_method = active_speckle_method(self)
    active_param = current_speckle_param(self)

    # Add speckle filter settings to the state dictionary
    st.update({
        "speckle_on":    (active_method is not None),
        "speckle_meth":  active_method,
        "speckle_param": active_param,
    })

    # Store the state dictionary in the appropriate list based on dimension
    if dimension == 0 and index < len(self.filter_states_dim0):
        self.filter_states_dim0[index] = st
    elif dimension == 1 and index < len(self.filter_states_dim1):
        self.filter_states_dim1[index] = st
    elif dimension == 2 and index < len(self.filter_states_dim2):
        self.filter_states_dim2[index] = st


# Restore filters control panel in Apply Filters
def load_ui_from_filter_state(self, dimension: int, index: int) -> None:
    """
    Restores the filter control UI (checkboxes, sliders, and speckle settings)
    based on the stored filter state for a given image dimension and index.

    Parameters:
        self (object): The GUI class instance
        dimension (int): 0 = Hologram, 1 = Amplitude, 2 = Phase
        index (int): Index of the image in the corresponding dimension list
    """

    # Retrieve the stored filter state dictionary
    if dimension == 0 and index < len(self.filter_states_dim0):
        st = self.filter_states_dim0[index]
    elif dimension == 1 and index < len(self.filter_states_dim1):
        st = self.filter_states_dim1[index]
    elif dimension == 2 and index < len(self.filter_states_dim2):
        st = self.filter_states_dim2[index]
    else:
        return

    # Restore basic filter checkboxes and sliders
    self.gamma_checkbox_var.set(st["gamma_on"])
    self.gamma_slider.set(st["gamma_val"])

    self.contrast_checkbox_var.set(st["contrast_on"])
    self.contrast_slider.set(st["contrast_val"])

    self.highpass_checkbox_var.set(st["highpass_on"])
    self.highpass_slider.set(st["highpass_val"])

    self.lowpass_checkbox_var.set(st["lowpass_on"])
    self.lowpass_slider.set(st["lowpass_val"])

    self.adaptative_eq_checkbox_var.set(st["adapt_eq_on"])

    #  Restore speckle filter method and parameter
    # First, reset all speckle method checkboxes
    for var in self.spk_vars:
        var.set(False)

    if st.get("speckle_on"):
        method = st.get("speckle_meth")
        param = st.get("speckle_param", 0)

        # Activate the corresponding speckle method checkbox
        if method in (0, 1, 2, 3):
            self.spk_vars[method].set(True)

        try:
            # Restore the correct parameter value in the matching entry widget,
            # and clear all others to avoid residual values.

            if method == 0:
                self.hybrid_k_entry.delete(0, "end")
                self.hybrid_k_entry.insert(0, str(param))
            else:
                self.hybrid_k_entry.delete(0, "end")

            if method == 1:
                self.mean_size_entry.delete(0, "end")
                self.mean_size_entry.insert(0, str(param))
            else:
                self.mean_size_entry.delete(0, "end")

            if method == 2:
                self.median_size_entry.delete(0, "end")
                self.median_size_entry.insert(0, str(param))
            else:
                self.median_size_entry.delete(0, "end")

            if method == 3:
                self.gauss_sigma_entry.delete(0, "end")
                self.gauss_sigma_entry.insert(0, str(param))
            else:
                self.gauss_sigma_entry.delete(0, "end")

        except Exception:
            # In case entry widgets are not initialized yet, skip safely
            pass


# Apply matplotlib colors
def apply_matplotlib_colormap(self, arr: np.ndarray, cmap_name: str) -> Image.Image:
    """
    Applies a matplotlib colormap to a grayscale image array.

    Parameters:
        arr (np.ndarray): Input grayscale image (values in 0–255).
        cmap_name (str): Name of the color map as selected in the GUI.

    Returns:
        Image.Image: PIL image in RGB format.
    """
    cmap_mpl_name = mpl_name(cmap_name)  # Convert to valid matplotlib name

    if cmap_mpl_name == "original":
        return Image.fromarray(arr.astype(np.uint8), mode="L")

    # Normalize array to [0, 1], unless image is constant
    span = np.ptp(arr)
    if span < 1e-9:
        normed = np.zeros_like(arr, dtype=np.float32)
    else:
        normed = (arr.astype(np.float32) - arr.min()) / span

    rgba = (plt.get_cmap(cmap_mpl_name)(normed) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3], mode="RGB")


# Color matplotlib
def update_colormap_display(self):
    """
    Re-renders the currently visible amplitude or phase frame
    with the selected colormap so the display stays in sync
    with user preference.
    """
    view = self.recon_view_var.get()

    if view == "Amplitude Reconstruction ":
        idx = getattr(self, "current_amp_index", 0)
        if idx < len(self.amplitude_arrays):
            pil = apply_matplotlib_colormap(self, self.amplitude_arrays[idx], self.colormap_amp_var.get())
            tkimg = self._preserve_aspect_ratio_right(pil)
            self.amplitude_frames[idx] = tkimg
            self.processed_label.configure(image=tkimg)

    elif view == "Phase Reconstruction ":
        idx = getattr(self, "current_phase_index", 0)
        if idx < len(self.phase_arrays):
            pil = apply_matplotlib_colormap(self, self.phase_arrays[idx], self.colormap_phase_var.get())
            tkimg = self._preserve_aspect_ratio_right(pil)
            self.phase_frames[idx] = tkimg
            self.processed_label.configure(image=tkimg)


# Apply colormap
def apply_colormap(self):
    """
    Applies the selected colormaps to all amplitude and phase image frames.
    Triggered when the user presses 'Apply' in the colormap section.
    """
    # Apply colormap to all amplitude frames
    for i, arr in enumerate(self.amplitude_arrays):
        pil = apply_matplotlib_colormap(self, arr, self.colormap_amp_var.get())
        tkimg = self._preserve_aspect_ratio_right(pil)
        self.amplitude_frames[i] = tkimg

    # Apply colormap to all phase frames
    for i, arr in enumerate(self.phase_arrays):
        pil = apply_matplotlib_colormap(self, arr, self.colormap_phase_var.get())
        tkimg = self._preserve_aspect_ratio_right(pil)
        self.phase_frames[i] = tkimg

    # Refresh the currently visible image
    update_colormap_display(self)

    print(f"Colormaps updated →  Amplitude: {self.colormap_amp_var.get()} │ "
          f"Phase: {self.colormap_phase_var.get()}")


# Color matplotlib
def mpl_name(gui_name: str) -> str:
    """
    Converts a user-selected GUI colormap name into a valid Matplotlib colormap name.

    Parameters:
        gui_name (str): The name as selected in the GUI dropdown (case-insensitive).

    Returns:
        str: A valid colormap name for use with matplotlib.pyplot.get_cmap().
    """
    lut = {
        "original": "original",
        "viridis":  "viridis",
        "plasma":   "plasma",
        "inferno":  "inferno",
        "magma":    "magma",
        "cividis":  "cividis",
        "hot":      "hot",
        "cool":     "cool",
        "wistia":   "Wistia",
    }

    return lut.get(gui_name.lower(), "original")


# Apply measurements in Speckle frame
def apply_speckle(app):
    """
    This function performs speckle analysis on a selected image type
    (Hologram, Amplitude, or Phase) by allowing the user to draw ROIs
    and computes speckle contrast in subdivided zones.
    """
    choice = app.speckle_var.get()

    # Try to retrieve zone, row, and column values; use defaults if invalid
    try:
        zone_count = int(app.zones_entry.get().strip())
    except:
        messagebox.showinfo(
            "Information",
            "Please  enter a valid Zones value."
        )
    try:
        rows_val = int(app.rows_entry.get().strip())
    except:
        messagebox.showinfo(
            "Information",
            "Please  enter a valid Rows value."
        )
    try:
        cols_val = int(app.cols_entry.get().strip())
    except:
        messagebox.showinfo(
            "Information",
            "Please  enter a valid Cols value."
        )

    # Select image to process
    if choice == 0:
        if app.current_holo_array is not None and app.current_holo_array.size > 0:
            speckle_data = app.current_holo_array.copy()
        else:
            messagebox.showinfo(
                "Information",
                "No hologram available."
            )
            return
        title_str = "Speckle from Hologram"
    elif choice == 1:
        idx = getattr(app, 'current_amp_index', 0)
        if not hasattr(app, 'amplitude_arrays') or idx >= len(app.amplitude_arrays):
            messagebox.showinfo(
                "Information",
                "No amplitude available."
            )
            return
        speckle_data = app.amplitude_arrays[idx].copy()
        title_str = "Speckle from Amplitude"
    else:
        idx = getattr(app, 'current_phase_index', 0)
        if not hasattr(app, 'phase_arrays') or idx >= len(app.phase_arrays):
            messagebox.showinfo(
                "Information",
                "No phase available."
            )
            return
        speckle_data = app.phase_arrays[idx].copy()
        title_str = "Speckle from Phase"

    # Normalize image to 8-bit for display
    smin, smax = speckle_data.min(), speckle_data.max()
    if abs(smax - smin) < 1e-9:
        disp_speckle = np.zeros_like(speckle_data, dtype=np.uint8)
    else:
        disp_speckle = ((speckle_data - smin) / (smax - smin) * 255).astype(np.uint8)

    # Micron calibration
    try:
        px_size = float(app.pixel_size_entry.get().strip())
    except:
        px_size = 0.0
    try:
        mag = float(app.magnification_entry.get().strip())
    except:
        mag = 0.0

    if choice == 0:
        µm_per_px = px_size if px_size > 0 else 0.0
    else:
        µm_per_px = px_size / mag if px_size > 0 and mag > 1e-6 else 0.0

    # Display image and allow ROI selection
    fig, ax = plt.subplots()
    ax.imshow(disp_speckle, cmap='gray', interpolation='nearest')

    _micron_axes(ax, width_px=disp_speckle.shape[1], height_px=disp_speckle.shape[0], µm_per_px=µm_per_px)
    ax.set_title(f"{title_str}: draw {zone_count} ROI(s)")

    subregions_info = []
    next_label = 1
    max_val = 0.0

    def calc_speckle_contrast(region):
        intensity_img = np.abs(region) ** 2
        mean_i = np.mean(intensity_img)
        std_i = np.std(intensity_img)
        return std_i / (mean_i + 1e-9)

    def subdivide_and_label(x1, y1, x2, y2, rows, cols):
        nonlocal next_label, max_val
        w_sub = (x2 - x1) // cols
        h_sub = (y2 - y1) // rows

        for r in range(rows + 1):
            y_line = y1 + r * h_sub
            ax.plot([x1, x2], [y_line, y_line], color='red', linewidth=1)
        for c in range(cols + 1):
            x_line = x1 + c * w_sub
            ax.plot([x_line, x_line], [y1, y2], color='red', linewidth=1)

        H, W = speckle_data.shape
        for rr in range(rows):
            for cc in range(cols):
                xx1 = x1 + cc * w_sub
                yy1 = y1 + rr * h_sub
                xx2 = min(xx1 + w_sub, W)
                yy2 = min(yy1 + h_sub, H)

                piece = speckle_data[yy1:yy2, xx1:xx2]
                sc_raw = calc_speckle_contrast(piece)
                if sc_raw > max_val:
                    max_val = sc_raw

                subregions_info.append({
                    "id": next_label,
                    "speckle_raw": sc_raw
                })

                cx = xx1 + (xx2 - xx1) // 2
                cy = yy1 + (yy2 - yy1) // 2
                ax.text(cx, cy, f"{next_label}", color='red',
                        ha='center', va='center', fontsize=10)

                next_label += 1

    zones_collected = []

    def onselect(eclick, erelease):
        if len(zones_collected) >= zone_count:
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        H, W = speckle_data.shape
        x1, x2 = sorted([max(0, x1), min(W, x2)])
        y1, y2 = sorted([max(0, y1), min(H, y2)])
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            print("ROI too small. Ignoring.")
            return

        rect = plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        subdivide_and_label(x1, y1, x2, y2, rows_val, cols_val)
        zones_collected.append((x1, y1, x2, y2))
        plt.draw()

        if len(zones_collected) == zone_count:
            table_rows = []
            for item in subregions_info:
                sc_raw = item["speckle_raw"]
                sc_norm = sc_raw / (max_val + 1e-9)
                table_rows.append({
                    "Zone": item["id"],
                    "Speckle Contrast": sc_raw,
                    "Speckle Contrast (norm)": sc_norm
                })

            if table_rows:
                avg_raw = np.mean([row["Speckle Contrast"] for row in table_rows])
                avg_norm = np.mean([row["Speckle Contrast (norm)"] for row in table_rows])
                table_rows.append({
                    "Zone": "Average",
                    "Speckle Contrast": avg_raw,
                    "Speckle Contrast (norm)": avg_norm
                })

            df = pd.DataFrame(table_rows)
            show_dataframe_in_table(app, df, "Speckle Results")
            toggle_selector.set_active(False)

    toggle_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )

    plt.show()


# Exclusivity for HMF and SPP filters
def speckle_exclusive_callback(self, idx_clicked: int):
    """
    Just one filter each time.
    """
    if self.spk_vars[idx_clicked].get():
        for i, var in enumerate(self.spk_vars):
            if i != idx_clicked:
                var.set(False)


# Apply Speckle filters in Speckle frame
def apply_speckle_filter(app):
    method = active_speckle_method(app)
    dim = app.speckle_filter_dim_var.get()

    idx = getattr(app, "current_amp_index", 0)

    # Reset stored iterations for speckle contrast plot
    app.speckle_iterations = []
    app.speckle_iterations_type = None

    if hasattr(app, "speckle_region_coords_spp"):
        delattr(app, "speckle_region_coords_spp")
    if hasattr(app, "speckle_region_coords_hmf"):
        delattr(app, "speckle_region_coords_hmf")
    if hasattr(app, "speckle_region_coords"):
        delattr(app, "speckle_region_coords")

    # ─────────────────────────────
    # Method 4 – SPP (Complex-object field)
    # ─────────────────────────────
    if method == 4:
        if not hasattr(app, "complex_fields") or not app.complex_fields:
            messagebox.showinfo(
                "Information",
                "Please reconstruct the hologram first."
            )
            return

        if hasattr(app, "original_complex_fields") and app.original_complex_fields and idx < len(
                app.original_complex_fields):
            base_field = app.original_complex_fields[idx]
        else:
            base_field = app.complex_fields[idx]

        field = app.complex_fields[idx]
        if not np.iscomplexobj(field):
            messagebox.showinfo(
                "Information",
                "Please reconstruct the hologram first."
            )
            return

        p = current_speckle_param(app)
        if p < 1:
            messagebox.showinfo(
                "Information",
                "Number of iterations must be greater than 0."
            )
            return

        # Apply SPP Filter — returns list of complex fields
        spp_iterations = spp_filter(base_field, max_iterations=p)
        filtered_field = spp_iterations[-1]

        # Store all iterations for speckle plot
        app.speckle_iterations = spp_iterations
        app.speckle_iterations_type = "spp"

        # Update the stored complex field
        app.complex_fields[idx] = filtered_field

        # Extract amplitude and phase
        amp = np.abs(filtered_field)
        amp_norm = 255 * (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
        phase = np.angle(filtered_field)
        phase_norm = 255 * (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)

        # Store updated amplitude and phase arrays
        app.amplitude_arrays[idx] = amp_norm.astype(np.uint8)
        app.phase_arrays[idx] = phase_norm.astype(np.uint8)

        # Display in viewer
        arr = app.amplitude_arrays[idx] if dim == 1 else app.phase_arrays[idx]
        cmap = app.colormap_amp_var.get() if dim == 1 else app.colormap_phase_var.get()

        pil = apply_matplotlib_colormap(app, arr, cmap)
        tkimg = app._preserve_aspect_ratio_right(pil)

        if dim == 1:
            app.amplitude_frames[idx] = tkimg
            app.filtered_amp_array = amp_norm
        else:
            app.phase_frames[idx] = tkimg
            app.filtered_phase_array = phase_norm

        app.processed_label.configure(image=tkimg)

        return

    # ─────────────────────────────
    # Methods 0 - 3 (Image domain)
    # ─────────────────────────────
    if dim == 1:
        if not app.amplitude_arrays:
            messagebox.showinfo(
                "Information",
                "Number amplitude image available."
            )
            return
        base_arr = app.original_amplitude_arrays[idx].astype(np.float32)
        tgt_arr = app.amplitude_arrays
        tgt_frames = app.amplitude_frames
        cmap_name = app.colormap_amp_var.get()
    else:
        if not app.phase_arrays:
            messagebox.showinfo(
                "Information",
                "Number phase image available."
            )
            return
        base_arr = app.original_phase_arrays[idx].astype(np.float32)
        tgt_arr = app.phase_arrays
        tgt_frames = app.phase_frames
        cmap_name = app.colormap_phase_var.get()

    update_cb = lambda tkimg: (
            app.recon_view_var.get() == ("Amplitude Reconstruction " if dim == 1 else "Phase Reconstruction ")
            and app.processed_label.configure(image=tkimg)
    )

    # If no filter is selected, restore original image
    if method is None:
        restored = np.clip(base_arr, 0, 255).astype(np.uint8)
        tgt_arr[idx] = restored
        pil = apply_matplotlib_colormap(app, restored, cmap_name)
        tkimg = app._preserve_aspect_ratio_right(pil)
        tgt_frames[idx] = tkimg
        update_cb(tkimg)
        return

    # Get numeric parameter and validate per method
    p = current_speckle_param(app)
    if method == 0 and p < 1:
        messagebox.showinfo(
            "Information",
            "Number of iterations must be greater than 0."
        )
        return
    if method == 1 and p < 1:
        messagebox.showinfo(
            "Information",
            "Kernel size must be greater than 0."
        )
        return
    if method == 2 and (p < 3 or p % 2 == 0):
        messagebox.showinfo(
            "Information",
            "Kernel size must be an odd integer greater than or equal to 3 for the median filter."
        )
        return
    if method == 3 and p <= 0:
        messagebox.showinfo(
            "Information",
            "Kernel size must be greater than 0."
        )
        return

    # Apply selected filter
    if method == 0:
        hmf_iterations = HybridMedianMean(base_arr, p)
        out = hmf_iterations[-1]

        # Store all iterations for speckle plot
        app.speckle_iterations = hmf_iterations
        app.speckle_iterations_type = "hmf"

    elif method == 1:
        out = uniform_filter(base_arr, size=p)
    elif method == 2:
        out = median_filter(base_arr, size=p)
    elif method == 3:
        out = gaussian_filter(base_arr, sigma=p)

    out = np.clip(out, 0, 255).astype(np.uint8)
    tgt_arr[idx] = out

    pil = apply_matplotlib_colormap(app, out, cmap_name)
    tkimg = app._preserve_aspect_ratio_right(pil)

    if dim == 1:
        app.filtered_amp_array = out
    else:
        app.filtered_phase_array = out

    tgt_frames[idx] = tkimg
    update_cb(tkimg)


# Hybrid Median Mean
def HybridMedianMean(sample, max_iterations):
    """
    Applies the hybrid median-mean method to reduce speckle noise.

    Parameters:
    - sample: input image (2D numpy array)
    - max_iterations: maximum number of iterations

    Returns:
    - List of intermediate results (2D arrays), one per iteration.
    """
    mean_image = sample.copy().astype(np.float64)
    intermediate_results = [mean_image.copy()]

    for i in range(max_iterations):
        kernel_size = 3 + (2 * i)
        filtered_image = ndimage.median_filter(sample, kernel_size, mode='constant', cval=0)
        mean_image = (mean_image + filtered_image) / 2
        intermediate_results.append(mean_image.copy())

    return intermediate_results


# Pointwise Phase Tuning
def spp_filter(sample, max_iterations):
    """
    Applies SPP (Stochastic Phase Processing) filter to a complex-valued sample.

    Parameters:
    - sample: complex numpy array
    - max_iterations: number of iterations

    Returns:
    - List of intermediate complex fields (complex 2D arrays), one per iteration.
    """
    real_part = np.real(sample).astype(np.float64, copy=True)
    imag_part = np.imag(sample).astype(np.float64, copy=True)


    real_min = np.min(real_part)
    real_max = np.max(real_part)

    accumulated_sample = np.zeros_like(sample, dtype=complex)
    intermediate_results = []

    for i in range(max_iterations):
        noise_mean = (real_max + real_min) / 2
        noise_std = (real_max - real_min) / 6
        noise = np.random.normal(noise_mean, noise_std, real_part.shape)

        new_real_part = real_part + noise
        new_complex_sample = new_real_part + 1j * imag_part

        accumulated_sample += new_complex_sample
        current_result = accumulated_sample / (i + 1)

        intermediate_results.append(current_result.copy())

    return intermediate_results


# Active speckle in Speckle frame
def active_speckle_method(app):
    """
    Returns the index of the currently selected speckle filter method.
    This is based on the state of the dynamically created checkboxes (spk_vars).
    Returns None if no filter is selected.
    """
    for i, var in enumerate(app.spk_vars):
        if var.get():
            return i
    return None


# Active speckle in Speckle frame
def current_speckle_param(app):
    """
    Returns the numeric parameter associated with the selected speckle filter.
    Reads from the corresponding entry in the list of parameter inputs (spk_param_entries).
    Returns 0 if no filter is active or input is invalid.
    """
    active = active_speckle_method(app)
    if active is None:
        return 0

    try:
        entry = app.spk_param_entries[active]
        # Gaussian uses float, others use integer
        return float(entry.get()) if active == 3 else int(entry.get())
    except (ValueError, IndexError):
        return 0


# All speckle comparison functions
def apply_speckle_comparison(self):
    active_options = []
    dim = self.speckle_filter_dim_var.get()
    idx = getattr(self, "current_amp_index", 0)

    if self.compare_side_by_side_var.get():
        active_options.append("Side by Side")
    if self.compare_speckle_plot_var.get():
        active_options.append("Speckle Plot")
    if self.compare_line_profile_var.get():
        active_options.append("Line Profile")

    if dim == 1:
        if not hasattr(self, "filtered_amp_array"):
            print("Amplitude filter has not been applied yet.")
            return
        filtered = self.filtered_amp_array
        original = self.original_amplitude_arrays[idx]
    else:
        if not hasattr(self, "filtered_phase_array"):
            print("Phase filter has not been applied yet.")
            return
        filtered = self.filtered_phase_array
        original = self.original_phase_arrays[idx]

    if "Line Profile" in active_options:
        if dim == 1:
            show_line_profile(self.original_amplitude_arrays[idx], self.filtered_amp_array)
        else:
            show_line_profile(self.original_phase_arrays[idx], self.filtered_phase_array)

    # Side by Side Comparison
    if "Side by Side" in active_options:
        show_side_by_side_comparison(self, original, filtered, title="Speckle Filter Result")

    # Speckle plot
    if "Speckle Plot" in active_options:
        if not hasattr(self, "speckle_iterations") or not self.speckle_iterations:
            print("No speckle data available for speckle plot. Apply a filter first (HMF or SPP).")
            messagebox.showinfo(
                "Information",
                "No speckle data available for speckle plot. Apply a filter first (HMF or SPP)."
            )
            return

        region_coords_attr = f"speckle_region_coords_{self.speckle_iterations_type}"
        needs_new_selection = not hasattr(self, region_coords_attr)

        if needs_new_selection:
            if hasattr(self, "_speckle_side_by_side_fig") and self._speckle_side_by_side_fig:
                try:
                    plt.close(self._speckle_side_by_side_fig)
                except Exception:
                    pass
                self._speckle_side_by_side_fig = None

            if hasattr(self, "_speckle_select_fig") and self._speckle_select_fig:
                try:
                    plt.close(self._speckle_select_fig)
                except Exception:
                    pass
                self._speckle_select_fig = None

            if self.speckle_iterations_type == "spp":
                img_for_selection = self.original_amplitude_arrays[self.current_amp_index]
            elif self.speckle_iterations_type == "hmf":
                img_for_selection = self.original_amplitude_arrays[self.current_amp_index]
            else:
                print("Unknown speckle iteration type.")
                return

            select_speckle_region(self, img_for_selection, callback=compare_speckle_plot_var)
            return

        self.speckle_region_coords = getattr(self, region_coords_attr)
        compare_speckle_plot_var(self)


# Show plot original Image and filtered image
def show_side_by_side_comparison(app, original: np.ndarray, filtered: np.ndarray, title: str = ""):
    # close previous figures
    if hasattr(app, "_speckle_side_by_side_fig") and app._speckle_side_by_side_fig:
        try:
            plt.close(app._speckle_side_by_side_fig)
        except Exception:
            pass
        app._speckle_side_by_side_fig = None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title("Filtered")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


# Profile in speckle
def show_line_profile(original, filtered):
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])

        region_original = original[y_start:y_end, x_start:x_end]
        region_filtered = filtered[y_start:y_end, x_start:x_end]

        profile_orig = region_original.sum(axis=0)
        profile_filt = region_filtered.sum(axis=0)
        pixel_pos = np.arange(x_start, x_end)

        # Unified figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(pixel_pos, profile_orig, label="Original", color='blue')
        ax1.set_title("Original Image Profile")
        ax1.set_xlabel("Pixel Position")
        ax1.set_ylabel("Vertical Sum")
        ax1.grid(True)

        ax2.plot(pixel_pos, profile_filt, label="Filtered", color='orange')
        ax2.set_title("Filtered Image Profile")
        ax2.set_xlabel("Pixel Position")
        ax2.set_ylabel("Vertical Sum")
        ax2.grid(True)

        fig.suptitle("Line Profile Comparison", fontsize=14)
        fig.tight_layout()
        plt.show()

        df = pd.DataFrame({
            "Pixel Position": pixel_pos,
            "Original Image": profile_orig,
            "Filtered Image": profile_filt
        })

        root = Toplevel()
        root.title("Line Profile Data Table")
        pt = Table(root, dataframe=df, showstatusbar=True)
        pt.show()

    fig, ax = plt.subplots()
    ax.imshow(original, cmap='gray')
    ax.set_title("Select region for line profile comparison")

    toggle_selector = RectangleSelector(
        ax,
        onselect=onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )

    plt.show(block=True)


# Select ROI for HMF and SPP
def select_speckle_region(app, image, callback=None):
    """
    Let the user select a rectangular ROI for speckle-contrast analysis.
    Coordinates are stored per filter type in app.speckle_region_coords_<type>
    and also mirrored to app.speckle_region_coords for compatibility.
    """

    def onselect(eclick, erelease):
        # Guard against clicks outside axes (xdata/ydata can be None)
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            print("Selection ignored: click was outside axis.")
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Sort coordinates (left<right, top<bottom)
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])

        # Enforce minimal ROI size
        if (x_end - x_start) < 2 or (y_end - y_start) < 2:
            print("ROI too small. Ignored.")
            return

        # Save coordinates by filter type and in a general attribute
        region_coords_attr = f"speckle_region_coords_{app.speckle_iterations_type}"
        coords = (x_start, x_end, y_start, y_end)
        setattr(app, region_coords_attr, coords)
        app.speckle_region_coords = coords  # general mirror

        print(f"Speckle region selected for {app.speckle_iterations_type.upper()}: "
              f"X({x_start}-{x_end}) Y({y_start}-{y_end})")
        print("Region saved. Click 'Apply' in Speckle Comparison again to generate the plot.")

        # Close only this selection figure
        try:
            if hasattr(app, "_speckle_select_fig") and app._speckle_select_fig:
                import matplotlib.pyplot as plt
                plt.close(app._speckle_select_fig)
        except Exception:
            pass

        # Trigger callback if provided
        if callback:
            callback(app)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Contrast stretch for 2D images to avoid saturation
    if len(image.shape) == 2:
        vmin, vmax = np.percentile(image, [2, 98])
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(image, cmap='gray')

    ax.set_title(f"Select region for {app.speckle_iterations_type.upper()} speckle contrast analysis")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Keep references on app to avoid GC issues and to close later
    app._speckle_select_fig = fig

    # Clean references when the window is closed
    def _on_close(_evt):
        if hasattr(app, "_speckle_selector"):
            app._speckle_selector = None
        if hasattr(app, "_speckle_select_fig"):
            app._speckle_select_fig = None
    fig.canvas.mpl_connect("close_event", _on_close)

    # Build the rectangle selector and keep a reference
    selector = RectangleSelector(
        ax,
        onselect=onselect,
        useblit=True,
        button=[1],        # left mouse button
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )
    app._speckle_selector = selector

    # Ensure the selector is active
    try:
        selector.set_active(True)
    except Exception:
        pass

    # Try to bring window to front (works on TkAgg)
    try:
        mngr = plt.get_current_fig_manager()
        try:
            mngr.window.attributes('-topmost', True)
            mngr.window.attributes('-topmost', False)
        except Exception:
            pass
    except Exception:
        pass

    # Block to keep focus and guarantee events reach the selector
    plt.show(block=True)


def compare_speckle_plot_var(app):
    """
    Computes and plots speckle contrast over iterations using the selected region.
    Requires:
    - app.speckle_iterations: list of filtered images/fields
    - app.speckle_region_coords: (x_start, x_end, y_start, y_end)
    - app.speckle_iterations_type: 'spp' or 'hmf'
    """
    if not hasattr(app, "speckle_iterations") or not app.speckle_iterations:
        print("No stored iterations for speckle contrast.")
        return

    if not hasattr(app, "speckle_region_coords"):
        print("No region selected for speckle contrast.")
        return

    x_start, x_end, y_start, y_end = app.speckle_region_coords
    region_contrasts = []

    print(f"Computing speckle contrast for {len(app.speckle_iterations)} iterations...")

    for i, iter_img in enumerate(app.speckle_iterations):
        try:
            if app.speckle_iterations_type == "spp":
                # SPP, iter_img corresponds to a complex field
                region = iter_img[y_start:y_end, x_start:x_end]
                contrast = calc_speckle_contrast(region)
            elif app.speckle_iterations_type == "hmf":
                # HMF, iter_img corresponds to an Intensity Image
                region = iter_img[y_start:y_end, x_start:x_end]
                mean_i = np.mean(region)
                std_i = np.std(region)
                contrast = std_i / (mean_i + 1e-9)
            else:
                print("Unknown iteration type.")
                return

            region_contrasts.append(contrast)

        except Exception as e:
            print(f"Error processing iteration {i + 1}: {e}")
            return

    # Normalize the contrast values to the first iteration
    region_contrasts = np.array(region_contrasts)
    if region_contrasts[0] > 0:
        region_contrasts /= region_contrasts[0]
    else:
        print("Warning: First iteration has zero contrast, cannot normalize.")

    # Plot the normalized speckle contrast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(region_contrasts) + 1), region_contrasts, marker='o', linewidth=2, markersize=6)
    ax.set_title(f"Normalized Speckle Contrast over Iterations ({app.speckle_iterations_type.upper()})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Contrast")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # add region information
    ax.text(0.02, 0.98, f"Region: X({x_start}-{x_end}) Y({y_start}-{y_end})",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Quantify speckle contrast
def calc_speckle_contrast(region):
    intensity_img = np.abs(region) ** 2
    mean_i = np.mean(intensity_img)
    std_i = np.std(intensity_img)
    return std_i / (mean_i + 1e-9)


def show_dataframe_in_table(parent,df, title="QPI Results"):
    """
    Displays a pandas DataFrame in a new Toplevel window using pandastable,
    without blocking the main Tkinter loop.
    """
    if df.empty:
        print("No data to display in the table.")
        return

    # Create a Toplevel so it runs inside the main app
    table_win = tk.Toplevel(parent)
    table_win.title(title)
    table_win.geometry("800x400")

    # Create a frame for the pandastable
    frame = tk.Frame(table_win)
    frame.pack(fill="both", expand=True)

    pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    pt.show()


def _micron_axes(ax, width_px, height_px, μm_per_px):
    """
    Configure the axis ticks and labels in microns (µm) without modifying the image content.

    Parameters:
    - ax: Matplotlib Axes object to apply the configuration to.
    - width_px: Width of the image in pixels.
    - height_px: Height of the image in pixels.
    - μm_per_px: Scale factor indicating how many microns correspond to one pixel.
    """

    if μm_per_px > 0:
        # choose 5 evenly spaced tick positions in pixel‐space
        x_px = np.linspace(0, width_px, 5)
        y_px = np.linspace(0, height_px, 5)

        ax.set_xticks(x_px)
        ax.set_xticklabels([f"{x*μm_per_px:.1f}" for x in x_px])
        ax.set_yticks(y_px)
        ax.set_yticklabels([f"{y*μm_per_px:.1f}" for y in y_px])

        # Lock the data limits back to full pixel range
        ax.set_xlim(0, width_px)
        ax.set_ylim(height_px, 0)

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
    else:
        ax.axis("off")


def _wire_scalebar_interaction(
        fig,
        ax,
        bar_line,
        bar_text,
        microns_per_pixel,
        busy_flag,
        ):

        MARGIN = 15
        current_len_um = [float(bar_text.get_text().split()[0])]
        dragging = {"active": False, "dx": 0.0, "dy": 0.0}

        # utility: redraw bar at new position / length
        def _update_bar(new_len_um, cx, cy):
            h, w = ax.get_images()[0].get_array().shape
            cx = np.clip(cx, MARGIN, w - MARGIN)
            cy = np.clip(cy, MARGIN, h - MARGIN)

            bar_px = new_len_um / microns_per_pixel
            if bar_px > w - 2*MARGIN:
                bar_px = w - 2*MARGIN
                new_len_um = bar_px * microns_per_pixel
            x0, x1 = cx - bar_px/2, cx + bar_px/2

            bar_line.set_data([x0, x1], [cy, cy])
            bar_text.set_position((cx, cy - 6))
            bar_text.set_text(f"{new_len_um:.1f} µm")
            current_len_um[0] = new_len_um
            fig.canvas.draw_idle()

        # Pick on bar / text
        def _on_pick(event):
            artist = event.artist
            if artist not in (bar_line, bar_text):
                return
            m = event.mouseevent

            # Double-click ⇒ numeric resize
            if m.dblclick:
                busy_flag["busy"] = True
                new_len = simpledialog.askfloat(
                    "Scale-bar length",
                    "New length (µm):",
                    initialvalue=current_len_um[0],
                    minvalue=0.1,
                )
                busy_flag["busy"] = False
                if new_len is not None:
                    cx, cy = bar_line.get_xydata().mean(axis=0)
                    _update_bar(new_len, cx, cy)
                return

            # Right-click ⇒ colour change
            if m.button == 3:
                colour = simpledialog.askstring(
                    "Scale-bar colour",
                    "Choose colour (white / black / red):",
                )
                if colour and colour.strip().lower() in {"white", "black", "red"}:
                    col = colour.strip().lower()
                    bar_line.set_color(col)
                    bar_text.set_color(col)
                    fig.canvas.draw_idle()
                return

            # left-click ⇒ start drag
            if m.button == 1:
                busy_flag["busy"] = True
                cx, cy = bar_line.get_xydata().mean(axis=0)
                dragging.update(active=True,
                                dx=m.xdata - cx,
                                dy=m.ydata - cy)

        # Mouse move while dragging
        def _on_motion(event):
            if dragging["active"]:
                _update_bar(
                    current_len_um[0],
                    event.xdata - dragging["dx"],
                    event.ydata - dragging["dy"],
                )

        # Button release ends drag
        def _on_release(event):
            if dragging["active"]:
                dragging["active"] = False
                busy_flag["busy"] = False

        # Right-click elsewhere centres bar there (optional)
        def _on_press_any(event):
            # only act if right-click, not on bar, but inside axes
            if event.inaxes != ax or event.button != 3:
                return
            if bar_line.contains(event)[0] or bar_text.contains(event)[0]:
                return     # handled by _on_pick already
            _update_bar(current_len_um[0], event.xdata, event.ydata)

        # register handlers
        fig.canvas.mpl_connect("pick_event",           _on_pick)
        fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("button_press_event",   _on_press_any)


def _show_popup_image(parent,arr: np.ndarray, title: str = "Speckle filtered"):
     win = tk.Toplevel(parent)
     win.title(title)
     im = Image.fromarray(arr)
     tk_img = ImageTk.PhotoImage(im)
     lbl = tk.Label(win, image=tk_img)
     lbl.image = tk_img
     lbl.pack()


def _refresh_zoom_view(self, refresh_ms: int = 100) -> None:
    if not getattr(self, "_zoom_live", False):
        return

    arr = self._get_current_array(self._zoom_target)
    if arr is None:
        self.after(refresh_ms, self._refresh_zoom_view)
        return

    # Apply ROI
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


def zoom_holo_view(self):
    choice = self.holo_view_var.get()
    self._open_zoom_view(choice)


def zoom_recon_view(self):
    choice = self.recon_view_var.get()
    target = "Phase" if choice.startswith("Phase") else "Amplitude"
    self._open_zoom_view(target)


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

