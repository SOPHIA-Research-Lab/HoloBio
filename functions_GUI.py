
# import libraries
import tkinter as tk
import customtkinter as ctk
import settings as st
import tools_GUI as tGUI


# Parameter reconstructions Frame
def create_param_with_arrow(parent, row, col, label_text, unit_list, entry_name_dict, unit_update_callback, entry_key=None):
    """
        Creates a label, an entry field and a dropdown unit button in a given parent frame.

        Parameters:
        - parent: the frame in which to place the widgets
        - row, col: grid placement
        - label_text: text to display in the label
        - unit_list: list of unit strings (e.g. ["µm", "nm", "mm"])
        - entry_name_dict: dict where entry widgets will be stored using string keys
        - unit_update_callback: function(label_widget, unit) -> None, to call on unit selection
    """
    # Label
    label = ctk.CTkLabel(parent, text=label_text)
    label.grid(row=row, column=col, padx=5, pady=(5, 5), sticky='w')

    # Entry + arrow
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.grid(row=row+1, column=col, padx=5, pady=5, sticky='w')

    entry = ctk.CTkEntry(container, width=70, placeholder_text='0.0')
    entry.grid(row=0, column=0, sticky='w')

    key = entry_key if entry_key else label_text
    entry_name_dict[key] = entry

    arrow_btn = ctk.CTkButton(container, width=30, text='▼')
    arrow_btn.grid(row=0, column=1, sticky='e')

    def on_arrow_click(event=None):
        menu = tk.Menu(parent, tearoff=0, font=("Helvetica", 14))
        for unit in unit_list:
            menu.add_command(
                label=unit,
                command=lambda u=unit: unit_update_callback(label, u)
            )
        menu.post(arrow_btn.winfo_rootx(), arrow_btn.winfo_rooty() + arrow_btn.winfo_height())

    arrow_btn.bind("<Button-1>", on_arrow_click)

    return label, entry


# Propagation Panel Frame
def create_propagate_panel(parent, attr_prefix="propagation", on_slider_change=None):
    """
    Enhanced propagation panel for microscopy holograms with unit conversion.
    All values are internally converted to micrometers (µm).
    """

    parent.grid_propagate(False)
    for col in range(4):
        parent.columnconfigure(col, weight=1)
    parent.grid_columnconfigure(2, weight=1, minsize=110)
    parent.grid_columnconfigure(3, weight=1, minsize=110)

    # Title
    ctk.CTkLabel(parent, text="Propagation", font=ctk.CTkFont(weight="bold"))\
        .grid(row=0, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="w")

    # Radio Buttons
    prop_mode_var = tk.StringVar(value="sweep")
    radio_frame = ctk.CTkFrame(parent, fg_color="transparent")
    radio_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=(0, 5), sticky="w")
    ctk.CTkRadioButton(radio_frame, text="Fixed Distance", variable=prop_mode_var, value="fixed").grid(row=0, column=0, padx=5)
    ctk.CTkRadioButton(radio_frame, text="Z-scan Sweep", variable=prop_mode_var, value="sweep").grid(row=0, column=1, padx=5)
    ctk.CTkRadioButton(radio_frame, text="Auto Focus", variable=prop_mode_var, value="auto").grid(row=0, column=2, padx=5)

    # Distance Labels
    dist_labels = {
        "distance": ctk.CTkLabel(parent, text="Distance (µm)"),
        "min": ctk.CTkLabel(parent, text="Min. (µm)"),
        "max": ctk.CTkLabel(parent, text="Max. (µm)")
    }
    ctk.CTkLabel(parent, text="Lateral M.").grid(row=2, column=0, padx=5, pady=(5, 0), sticky="n")
    dist_labels["distance"].grid(row=2, column=1, padx=5, pady=(5, 0), sticky="n")
    dist_labels["min"].grid(row=2, column=2, padx=5, pady=(5, 0), sticky="n")
    dist_labels["max"].grid(row=2, column=3, padx=5, pady=(5, 0), sticky="n")

    # Entry fields
    magnification_entry = ctk.CTkEntry(parent, width=80, placeholder_text="40x", justify="center")
    magnification_entry.grid(row=3, column=0, padx=5, pady=(0, 10))
    fixed_distance_entry = ctk.CTkEntry(parent, width=80, placeholder_text="0.0")
    fixed_distance_entry.grid(row=3, column=1, padx=5, pady=(0, 10))
    min_entry = ctk.CTkEntry(parent, width=80, placeholder_text="0.0")
    min_entry.grid(row=3, column=2, padx=5, pady=(0, 10))
    max_entry = ctk.CTkEntry(parent, width=80, placeholder_text="100.0")
    max_entry.grid(row=3, column=3, padx=5, pady=(0, 10))

    # Units Dropdown
    unit_var = tk.StringVar(value="µm")
    previous_unit = {"val": "µm"}

    ctk.CTkLabel(parent, text="Units:").grid(row=4, column=0, padx=5, pady=(5, 0), sticky="e")
    unit_selector = ctk.CTkOptionMenu(
        parent,
        variable=unit_var,
        values=["µm", "mm", "cm"],
        width=100
    )
    unit_selector.grid(row=4, column=1, padx=1, pady=(0, 0), sticky="e")

    unit_scales = {"µm": 1.0, "mm": 1000.0, "cm": 10000.0}

    def convert_entries_on_unit_change(*_):
        old_unit = previous_unit["val"]
        new_unit = unit_var.get()
        if old_unit == new_unit:
            return
        try:
            factor = unit_scales[old_unit] / unit_scales[new_unit]
            for entry in [fixed_distance_entry, min_entry, max_entry]:
                try:
                    val = float(entry.get())
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{val * factor:.3f}")
                except ValueError:
                    continue
        except KeyError:
            pass
        previous_unit["val"] = new_unit

        # Update visible labels
        name = "Distance" if prop_mode_var.get() == "fixed" else "Step"
        dist_labels["distance"].configure(text=f"{name} ({new_unit})")
        dist_labels["min"].configure(text=f"Min. ({new_unit})")
        dist_labels["max"].configure(text=f"Max. ({new_unit})")
        current_value_label.configure(text=f"{propagate_slider.get():.1f} {new_unit}")

    unit_var.trace_add("write", convert_entries_on_unit_change)

    # Slider and Label
    propagate_slider = ctk.CTkSlider(
        parent, from_=0, to=100, number_of_steps=100,
        width=250,
        command=lambda val: (
            current_value_label.configure(
                text=f"{val / (float(magnification_entry.get().replace('x', '').strip())**2):.2f} {unit_var.get()}"
                if magnification_entry.get().strip() else f"{val:2f}{unit_var.get()}"),
            on_slider_change(val) if on_slider_change else None
        )
    )
    propagate_slider.grid(row=6, column=0, columnspan=3, padx=(10, 5), pady=5, sticky="ew")

    current_value_label = ctk.CTkLabel(parent, text=f"0.0 {unit_var.get()}")
    current_value_label.grid(row=6, column=3, padx=(0, 10), sticky="e")

    # Enable/Disable Fields by Mode
    def set_widget_state(widget, enabled=True):
        state = "normal" if enabled else "disabled"
        widget.configure(state=state)
        if isinstance(widget, ctk.CTkEntry):
            widget.configure(text_color=("black", "white") if enabled else "gray50")

    # Helper to show/hide without losing the position in the grid
    def _show(widget, show=True):
        if not widget: return
        (widget.grid() if show else widget.grid_remove())

    def update_visibility(*_):
        mode = prop_mode_var.get()

        # Enable/Disable entries according to mode
        set_widget_state(fixed_distance_entry, mode in ("fixed", "sweep", "auto"))
        set_widget_state(min_entry, mode in ("sweep", "auto"))
        set_widget_state(max_entry, mode in ("sweep", "auto"))

        fixed_distance_entry.configure(placeholder_text=("0.0" if mode == "fixed" else "step"))

        # Show/Hide Min/Max only on fixed
        only_distance = (mode == "fixed")
        _show(dist_labels["min"], not only_distance)
        _show(dist_labels["max"], not only_distance)
        _show(min_entry, not only_distance)
        _show(max_entry, not only_distance)

        # Slider active only in sweep
        propagate_slider.configure(state=("normal" if mode == "sweep" else "disabled"))

        # Adjust slider range when applicable
        if mode == "sweep":
            try:
                min_val = float(min_entry.get())
                max_val = float(max_entry.get())
                if min_val < max_val:
                    propagate_slider.configure(from_=min_val, to=max_val)
            except ValueError:
                pass

        name = "Distance" if mode == "fixed" else "Step"
        dist_labels["distance"].configure(text=f"{name} ({unit_var.get()})")

    prop_mode_var.trace_add("write", update_visibility)
    update_visibility()

    # Autofocus Metric Selector
    metric_var = tk.StringVar(value="Normalized Variance")
    ctk.CTkLabel(parent, text="Autofocus Metric:")\
        .grid(row=7, column=0, columnspan=2, padx=5, pady=(10, 2), sticky="w")
    metric_selector = ctk.CTkOptionMenu(
        parent,
        variable=metric_var,
        values=["Normalized Variance", "Tenengrad"],
        width=180
    )
    metric_selector.grid(row=7, column=1, columnspan=2, padx=10, pady=(10, 2), sticky="e")
    plot_metric_var = tk.BooleanVar(value=False)
    ctk.CTkCheckBox(parent, text="Show Curve", variable=plot_metric_var)\
        .grid(row=7, column=3, padx=(5, 0), pady=(10, 2), sticky="e")

    # Apply Button
    apply_button = ctk.CTkButton(
        parent,
        text="Apply",
        width=100,
        command=lambda: print("--)")
    )
    apply_button.grid(row=8, column=0, columnspan=4, padx=10, pady=(10, 10), sticky="w")

    return {
        f"{attr_prefix}_magnification": magnification_entry,
        f"{attr_prefix}_mode_var": prop_mode_var,
        f"{attr_prefix}_fixed_distance": fixed_distance_entry,
        f"{attr_prefix}_min_entry": min_entry,
        f"{attr_prefix}_max_entry": max_entry,
        f"{attr_prefix}_slider": propagate_slider,
        f"{attr_prefix}_current_label": current_value_label,
        f"{attr_prefix}_apply_button": apply_button,
        f"{attr_prefix}_unit_var": unit_var,
        f"{attr_prefix}_unit_selector": unit_selector,
        f"{attr_prefix}_metric_var": metric_var,
        f"{attr_prefix}_metric_selector": metric_selector,
        f"{attr_prefix}_plot_metric_var": plot_metric_var
    }


# Toolbar panel
def build_toolbar(app):
    """
    Creates –in *app.toolbar_frame*– the top toolbar with the
    fixed “Load” drop-down, Tools, Save, Main Menu, Home and Theme.
    """
    app.toolbar_frame = ctk.CTkFrame(
        app.viewing_frame, corner_radius=0,
        fg_color=("gray85", "gray15")
    )
    app.toolbar_frame.grid(row=0, column=0, padx=15, pady=(8, 4), sticky="ew")
    for col in range(6):
        app.toolbar_frame.grid_columnconfigure(col, weight=1)

    # Load (option-menu)
    app.load_menu = ctk.CTkOptionMenu(
        app.toolbar_frame,
        values=app.get_load_menu_values(),
        command=app._on_load_select,
        width=100, corner_radius=5
    )
    app.load_menu.grid(row=0, column=0, padx=3, sticky="ew")
    app.load_menu.set("Load")

    # Tools
    app.tools_menu = ctk.CTkOptionMenu(
        app.toolbar_frame, values=["Bio-Analysis", "Filters", "Speckle"],
        command=app._on_tools_select, width=100, corner_radius=5
    )
    app.tools_menu.grid(row=0, column=1, padx=3, sticky="ew")
    app.tools_menu.set("Tools")

    # Save
    app.save_menu = ctk.CTkOptionMenu(
        app.toolbar_frame, values=["Save FT", "Save Phase", "Save Amplitude"],
        command=app._on_save_select, width=100, corner_radius=5
    )
    app.save_menu.grid(row=0, column=2, padx=3, sticky="ew")
    app.save_menu.set("Save")

    # Navigation shortcuts
    ctk.CTkButton(
        app.toolbar_frame, text="Main Menu", width=100,
        corner_radius=5, command=lambda: app.change_menu_to("home")
    ).grid(row=0, column=3, padx=3, sticky="ew")

    ctk.CTkButton(
        app.toolbar_frame, text="Home", width=100,
        corner_radius=5, command=app.open_main_menu
    ).grid(row=0, column=4, padx=3, sticky="ew")

    # Light / Dark selector
    app.theme_menu = ctk.CTkOptionMenu(
        app.toolbar_frame, values=["Light", "Dark"],
        command=app._on_theme_select, width=100, corner_radius=5
    )
    app.theme_menu.grid(row=0, column=5, padx=3, sticky="ew")
    app.theme_menu.set("Theme")


# Views panels for hologram/fft and Phase/Amplitude
def build_two_views_panel(app):
    """
    Re-creates the twin viewers (left = Hologram/FT, right = Phase/Amp)
    with Zoom buttons aligned at the right edge of the titles.
    """
    # root container inside *app.viewing_frame
    app.two_views_frame = ctk.CTkFrame(
        app.viewing_frame, corner_radius=8,
        fg_color=("gray85", "gray15")
    )
    app.two_views_frame.grid(row=1, column=0, padx=10, pady=(40, 40), sticky="nsew")
    app.two_views_frame.grid_rowconfigure(0, weight=1)
    app.two_views_frame.grid_columnconfigure(0, weight=1)
    app.two_views_frame.grid_columnconfigure(1, weight=1)

    # LEFT VIEWER  (Hologram / Fourier-Transform)
    app.left_frame = ctk.CTkFrame(
        app.two_views_frame, width=app.viewbox_width,
        height=app.viewbox_height, fg_color=("gray80", "gray20")
    )
    app.left_frame.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="nsew")
    app.left_frame.grid_propagate(False)
    app.left_frame.grid_rowconfigure(3, weight=1)
    app.left_frame.grid_rowconfigure(4, minsize=50)
    app.left_frame.grid_columnconfigure(0, weight=1)
    app.left_frame.grid_columnconfigure(1, weight=0)

    # Radio-buttons, dropdowns
    app.holo_view_var = tk.StringVar(value="Hologram")
    app.radio_holo = ctk.CTkRadioButton(
        app.left_frame, text="Hologram",
        variable=app.holo_view_var, value="Hologram",
        command=app.update_left_view
    )
    app.radio_holo.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")

    app.zoom_holo_button = ctk.CTkButton(
        app.left_frame, text="Zoom 🔍", width=60,
        command=app.zoom_holo_view
    )
    # ⬇️ Moved to the title row
    # app.zoom_holo_button.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="e")
    app.zoom_holo_button.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="e")

    app.radio_ft = ctk.CTkRadioButton(
        app.left_frame, text="Fourier Transform",
        variable=app.holo_view_var, value="Fourier Transform",
        command=app.update_left_view
    )
    app.radio_ft.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")

    app.ft_mode_var = tk.StringVar(value="With logarithmic scale")
    app.ft_mode_button = ctk.CTkButton(
        app.left_frame, text="▼", width=25,
        command=app._show_ft_mode_menu
    )
    app.ft_mode_button.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="e")

    app.captured_title_label = ctk.CTkLabel(
        app.left_frame, text="Hologram",
        font=ctk.CTkFont(size=14, weight="bold")
    )
    # ⬇️ No columnspan; title at col 0, Zoom en col 1
    app.captured_title_label.grid(row=2, column=0, padx=5, pady=5, sticky="n")

    app.captured_label = ctk.CTkLabel(
        app.left_frame, text="", image=app.holo_views[0][1]
    )
    app.captured_label.grid(row=3, column=0, columnspan=2,
                            padx=5, pady=5, sticky="nsew")

    # Arrow navigation
    app.left_arrow_holo = ctk.CTkButton(
        app.left_frame, text="⪪", width=30,
        command=app.previous_hologram_view
    )
    app.right_arrow_holo = ctk.CTkButton(
        app.left_frame, text="⪫", width=30,
        command=app.next_hologram_view
    )
    app.left_arrow_holo.grid(row=4, column=0, sticky="w", padx=20, pady=5)
    app.right_arrow_holo.grid(row=4, column=1, sticky="e", padx=20, pady=5)
    app.hide_holo_arrows()             # start hidden

    # FT coordinate label (placed later)
    app.ft_coord_label = ctk.CTkLabel(
        app.left_frame, text="",
        font=ctk.CTkFont(size=10),
        fg_color=("gray80", "gray20"),
        text_color=("black", "white"),
        corner_radius=4, padx=6, pady=2
    )

    # RIGHT VIEWER  (Phase / Amplitude)
    app.right_frame = ctk.CTkFrame(
        app.two_views_frame, width=app.viewbox_width,
        height=app.viewbox_height, fg_color=("gray80", "gray20")
    )
    app.right_frame.grid(row=0, column=1, padx=5, pady=(0, 5), sticky="nsew")
    app.right_frame.grid_propagate(False)
    app.right_frame.grid_rowconfigure(3, weight=1)
    app.right_frame.grid_rowconfigure(4, minsize=50)
    app.right_frame.grid_columnconfigure(0, weight=1)
    app.right_frame.grid_columnconfigure(1, weight=0)

    app.recon_view_var = tk.StringVar(value="Phase Reconstruction ")

    # Phase Reconstruction (row 0) sub-frame
    right_row0 = ctk.CTkFrame(app.right_frame, fg_color="transparent")
    right_row0.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
    right_row0.grid_columnconfigure(0, weight=1)
    right_row0.grid_columnconfigure(1, weight=0)

    app.recon_view_var = tk.StringVar(value="Phase Reconstruction ")
    app.radio_phase = ctk.CTkRadioButton(
        right_row0, text="Phase Reconstruction ",
        variable=app.recon_view_var, value="Phase Reconstruction ",
        command=app.update_right_view
    )
    app.radio_phase.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")

    app.unwrap_mode_var = tk.StringVar(value="Unwrapping")
    app.unwrap_mode_button = ctk.CTkButton(
        right_row0, text="▼", width=25,
        command=getattr(app, "_show_unwrap_mode_menu", lambda: None)
    )
    app.unwrap_mode_button.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="e")

    app.zoom_recon_button = ctk.CTkButton(
        app.right_frame, text="Zoom 🔍", width=60,
        command=app.zoom_recon_view
    )
    # ⬇️ Moved to the title row
    # app.zoom_recon_button.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="e")
    app.zoom_recon_button.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="e")

    app.radio_amp = ctk.CTkRadioButton(
        app.right_frame, text="Amplitude Reconstruction ",
        variable=app.recon_view_var, value="Amplitude Reconstruction ",
        command=app.update_right_view
    )
    app.radio_amp.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")

    app.amp_mode_var = tk.StringVar(value="Amplitude")
    app.amp_mode_button = ctk.CTkButton(
        app.right_frame, text="▼", width=25,
        command=app._show_amp_mode_menu
    )
    app.amp_mode_button.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="e")

    app.processed_title_label = ctk.CTkLabel(
        app.right_frame, text="Phase Reconstruction ",
        font=ctk.CTkFont(size=14, weight="bold")
    )
    # ⬇️ No columnspan; title at col 0, Zoom en col 1
    app.processed_title_label.grid(row=2, column=0, padx=5, pady=5, sticky="n")

    app.processed_label = ctk.CTkLabel(
        app.right_frame, text="", image=app.recon_views[0][1]
    )
    app.processed_label.grid(row=3, column=0, columnspan=2,
                             padx=5, pady=5, sticky="nsew")


# Frame Bio-Analysis
def init_bio_analysis_frame(parent, apply_dimensions_callback, apply_qpi_callback,
                             update_qpi_placeholder_callback,
                             add_structure_quantification_callback,
                             apply_microstructure_callback):

    # Main container for the Bio-Analysis section
    parent.bio_frame = ctk.CTkFrame(parent, corner_radius=8)
    parent.bio_frame.grid_propagate(False)

    # Scrollable container
    parent.bio_container = ctk.CTkFrame(parent.bio_frame, corner_radius=8, width=420)
    parent.bio_container.grid_propagate(False)
    parent.bio_container.pack(fill="both", expand=True)

    # Vertical scrollbar
    parent.bio_scrollbar = ctk.CTkScrollbar(parent.bio_container, orientation='vertical')
    parent.bio_scrollbar.grid(row=0, column=0, sticky='ns')

    # Canvas that holds the content
    parent.bio_canvas = ctk.CTkCanvas(parent.bio_container, width=st.PARAMETER_FRAME_WIDTH)
    parent.bio_canvas.grid(row=0, column=1, sticky='nsew')

    parent.bio_container.grid_rowconfigure(0, weight=1)
    parent.bio_container.grid_columnconfigure(1, weight=1)
    parent.bio_canvas.configure(yscrollcommand=parent.bio_scrollbar.set)
    parent.bio_scrollbar.configure(command=parent.bio_canvas.yview)

    # Frame inside the canvas where widgets will be placed
    parent.bio_inner_frame = ctk.CTkFrame(parent.bio_canvas)
    parent.bio_canvas.create_window((0, 0), window=parent.bio_inner_frame, anchor='nw')

    # Title
    parent.main_title_bio = ctk.CTkLabel(
        parent.bio_inner_frame,
        text='Bio-Analysis',
        font=ctk.CTkFont(size=15, weight='bold')
    )
    parent.main_title_bio.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

    # DIMENSIONS SECTION
    parent.dimensions_frame = ctk.CTkFrame(
        parent.bio_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=st.PARAMETER_FRAME_HEIGHT * 2
    )
    parent.dimensions_frame.grid(row=1, column=0, sticky='ew', pady=2)
    parent.dimensions_frame.grid_propagate(False)
    for c in range(3):
        parent.dimensions_frame.columnconfigure(c, weight=1)

    ctk.CTkLabel(
        parent.dimensions_frame,
        text='Dimensions',
        font=ctk.CTkFont(weight='bold')
    ).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # Radio buttons to select image type
    parent.dimensions_var = tk.IntVar(value=0)
    for idx, label in enumerate(['Hologram', 'Amplitude', 'Phase']):
        ctk.CTkRadioButton(
            parent.dimensions_frame,
            text=label,
            variable=parent.dimensions_var,
            value=idx
        ).grid(row=1, column=idx, padx=5, pady=5, sticky='w')

    # Inter frame
    dime_frame = ctk.CTkFrame(parent.dimensions_frame, fg_color="transparent")
    dime_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    for c in range(4):
        dime_frame.columnconfigure(c, weight=1)

    # Pixel size input
    ctk.CTkLabel(dime_frame, text='Pixel Size (µm)').grid(row=0, column=0, padx=5, sticky='w')
    parent.pixel_size_entry = ctk.CTkEntry(dime_frame, width=50, placeholder_text='0.0')
    parent.pixel_size_entry.grid(row=0, column=1, padx=5, sticky='w')

    # Effective magnification input
    ctk.CTkLabel(dime_frame, text='Lateral Magnification').grid(row=0, column=2, padx=5, sticky='w')
    parent.magnification_entry = ctk.CTkEntry(dime_frame, width=50, placeholder_text='40x')
    parent.magnification_entry.grid(row=0, column=3, padx=5, sticky='w')

    # Apply button for dimensions
    ctk.CTkButton(
        parent.dimensions_frame,
        text='Apply',
        width=100,
        command=apply_dimensions_callback
    ).grid(row=3, column=0, padx=10, pady=5, sticky='w')

    # QPI MEASUREMENTS SECTION
    parent.QPI_frame = ctk.CTkFrame(
        parent.bio_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=int(st.PARAMETER_FRAME_HEIGHT * 2.5)
    )
    parent.QPI_frame.grid(row=2, column=0, sticky='ew', pady=2)
    parent.QPI_frame.grid_propagate(False)

    for c in range(4):
        parent.QPI_frame.columnconfigure(c, weight=1)

    # Section title
    ctk.CTkLabel(
        parent.QPI_frame,
        text='QPI Measurements',
        font=ctk.CTkFont(weight='bold')
    ).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # QPI type (Line vs Circle)
    parent.qpi_type_var = tk.IntVar(value=0)
    ctk.CTkRadioButton(parent.QPI_frame, text='ROI Lineal', variable=parent.qpi_type_var, value=0).grid(row=1, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkRadioButton(parent.QPI_frame, text='ROI Circular', variable=parent.qpi_type_var, value=1).grid(row=1, column=1, padx=5, pady=5, sticky='w')

    # Measurement mode dropdown with dynamic enabling
    parent.option_meas_var = tk.StringVar(value="Thickness")

    def on_option_meas_change(selected_option=None):
        if parent.option_meas_var.get() == "Thickness":
            parent.thickness_entry.configure(state="disabled")
            parent.ind_sample_entry.configure(state="normal")
            parent.ind_medium_entry.configure(state="normal")
        else:
            parent.thickness_entry.configure(state="normal")
            parent.ind_sample_entry.configure(state="disabled")
            parent.ind_medium_entry.configure(state="disabled")

    parent.option_meas_menu = ctk.CTkOptionMenu(
        parent.QPI_frame,
        variable=parent.option_meas_var,
        values=["Thickness", "Index"],
        command=on_option_meas_change
    )
    parent.option_meas_menu.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky='w')

    # QPI parameters
    measurement_frame = ctk.CTkFrame(parent.QPI_frame, fg_color="transparent")
    measurement_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    for c in range(4):
        measurement_frame.columnconfigure(c, weight=1)

    ctk.CTkLabel(measurement_frame, text='Zones').grid(row=0, column=0, padx=5, sticky='')
    parent.zones_get_qpi_entry = ctk.CTkEntry(measurement_frame, width=70, placeholder_text='1-10')
    parent.zones_get_qpi_entry.grid(row=1, column=0, padx=5, pady=5)


    ctk.CTkLabel(measurement_frame, text='Ind. Sample').grid(row=0, column=1, padx=5, sticky='')
    parent.ind_sample_entry = ctk.CTkEntry(measurement_frame, width=70, placeholder_text='1.33')
    parent.ind_sample_entry.grid(row=1, column=1, padx=5, pady=5)

    ctk.CTkLabel(measurement_frame, text='Ref. Medium').grid(row=0, column=2, padx=5, sticky='')
    parent.ind_medium_entry = ctk.CTkEntry(measurement_frame, width=70, placeholder_text='1.00')
    parent.ind_medium_entry.grid(row=1, column=2, padx=5, pady=5)

    ctk.CTkLabel(measurement_frame, text='Thickness (µm)').grid(row=0, column=3, padx=5, sticky='')
    parent.thickness_entry = ctk.CTkEntry(measurement_frame, width=70, placeholder_text='10.0')
    parent.thickness_entry.grid(row=1, column=3, padx=5, pady=5)

    # Apply button for QPI
    ctk.CTkButton(
        parent.QPI_frame,
        text='Apply',
        width=100,
        command=apply_qpi_callback
    ).grid(row=3, column=0, padx=10, pady=5, sticky='w')

    # Microstructure Metrics
    parent.microstructure_frame = ctk.CTkFrame(
        parent.bio_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=st.PARAMETER_FRAME_HEIGHT * 5
    )
    parent.microstructure_frame.grid(row=4, column=0, sticky='ew', pady=2)
    parent.microstructure_frame.grid_propagate(False)
    for c in range(3):
        parent.microstructure_frame.columnconfigure(c, weight=1)

    # Title label
    ctk.CTkLabel(
        parent.microstructure_frame,
        text='Microstructure Metrics',
        font=ctk.CTkFont(weight='bold')
    ).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # Radio buttons to select Amplitude or Phase
    parent.imageProcess_var = tk.IntVar(value=0)
    for idx, label in enumerate(['Amplitude', 'Phase']):
        ctk.CTkRadioButton(
            parent.microstructure_frame,
            text=label,
            variable=parent.imageProcess_var,
            value=idx
        ).grid(row=1, column=idx, padx=5, pady=5, sticky='w')

    # Radio buttons for thresholding methods
    parent.thresh_method_var = tk.StringVar(value="Otsu")
    methods = ['Otsu', 'Manual', 'Adaptive']
    for idx, method in enumerate(methods):
        ctk.CTkRadioButton(
            parent.microstructure_frame,
            text=method,
            variable=parent.thresh_method_var,
            value=method
        ).grid(row=2, column=idx, padx=5, pady=5, sticky='w')

    # Entry fields for thresholding parameters
    entry_labels = ['Min Area', 'Max Area', 'Ind. Sample', 'Ind. Medium', 'Threshold']
    parent.micro_entries = {}

    entry_frame = ctk.CTkFrame(parent.microstructure_frame, fg_color="transparent")
    entry_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
    for i, label in enumerate(entry_labels):
        ctk.CTkLabel(entry_frame, text=label).grid(row=0, column=i, padx=5, pady=2)
        entry = ctk.CTkEntry(entry_frame, width=60, placeholder_text='...')
        entry.grid(row=1, column=i, padx=5, pady=2)
        parent.micro_entries[label] = entry

    # Frame for checkboxes with aligned text and dotted label
    parent.struct_frame = ctk.CTkFrame(
        parent.microstructure_frame,
        fg_color="transparent"
    )
    parent.struct_frame.grid(row=4, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
    for c in range(2):
        parent.struct_frame.columnconfigure(c, weight=1)

    # Define BooleanVars
    parent.count_particles_var = tk.BooleanVar(value=False)
    parent.particles_areas_var = tk.BooleanVar(value=False)
    parent.automatic_profile_var = tk.BooleanVar(value=False)
    parent.profile_thickness_var = tk.BooleanVar(value=False)

    # Helper function for checkbox rows with dotted text
    def _add_opt(r, text, var):
        label_width = 90
        padded_text = text + '.' * max(0, label_width - len(text))
        label = ctk.CTkLabel(
            parent.struct_frame,
            text=padded_text,
            anchor='w',
            font=ctk.CTkFont(size=13)
        )
        label.grid(row=r, column=0, sticky='w', padx=(5, 0))
        checkbox = ctk.CTkCheckBox(parent.struct_frame, text="", variable=var)
        checkbox.grid(row=r, column=1, sticky='w', padx=(10, 5))

    # Add all checkbox options
    _add_opt(0, "Count Particles", parent.count_particles_var)
    _add_opt(1, "Particles Areas", parent.particles_areas_var)
    _add_opt(2, "Automatic Phase Profile", parent.automatic_profile_var)
    _add_opt(3, "Thickness Estimation", parent.profile_thickness_var)

    # Apply button for microstructure analysis
    ctk.CTkButton(
        parent.microstructure_frame,
        text='Apply',
        width=100,
        command=apply_microstructure_callback
    ).grid(row=5, column=0, padx=10, pady=5, sticky='w')

    # Trigger initial state
    on_option_meas_change()

    # Update scroll region
    parent.bio_inner_frame.update_idletasks()
    parent.bio_canvas.config(scrollregion=parent.bio_canvas.bbox("all"))


# Frame Filters
def init_filters_frame(self):
    self.filters_frame = ctk.CTkFrame(self, corner_radius=8)
    self.filters_frame.grid_propagate(False)

    self.filters_container = ctk.CTkFrame(self.filters_frame, corner_radius=8, width=420)
    self.filters_container.grid_propagate(False)
    self.filters_container.pack(fill="both", expand=True)

    self.filters_scrollbar = ctk.CTkScrollbar(self.filters_container, orientation='vertical')
    self.filters_scrollbar.grid(row=0, column=0, sticky='ns')

    self.filters_canvas = ctk.CTkCanvas(self.filters_container, width=st.PARAMETER_FRAME_WIDTH)
    self.filters_canvas.grid(row=0, column=1, sticky='nsew')

    self.filters_container.grid_rowconfigure(0, weight=1)
    self.filters_container.grid_columnconfigure(1, weight=1)
    self.filters_canvas.configure(yscrollcommand=self.filters_scrollbar.set)
    self.filters_scrollbar.configure(command=self.filters_canvas.yview)

    self.filters_inner_frame = ctk.CTkFrame(self.filters_canvas)
    self.filters_canvas.create_window((0, 0), window=self.filters_inner_frame, anchor='nw')

    self.main_title_filters = ctk.CTkLabel(
        self.filters_inner_frame,
        text='Filters',
        font=ctk.CTkFont(size=15, weight='bold')
    )
    self.main_title_filters.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

    self.dimensions_frame = ctk.CTkFrame(
        self.filters_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=st.PARAMETER_FRAME_HEIGHT * 4.0
    )
    self.dimensions_frame.grid(row=1, column=0, sticky='ew', pady=2)
    self.dimensions_frame.grid_propagate(False)
    self.dimensions_frame.columnconfigure(0, weight=1)
    self.dimensions_frame.columnconfigure(1, weight=1)
    self.dimensions_frame.columnconfigure(2, weight=1)

    # Adjust Image Filters
    dims_title = ctk.CTkLabel(
        self.dimensions_frame,
        text='Adjust Image Filters',
        font=ctk.CTkFont(weight="bold")
    )
    dims_title.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    dims_apply_btn = ctk.CTkButton(
        self.dimensions_frame,
        text='Apply',
        width=100,
        command=self.apply_filters
    )
    dims_apply_btn.grid(row=7, column=0, padx=10, pady=5, sticky='w')

    self.filters_dimensions_var = tk.IntVar(value=0)
    self.filters_dimensions_var.trace_add("write", self.on_filters_dimensions_change)

    ctk.CTkRadioButton(self.dimensions_frame, text='Hologram', variable=self.filters_dimensions_var, value=0).grid(row=1, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkRadioButton(self.dimensions_frame, text='Amplitude', variable=self.filters_dimensions_var, value=1).grid(row=1, column=1, padx=5, pady=5, sticky='w')
    ctk.CTkRadioButton(self.dimensions_frame, text='Phase', variable=self.filters_dimensions_var, value=2).grid(row=1, column=2, padx=5, pady=5, sticky='w')

    # Gamma Filter
    ctk.CTkLabel(self.dimensions_frame, text='Gamma Filter').grid(row=2, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkCheckBox(self.dimensions_frame, text='', variable=self.gamma_checkbox_var).grid(row=2, column=2, padx=5, pady=5, sticky='w')
    self.gamma_slider = ctk.CTkSlider(self.dimensions_frame, from_=st.MIN_GAMMA, to=st.MAX_GAMMA, command=self.adjust_gamma, width=10)
    self.gamma_slider.set(0)
    self.gamma_slider.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

    # Contrast Filter
    ctk.CTkLabel(self.dimensions_frame, text='Contrast Filter').grid(row=3, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkCheckBox(self.dimensions_frame, text='', variable=self.contrast_checkbox_var).grid(row=3, column=2, padx=5, pady=5, sticky='w')
    self.contrast_slider = ctk.CTkSlider(self.dimensions_frame, from_=st.MIN_CONTRAST, to=st.MAX_CONTRAST, command=self.adjust_contrast, width=10)
    self.contrast_slider.set(1)
    self.contrast_slider.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

    # High-Pass Filter
    ctk.CTkLabel(self.dimensions_frame, text='High-Pass Filter').grid(row=4, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkCheckBox(self.dimensions_frame, text='', variable=self.highpass_checkbox_var).grid(row=4, column=2, padx=5, pady=5, sticky='w')
    self.highpass_slider = ctk.CTkSlider(self.dimensions_frame, from_=st.MIN_CUTOFF, to=st.MAX_CUTOFF, command=self.adjust_highpass, width=10)
    self.highpass_slider.set(st.DEFAULT_CUTOFF)
    self.highpass_slider.grid(row=4, column=1, padx=5, pady=5, sticky='ew')

    # Low-Pass Filter
    ctk.CTkLabel(self.dimensions_frame, text='Low-Pass Filter').grid(row=5, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkCheckBox(self.dimensions_frame, text='', variable=self.lowpass_checkbox_var).grid(row=5, column=2, padx=5, pady=5, sticky='w')
    self.lowpass_slider = ctk.CTkSlider(self.dimensions_frame, from_=st.MIN_CUTOFF, to=st.MAX_CUTOFF, command=self.adjust_lowpass, width=10)
    self.lowpass_slider.set(st.DEFAULT_CUTOFF)
    self.lowpass_slider.grid(row=5, column=1, padx=5, pady=5, sticky='ew')

    # Adaptive Equalization
    ctk.CTkLabel(self.dimensions_frame, text='Adaptive').grid(row=6, column=0, padx=5, pady=5, sticky='w')
    ctk.CTkCheckBox(self.dimensions_frame, text='', variable=self.adaptative_eq_checkbox_var).grid(row=6, column=2, padx=5, pady=5, sticky='w')

    # Visualization Color Mode
    self.colormap_frame = ctk.CTkFrame(
        self.filters_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=160
    )
    self.colormap_frame.grid(row=2, column=0, pady=10, sticky="ew")
    self.colormap_frame.grid_propagate(True)

    for c in range(4):
        self.colormap_frame.columnconfigure(c, weight=0)

    title = ctk.CTkLabel(
        self.colormap_frame,
        text="Visualization Color Mode",
        font=ctk.CTkFont(weight="bold")
    )
    title.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")

    # Amplitude selector
    amp_lbl = ctk.CTkLabel(self.colormap_frame, text="Amplitude")
    amp_lbl.grid(row=1, column=0, padx=(5, 0), pady=5, sticky="e")

    amp_menu = ctk.CTkOptionMenu(
        self.colormap_frame,
        values=self.available_colormaps,
        variable=self.colormap_amp_var,
        width=100
    )
    amp_menu.grid(row=1, column=1, padx=(2, 10), pady=5, sticky="w")

    # Phase selector
    ph_lbl = ctk.CTkLabel(self.colormap_frame, text="Phase")
    ph_lbl.grid(row=1, column=2, padx=(5, 0), pady=5, sticky="e")

    ph_menu = ctk.CTkOptionMenu(
        self.colormap_frame,
        values=self.available_colormaps,
        variable=self.colormap_phase_var,
        width=100
    )
    ph_menu.grid(row=1, column=3, padx=(2, 5), pady=5, sticky="w")

    # Apply button
    apply_btn = ctk.CTkButton(
        self.colormap_frame,
        text="Apply",
        width=100,
        command=self.apply_colormap
    )
    apply_btn.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    self.filters_inner_frame.update_idletasks()
    self.filters_canvas.config(scrollregion=self.filters_canvas.bbox("all"))


# Frame speckle
def init_speckles_frame(self):
    """
    Initializes the speckle analysis and filtering panel within the GUI.
    Includes speckle metric controls, selection between amplitude/phase,
    zone/grid inputs, and speckle filtering options.
    """
    self.speckles_frame = ctk.CTkFrame(self, corner_radius=8)
    self.speckles_frame.grid_propagate(False)

    self.speckles_container = ctk.CTkFrame(self.speckles_frame, corner_radius=8, width=420)
    self.speckles_container.grid_propagate(False)
    self.speckles_container.pack(fill="both", expand=True)

    self.speckles_scrollbar = ctk.CTkScrollbar(self.speckles_container, orientation='vertical')
    self.speckles_scrollbar.grid(row=0, column=0, sticky='ns')

    self.speckles_canvas = ctk.CTkCanvas(self.speckles_container, width=st.PARAMETER_FRAME_WIDTH)
    self.speckles_canvas.grid(row=0, column=1, sticky='nsew')

    self.speckles_container.grid_rowconfigure(0, weight=1)
    self.speckles_container.grid_columnconfigure(1, weight=1)

    self.speckles_canvas.configure(yscrollcommand=self.speckles_scrollbar.set)
    self.speckles_scrollbar.configure(command=self.speckles_canvas.yview)

    self.speckles_inner_frame = ctk.CTkFrame(self.speckles_canvas)
    self.speckles_canvas.create_window((0, 0), window=self.speckles_inner_frame, anchor='nw')

    # Title
    self.main_title_speckles = ctk.CTkLabel(
        self.speckles_inner_frame,
        text='Speckle',
        font=ctk.CTkFont(size=15, weight='bold')
    )
    self.main_title_speckles.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

    # Speckle Measurements
    self.speckle_frame = ctk.CTkFrame(
        self.speckles_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=st.PARAMETER_FRAME_HEIGHT * 2.0
    )
    self.speckle_frame.grid(row=3, column=0, sticky='ew', pady=2)
    self.speckle_frame.grid_propagate(False)
    for c in range(3):
        self.speckle_frame.columnconfigure(c, weight=1)

    ctk.CTkLabel(
        self.speckle_frame,
        text='Speckle Measurements',
        font=ctk.CTkFont(weight='bold')
    ).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # Radio buttons for data type selection
    self.speckle_var = tk.IntVar(value=0)
    for idx, label in enumerate(['Hologram', 'Amplitude', 'Phase']):
        ctk.CTkRadioButton(
            self.speckle_frame,
            text=label,
            variable=self.speckle_var,
            value=idx
        ).grid(row=1, column=idx, padx=5, pady=5, sticky='w')

    # Zones, Rows, Cols entries in same row
    measu_frame = ctk.CTkFrame(self.speckle_frame, fg_color="transparent")
    measu_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
    measu_frame.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

    # Labels
    ctk.CTkLabel(measu_frame, text='Zones').grid(row=0, column=0, sticky='e', padx=(0, 2))
    ctk.CTkLabel(measu_frame, text='Rows').grid(row=0, column=2, sticky='e', padx=(10, 2))
    ctk.CTkLabel(measu_frame, text='Cols').grid(row=0, column=4, sticky='e', padx=(10, 2))

    # Entries
    self.zones_entry = ctk.CTkEntry(measu_frame, width=50, placeholder_text='0')
    self.zones_entry.grid(row=0, column=1, sticky='w')
    self.rows_entry = ctk.CTkEntry(measu_frame, width=50, placeholder_text='0')
    self.rows_entry.grid(row=0, column=3, sticky='w')
    self.cols_entry = ctk.CTkEntry(measu_frame, width=50, placeholder_text='0')
    self.cols_entry.grid(row=0, column=5, sticky='w')

    ctk.CTkButton(
        self.speckle_frame,
        text='Apply',
        width=100,
        command=self.apply_speckle
    ).grid(row=4, column=0, padx=10, pady=5, sticky='w')

    # Speckle Filters
    self.speckle_filters_frame = ctk.CTkFrame(
        self.speckles_inner_frame, width=st.PARAMETER_FRAME_WIDTH, height=285
    )
    self.speckle_filters_frame.grid(row=4, column=0, sticky="ew", pady=2)
    self.speckle_filters_frame.grid_propagate(True)
    for c in range(2):
        self.speckle_filters_frame.columnconfigure(c, weight=1)

    ctk.CTkLabel(
        self.speckle_filters_frame,
        text="Speckle Filters",
        font=ctk.CTkFont(weight="bold")
    ).grid(row=0, column=0, columnspan=3, padx=5, pady=10, sticky="w")

    self.speckle_filter_dim_var = tk.IntVar(value=1)
    ctk.CTkRadioButton(self.speckle_filters_frame, text="Amplitude",
                       variable=self.speckle_filter_dim_var, value=1).grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ctk.CTkRadioButton(self.speckle_filters_frame, text="Phase",
                       variable=self.speckle_filter_dim_var, value=2).grid(row=1, column=1, padx=5, pady=5, sticky="w")

    # Speckle method checkboxes
    self.spk_vars = []
    self.spk_param_entries = []
    method_configs = [
        ("HMF", "Iterations:"),
        ("Mean Filter", "Kernel:"),
        ("Median Filter", "Kernel:"),
        ("Gaussian Filter", "Sigma:"),
        ("SPP Filter", "Iterations:")
    ]

    for i, (label_text, param_label) in enumerate(method_configs):
        var = tk.BooleanVar(value=False)
        self.spk_vars.append(var)

        # Checkbox
        ctk.CTkCheckBox(
            self.speckle_filters_frame,
            text=label_text,
            variable=var,
            command=lambda idx=i: self.speckle_exclusive_callback(idx)
        ).grid(row=2 + i, column=0, padx=(10, 2), pady=3, sticky="w")

        # Label (parameter name)
        ctk.CTkLabel(
            self.speckle_filters_frame,
            text=param_label
        ).grid(row=2 + i, column=1, padx=(10, 2), pady=3, sticky="e")

        # Entry (parameter value)
        entry = ctk.CTkEntry(self.speckle_filters_frame, width=60, placeholder_text="0")
        entry.grid(row=2 + i, column=2, padx=2, pady=3, sticky="w")
        self.spk_param_entries.append(entry)

    ctk.CTkButton(
        self.speckle_filters_frame,
        text="Apply",
        width=100,
        command=self.apply_speckle_filter
    ).grid(row=7, column=0, padx=10, pady=5, sticky="w")

    # Speckle Comparison
    self.comparison_frame = ctk.CTkFrame(
        self.speckles_inner_frame,
        width=st.PARAMETER_FRAME_WIDTH,
        height=st.PARAMETER_FRAME_HEIGHT * 1.5
    )
    self.comparison_frame.grid(row=8, column=0, sticky='ew', pady=2)
    self.comparison_frame.grid_propagate(False)
    for c in range(3):
        self.comparison_frame.columnconfigure(c, weight=1)

    ctk.CTkLabel(
        self.comparison_frame,
        text='Speckle Comparison',
        font=ctk.CTkFont(weight='bold')
    ).grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # Checkboxes
    self.side_by_side_var = tk.BooleanVar()
    self.speckle_plot_var = tk.BooleanVar()
    self.line_profile_var = tk.BooleanVar()

    ctk.CTkCheckBox(
        self.comparison_frame,
        text="Side by Side",
        variable=self.compare_side_by_side_var
    ).grid(row=9, column=1, padx=10, pady=3, sticky='w')

    ctk.CTkCheckBox(
        self.comparison_frame,
        text="Speckle Plot",
        variable=self.compare_speckle_plot_var
    ).grid(row=9, column=2, padx=10, pady=3, sticky='w')

    ctk.CTkCheckBox(
        self.comparison_frame,
        text="Profile",
        variable=self.compare_line_profile_var
    ).grid(row=9, column=3, padx=10, pady=3, sticky='w')

    ctk.CTkButton(
        self.comparison_frame,
        text="Apply",
        width=100,
        command=lambda: tGUI.apply_speckle_comparison(self)
    ).grid(row=11, column=1, padx=0, pady=5, sticky="w")
