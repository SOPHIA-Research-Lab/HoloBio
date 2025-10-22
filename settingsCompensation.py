# compensation_settings_panel.py
# -*- coding: utf-8 -*-

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import re
import math


def create_compensation_settings(parent,
                                 init_semi_s=5,
                                 init_semi_step=0.2,
                                 init_tudhm_step=2.0,
                                 init_tudhm_method="TNC",
                                 init_limit="256",
                                 init_piston=True,
                                 init_pca=False):
    """
    Open the Compensation Method Settings dialog and return user selections.
    """

    # Result container mutated by callbacks
    result = {"value": None}

    # Create modal top-level window
    settings_window = ctk.CTkToplevel(parent)
    settings_window.title("Compensation Method Settings")
    settings_window.geometry("320x420")
    settings_window.transient(parent)
    settings_window.grab_set()

    # Center on screen
    settings_window.update_idletasks()
    sw, sh = settings_window.winfo_screenwidth(), settings_window.winfo_screenheight()
    w, h = 320, 420
    x = (sw // 2) - (w // 2)
    y = (sh // 2) - (h // 2)
    settings_window.geometry(f"{w}x{h}+{x}+{y}")

    # Tabs
    tabview = ctk.CTkTabview(settings_window)
    tabview.pack(fill="both", expand=True, padx=10, pady=(10, 0))
    tab1 = tabview.add("Semi-Heuristic")
    tab2 = tabview.add("Tu-DHM")
    tab3 = tabview.add("Vortex Legendre")

    # ---------------- Semi-Heuristic tab ----------------
    frame1 = ctk.CTkFrame(tab1)
    frame1.pack(fill="x", padx=12, pady=(8, 12))

    size_var = tk.StringVar(value=str(init_semi_s))
    step_var = tk.StringVar(value=str(init_semi_step))

    ctk.CTkLabel(frame1, text="Size search:").grid(row=0, column=0, padx=(10, 6), pady=(10, 6), sticky="w")
    ent_size = ctk.CTkEntry(frame1, textvariable=size_var, width=120, placeholder_text="e.g. 21")
    ent_size.grid(row=0, column=1, padx=(0, 10), pady=(10, 6), sticky="w")

    ctk.CTkLabel(frame1, text="Step:").grid(row=1, column=0, padx=(10, 6), pady=(6, 10), sticky="w")
    ent_step = ctk.CTkEntry(frame1, textvariable=step_var, width=120, placeholder_text="e.g. 0.2")
    ent_step.grid(row=1, column=1, padx=(0, 10), pady=(6, 10), sticky="w")

    frame1.grid_columnconfigure(2, weight=1)

    # ---------------- Tu-DHM tab ----------------
    frame2 = ctk.CTkFrame(tab2)
    frame2.pack(fill="x", padx=12, pady=(8, 12))

    step_var2 = tk.StringVar(value=str(init_tudhm_step))
    # Use SciPy-valid values; show pretty labels where needed
    optimizer_options = [
        ("L-BFGS-B", "L-BFGS-B"),
        ("Powell", "Powell"),
        ("TNC", "TNC"),
        ("Nelder-Mead", "Nelder-Mead"),
        ("COBYLA", "COBYLA"),
        ("SLSQP", "SLSQP"),
        ("Trust-constr", "trust-constr"),
    ]
    # Normalize initial method to exact stored value
    init_method_norm = init_tudhm_method if init_tudhm_method != "Trust-constr" else "trust-constr"
    optimizer_var = tk.StringVar(value=init_method_norm)

    ctk.CTkLabel(frame2, text="Step:").grid(row=0, column=0, padx=(10, 6), pady=(10, 6), sticky="w")
    ent_step2 = ctk.CTkEntry(frame2, textvariable=step_var2, width=120, placeholder_text="e.g. 0.1")
    ent_step2.grid(row=0, column=1, padx=(0, 10), pady=(10, 6), sticky="w")

    ctk.CTkLabel(frame2, text="Optimizer Method:", font=ctk.CTkFont(size=13, weight="bold"))\
        .grid(row=1, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

    for i, (label, value) in enumerate(optimizer_options, start=2):
        rb = ctk.CTkRadioButton(frame2, text=label, variable=optimizer_var, value=value)
        rb.grid(row=i, column=0, columnspan=2, padx=20, pady=2, sticky="w")

    frame2.grid_columnconfigure(2, weight=1)

    # ---------------- Vortex Legendre tab ----------------
    frame3 = ctk.CTkFrame(tab3)
    frame3.pack(fill="x", padx=12, pady=(8, 12))

    limit_var = tk.StringVar(value=str(init_limit))
    ctk.CTkLabel(frame3, text="Limit:").grid(row=0, column=0, padx=(10, 6), pady=(10, 6), sticky="w")
    ctk.CTkOptionMenu(frame3, variable=limit_var,
                      values=["64", "128", "256", "512", "1024"], width=100)\
        .grid(row=0, column=1, padx=(0, 10), pady=(10, 6), sticky="w")

    ctk.CTkLabel(frame3, text="Options:", font=ctk.CTkFont(size=13, weight="bold"))\
        .grid(row=1, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="w")

    piston_var = tk.BooleanVar(value=bool(init_piston))
    pca_var = tk.BooleanVar(value=bool(init_pca))

    ctk.CTkSwitch(frame3, text="Piston Compensation", variable=piston_var, onvalue=True, offvalue=False)\
        .grid(row=2, column=0, columnspan=2, padx=20, pady=5, sticky="w")
    ctk.CTkSwitch(frame3, text="PCA", variable=pca_var, onvalue=True, offvalue=False)\
        .grid(row=3, column=0, columnspan=2, padx=20, pady=5, sticky="w")

    frame3.grid_columnconfigure(2, weight=1)

    # ---------------- Buttons ----------------
    btn_frame = ctk.CTkFrame(settings_window)
    btn_frame.pack(fill="x", padx=10, pady=10)
    inner = ctk.CTkFrame(btn_frame, fg_color="transparent")
    inner.pack(anchor="center")

    def _validate_and_collect():
        # Semi-Heuristic validations
        raw_size = size_var.get().strip()
        raw_step = step_var.get().strip().replace(",", ".")
        if not re.fullmatch(r"[+-]?\d+", raw_size or ""):
            messagebox.showinfo("Invalid input", "Size search must be an odd positive integer.")
            return None
        s_val = int(raw_size)
        if s_val <= 0 or (s_val % 2 == 0):
            messagebox.showinfo("Invalid value", "Size search must be an odd positive integer.")
            return None
        try:
            step_val = float(raw_step)
        except ValueError:
            messagebox.showinfo("Invalid input", "Step must be a decimal number between 0 and 1.")
            return None
        if not (0 < step_val < 1) or math.isnan(step_val) or math.isinf(step_val):
            messagebox.showinfo("Invalid value", "Step must be between 0 and 1 (exclusive).")
            return None

        # Tu-DHM validations
        raw_step2 = step_var2.get().strip().replace(",", ".")
        try:
            step2_val = float(raw_step2)
        except ValueError:
            messagebox.showinfo("Invalid input", "Tu-DHM Step must be a number.")
            return None

        if math.isnan(step2_val) or math.isinf(step2_val):
            messagebox.showinfo("Invalid value", "Tu-DHM Step must be a finite number.")
            return None

        data = {
            "semi": {"s": s_val, "step": step_val},
            "tudhm": {"step": step2_val, "method": optimizer_var.get()},
            "vortex_legendre": {"limit": int(limit_var.get()), "piston": bool(piston_var.get()), "pca": bool(pca_var.get())}
        }
        return data

    def on_accept():
        data = _validate_and_collect()
        if data is None:
            return
        result["value"] = data
        settings_window.destroy()

    def on_cancel():
        result["value"] = None
        settings_window.destroy()

    ctk.CTkButton(inner, text="Accept", width=110, command=on_accept)\
        .grid(row=0, column=0, padx=10, pady=5)
    ctk.CTkButton(inner, text="Cancel", width=110, command=on_cancel)\
        .grid(row=0, column=1, padx=10, pady=5)

    # Block until the dialog is closed, then return the result
    settings_window.wait_window()
    return result["value"]
