import customtkinter as ctk

# Set appearance mode and theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class MainMenu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HoloBio")

        # Optimized window size - much smaller and more appropriate
        self.geometry("800x600")
        self.resizable(False, False)  # Fixed size for better control

        # Center the window on screen
        self.center_window()

        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Content
        self.grid_rowconfigure(2, weight=0)  # Footer

        # Header with application title and description
        self.create_header()

        # Main content area
        self.create_main_content()

        # Optional footer
        self.create_footer()

    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = 800
        height = 600
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def create_header(self):
        """Create header section with title and description"""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=40, pady=(30, 20))

        # Main title
        title_label = ctk.CTkLabel(
            header_frame,
            text="HoloBio",
            font=ctk.CTkFont(family="Arial", size=32, weight="bold"),
            text_color="#1f538d"
        )
        title_label.pack(pady=(0, 5))

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Digital Holographic Microscopy Platform",
            font=ctk.CTkFont(family="Arial", size=16),
            text_color="#666666"
        )
        subtitle_label.pack()

    def create_main_content(self):
        """Create the main content with optimized sections"""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=40, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)

        # Real Time Section
        rt_frame = ctk.CTkFrame(
            main_frame,
            fg_color="#f0f0f0",
            corner_radius=15,
            height=180
        )
        rt_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        rt_frame.grid_columnconfigure((0, 1), weight=1)
        rt_frame.grid_propagate(False)  # Maintain fixed height

        # Real Time title with icon
        rt_title = ctk.CTkLabel(
            rt_frame,
            text="Real-Time Hologram Processing",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color="#1f538d"
        )
        rt_title.grid(row=0, column=0, columnspan=2, pady=(20, 15))

        # Real Time buttons
        btn_font = ctk.CTkFont(family="Arial", size=16, weight="bold")

        btn_rt_dhm = ctk.CTkButton(
            rt_frame,
            text="DHM",
            command=self.open_interface4,
            width=180,
            height=60,
            border_width=2,
            border_color="#1f538d",
            font=btn_font,
            hover_color="#1565c0"
        )
        btn_rt_dhm.grid(row=1, column=0, padx=20, pady=(0, 20))

        btn_rt_dlhm = ctk.CTkButton(
            rt_frame,
            text="DLHM",
            command=self.open_interface2,
            width=180,
            height=60,
            border_width=2,
            border_color="#1f538d",
            font=btn_font,
            hover_color="#1565c0"
        )
        btn_rt_dlhm.grid(row=1, column=1, padx=20, pady=(0, 20))

        # Reconstruction Tools Section
        rec_frame = ctk.CTkFrame(
            main_frame,
            fg_color="#f0f0f0",
            corner_radius=15,
            height=180
        )
        rec_frame.grid(row=1, column=0, sticky="ew", pady=(15, 0))
        rec_frame.grid_columnconfigure((0, 1), weight=1)
        rec_frame.grid_propagate(False)  # Maintain fixed height

        # Reconstruction title with icon
        rec_title = ctk.CTkLabel(
            rec_frame,
            text="Offline Hologram Processing",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color="#1f538d"
        )
        rec_title.grid(row=0, column=0, columnspan=2, pady=(20, 15))

        # Reconstruction buttons
        btn_rec_dhm = ctk.CTkButton(
            rec_frame,
            text="DHM",
            command=self.open_interface1,
            width=180,
            height=60,
            border_width=2,
            border_color="#1f538d",
            font=btn_font,
            hover_color="#1565c0"
        )
        btn_rec_dhm.grid(row=1, column=0, padx=20, pady=(0, 20))

        btn_rec_dlhm = ctk.CTkButton(
            rec_frame,
            text="DLHM",
            command=self.open_interface3,
            width=180,
            height=60,
            border_width=2,
            border_color="#1f538d",
            font=btn_font,
            hover_color="#1565c0"
        )
        btn_rec_dlhm.grid(row=1, column=1, padx=20, pady=(0, 20))

    def create_footer(self):
        """Create footer with additional information"""
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.grid(row=2, column=0, sticky="ew", padx=40, pady=(20, 30))

        footer_label = ctk.CTkLabel(
            footer_frame,
            text="Select processing mode and microscopy type to begin",
            font=ctk.CTkFont(family="Arial", size=12),
            text_color="#888888"
        )
        footer_label.pack()

    def show_loading(self, message="Loading..."):
        """Show loading indicator while switching interfaces"""
        loading_window = ctk.CTkToplevel(self)
        loading_window.title("Loading")
        loading_window.geometry("300x100")
        loading_window.transient(self)
        loading_window.grab_set()

        # Center loading window
        loading_window.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 150
        y = self.winfo_y() + (self.winfo_height() // 2) - 50
        loading_window.geometry(f"300x100+{x}+{y}")

        loading_label = ctk.CTkLabel(
            loading_window,
            text=message,
            font=ctk.CTkFont(family="Arial", size=14)
        )
        loading_label.pack(expand=True)

        loading_window.update()
        return loading_window

    def safe_interface_switch(self, module_name, interface_name="App", custom_message=None):
        """Safely switch to another interface with error handling"""
        try:
            message = custom_message if custom_message else f"Loading {module_name}..."
            loading = self.show_loading(message)

            from importlib import import_module, reload
            dhm_mod = import_module(module_name)
            dhm_mod = reload(dhm_mod)

            loading.destroy()
            self.destroy()

            PrimaryApp = getattr(dhm_mod, interface_name)
            PrimaryApp().mainloop()

        except ImportError as e:
            loading.destroy()
            self.show_error(f"Module not found: {module_name}\n{str(e)}")
        except AttributeError as e:
            loading.destroy()
            self.show_error(f"Interface not found: {interface_name}\n{str(e)}")
        except Exception as e:
            loading.destroy()
            self.show_error(f"Error loading interface: {str(e)}")

    def show_error(self, message):
        """Show error dialog"""
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        error_window.geometry("400x200")
        error_window.transient(self)
        error_window.grab_set()

        # Center error window
        error_window.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 200
        y = self.winfo_y() + (self.winfo_height() // 2) - 100
        error_window.geometry(f"400x200+{x}+{y}")

        error_label = ctk.CTkLabel(
            error_window,
            text=message,
            font=ctk.CTkFont(family="Arial", size=12),
            wraplength=350
        )
        error_label.pack(expand=True, padx=20, pady=20)

        ok_button = ctk.CTkButton(
            error_window,
            text="OK",
            command=error_window.destroy,
            width=100
        )
        ok_button.pack(pady=(0, 20))

    # Interface opening methods with improved error handling
    def open_interface1(self):
        """Open DHM Post-Processing interface"""
        self.safe_interface_switch("holobio.main_DHM_PP", custom_message="Loading DHM offline module...")

    def open_interface2(self):
        """Open DLHM Real-Time interface"""
        self.safe_interface_switch("holobio.main_DLHM_RT", custom_message="Loading DLHM real-time module...")

    def open_interface3(self):
        """Open DLHM Post-Processing interface"""
        self.safe_interface_switch("holobio.main_DLHM_PP", custom_message="Loading DLHM offline module...")

    def open_interface4(self):
        """Open DHM Real-Time interface"""
        self.safe_interface_switch("holobio.main_DHM_RT", custom_message="Loading DHM real-time module...")


def main():
    app = MainMenu()
    app.mainloop()

if __name__ == "__main__":
    main()