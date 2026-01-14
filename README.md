# HoloBio: Digital Holographic Microscopy Library

**Version:** 1.0.1
**Author:** SOPHIA-Research-Lab
**License:** GPL-3.0

HoloBio is an open-source Python library and GUI for quantitative analysis in Digital Holographic Microscopy (DHM). It supports real-time and offline modes, various optical setups (lens-based and lensless), and provides advanced tools for analyzing biological samples.



## Installation

You can install `holoBio` directly from PyPI (Python 3.8+ required):

```bash
pip install holoBio
```

Alternatively, to install from source:

1.  Clone the repository:
    ```bash
    git clone https://github.com/SOPHIA-Research-Lab/HoloBio.git
    cd HoloBio
    ```
2.  Install dependencies and the package:
    ```bash
    pip install .
    ```

---

## Usage

Once installed, you can launch the main graphical interface by running the following command in your terminal:

```bash
holobio
```

Or, if the command is not found (due to PATH issues):

```bash
python -m holobio
```

This will open the main menu where you can select between DHM and DLHM modes, and between Real-Time and Post-Processing interfaces.

---

## Project Structure

The source code is organized in the `holobio/` package directory:

### Core Application
- **`Main_.py`**: The entry point of the application. Initializes the main menu window and handles navigation between different modules.
- **`__init__.py`**: Marks the directory as a Python package.

### Main Interfaces (GUI)
- **`main_DHM_PP.py`**: **DHM Post-Processing**. Interface for loading and analyzing existing holograms.
- **`main_DHM_RT.py`**: **DHM Real-Time**. Interface for capturing and analyzing holograms live from a camera.
- **`main_DLHM_PP.py`**: **DLHM Post-Processing**. Specialized interface for Lensless Digital Holographic Microscopy (offline).
- **`main_DLHM_RT.py`**: **DLHM Real-Time**. Specialized interface for Lensless Digital Holographic Microscopy (live).

### Algorithms & Methods
- **`pyDHM_methods.py`**: Core numerical methods for reconstruction, propagation, and filtering.
- **`phaseShifting.py`**: Implementation of phase-shifting algorithms for phase retrieval.
- **`unwrap_methods.py`**: Wrapper functions for different phase unwrapping techniques.
- **`unwrapping.py`**: Implementation of specific phase unwrapping algorithms (e.g., weighted least squares).
- **`settingsCompensation.py`**: GUI and logic for configuring phase compensation parameters (e.g., for tilt/aberration removal).
- **`parallel_rc.py`**: Parallel processing implementations for faster reconstruction.
- **`track_particles_kalman.py`**: Kalman filter implementation for tracking moving particles in real-time or video sequences.

### Utilities
- **`tools_GUI.py`**: Common GUI widgets, dialogs, and helper functions reused across different interfaces.
- **`functions_GUI.py`**: Additional functional logic for GUI interactions.
- **`tools_microstructure.py`**: Tools for morphological analysis (segmentation, particle counting, area measurement).
- **`utilities.py`**: General purpose utility functions (image I/O, array manipulation).
- **`settings.py`**: Global configuration and constants.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE` file for details.

## Additional Resources

For more detailed information, please refer to the **User Guide** and **Scientific Paper** located in the `Complementary_info` directory of this repository:

*   📄 **User Guide:** `Complementary_info/User_Manual.pdf` - Comprehensive manual covering installation, mode selection (Real-Time vs Offline), and advanced reconstruction procedures.
*   📄 **Paper:** `Complementary_info/HoloBio_paper.pdf` - Detailed scientific background and validation of the tool.

## Support or Contact

If you have any questions or need support, please contact:

| Researcher | Email | Google Scholar |
| :--- | :--- | :--- |
| **Ana Doblas** | adoblas@umassd.edu | [Profile](https://scholar.google.es/citations?user=PvvDEMYAAAAJ&hl=en) |
| **Raul Castañeda** | racastaneq@eafit.edu.co | [Profile](https://scholar.google.com/citations?user=RBtkL1oAAAAJ&hl=es) |

## Citation

If you use HoloBio in your research, please refer to the following manuscript:

> W. Mona, M. J. Gil-Herrera, E. Mazo, D. Córdoba, S. Obando, M. J. Lopera, R. Restrepo, C. Trujillo, A. Doblas, and R. Castañeda.  
> **"HoloBio: A Holographic Microscopy Tool for Quantitative Biological Analysis"**  
> *Under review, PLOS Computational Biology.*

