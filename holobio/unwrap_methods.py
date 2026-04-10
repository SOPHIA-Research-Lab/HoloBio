
import numpy as np
from skimage.restoration import unwrap_phase as sk_unwrap
from . import unwrapping as uw


def apply_unwrap(raw_phase: np.ndarray, method_name: str | None) -> np.ndarray:
    """
    Apply the selected unwrapping method to raw_phase (in radians, wrapped in [-pi, pi]).
    """
    raw_phase_1 = grayscaleToPhase(raw_phase)

    if method_name == "Skimage Unwrap":
        return sk_unwrap(raw_phase_1)

    elif method_name == "WPhU":
        return uw.phase_unwrap(raw_phase_1)

    elif method_name == "Original":
        return raw_phase

    return raw_phase


def grayscaleToPhase(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image (0–255 or 0–1) into a phase map in radians [-π, π].
    """
    image = np.array(image)
    img = image.astype(float)

    # Normalize to [0, 1] if values are in [0, 255]
    if img.max() > 1.0:
        img = img / 255.0

    # Scale to [-π, π]
    phase = img * (2 * np.pi) - np.pi
    return phase