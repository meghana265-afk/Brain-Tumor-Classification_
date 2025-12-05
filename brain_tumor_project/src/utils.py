"""
utils.py
Small helper utilities used across the project. Each function is commented
to explain purpose and expected inputs/outputs.
"""

import os


def count_images(folder):
    """Count total image files in `folder` recursively.

    This function walks the directory tree and counts files matching common
    image extensions. Useful for quickly verifying dataset sizes.

    Args:
      folder (str or Path): root directory to scan

    Returns:
      int: number of image files found
    """

    total = 0

    # Walk through all directories and files under `folder`
    for root, dirs, files in os.walk(folder):
        for f in files:
            # Count common image file extensions (case-insensitive)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1

    return total
