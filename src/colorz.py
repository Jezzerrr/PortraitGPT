import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from skimage.filters.rank import gradient
from skimage.morphology import dilation, disk
from skimage.segmentation import slic
from skimage.measure import regionprops


def get_matplotlib_palette(palette_name='viridis', n_colors=40, randomize=True):
    """
    Matplotlib color palettes: You can use matplotlib's built-in color palettes

    Args:
        palette_name: Name of the matplotlib colormap
            Examples: 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                      'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
                      'Blues', 'Greens', 'Reds', 'Oranges', 'Purples'
        n_colors: Number of colors to get from the palette
        randomize: If True, randomly sample colors from the palette

    Returns:
        List of RGB tuples with values from 0-255
    """

    cmap = plt.cm.get_cmap(palette_name, 256)  # Use max resolution

    if randomize:
        # Randomly sample from the colormap
        indices = np.sort(np.random.choice(256, size=n_colors, replace=False))
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in cmap(indices / 255)]
    else:
        # Evenly sample from the colormap
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in cmap(np.linspace(0, 1, n_colors))]


def get_seaborn_palette(palette_name='husl', n_colors=40, randomize=True):
    """
    Seaborn color palettes: Aesthetically pleasing palettes

    Args:
        palette_name: Name of the seaborn palette
            Examples: 'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
                      'husl', 'hls', 'rocket', 'mako', 'flare', 'crest'
        n_colors: Number of colors to get from the palette
        randomize: If True, randomly sample colors from the palette

    Returns:
        List of RGB tuples with values from 0-255
    """
    if randomize:
        # Generate more colors than needed, then randomly sample
        full_palette = sns.color_palette(palette_name, n_colors * 2)
        indices = np.random.choice(len(full_palette), size=n_colors, replace=False)
        palette = [full_palette[i] for i in indices]
    else:
        palette = sns.color_palette(palette_name, n_colors)

    return [(int( r *255), int( g *255), int( b *255)) for r, g, b in palette]


def get_colorbrewer_palette(palette_name='Set3', n_colors=40, randomize=True):
    """
    ColorBrewer palettes: Designed to be perceptually uniform and accessible

    Args:
        palette_name: Name of the ColorBrewer palette
            Examples: 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
                      'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn'
        n_colors: Number of colors to get from the palette
        randomize: If True, randomly sample colors from the palette

    Returns:
        List of RGB tuples with values from 0-255
    """
    # Get the maximum number of colors for this palette
    max_colors = min(plt.cm.get_cmap(palette_name).N, 256)
    cmap = plt.cm.get_cmap(palette_name, max_colors)

    if randomize:
        indices = np.sort(np.random.choice(max_colors, size=min(n_colors, max_colors), replace=False))
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in cmap(indices / max_colors)]
    else:
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in cmap(np.linspace(0, 1, min(n_colors, max_colors)))]


def get_distinct_colors(n_colors=40, randomize=True):
    """
    Pre-defined set of visually distinct colors

    Args:
        n_colors: Number of colors to return
        randomize: If True, randomly sample colors from the full set

    Returns:
        List of RGB tuples with values from 0-255
    """
    distinct_colors = [
        [31, 119, 180],    # Blue
        [255, 127, 14],    # Orange
        [44, 160, 44],     # Green
        [214, 39, 40],     # Red
        [148, 103, 189],   # Purple
        [140, 86, 75],     # Brown
        [227, 119, 194],   # Pink
        [127, 127, 127],   # Gray
        [188, 189, 34],    # Olive
        [23, 190, 207],    # Cyan
        [255, 152, 150],   # Light pink
        [214, 39, 40],     # Crimson
        [44, 160, 44],     # Forest green
        [152, 223, 138],   # Light green
        [31, 119, 180],    # Steel blue
        [255, 187, 120],   # Light orange
        [148, 103, 189],   # Violet
        [197, 176, 213],   # Lavender
        [140, 86, 75],     # Sienna
        [196, 156, 148],   # Tan
        [255, 215, 0],     # Gold
        [220, 20, 60],     # Crimson
        [50, 205, 50],     # Lime green
        [0, 191, 255],     # Deep sky blue
        [255, 105, 180],   # Hot pink
        [154, 205, 50],    # Yellow green
        [138, 43, 226],    # Blue violet
        [210, 105, 30],    # Chocolate
        [0, 139, 139],     # Dark cyan
        [85, 107, 47],     # Dark olive green
    ]

    if randomize and n_colors <= len(distinct_colors):
        return random.sample(distinct_colors, n_colors)
    else:
        return distinct_colors[:n_colors]


def get_tableau_palette(palette_name='Tableau_10', n_colors=40, randomize=True):
    """
    Tableau color palettes: Professional-looking color schemes

    Args:
        palette_name: Name of the Tableau palette
            Examples: 'Tableau_10', 'Tableau_20', 'ColorBlind_10', 'TableauLight_10',
                      'TableauMedium_10', 'TableauDark_10'
        n_colors: Number of colors to get from the palette
        randomize: If True, randomly sample colors from the palette

    Returns:
        List of RGB tuples with values from 0-255
    """
    tableau_palettes = {
        'Tableau_10': [
            [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
            [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]
        ],
        'Tableau_20': [
            [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
            [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
            [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
            [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
            [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]
        ],
        'ColorBlind_10': [
            [0, 107, 164], [255, 128, 14], [171, 171, 171], [89, 89, 89], [95, 158, 209],
            [200, 82, 0], [137, 137, 137], [163, 200, 236], [255, 188, 121], [207, 207, 207]
        ],
        'TableauLight_10': [
            [117, 170, 219], [231, 173, 94], [152, 217, 148], [242, 142, 130], [179, 151, 215],
            [180, 141, 122], [235, 171, 208], [168, 168, 168], [195, 196, 125], [110, 206, 220]
        ],
        'TableauMedium_10': [
            [74, 144, 226], [234, 147, 46], [106, 206, 88], [237, 102, 93], [159, 118, 195],
            [165, 120, 98], [231, 132, 196], [143, 143, 143], [179, 179, 89], [64, 188, 204]
        ],
        'TableauDark_10': [
            [31, 119, 180], [200, 80, 0], [0, 140, 0], [200, 0, 0], [106, 60, 155],
            [122, 80, 58], [187, 62, 145], [77, 77, 77], [142, 142, 0], [0, 130, 143]
        ],
    }

    if palette_name not in tableau_palettes:
        raise ValueError(f"Palette '{palette_name}' not found. Available palettes: {', '.join(tableau_palettes.keys())}")

    palette = tableau_palettes[palette_name]

    if randomize:
        indices = np.random.choice(len(palette), size=min(n_colors, len(palette)), replace=False)
        return [palette[i] for i in indices]
    else:
        return palette[:n_colors]


def get_nature_palette(palette_type='forest', n_colors=40, randomize=True):
    """
    Nature-inspired color palettes

    Args:
        palette_type: Type of nature palette ('forest', 'ocean', 'sunset', 'autumn', or 'spring')
        n_colors: Number of colors to get from the palette
        randomize: If True, randomly sample colors from the palette

    Returns:
        List of RGB tuples with values from 0-255
    """
    nature_palettes = {
        'forest': [
            [34, 120, 15],   # Forest Green
            [73, 97, 35],    # Moss Green
            [47, 79, 47],    # Dark Forest Green
            [138, 154, 91],  # Olive
            [200, 212, 126], # Light Olive
            [99, 135, 74],   # Medium Green
            [67, 87, 50],    # Pine Green
            [149, 185, 79],  # Leaf Green
            [92, 64, 51],    # Brown
            [161, 136, 127], # Tan
        ],
        'ocean': [
            [0, 105, 148],   # Deep Blue
            [0, 146, 199],   # Medium Blue
            [0, 187, 249],   # Light Blue
            [82, 195, 200],  # Teal
            [142, 216, 216], # Light Teal
            [182, 227, 227], # Pale Blue
            [55, 93, 147],   # Navy Blue
            [6, 71, 98],     # Dark Blue
            [12, 115, 137],  # Ocean Blue
            [164, 200, 225], # Sky Blue
        ],
        'sunset': [
            [255, 102, 0],   # Orange
            [255, 153, 0],   # Gold
            [255, 187, 51],  # Amber
            [245, 64, 33],   # Sunset Red
            [243, 121, 52],  # Vermilion
            [247, 157, 101], # Light Coral
            [155, 79, 150],  # Purple
            [97, 64, 117],   # Deep Purple
            [247, 201, 153], # Peach
            [96, 20, 17],    # Burgundy
        ],
        'autumn': [
            [194, 96, 37],   # Rust
            [156, 86, 95],   # Marsala
            [176, 87, 60],   # Copper
            [229, 139, 77],  # Burnt Orange
            [204, 163, 108], # Camel
            [110, 40, 30],   # Brown
            [242, 196, 12],  # Mustard
            [168, 47, 32],   # Brick Red
            [201, 138, 81],  # Light Brown
            [143, 90, 45],   # Chocolate
        ],
        'spring': [
            [87, 193, 144],  # Mint
            [140, 211, 164], # Light Green
            [230, 217, 120], # Lemon
            [242, 175, 198], # Pink
            [182, 215, 228], # Baby Blue
            [249, 239, 158], # Cream
            [255, 171, 76],  # Peach
            [176, 212, 123], # Lime
            [224, 185, 219], # Lavender
            [212, 150, 142], # Salmon
        ]
    }

    if palette_type not in nature_palettes:
        raise ValueError(f"Palette '{palette_type}' not found. Available palettes: {', '.join(nature_palettes.keys())}")

    palette = nature_palettes[palette_type]

    if randomize:
        return random.sample(palette, min(n_colors, len(palette)))
    else:
        return palette[:n_colors]
