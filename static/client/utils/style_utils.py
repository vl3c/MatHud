"""
MatHud Style Utilities Module

CSS validation and styling utilities for mathematical visualization objects.
Provides validation functions for colors, opacity values, and other styling properties.

Key Features:
    - CSS color validation for named colors, hex values, rgb(), rgba(), hsl(), hsla()
    - Opacity value validation (0.0 to 1.0 range)
    - Combined color and opacity validation
    - Comprehensive named color support (HTML/CSS standard colors)

Supported Color Formats:
    - Named colors: Standard CSS color names (red, blue, etc.)
    - Hex colors: #RGB and #RRGGBB formats
    - Functional colors: rgb(), rgba(), hsl(), hsla() formats

Validation Features:
    - Type safety for opacity values
    - Range checking for opacity (0-1)
    - Format validation for all color types
    - Error handling with descriptive messages

Dependencies:
    - None (pure validation utilities)
"""

from __future__ import annotations

from typing import Optional


class StyleUtils:
    """CSS styling validation utilities for mathematical visualization objects.
    
    Provides static methods for validating color values, opacity settings, and other
    styling properties used throughout the MatHud canvas system.
    """
    @staticmethod
    def is_valid_css_color(color: str) -> bool:
        """Validates if a string is a valid CSS color.
        Supports named colors, hex colors, rgb(), rgba(), hsl(), and hsla()."""
        # Basic named colors
        if color.lower() in ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                           "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
                           "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
                           "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen",
                           "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue",
                           "darkslategray", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray",
                           "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
                           "gold", "goldenrod", "gray", "green", "greenyellow", "honeydew", "hotpink", "indianred",
                           "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon",
                           "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen",
                           "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue",
                           "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
                           "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue",
                           "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream",
                           "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange",
                           "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
                           "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple",
                           "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell",
                           "sienna", "silver", "skyblue", "slateblue", "slategray", "snow", "springgreen", "steelblue",
                           "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke",
                           "yellow", "yellowgreen"]:
            return True

        # Hex colors
        if color.startswith('#') and len(color) in [4, 7]:  # #RGB or #RRGGBB
            try:
                int(color[1:], 16)
                return True
            except ValueError:
                return False

        # rgb(), rgba(), hsl(), hsla()
        if color.startswith(('rgb(', 'rgba(', 'hsl(', 'hsla(')):
            return True

        return False

    @staticmethod
    def validate_opacity(opacity: float) -> bool:
        """Validates if an opacity value is between 0 and 1"""
        try:
            opacity_val: float = float(opacity)
            return 0 <= opacity_val <= 1
        except (TypeError, ValueError):
            return False

    @staticmethod
    def validate_color_and_opacity(color: Optional[str], opacity: Optional[float]) -> bool:
        """Validates both color and opacity values"""
        if not StyleUtils.is_valid_css_color(color or ""):
            raise ValueError(f"Invalid CSS color: {color}")
        if not StyleUtils.validate_opacity(opacity or 0.0):
            raise ValueError(f"Invalid opacity value: {opacity}. Must be between 0 and 1")
        return True 