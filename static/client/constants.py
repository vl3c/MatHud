"""
MatHud Client-Side Constants Configuration

Global configuration values for the mathematical canvas visualization system.
Defines styling, interaction thresholds, and performance parameters for consistent behavior.

Categories:
    - Visual Styling: Point sizes, colors, fonts
    - User Interaction: Click thresholds, zoom factors
    - Angle Visualization: Arc display and text positioning
    - Performance: Event throttling for smooth interactions

Dependencies:
    - None (pure configuration module)
"""

from __future__ import annotations

# ===== VISUAL STYLING CONSTANTS =====
# Default appearance settings for canvas elements
default_point_size: int = 2
default_color: str = "black"
default_font_size: int = 16
default_font_family: str = "Inter, sans-serif"
point_label_font_size: float = default_font_size * 5 / 8  # 5/8 ratio for readable point labels
default_label_font_size: float = default_font_size * 0.875
label_min_screen_font_px: float = 0.0
label_vanish_threshold_px: float = 2.0
label_text_max_length: int = 160
label_line_wrap_threshold: int = 40
default_label_rotation_degrees: float = 0.0
successful_call_message: str = "Call successful!"

# ===== USER INTERACTION CONSTANTS =====
# Timing and behavior thresholds for user interactions
double_click_threshold_s: float = 0.2  # Maximum time between clicks for double-click detection

# ===== ANGLE VISUALIZATION CONSTANTS =====
# Specialized settings for angle display and measurement
DEFAULT_ANGLE_COLOR: str = "blue"
DEFAULT_ANGLE_ARC_SCREEN_RADIUS: int = 15  # Arc radius in pixels for angle indicators
DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR: float = 1.8  # Text positioning relative to arc radius

# ===== ZOOM AND NAVIGATION CONSTANTS =====
# Scaling factors for canvas zoom operations
zoom_in_scale_factor: float = 1.1  # 10% increase per zoom in action
zoom_out_scale_factor: float = 0.9  # 10% decrease per zoom out action

# ===== RENDERER SELECTION =====
# Default rendering backend used by Canvas when none is specified
DEFAULT_RENDERER_MODE: str = "canvas2d"  # other options: "svg", "webgl"

# ===== PERFORMANCE OPTIMIZATION CONSTANTS =====
# Event throttling settings for smooth user experience
mousemove_throttle_ms: int = 8  # Mouse movement throttling (8ms = ~120fps for smooth panning)