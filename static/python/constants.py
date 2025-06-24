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

# ===== VISUAL STYLING CONSTANTS =====
# Default appearance settings for canvas elements
default_point_size = 2
default_color = "black"
default_font_size = 16
point_label_font_size = default_font_size * 5/8  # 5/8 ratio for readable point labels
successful_call_message = "Call successful!"

# ===== USER INTERACTION CONSTANTS =====
# Timing and behavior thresholds for user interactions
double_click_threshold_s = 0.2  # Maximum time between clicks for double-click detection

# ===== ANGLE VISUALIZATION CONSTANTS =====
# Specialized settings for angle display and measurement
DEFAULT_ANGLE_COLOR = "blue"
DEFAULT_ANGLE_ARC_SCREEN_RADIUS = 15  # Arc radius in pixels for angle indicators
DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR = 1.8  # Text positioning relative to arc radius

# ===== ZOOM AND NAVIGATION CONSTANTS =====
# Scaling factors for canvas zoom operations
zoom_in_scale_factor = 1.1   # 10% increase per zoom in action
zoom_out_scale_factor = 0.9  # 10% decrease per zoom out action

# ===== PERFORMANCE OPTIMIZATION CONSTANTS =====
# Event throttling settings for smooth user experience
mousemove_throttle_ms = 8  # Mouse movement throttling (8ms = ~120fps for smooth panning)