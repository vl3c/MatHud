import os
import sys

# Add the site-packages directory to Python path
SITE_PACKAGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Go up one level from Tests
    'static', 'client'
)

# Add the path if it's not already there
if SITE_PACKAGES_PATH not in sys.path:
    sys.path.append(SITE_PACKAGES_PATH) 