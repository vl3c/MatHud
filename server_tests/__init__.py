# This file makes the Tests directory a Python package

import os
import sys

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add the site-packages directory to Python path
SITE_PACKAGES_PATH = os.path.join(
    PROJECT_ROOT,
    'static', 'python'
)
if SITE_PACKAGES_PATH not in sys.path:
    sys.path.append(SITE_PACKAGES_PATH)
