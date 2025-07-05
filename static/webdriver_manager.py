"""
MatHud Selenium WebDriver Management

Manages headless Firefox WebDriver for capturing canvas images for vision system.
Handles SVG state injection, page configuration, and screenshot capture operations.

Dependencies:
    - selenium: WebDriver automation framework
    - selenium.webdriver: Firefox WebDriver and configuration
    - selenium.webdriver.support: WebDriverWait and expected conditions
    - logging: Error logging for WebDriver operations
    - os: File system operations for screenshot storage
    - time: Timing delays for WebDriver operations
"""

import logging
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class WebDriverManager:
    """Manages Selenium WebDriver operations for capturing math visualizations.
    
    Provides headless Firefox WebDriver for injecting SVG states and capturing
    canvas screenshots for the vision system. Handles page configuration,
    error recovery, and resource cleanup.
    """
    
    def __init__(self, base_url="http://127.0.0.1:5000/"):
        """Initialize WebDriverManager with specified base URL.
        
        Args:
            base_url: Base URL for the MatHud application (default: localhost:5000)
        """
        self.base_url = base_url
        self.driver = None
        self._setup_driver()
    
    def capture_svg_state(self, svg_state):
        """Update SVG state and capture the canvas.
        
        Main entry point for vision system image capture. Updates the SVG
        content and takes a screenshot for AI analysis.
        
        Args:
            svg_state: Dictionary containing SVG content, dimensions, viewBox, and transform
        """
        try:
            self.update_svg_state(svg_state)
            self.capture_canvas()
        except Exception as e:
            print(f"Failed to capture canvas: {str(e)}")
            logging.error(f"Failed to capture canvas: {str(e)}")
    
    def update_svg_state(self, svg_state):
        """Update the SVG content and attributes with the provided state.
        
        Injects SVG content into the headless browser and configures container
        dimensions and SVG attributes for accurate rendering.
        
        Args:
            svg_state: Dictionary with SVG content, dimensions, viewBox, and transform
        """
        print("Loading SVG state...")
        self.driver.execute_script("""
            const svg = document.getElementById('math-svg');
            const container = document.querySelector('.math-container');
            
            // Set the SVG content
            svg.outerHTML = arguments[0].content;
            
            // Set container dimensions
            container.style.width = arguments[0].dimensions.width + 'px';
            container.style.height = arguments[0].dimensions.height + 'px';
            
            // Set SVG attributes
            const newSvg = document.getElementById('math-svg');  // Get reference to new SVG after outerHTML
            if (arguments[0].viewBox) {
                newSvg.setAttribute('viewBox', arguments[0].viewBox);
            }
            if (arguments[0].transform) {
                newSvg.setAttribute('transform', arguments[0].transform);
            }
            
            return true;  // Confirm execution
        """, svg_state)
        time.sleep(1)  # Give time for the SVG to be redrawn

    def _setup_driver(self):
        """Initialize the Firefox WebDriver with headless mode.
        
        Sets up headless Firefox WebDriver with retry logic and page configuration.
        Handles navigation to the application URL and configures page layout.
        """
        print("Initializing WebDriver...")
        firefox_options = Options()
        firefox_options.add_argument('--headless')
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.driver is None:
                    self.driver = webdriver.Firefox(options=firefox_options)
                    print("WebDriver started successfully.")
                
                time.sleep(3)  # Wait for initialization
                
                print(f"Attempting to navigate (attempt {attempt + 1}/{max_retries})...")
                self.driver.get(self.base_url)
                
                self._configure_page_layout()
                print("WebDriver navigation successful.")
                return
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if self.driver:
                    self.driver.quit()
                    self.driver = None
                
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("All attempts failed.")
                    raise
    
    def _configure_page_layout(self):
        """Configure the page layout by hiding chat container and adjusting math container.
        
        Optimizes the page layout for screenshot capture by hiding UI elements
        and maximizing the math container visibility.
        """
        self.driver.execute_script("""
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                chatContainer.style.display = 'none';
            }
            const mathContainer = document.querySelector('.math-container');
            if (mathContainer) {
                mathContainer.style.width = '100%';
            }
        """)
    
    def capture_canvas(self):
        """Capture the math visualization canvas as a PNG image.
        
        Takes a screenshot of the math container and saves it to canvas_snapshots/canvas.png.
        Handles element waiting, content verification, and size configuration.
        """
        print("\nStarting capture_canvas...")
        try:
            snapshots_dir = "canvas_snapshots"
            if not os.path.exists(snapshots_dir):
                os.makedirs(snapshots_dir)
            
            self._wait_for_svg_elements()
            self._verify_svg_content()
            dimensions = self._get_container_dimensions()
            self._configure_svg_size(dimensions)
            
            time.sleep(1)  # Give time for changes to take effect
            
            canvas_path = os.path.join(snapshots_dir, "canvas.png")
            container = self.driver.find_element(By.CLASS_NAME, "math-container")
            container.screenshot(canvas_path)
            print(f"Canvas capture completed successfully (dimensions: {dimensions['width']}x{dimensions['height']})")
            
        except Exception as e:
            print(f"Error in capture_canvas: {str(e)}")
            logging.error(f"Error in capture_canvas: {str(e)}")
    
    def _wait_for_svg_elements(self):
        """Wait for SVG elements to be present and visible.
        
        Uses WebDriverWait to ensure SVG elements are loaded before capture.
        """
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "math-svg"))
        )
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((
                By.CSS_SELECTOR, 
                "#math-svg > *"
            ))
        )
    
    def _verify_svg_content(self):
        """Verify that the SVG content is not empty.
        
        Raises:
            Exception: If SVG content is empty
        """
        svg = self.driver.find_element(By.ID, "math-svg")
        if not svg.get_attribute("innerHTML").strip():
            raise Exception("SVG content is empty")
    
    def _get_container_dimensions(self):
        """Get the actual rendered dimensions of the container.
        
        Returns:
            dict: Container dimensions with 'width' and 'height' keys
        """
        return self.driver.execute_script("""
            var container = document.querySelector('.math-container');
            var rect = container.getBoundingClientRect();
            return {
                width: rect.width,
                height: rect.height
            };
        """)
    
    def _configure_svg_size(self, dimensions):
        """Configure the SVG and container size with the given dimensions.
        
        Sets container and SVG styling for optimal screenshot capture.
        
        Args:
            dimensions: Dictionary with 'width' and 'height' keys
        """
        self.driver.execute_script("""
            var container = document.querySelector('.math-container');
            var svg = document.getElementById('math-svg');
            
            // Set container style
            container.style.width = arguments[0] + 'px';
            container.style.height = arguments[1] + 'px';
            
            // Set SVG attributes
            svg.setAttribute('width', arguments[0]);
            svg.setAttribute('height', arguments[1]);
            svg.setAttribute('viewBox', '0 0 ' + arguments[0] + ' ' + arguments[1]);
            svg.style.width = '100%';
            svg.style.height = '100%';
            
            // Force all SVG elements to be visible
            var elements = svg.getElementsByTagName('*');
            for(var i=0; i < elements.length; i++) {
                elements[i].style.visibility = 'visible';
                elements[i].style.opacity = '1';
            }
        """, dimensions['width'], dimensions['height'])
    
    def cleanup(self):
        """Clean up WebDriver resources.
        
        Properly closes the WebDriver and releases system resources.
        """
        if self.driver:
            self.driver.quit()
            self.driver = None 