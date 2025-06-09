#!/usr/bin/env python3
"""
MatHud Architecture Diagram Generation

Generates comprehensive architecture diagrams using the Python diagrams library:
- System Overview: Overall MatHud system flow
- AI Integration: How OpenAI API integrates with the app
- WebDriver Flow: Vision capture and processing workflow  
- Data Flow: User interaction → AI → Response pipeline

Dependencies:
    pip install diagrams
"""

import os
from pathlib import Path


class ArchitectureDiagramGenerator:
    # Font configuration constants
    DIAGRAM_FONT = "Arial"
    DIAGRAM_FONT_SIZE = "12"  # Larger font size as requested
    
    def __init__(self, png_dir="../generated_png", svg_dir="../generated_svg", formats=["png", "svg"]):
        # Convert to absolute paths to avoid issues with cwd changes
        self.png_dir = Path(png_dir).resolve()
        self.svg_dir = Path(svg_dir).resolve()
        self.formats = formats
        
        # Create output directories
        if "png" in formats:
            self.png_dir.mkdir(exist_ok=True)
        if "svg" in formats:
            self.svg_dir.mkdir(exist_ok=True)
    
    def get_output_dir(self, fmt):
        """Get the appropriate output directory for the given format."""
        if fmt == "png":
            return self.png_dir
        elif fmt == "svg":
            return self.svg_dir
        else:
            return self.png_dir
    
    def _post_process_svg_fonts(self, svg_file):
        """Post-process SVG file to ensure configured font is used."""
        try:
            if not svg_file.exists():
                return
                
            content = svg_file.read_text(encoding='utf-8')
            
            # Replace common serif fonts with configured font
            font_replacements = [
                ('Times New Roman', self.DIAGRAM_FONT),
                ('Times', self.DIAGRAM_FONT),
                ('serif', f'{self.DIAGRAM_FONT},sans-serif'),
                ('font-family="Times-Roman"', f'font-family="{self.DIAGRAM_FONT}"'),
                ('font-family=\'Times-Roman\'', f'font-family=\'{self.DIAGRAM_FONT}\''),
                ('Times-Roman', self.DIAGRAM_FONT),
                ('TimesNewRoman', self.DIAGRAM_FONT)
            ]
            
            modified = False
            for old_font, new_font in font_replacements:
                if old_font in content:
                    content = content.replace(old_font, new_font)
                    modified = True
            
            if modified:
                svg_file.write_text(content, encoding='utf-8')
                print(f"  ✓ Updated fonts to {self.DIAGRAM_FONT} in: {svg_file.name}")
                
        except Exception as e:
            print(f"  ⚠ Could not update fonts in {svg_file.name}: {e}")

    def generate_system_overview_diagram(self):
        """Generate overall MatHud system architecture diagram."""
        try:
            from diagrams import Diagram, Cluster
            from diagrams.onprem.client import Client
            from diagrams.programming.framework import Flask
            from diagrams.programming.language import Python
            from diagrams.onprem.network import Internet
            from diagrams.onprem.database import Mongodb
            from diagrams.onprem.inmemory import Redis
            from diagrams.saas.chat import Slack  # Using available chat service icon
            
            # Generate for each format
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
                diagram_path = output_dir / "system_overview"
                
                with Diagram("MatHud System Overview", 
                            filename=str(diagram_path), 
                            show=False, 
                            direction="TB",
                            outformat=fmt,
                            graph_attr={
                                "fontname": self.DIAGRAM_FONT, 
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "dpi": "150"  # Higher DPI for smaller icons
                            },
                            node_attr={
                                "fontname": self.DIAGRAM_FONT,
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "width": "1.2",  # Smaller icon width
                                "height": "1.2"  # Smaller icon height
                            }):
                    
                    # User Interface Layer
                    user = Client("User Browser")
                    
                    with Cluster("Frontend Layer"):
                        brython = Python("Brython\n(Client-side Python)")
                        canvas = Python("SVG Canvas\n& Drawing System")
                        ui = Python("User Interface\n& Controls")
                    
                    with Cluster("Flask Backend"):
                        flask_app = Flask("Flask Server\n(app.py)")
                        app_manager = Python("App Manager\n(Dependency Injection)")
                        routes = Python("Routes\n(API Endpoints)")
                    
                    with Cluster("Core Services"):
                        ai_service = Python("OpenAI API\nManager")
                        webdriver_service = Python("WebDriver\nManager")
                        workspace_service = Python("Workspace\nManager")
                        log_service = Python("Log Manager")
                        tool_processor = Python("Tool Call\nProcessor")
                    
                    with Cluster("External Services"):
                        openai_api = Slack("OpenAI API\n(GPT-4, Vision)")
                        selenium = Python("Selenium\n+ Firefox")
                        internet = Internet("Internet")
                    
                    with Cluster("Data Storage"):
                        workspaces = Mongodb("JSON\nWorkspaces")
                        logs = Redis("Application\nLogs")
                        snapshots = Python("Canvas\nSnapshots")
                    
                    # Connections - User to Frontend
                    user >> [brython, ui]
                    brython >> canvas
                    ui >> canvas
                    
                    # Frontend to Backend
                    [brython, ui] >> flask_app
                    flask_app >> app_manager
                    app_manager >> routes
                    
                    # Backend to Services  
                    routes >> [ai_service, webdriver_service, workspace_service, log_service]
                    ai_service >> tool_processor
                    
                    # Services to External
                    ai_service >> openai_api
                    webdriver_service >> selenium
                    openai_api >> internet
                    selenium >> internet
                    
                    # Services to Storage
                    workspace_service >> workspaces
                    log_service >> logs
                    webdriver_service >> snapshots
                    
                print(f"  ✓ System overview diagram: {diagram_path}.{fmt}")
                
                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._post_process_svg_fonts(output_dir / f'system_overview.{fmt}')
            
        except Exception as e:
            print(f"  ✗ System overview diagram failed: {e}")
    
    def generate_ai_integration_diagram(self):
        """Generate AI integration flow diagram."""
        try:
            from diagrams import Diagram, Cluster, Edge
            from diagrams.onprem.client import Client
            from diagrams.programming.language import Python
            from diagrams.onprem.network import Internet
            from diagrams.saas.chat import Slack
            
            # Generate for each format
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
                diagram_path = output_dir / "ai_integration"
                
                with Diagram("MatHud AI Integration Flow", 
                            filename=str(diagram_path), 
                            show=False,
                            direction="LR",
                            outformat=fmt,
                            graph_attr={
                                "fontname": self.DIAGRAM_FONT, 
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "dpi": "150"
                            },
                            node_attr={
                                "fontname": self.DIAGRAM_FONT,
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "width": "1.2",
                                "height": "1.2"
                            }):
                    
                    # Input Sources
                    with Cluster("User Input"):
                        user_text = Client("Text Input\n(Math Problems)")
                        canvas_state = Python("Canvas State\n(SVG)")
                        vision_data = Python("Vision Capture\n(Screenshot)")
                    
                    # Processing Pipeline
                    with Cluster("AI Processing Pipeline"):
                        message_prep = Python("Message\nPreparation")
                        openai_api = Python("OpenAI API\nManager")
                        model_selection = Python("Model Selection\n(GPT-4/Vision)")
                        
                    with Cluster("OpenAI Services"):
                        gpt4 = Slack("GPT-4\n(Text Processing)")
                        gpt4_vision = Slack("GPT-4 Vision\n(Image Analysis)")
                        function_calls = Python("Function Calls\n& Tool Usage")
                    
                    # Response Processing
                    with Cluster("Response Processing"):
                        tool_processor = Python("Tool Call\nProcessor")
                        response_formatter = Python("Response\nFormatter")
                        
                    # Output
                    with Cluster("Output"):
                        ai_response = Python("AI Response\n(Text)")
                        tool_calls = Python("Tool Calls\n(JSON)")
                        canvas_updates = Python("Canvas Updates\n(Drawing Commands)")
                    
                    # Flow connections
                    [user_text, canvas_state, vision_data] >> message_prep
                    message_prep >> openai_api
                    openai_api >> model_selection
                    
                    model_selection >> Edge(label="text") >> gpt4
                    model_selection >> Edge(label="vision") >> gpt4_vision
                    
                    gpt4 >> function_calls
                    gpt4_vision >> function_calls
                    
                    function_calls >> tool_processor
                    tool_processor >> response_formatter
                    
                    response_formatter >> [ai_response, tool_calls, canvas_updates]
                    
                print(f"  ✓ AI integration diagram: {diagram_path}.{fmt}")
                
                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._post_process_svg_fonts(output_dir / f'ai_integration.{fmt}')
            
        except Exception as e:
            print(f"  ✗ AI integration diagram failed: {e}")
    
    def generate_webdriver_flow_diagram(self):
        """Generate WebDriver vision capture workflow diagram."""
        try:
            from diagrams import Diagram, Cluster, Edge
            from diagrams.programming.language import Python
            from diagrams.onprem.client import Client
            from diagrams.onprem.network import Internet
            
            # Generate for each format
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
                diagram_path = output_dir / "webdriver_flow"
                
                with Diagram("MatHud WebDriver Vision Flow", 
                            filename=str(diagram_path), 
                            show=False,
                            direction="TB",
                            outformat=fmt,
                            graph_attr={
                                "fontname": self.DIAGRAM_FONT, 
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "dpi": "150"
                            },
                            node_attr={
                                "fontname": self.DIAGRAM_FONT,
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "width": "1.2",
                                "height": "1.2"
                            }):
                    
                    # Trigger
                    user_request = Client("User Requests\nVision Analysis")
                    
                    with Cluster("WebDriver Initialization"):
                        webdriver_manager = Python("WebDriver\nManager")
                        firefox_setup = Python("Firefox Browser\nSetup")
                        
                    with Cluster("Vision Capture Process"):
                        svg_injection = Python("SVG State\nInjection")
                        page_render = Python("Page Rendering\n& Layout")
                        screenshot = Python("Screenshot\nCapture")
                        image_processing = Python("Image\nProcessing")
                        
                    with Cluster("Integration with AI"):
                        image_encoding = Python("Base64\nEncoding")
                        openai_vision = Python("OpenAI Vision\nAPI Call")
                        vision_analysis = Python("Vision Analysis\nResults")
                        
                    with Cluster("Storage & Cleanup"):
                        snapshot_storage = Python("Canvas\nSnapshot Storage")
                        cleanup = Python("Browser\nCleanup")
                        
                    # Flow
                    user_request >> webdriver_manager
                    webdriver_manager >> firefox_setup
                    firefox_setup >> svg_injection
                    
                    svg_injection >> page_render
                    page_render >> screenshot  
                    screenshot >> image_processing
                    
                    image_processing >> image_encoding
                    image_encoding >> openai_vision
                    openai_vision >> vision_analysis
                    
                    image_processing >> snapshot_storage
                    webdriver_manager >> cleanup
                    
                print(f"  ✓ WebDriver flow diagram: {diagram_path}.{fmt}")
                
                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._post_process_svg_fonts(output_dir / f'webdriver_flow.{fmt}')
            
        except Exception as e:
            print(f"  ✗ WebDriver flow diagram failed: {e}")
    
    def generate_data_flow_diagram(self):
        """Generate end-to-end data flow diagram."""
        try:
            from diagrams import Diagram, Cluster, Edge
            from diagrams.onprem.client import Client
            from diagrams.programming.language import Python
            from diagrams.onprem.database import Mongodb
            from diagrams.onprem.inmemory import Redis
            from diagrams.saas.chat import Slack
            
            # Generate for each format
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
                diagram_path = output_dir / "data_flow"
                
                with Diagram("MatHud Data Flow Pipeline", 
                            filename=str(diagram_path), 
                            show=False,
                            direction="LR",
                            outformat=fmt,
                            graph_attr={
                                "fontname": self.DIAGRAM_FONT, 
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "dpi": "150"
                            },
                            node_attr={
                                "fontname": self.DIAGRAM_FONT,
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "width": "1.2",
                                "height": "1.2"
                            }):
                    
                    # Input Stage
                    with Cluster("Input Stage"):
                        user_input = Client("User Input")
                        canvas_state = Python("Canvas State\n(SVG)")
                        
                    # Processing Stage
                    with Cluster("Processing Stage"):
                        flask_routes = Python("Flask Routes\n(/send_message)")
                        request_validation = Python("Request\nValidation")
                        message_processing = Python("Message\nProcessing")
                        
                    # AI Processing
                    with Cluster("AI Processing"):
                        vision_check = Python("Vision\nRequired?")
                        webdriver_capture = Python("WebDriver\nCapture")
                        openai_request = Python("OpenAI\nRequest")
                        ai_response = Slack("AI Response\nProcessing")
                        
                    # Response Stage
                    with Cluster("Response Processing"):
                        tool_call_processing = Python("Tool Call\nProcessing")
                        response_formatting = Python("Response\nFormatting")
                        logging = Python("Logging")
                        
                    # Storage & Output
                    with Cluster("Storage & Output"):
                        workspace_save = Mongodb("Workspace\nStorage")
                        log_storage = Redis("Log\nStorage")
                        client_response = Client("Client\nResponse")
                        
                    # Data Flow
                    user_input >> flask_routes
                    canvas_state >> flask_routes
                    flask_routes >> request_validation
                    request_validation >> message_processing
                    
                    message_processing >> vision_check
                    vision_check >> Edge(label="yes") >> webdriver_capture
                    vision_check >> Edge(label="no") >> openai_request
                    webdriver_capture >> openai_request
                    
                    openai_request >> ai_response
                    ai_response >> tool_call_processing
                    tool_call_processing >> response_formatting
                    response_formatting >> logging
                    
                    logging >> log_storage
                    response_formatting >> client_response
                    
                    # Workspace operations (separate flow)
                    canvas_state >> workspace_save
                    
                print(f"  ✓ Data flow diagram: {diagram_path}.{fmt}")
                
                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._post_process_svg_fonts(output_dir / f'data_flow.{fmt}')
            
        except Exception as e:
            print(f"  ✗ Data flow diagram failed: {e}")

    def generate_all_architecture_diagrams(self):
        """Generate all architecture diagrams."""
        print("Generating architecture diagrams...")
        
        try:
            self.generate_system_overview_diagram()
            self.generate_ai_integration_diagram() 
            self.generate_webdriver_flow_diagram()
            self.generate_data_flow_diagram()
            
        except ImportError as e:
            print(f"  ✗ Missing diagrams library: {e}")
            print("    Install with: pip install diagrams")
        except Exception as e:
            print(f"  ✗ Error generating architecture diagrams: {e}")


def main():
    """Main function for standalone execution."""
    generator = ArchitectureDiagramGenerator()
    generator.generate_all_architecture_diagrams()


if __name__ == '__main__':
    main()
