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
                                "dpi": "150"
                            },
                            node_attr={
                                "fontname": self.DIAGRAM_FONT,
                                "fontsize": self.DIAGRAM_FONT_SIZE,
                                "width": "1.2",
                                "height": "1.2"
                            }):
                    
                    # User Interface Layer
                    user = Client("User Browser")
                    
                    with Cluster("Frontend Layer (Brython)"):
                        brython = Python("Brython Runtime\n(Python in Browser)")
                        canvas = Python("SVG Canvas System\n& Drawable Managers")
                        ui = Python("Chat Interface\n(Markdown + MathJax)")
                        event_handler = Python("Canvas Event\nHandler")
                    
                    with Cluster("Flask Backend Server"):
                        flask_app = Flask("Flask Application\n(app.py)")
                        app_manager = Python("App Manager\n(Dependency Injection)")
                        routes = Python("API Routes\n(/send_message, /save_workspace)")
                    
                    with Cluster("Core Service Managers"):
                        ai_service = Python("OpenAI API\nManager")
                        webdriver_service = Python("WebDriver Manager\n(Selenium + Firefox)")
                        workspace_service = Python("Workspace Manager\n(Server-side)")
                        log_service = Python("Log Manager\n(Session-based)")
                        tool_processor = Python("Tool Call\nProcessor")
                    
                    with Cluster("Client-Side Managers (Brython)"):
                        drawable_manager = Python("Drawable Manager\n(Central Orchestrator)")
                        shape_managers = Python("Shape Managers\n(Point, Segment, etc.)")
                        undo_manager = Python("Undo/Redo\nManager")
                        workspace_client = Python("Workspace Manager\n(Client-side)")
                    
                    with Cluster("External Services"):
                        openai_api = Slack("OpenAI API\n(GPT-4, Vision)")
                        selenium = Python("Firefox Browser\n(Headless)")
                        internet = Internet("Internet")
                    
                    with Cluster("Data & Storage"):
                        workspaces = Mongodb("JSON Workspaces\n(File System)")
                        logs = Redis("Application Logs\n(Timestamped)")
                        snapshots = Python("Canvas Snapshots\n(PNG Images)")
                        state_data = Python("Canvas State\n(SVG + Metadata)")
                    
                    with Cluster("Testing Systems"):
                        server_tests = Python("Server Tests\n(pytest)")
                        client_tests = Python("Client Tests\n(Brython unittest)")
                    
                    # User to Frontend connections
                    user >> [brython, ui, event_handler]
                    brython >> canvas
                    ui >> brython
                    event_handler >> canvas
                    
                    # Frontend to Backend
                    [brython, ui] >> flask_app
                    flask_app >> app_manager
                    app_manager >> routes
                    
                    # Backend to Services  
                    routes >> [ai_service, webdriver_service, workspace_service, log_service]
                    ai_service >> tool_processor
                    
                    # Client-side manager coordination
                    canvas >> drawable_manager
                    drawable_manager >> shape_managers
                    drawable_manager >> undo_manager
                    brython >> workspace_client
                    
                    # Services to External
                    ai_service >> openai_api
                    webdriver_service >> selenium
                    openai_api >> internet
                    selenium >> internet
                    
                    # Data flows
                    workspace_service >> workspaces
                    workspace_client >> workspace_service
                    log_service >> logs
                    webdriver_service >> snapshots
                    canvas >> state_data
                    
                    # Testing connections
                    flask_app >> server_tests
                    brython >> client_tests
                    
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
                
                with Diagram("MatHud AI Integration & Function Call Flow", 
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
                        user_text = Client("User Message\n(Math Problems)")
                        canvas_state = Python("Canvas State\n(SVG + Computations)")
                        vision_toggle = Python("Vision Toggle\n(UI Control)")
                    
                    # Client Processing
                    with Cluster("Client-Side Processing (Brython)"):
                        ai_interface = Python("AI Interface\n(Main Controller)")
                        function_registry = Python("Function Registry\n(42 Available Functions)")
                        markdown_parser = Python("Markdown Parser\n+ MathJax Renderer")
                        
                    # Backend Processing Pipeline
                    with Cluster("Backend AI Pipeline"):
                        flask_route = Python("Flask Route\n(/send_message)")
                        message_prep = Python("Message Preparation\n+ Context Assembly")
                        model_selection = Python("AI Model Selection\n(GPT-4, GPT-4o, etc.)")
                        
                    with Cluster("OpenAI Services"):
                        gpt4 = Slack("GPT-4\n(Text + Function Calls)")
                        gpt4_vision = Slack("GPT-4 Vision\n(Image Analysis)")
                        function_calls = Python("Function Definitions\n(42 Tools Available)")
                    
                    # Function Execution (Client-Side)
                    with Cluster("Function Call Execution (Brython)"):
                        process_calls = Python("ProcessFunctionCalls\n(Facade Pattern)")
                        result_processor = Python("Result Processor\n(Execution Engine)")
                        expression_evaluator = Python("Expression Evaluator\n(Math Operations)")
                        drawable_operations = Python("Drawable Operations\n(Canvas Manipulation)")
                        
                    # Response Processing
                    with Cluster("Response & UI Update"):
                        tool_results = Python("Tool Call Results\n(JSON)")
                        ui_update = Python("Chat UI Update\n(Markdown Rendering)")
                        canvas_update = Python("Canvas Update\n(SVG Manipulation)")
                    
                    # Flow connections
                    [user_text, canvas_state, vision_toggle] >> ai_interface
                    ai_interface >> flask_route
                    flask_route >> message_prep
                    message_prep >> model_selection
                    
                    model_selection >> Edge(label="text") >> gpt4
                    model_selection >> Edge(label="vision") >> gpt4_vision
                    
                    gpt4 >> function_calls
                    gpt4_vision >> function_calls
                    
                    function_calls >> process_calls
                    process_calls >> [result_processor, expression_evaluator, drawable_operations]
                    
                    [result_processor, expression_evaluator, drawable_operations] >> tool_results
                    tool_results >> Edge(label="back to AI") >> flask_route
                    tool_results >> [ui_update, canvas_update]
                    
                    # UI rendering
                    ui_update >> markdown_parser
                    function_registry >> process_calls
                    
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
                
                with Diagram("MatHud Vision System & WebDriver Flow", 
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
                    with Cluster("Vision Request Trigger"):
                        user_request = Client("User Enables Vision\n+ Sends Message")
                        vision_check = Python("Vision Toggle\nStatus Check")
                        canvas_state = Python("Current Canvas State\n(SVG Content)")
                    
                    with Cluster("Client-Side State Capture"):
                        svg_extraction = Python("SVG State Extraction\n(Content + Dimensions)")
                        ajax_request = Python("AJAX Request\nto /send_message")
                        
                    with Cluster("Backend WebDriver Management"):
                        route_handler = Python("Flask Route Handler\n(/send_message)")
                        webdriver_init = Python("WebDriver Manager\nInitialization")
                        firefox_setup = Python("Headless Firefox\nLaunch + Configuration")
                        
                    with Cluster("Vision Capture Process"):
                        svg_injection = Python("SVG State Injection\ninto Browser DOM")
                        page_setup = Python("Page Layout\n+ Style Application")
                        screenshot = Python("Canvas Screenshot\nCapture (PNG)")
                        image_save = Python("Save to CanvasSnapshots/\ncanvas.png")
                        
                    with Cluster("AI Vision Integration"):
                        image_encoding = Python("Base64 Image\nEncoding")
                        openai_payload = Python("Multimodal Payload\nPreparation")
                        vision_analysis = Python("OpenAI Vision API\nImage Analysis")
                        contextual_response = Python("AI Response with\nVisual Context")
                        
                    with Cluster("Cleanup & Error Handling"):
                        browser_cleanup = Python("WebDriver Browser\nCleanup")
                        error_fallback = Python("Vision Failure\nFallback (Text-only)")
                        
                    # Flow
                    user_request >> vision_check
                    vision_check >> Edge(label="enabled") >> canvas_state
                    canvas_state >> svg_extraction
                    svg_extraction >> ajax_request
                    
                    ajax_request >> route_handler
                    route_handler >> webdriver_init
                    webdriver_init >> firefox_setup
                    
                    firefox_setup >> svg_injection
                    svg_injection >> page_setup
                    page_setup >> screenshot
                    screenshot >> image_save
                    
                    image_save >> image_encoding
                    image_encoding >> openai_payload
                    openai_payload >> vision_analysis
                    vision_analysis >> contextual_response
                    
                    webdriver_init >> browser_cleanup
                    route_handler >> Edge(label="on error") >> error_fallback
                    
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

    def generate_manager_architecture_diagram(self):
        """Generate detailed manager pattern architecture diagram."""
        try:
            from diagrams import Diagram, Cluster, Edge
            from diagrams.programming.language import Python
            from diagrams.onprem.client import Client
            
            # Generate for each format
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
                diagram_path = output_dir / "manager_architecture"
                
                with Diagram("MatHud Manager Pattern Architecture", 
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
                    
                    # Central Coordinator
                    with Cluster("Central Canvas System"):
                        canvas = Python("Canvas\n(SVG Manipulation)")
                        drawable_manager = Python("DrawableManager\n(Central Orchestrator)")
                        drawables_container = Python("DrawablesContainer\n(Object Storage)")
                    
                    # Specialized Shape Managers
                    with Cluster("Specialized Shape Managers"):
                        point_mgr = Python("PointManager")
                        segment_mgr = Python("SegmentManager")
                        vector_mgr = Python("VectorManager")
                        triangle_mgr = Python("TriangleManager")
                        rectangle_mgr = Python("RectangleManager")
                        circle_mgr = Python("CircleManager")
                        ellipse_mgr = Python("EllipseManager")
                        angle_mgr = Python("AngleManager")
                        function_mgr = Python("FunctionManager")
                        area_mgr = Python("ColoredAreaManager")
                    
                    # Supporting Managers
                    with Cluster("Supporting Management Systems"):
                        undo_redo = Python("UndoRedo\n(State History)")
                        dependency_mgr = Python("DrawableDependency\n(Object Dependencies)")
                        transform_mgr = Python("Transformations\n(Geometric Operations)")
                        coord_system = Python("CartesianSystem2Axis\n(Coordinate Management)")
                    
                    # Drawable Objects
                    with Cluster("Drawable Object Classes"):
                        point_obj = Python("Point")
                        segment_obj = Python("Segment")
                        vector_obj = Python("Vector")
                        triangle_obj = Python("Triangle")
                        rectangle_obj = Python("Rectangle")
                        circle_obj = Python("Circle")
                        ellipse_obj = Python("Ellipse")
                        angle_obj = Python("Angle")
                        function_obj = Python("Function")
                        colored_area_obj = Python("ColoredArea")
                    
                    # Name Generation System
                    with Cluster("Name Generation System"):
                        name_generator = Python("Name Generator\n(Base Class)")
                        drawable_names = Python("DrawableNameGenerator")
                        point_names = Python("PointNameGenerator")
                        function_names = Python("FunctionNameGenerator")
                    
                    # Connections - Canvas to DrawableManager
                    canvas >> drawable_manager
                    drawable_manager >> drawables_container
                    drawable_manager >> [undo_redo, dependency_mgr, transform_mgr]
                    canvas >> coord_system
                    
                    # DrawableManager coordinates all shape managers
                    drawable_manager >> [point_mgr, segment_mgr, vector_mgr, triangle_mgr, rectangle_mgr]
                    drawable_manager >> [circle_mgr, ellipse_mgr, angle_mgr, function_mgr, area_mgr]
                    
                    # Shape managers create and manage their respective objects
                    point_mgr >> point_obj
                    segment_mgr >> segment_obj
                    vector_mgr >> vector_obj
                    triangle_mgr >> triangle_obj
                    rectangle_mgr >> rectangle_obj
                    circle_mgr >> circle_obj
                    ellipse_mgr >> ellipse_obj
                    angle_mgr >> angle_obj
                    function_mgr >> function_obj
                    area_mgr >> colored_area_obj
                    
                    # Name generation
                    [point_mgr, segment_mgr, vector_mgr, triangle_mgr, rectangle_mgr] >> name_generator
                    [circle_mgr, ellipse_mgr, angle_mgr, function_mgr, area_mgr] >> name_generator
                    name_generator >> [drawable_names, point_names, function_names]
                    
                    # Dependencies
                    dependency_mgr >> [point_obj, segment_obj, vector_obj, triangle_obj, rectangle_obj]
                    dependency_mgr >> [circle_obj, ellipse_obj, angle_obj, function_obj, colored_area_obj]
                    
                print(f"  ✓ Manager architecture diagram: {diagram_path}.{fmt}")
                
                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._post_process_svg_fonts(output_dir / f'manager_architecture.{fmt}')
            
        except Exception as e:
            print(f"  ✗ Manager architecture diagram failed: {e}")

    def generate_all_architecture_diagrams(self):
        """Generate all architecture diagrams."""
        print("Generating architecture diagrams...")
        
        try:
            self.generate_system_overview_diagram()
            self.generate_ai_integration_diagram() 
            self.generate_webdriver_flow_diagram()
            self.generate_data_flow_diagram()
            self.generate_manager_architecture_diagram()  # New detailed manager diagram
            
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
