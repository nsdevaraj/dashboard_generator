import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from src.config import config
from src.api_key_manager import APIKeyManager
from src.csv_processor import CSVDataProcessor
from src.dashboard_generator import DashboardDesignGenerator
from src.visualization_components import VisualizationComponents
from src.interactive_components import InteractiveComponents, ResponsiveLayouts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardGeneratorUI:
    """
    Main user interface for the Dashboard Generator application.
    Provides CSV upload, OpenAI API configuration, and dashboard generation.
    """
    
    def __init__(self):
        """Initialize the Dashboard Generator UI."""
        # Initialize components
        self.api_key_manager = APIKeyManager()
        self.csv_processor = CSVDataProcessor()
        self.dashboard_generator = DashboardDesignGenerator(csv_processor=self.csv_processor)
        self.viz_components = VisualizationComponents()
        self.interactive = InteractiveComponents()
        self.responsive = ResponsiveLayouts()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.app.title = "Dashboard Design Generator"
        
        # Set app instance for interactive components
        self.interactive.set_app(self.app)
        
        # Set up app layout
        self.app.layout = self.create_layout()
        
        # Register callbacks
        self.register_callbacks()
    
    def create_layout(self) -> html.Div:
        """
        Create the main application layout.
        
        Returns:
            dash.html.Div: Main application layout.
        """
        # Create navigation bar
        navbar = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Dashboard Design Generator", className="ms-2")),
                ], align="center"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Home", href="#")),
                        dbc.NavItem(dbc.NavLink("Documentation", href="#")),
                        dbc.NavItem(dbc.NavLink("About", href="#")),
                    ], className="ms-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]),
            color="primary",
            dark=True,
            className="mb-4",
        )
        
        # Create API key configuration section
        api_key_section = dbc.Card([
            dbc.CardHeader("OpenAI API Configuration"),
            dbc.CardBody([
                dbc.Form([
                    dbc.Label("API Key"),
                    dbc.Input(
                        id="api-key-input",
                        type="password",
                        placeholder="Enter your OpenAI API key",
                    ),
                    dbc.FormText(
                        "Your API key is stored locally and never sent to our servers."
                    ),
                ]),
                dbc.Button(
                    "Save API Key",
                    id="save-api-key-button",
                    color="primary",
                    className="mt-3",
                ),
                html.Div(id="api-key-status", className="mt-2"),
            ]),
        ], className="mb-4")
        
        # Create CSV upload section
        csv_upload_section = dbc.Card([
            dbc.CardHeader("Upload CSV Data"),
            dbc.CardBody([
                dcc.Upload(
                    id="upload-csv",
                    children=html.Div([
                        "Drag and Drop or ",
                        html.A("Select a CSV File")
                    ]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px 0"
                    },
                    multiple=False
                ),
                html.Div(id="upload-status", className="mt-2"),
                html.Div(id="csv-preview", className="mt-3"),
            ]),
        ], className="mb-4")
        
        # Create dashboard requirements section
        dashboard_requirements_section = dbc.Card([
            dbc.CardHeader("Dashboard Requirements"),
            dbc.CardBody([
                dbc.Form([
                    dbc.Label("Dashboard Title"),
                    dbc.Input(
                        id="dashboard-title-input",
                        type="text",
                        placeholder="Enter a title for your dashboard",
                    ),
                ]),
                dbc.Form([
                    dbc.Label("Dashboard Purpose"),
                    dbc.Textarea(
                        id="dashboard-purpose-input",
                        placeholder="Describe the purpose of your dashboard",
                        style={"height": "100px"},
                    ),
                ]),
                dbc.Form([
                    dbc.Label("Key Performance Indicators (KPIs)"),
                    dbc.Textarea(
                        id="dashboard-kpis-input",
                        placeholder="Enter KPIs to focus on (one per line)",
                        style={"height": "100px"},
                    ),
                    dbc.FormText(
                        "Enter each KPI on a new line. Example: Sales Growth, Customer Retention, etc."
                    ),
                ]),
                dbc.Form([
                    dbc.Label("Number of Dashboards"),
                    dbc.Input(
                        id="num-dashboards-input",
                        type="number",
                        min=1,
                        max=5,
                        step=1,
                        value=1,
                    ),
                    dbc.FormText(
                        "How many distinct dashboards would you like to generate?"
                    ),
                ]),
                dbc.Form([
                    dbc.Label("Additional Requirements"),
                    dbc.Textarea(
                        id="additional-requirements-input",
                        placeholder="Any additional requirements or preferences",
                        style={"height": "100px"},
                    ),
                ]),
                dbc.Button(
                    "Generate Dashboard Designs",
                    id="generate-button",
                    color="success",
                    className="mt-3",
                ),
                html.Div(id="generation-status", className="mt-2"),
            ]),
        ], className="mb-4")
        
        # Create dashboard preview section
        dashboard_preview_section = dbc.Card([
            dbc.CardHeader("Dashboard Designs"),
            dbc.CardBody([
                html.Div(id="dashboard-designs-container"),
                html.Div(id="dashboard-preview-container"),
                dbc.Button(
                    "Export Dashboard Designs",
                    id="export-button",
                    color="info",
                    className="mt-3",
                    style={"display": "none"},
                ),
                dcc.Download(id="download-designs"),
            ]),
        ], className="mb-4")
        
        # Create footer
        footer = html.Footer([
            html.Hr(),
            html.P(
                "Dashboard Design Generator © 2025 | Powered by OpenAI",
                className="text-center text-muted",
            ),
        ])
        
        # Combine all sections into main layout
        return html.Div([
            navbar,
            dbc.Container([
                html.H1("Generate Dashboard Designs with AI", className="text-center mb-4"),
                html.P(
                    "Upload your CSV data and let AI design beautiful, insightful dashboards tailored to your needs.",
                    className="text-center mb-4",
                ),
                dbc.Row([
                    dbc.Col([
                        api_key_section,
                        csv_upload_section,
                    ], md=6),
                    dbc.Col([
                        dashboard_requirements_section,
                    ], md=6),
                ]),
                dashboard_preview_section,
                footer,
            ], fluid=True),
        ])
    
    def register_callbacks(self):
        """Register callbacks for the Dash application."""
        # Register callback for API key saving
        @self.app.callback(
            Output("api-key-status", "children"),
            Input("save-api-key-button", "n_clicks"),
            State("api-key-input", "value"),
            prevent_initial_call=True
        )
        def save_api_key(n_clicks, api_key):
            if api_key is None or api_key.strip() == "":
                return html.Div("Please enter an API key.", className="text-danger")
            
            # Save API key
            success = self.api_key_manager.save_api_key(api_key)
            
            if success:
                # Initialize OpenAI client
                self.dashboard_generator.initialize()
                return html.Div("API key saved successfully!", className="text-success")
            else:
                return html.Div("Failed to save API key.", className="text-danger")
        
        # Register callback for CSV upload
        @self.app.callback(
            [
                Output("upload-status", "children"),
                Output("csv-preview", "children"),
            ],
            Input("upload-csv", "contents"),
            State("upload-csv", "filename"),
            prevent_initial_call=True
        )
        def process_csv_upload(contents, filename):
            if contents is None:
                return html.Div("No file uploaded."), None
            
            if not filename.endswith('.csv'):
                return html.Div("Please upload a CSV file.", className="text-danger"), None
            
            # Decode and save the uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            try:
                # Save the file temporarily
                temp_file_path = os.path.join(os.getcwd(), "temp_upload.csv")
                with open(temp_file_path, 'wb') as f:
                    f.write(decoded)
                
                # Load the CSV data
                success = self.csv_processor.load_csv(temp_file_path)
                
                if not success:
                    return html.Div("Failed to load CSV data.", className="text-danger"), None
                
                # Clean the data
                self.csv_processor.clean_data()
                
                # Analyze the data
                data_summary = self.csv_processor.analyze_data()
                
                if not data_summary:
                    return html.Div("Failed to analyze data.", className="text-danger"), None
                
                # Create a preview table
                df_preview = self.csv_processor.df.head(5)
                
                preview_table = dbc.Table.from_dataframe(
                    df_preview,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mt-3"
                )
                
                # Create summary cards
                summary_cards = []
                
                # Row count card
                row_count = data_summary.get("row_count", 0)
                summary_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Rows", className="card-title"),
                            html.P(f"{row_count:,}", className="card-text"),
                        ])
                    ], className="text-center")
                )
                
                # Column count card
                col_count = data_summary.get("column_count", 0)
                summary_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Columns", className="card-title"),
                            html.P(f"{col_count:,}", className="card-text"),
                        ])
                    ], className="text-center")
                )
                
                # Data types card
                data_types = data_summary.get("data_types", {})
                type_counts = {}
                for dtype in data_types.values():
                    type_counts[dtype] = type_counts.get(dtype, 0) + 1
                
                type_text = ", ".join([f"{count} {dtype}" for dtype, count in type_counts.items()])
                summary_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Data Types", className="card-title"),
                            html.P(type_text, className="card-text"),
                        ])
                    ], className="text-center")
                )
                
                # Create summary row
                summary_row = dbc.Row([
                    dbc.Col(card, width=4) for card in summary_cards
                ], className="mb-4")
                
                # Combine preview and summary
                preview_content = html.Div([
                    html.H5(f"Preview of {filename}"),
                    summary_row,
                    html.Div(
                        preview_table,
                        style={"maxHeight": "300px", "overflow": "auto"}
                    ),
                ])
                
                return html.Div(f"File '{filename}' uploaded successfully!", className="text-success"), preview_content
                
            except Exception as e:
                logger.error(f"Error processing CSV upload: {e}")
                return html.Div(f"Error processing file: {str(e)}", className="text-danger"), None
        
        # Register callback for dashboard generation
        @self.app.callback(
            [
                Output("generation-status", "children"),
                Output("dashboard-designs-container", "children"),
                Output("export-button", "style"),
            ],
            Input("generate-button", "n_clicks"),
            [
                State("dashboard-title-input", "value"),
                State("dashboard-purpose-input", "value"),
                State("dashboard-kpis-input", "value"),
                State("num-dashboards-input", "value"),
                State("additional-requirements-input", "value"),
            ],
            prevent_initial_call=True
        )
        def generate_dashboard_designs(n_clicks, title, purpose, kpis, num_dashboards, additional_requirements):
            if not self.api_key_manager.is_api_key_set():
                return html.Div("Please set your OpenAI API key first.", className="text-danger"), None, {"display": "none"}
            
            if not hasattr(self.csv_processor, 'df') or self.csv_processor.df is None:
                return html.Div("Please upload a CSV file first.", className="text-danger"), None, {"display": "none"}
            
            # Prepare requirements
            kpi_list = []
            if kpis:
                kpi_list = [kpi.strip() for kpi in kpis.split('\n') if kpi.strip()]
            
            requirements = {
                "title": title if title else "Dashboard",
                "purpose": purpose if purpose else "Visualize data insights",
                "kpis": kpi_list,
                "num_dashboards": num_dashboards if num_dashboards else 1,
                "additional_requirements": additional_requirements if additional_requirements else ""
            }
            
            try:
                # Ensure data is analyzed before generating designs
                if not self.csv_processor.data_summary:
                    logger.info("Analyzing data before generating designs...")
                    # First clean the data
                    self.csv_processor.clean_data()
                    # Then analyze the data
                    data_summary = self.csv_processor.analyze_data()
                    
                    if not data_summary:
                        return html.Div("Failed to analyze data. Please try uploading the CSV file again.", className="text-danger"), None, {"display": "none"}
                    
                    # Store the data summary
                    self.csv_processor.data_summary = data_summary
                
                # Generate dashboard designs
                dashboard_designs = self.dashboard_generator.generate_dashboard_designs(requirements)
                
                if not dashboard_designs:
                    return html.Div("Failed to generate dashboard designs.", className="text-danger"), None, {"display": "none"}
                
                # Save designs to file
                designs_file_path = os.path.join(os.getcwd(), "dashboard_designs.json")
                self.dashboard_generator.save_dashboard_designs(designs_file_path)
                
                # Create design cards
                design_cards = []
                
                for i, dashboard in enumerate(dashboard_designs.get("dashboards", [])):
                    dashboard_title = dashboard.get("title", f"Dashboard {i+1}")
                    dashboard_purpose = dashboard.get("purpose", "")
                    visualizations = dashboard.get("visualizations", [])
                    
                    # Create visualization list
                    viz_items = []
                    for j, viz in enumerate(visualizations[:5]):  # Limit to first 5 visualizations
                        viz_title = viz.get("title", f"Visualization {j+1}")
                        chart_type = viz.get("chart_type", "")
                        rationale = viz.get("rationale", "")
                        
                        viz_items.append(
                            dbc.ListGroupItem([
                                html.H6(viz_title),
                                html.P(f"Chart Type: {chart_type.capitalize()}"),
                                html.Small(rationale, className="text-muted"),
                            ])
                        )
                    
                    if len(visualizations) > 5:
                        viz_items.append(
                            dbc.ListGroupItem(f"+ {len(visualizations) - 5} more visualizations")
                        )
                    
                    # Create design card
                    design_cards.append(
                        dbc.Card([
                            dbc.CardHeader(dashboard_title),
                            dbc.CardBody([
                                html.P(dashboard_purpose, className="card-text"),
                                html.H6("Visualizations:"),
                                dbc.ListGroup(viz_items, className="mb-3"),
                                dbc.Button(
                                    "Preview Dashboard",
                                    id=f"preview-button-{i}",
                                    color="primary",
                                    className="mr-2",
                                ),
                                dbc.Button(
                                    "Generate Code",
                                    id=f"code-button-{i}",
                                    color="secondary",
                                ),
                            ]),
                        ], className="mb-4")
                    )
                
                # Create design container
                designs_container = html.Div([
                    html.H4("Generated Dashboard Designs"),
                    html.Div(design_cards),
                ])
                
                return html.Div("Dashboard designs generated successfully!", className="text-success"), designs_container, {"display": "block"}
                
            except Exception as e:
                logger.error(f"Error generating dashboard designs: {e}")
                return html.Div(f"Error generating designs: {str(e)}", className="text-danger"), None, {"display": "none"}
        
        # Register callback for design export
        @self.app.callback(
            Output("download-designs", "data"),
            Input("export-button", "n_clicks"),
            prevent_initial_call=True
        )
        def export_designs(n_clicks):
            if not hasattr(self.dashboard_generator, 'dashboard_designs') or self.dashboard_generator.dashboard_designs is None:
                return None
            
            # Export designs as JSON
            return dcc.send_file(
                os.path.join(os.getcwd(), "dashboard_designs.json"),
                filename="dashboard_designs.json"
            )
        
        # Register single callback for all preview and code generation buttons
        @self.app.callback(
            Output("dashboard-preview-container", "children"),
            [
                Input(f"preview-button-{i}", "n_clicks") for i in range(10)  # Increased range to handle more dashboards
            ] + [
                Input(f"code-button-{i}", "n_clicks") for i in range(10)  # Increased range to handle more dashboards
            ] + [
                Input("close-preview-button", "n_clicks"),
                Input("close-code-button", "n_clicks")
            ],
            prevent_initial_call=True
        )
        def handle_preview_and_code(*args):
            ctx = dash.callback_context
            if not ctx.triggered:
                return None
            
            # Get the button ID that triggered the callback
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle close buttons
            if button_id == "close-preview-button" or button_id == "close-code-button":
                return None
            
            # Extract the index from the button ID
            if button_id.startswith("preview-button-") or button_id.startswith("code-button-"):
                i = int(button_id.split('-')[-1])
                is_preview = button_id.startswith("preview-button-")
                
                if not hasattr(self.dashboard_generator, 'dashboard_designs') or self.dashboard_generator.dashboard_designs is None:
                    return html.Div("No dashboard designs available.")
                
                dashboards = self.dashboard_generator.dashboard_designs.get("dashboards", [])
                if i >= len(dashboards):
                    return html.Div("Dashboard not found.")
                
                dashboard = dashboards[i]
                dashboard_title = dashboard.get("title", f"Dashboard {i+1}")
                dashboard_purpose = dashboard.get("purpose", "")
                visualizations = dashboard.get("visualizations", [])
                
                if is_preview:
                    # Create mock visualizations for preview
                    viz_components = []
                    for j, viz in enumerate(visualizations):
                        viz_title = viz.get("title", f"Visualization {j+1}")
                        chart_type = viz.get("chart_type", "bar")
                        data_columns = viz.get("data_columns", [])
                        
                        # Create a sample figure based on chart type
                        fig = None
                        if chart_type == "bar":
                            if len(data_columns) >= 2:
                                fig = self.viz_components.create_bar_chart(
                                    self.csv_processor.df, 
                                    data_columns[0], 
                                    data_columns[1],
                                    title=viz_title
                                )
                        elif chart_type == "line":
                            if len(data_columns) >= 2:
                                fig = self.viz_components.create_line_chart(
                                    self.csv_processor.df, 
                                    data_columns[0], 
                                    data_columns[1],
                                    title=viz_title
                                )
                        elif chart_type == "pie":
                            if len(data_columns) >= 2:
                                fig = self.viz_components.create_pie_chart(
                                    self.csv_processor.df, 
                                    data_columns[0], 
                                    data_columns[1],
                                    title=viz_title
                                )
                        elif chart_type == "scatter":
                            if len(data_columns) >= 2:
                                fig = self.viz_components.create_scatter_plot(
                                    self.csv_processor.df, 
                                    data_columns[0], 
                                    data_columns[1],
                                    title=viz_title
                                )
                        
                        if fig:
                            viz_components.append(
                                html.Div([
                                    html.H5(viz_title),
                                    dcc.Graph(figure=fig),
                                ], className="visualization-container")
                            )
                    
                    # Create preview
                    return html.Div([
                        html.H4(f"Preview: {dashboard_title}"),
                        html.P(dashboard_purpose, className="mb-4"),
                        self.responsive.create_grid_layout(viz_components),
                        dbc.Button(
                            "Close Preview",
                            id="close-preview-button",
                            color="secondary",
                            className="mt-3",
                        ),
                    ])
                else:
                    # Generate code
                    code = self.dashboard_generator.generate_layout_code(i, framework='dash')
                    
                    if not code:
                        return html.Div("Failed to generate code.")
                    
                    # Save code to file
                    code_file_path = os.path.join(os.getcwd(), f"dashboard_{i+1}_code.py")
                    with open(code_file_path, 'w') as f:
                        f.write(code)
                    
                    # Create code preview
                    return html.Div([
                        html.H4(f"Generated Code for Dashboard {i+1}"),
                        html.Div([
                            dcc.Textarea(
                                value=code,
                                style={
                                    "width": "100%",
                                    "height": "500px",
                                    "fontFamily": "monospace",
                                    "fontSize": "12px",
                                    "whiteSpace": "pre",
                                    "overflowX": "auto",
                                },
                                readOnly=True,
                            ),
                        ], className="code-container"),
                        dbc.Button(
                            "Download Code",
                            id="download-code-button",
                            color="primary",
                            className="mt-3 mr-2",
                        ),
                        dbc.Button(
                            "Close Code",
                            id="close-code-button",
                            color="secondary",
                            className="mt-3",
                        ),
                        dcc.Download(id="download-code"),
                    ])
            
            return None
        
        # Register callback for code download
        @self.app.callback(
            Output("download-code", "data"),
            Input("download-code-button", "n_clicks"),
            prevent_initial_call=True
        )
        def download_code(n_clicks):
            if n_clicks is None:
                return None
            # Find the most recently generated code file
            code_files = [f for f in os.listdir(os.getcwd()) if f.startswith("dashboard_") and f.endswith("_code.py")]
            if not code_files:
                return None
            latest_file = max(code_files, key=os.path.getctime)
            return dcc.send_file(
                os.path.join(os.getcwd(), latest_file),
                filename=latest_file
            )
    
    def run_server(self, debug=False, port=8050, host="0.0.0.0"):
        """
        Run the Dash application server.
        
        Args:
            debug (bool): Whether to run in debug mode.
            port (int): Port to run the server on.
            host (str): Host to run the server on.
        """
        self.app.run_server(debug=debug, port=port, host=host)


if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Create UI and run server
    ui = DashboardGeneratorUI()
    ui.run_server(debug=True)
