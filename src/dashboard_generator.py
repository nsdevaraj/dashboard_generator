import os
import json
import logging
from typing import Dict, List, Any, Optional

from src.openai_client import OpenAIClient
from src.prompt_engineering import PromptEngineering
from src.csv_processor import CSVDataProcessor
from src.data_transformer import DataTransformer
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardDesignGenerator:
    """
    Generates dashboard designs based on CSV data analysis and OpenAI API.
    Combines data processing with AI-powered design generation.
    """
    
    def __init__(self, csv_processor=None):
        """
        Initialize the dashboard design generator.
        
        Args:
            csv_processor (CSVDataProcessor, optional): CSV processor instance to use.
        """
        self.openai_client = OpenAIClient()
        self.prompt_engineering = PromptEngineering()
        self.csv_processor = csv_processor or CSVDataProcessor()
        self.data_transformer = DataTransformer()
        self.dashboard_designs = None
        
    def initialize(self) -> bool:
        """
        Initialize the dashboard design generator components.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Initialize OpenAI client
            if not self.openai_client.initialize():
                logger.error("Failed to initialize OpenAI client")
                return False
                
            logger.info("Dashboard design generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing dashboard design generator: {e}")
            return False
    
    def load_csv_data(self, file_path: str) -> bool:
        """
        Load CSV data from a file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            # Load CSV data
            if not self.csv_processor.load_csv(file_path):
                logger.error(f"Failed to load CSV data from {file_path}")
                return False
                
            # Validate data
            validation_results = self.csv_processor.validate_data()
            if not validation_results["valid"]:
                logger.error(f"CSV data validation failed: {validation_results['issues']}")
                return False
                
            # Clean data
            if not self.csv_processor.clean_data():
                logger.error("Failed to clean CSV data")
                return False
                
            # Set the cleaned dataframe for the data transformer
            self.data_transformer.set_dataframe(self.csv_processor.df)
                
            logger.info(f"CSV data loaded and processed successfully from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return False
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Analyze the loaded CSV data.
        
        Returns:
            dict: Data analysis summary.
        """
        try:
            # Analyze data
            data_summary = self.csv_processor.analyze_data()
            if "error" in data_summary:
                logger.error(f"Failed to analyze data: {data_summary['error']}")
                return None
                
            logger.info("Data analysis completed successfully")
            return data_summary
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return None
    
    def generate_dashboard_designs(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dashboard designs based on data analysis and requirements.
        
        Args:
            requirements (dict): Requirements for the dashboard design.
            
        Returns:
            dict: Generated dashboard designs.
        """
        try:
            # Analyze data
            data_summary = self.analyze_data()
            if not data_summary:
                logger.error("Failed to generate dashboard designs: No data analysis available")
                return None
                
            # Get visualization recommendations
            viz_recommendations = self.csv_processor.get_visualization_recommendations()
            if "error" in viz_recommendations:
                logger.error(f"Failed to get visualization recommendations: {viz_recommendations['error']}")
                return None
                
            # Combine data summary and visualization recommendations
            data_analysis = {
                **data_summary,
                "visualization_recommendations": viz_recommendations
            }
                
            # Create prompt for dashboard design generation
            prompt = self.prompt_engineering.create_dashboard_design_prompt(data_analysis, requirements)
            
            # Generate dashboard designs using OpenAI API
            response = self.openai_client.generate_dashboard_design(data_analysis, requirements)
            if not response:
                logger.error("Failed to generate dashboard designs: No response from OpenAI API")
                return None
                
            # Parse the response
            dashboard_designs = self.prompt_engineering.parse_dashboard_design_response(response.get("raw_content", ""))
            if not dashboard_designs:
                logger.error("Failed to parse dashboard design response")
                return None
                
            # Store the generated designs
            self.dashboard_designs = dashboard_designs
                
            logger.info("Dashboard designs generated successfully")
            return dashboard_designs
            
        except Exception as e:
            logger.error(f"Error generating dashboard designs: {e}")
            return None
    
    def save_dashboard_designs(self, file_path: str) -> bool:
        """
        Save the generated dashboard designs to a file.
        
        Args:
            file_path (str): Path to save the dashboard designs.
            
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            if not self.dashboard_designs:
                logger.error("No dashboard designs to save")
                return False
                
            with open(file_path, 'w') as f:
                json.dump(self.dashboard_designs, f, indent=2)
                
            logger.info(f"Dashboard designs saved successfully to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving dashboard designs: {e}")
            return False
    
    def generate_layout_code(self, dashboard_index: int = 0, 
                           framework: str = 'dash') -> str:
        """
        Generate code for the dashboard layout based on the design.
        
        Args:
            dashboard_index (int): Index of the dashboard to generate code for.
            framework (str): Framework to generate code for ('dash', 'plotly', etc.).
            
        Returns:
            str: Generated code for the dashboard layout.
        """
        try:
            if not self.dashboard_designs or "dashboards" not in self.dashboard_designs:
                logger.error("No dashboard designs available")
                return None
                
            if dashboard_index >= len(self.dashboard_designs["dashboards"]):
                logger.error(f"Dashboard index {dashboard_index} out of range")
                return None
                
            dashboard = self.dashboard_designs["dashboards"][dashboard_index]
            
            # Generate code based on framework
            if framework == 'dash':
                return self._generate_dash_layout_code(dashboard)
            elif framework == 'plotly':
                return self._generate_plotly_layout_code(dashboard)
            else:
                logger.error(f"Unsupported framework: {framework}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating layout code: {e}")
            return None
    
    def _generate_dash_layout_code(self, dashboard: Dict[str, Any]) -> str:
        """
        Generate Dash layout code for a dashboard design.
        
        Args:
            dashboard (dict): Dashboard design.
            
        Returns:
            str: Generated Dash layout code.
        """
        try:
            title = dashboard.get("title", "Dashboard")
            visualizations = dashboard.get("visualizations", [])
            
            # Generate imports
            code = """import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__, title="{title}")

# Load and process data
# TODO: Replace with actual data loading code
df = pd.read_csv('your_data.csv')

# Create visualizations
""".format(title=title)
            
            # Generate visualization code
            viz_components = []
            for i, viz in enumerate(visualizations):
                viz_title = viz.get("title", f"Visualization {i+1}")
                chart_type = viz.get("chart_type", "bar")
                data_columns = viz.get("data_columns", [])
                
                viz_id = f"viz-{i+1}"
                viz_var = f"fig_{i+1}"
                
                # Generate code based on chart type
                if chart_type == "bar":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
{viz_var} = px.bar(df, x='{data_columns[0]}', y='{data_columns[1]}', title='{viz_title}')
"""
                    else:
                        code += f"""
# {viz_title}
{viz_var} = px.bar(df, title='{viz_title}')  # TODO: Specify x and y columns
"""
                
                elif chart_type == "line":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
{viz_var} = px.line(df, x='{data_columns[0]}', y='{data_columns[1]}', title='{viz_title}')
"""
                    else:
                        code += f"""
# {viz_title}
{viz_var} = px.line(df, title='{viz_title}')  # TODO: Specify x and y columns
"""
                
                elif chart_type == "pie":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
{viz_var} = px.pie(df, names='{data_columns[0]}', values='{data_columns[1]}', title='{viz_title}')
"""
                    else:
                        code += f"""
# {viz_title}
{viz_var} = px.pie(df, title='{viz_title}')  # TODO: Specify names and values columns
"""
                
                elif chart_type == "scatter":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
{viz_var} = px.scatter(df, x='{data_columns[0]}', y='{data_columns[1]}', title='{viz_title}')
"""
                    else:
                        code += f"""
# {viz_title}
{viz_var} = px.scatter(df, title='{viz_title}')  # TODO: Specify x and y columns
"""
                
                elif chart_type == "heatmap":
                    if len(data_columns) >= 3:
                        code += f"""
# {viz_title}
pivot_table = pd.pivot_table(df, values='{data_columns[2]}', index='{data_columns[0]}', columns='{data_columns[1]}')
{viz_var} = px.imshow(pivot_table, title='{viz_title}')
"""
                    else:
                        code += f"""
# {viz_title}
{viz_var} = px.imshow(df.corr(), title='{viz_title}')  # TODO: Specify appropriate pivot table
"""
                
                else:
                    code += f"""
# {viz_title}
{viz_var} = px.bar(df, title='{viz_title}')  # TODO: Replace with appropriate chart type
"""
                
                # Add to visualization components
                viz_components.append(f"""
    html.Div([
        html.H3('{viz_title}'),
        dcc.Graph(id='{viz_id}', figure={viz_var})
    ], className='viz-container')
""")
            
            # Generate layout
            code += """
# Define the layout
app.layout = html.Div([
    html.H1("{title}", className='dashboard-title'),
    html.Div([
        html.P("{purpose}")
    ], className='dashboard-description'),
    
    html.Div([
""".format(title=title, purpose=dashboard.get("purpose", "Dashboard purpose"))
            
            # Add visualization components to layout
            code += ",\n".join(viz_components)
            
            # Complete layout
            code += """
    ], className='dashboard-content')
], className='dashboard-container')

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .dashboard-title {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 20px;
            }
            .dashboard-description {
                text-align: center;
                margin-bottom: 30px;
                color: #7f8c8d;
            }
            .dashboard-content {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }
            .viz-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 15px;
            }
            @media (max-width: 768px) {
                .dashboard-content {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
"""
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating Dash layout code: {e}")
            return None
    
    def _generate_plotly_layout_code(self, dashboard: Dict[str, Any]) -> str:
        """
        Generate Plotly layout code for a dashboard design.
        
        Args:
            dashboard (dict): Dashboard design.
            
        Returns:
            str: Generated Plotly layout code.
        """
        try:
            title = dashboard.get("title", "Dashboard")
            visualizations = dashboard.get("visualizations", [])
            
            # Generate imports
            code = """import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Load and process data
# TODO: Replace with actual data loading code
df = pd.read_csv('your_data.csv')

# Create subplot grid
fig = make_subplots(
    rows={rows}, cols={cols},
    subplot_titles=[{subplot_titles}],
    specs={specs}
)
""".format(
                rows=min(len(visualizations), 3),
                cols=min(2, max(1, len(visualizations) // 2)),
                subplot_titles=", ".join([f"'{viz.get('title', f'Visualization {i+1}')}'" for i, viz in enumerate(visualizations)]),
                specs=[[{"type": "xy"} for _ in range(min(2, max(1, len(visualizations) // 2)))] for _ in range(min(len(visualizations), 3))]
            )
            
            # Generate visualization code
            for i, viz in enumerate(visualizations):
                viz_title = viz.get("title", f"Visualization {i+1}")
                chart_type = viz.get("chart_type", "bar")
                data_columns = viz.get("data_columns", [])
                
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                if row > 3:  # Limit to 3 rows
                    continue
                
                # Generate code based on chart type
                if chart_type == "bar":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
fig.add_trace(
    go.Bar(
        x=df['{data_columns[0]}'],
        y=df['{data_columns[1]}'],
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                    else:
                        code += f"""
# {viz_title}
# TODO: Specify x and y columns
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df.iloc[:, 0],
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                
                elif chart_type == "line":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
fig.add_trace(
    go.Scatter(
        x=df['{data_columns[0]}'],
        y=df['{data_columns[1]}'],
        mode='lines',
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                    else:
                        code += f"""
# {viz_title}
# TODO: Specify x and y columns
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df.iloc[:, 0],
        mode='lines',
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                
                elif chart_type == "pie":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
fig.add_trace(
    go.Pie(
        labels=df['{data_columns[0]}'],
        values=df['{data_columns[1]}'],
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                    else:
                        code += f"""
# {viz_title}
# TODO: Specify labels and values columns
fig.add_trace(
    go.Pie(
        labels=df.iloc[:, 0],
        values=df.iloc[:, 1],
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                
                elif chart_type == "scatter":
                    if len(data_columns) >= 2:
                        code += f"""
# {viz_title}
fig.add_trace(
    go.Scatter(
        x=df['{data_columns[0]}'],
        y=df['{data_columns[1]}'],
        mode='markers',
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                    else:
                        code += f"""
# {viz_title}
# TODO: Specify x and y columns
fig.add_trace(
    go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, 1],
        mode='markers',
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
                
                else:
                    code += f"""
# {viz_title}
# TODO: Replace with appropriate chart type
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df.iloc[:, 0],
        name='{viz_title}'
    ),
    row={row}, col={col}
)
"""
            
            # Update layout
            code += f"""
# Update layout
fig.update_layout(
    title_text='{title}',
    height={len(visualizations) * 300},
    width=1000,
    showlegend=True
)

# Show the figure
fig.show()

# Save the figure
fig.write_html('dashboard.html')
"""
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating Plotly layout code: {e}")
            return None
