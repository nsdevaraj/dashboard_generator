import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractiveComponents:
    """
    Provides interactive components for dashboard filtering and data exploration.
    Creates filter controls, selectors, and interactive elements.
    """
    
    def __init__(self, app=None):
        """
        Initialize the interactive components.
        
        Args:
            app (dash.Dash, optional): Dash application instance.
        """
        self.app = app
    
    def set_app(self, app):
        """
        Set the Dash application instance.
        
        Args:
            app (dash.Dash): Dash application instance.
        """
        self.app = app
    
    def create_date_range_filter(self, id_prefix: str, 
                               label: str = "Date Range",
                               start_date=None, 
                               end_date=None) -> html.Div:
        """
        Create a date range filter component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            label (str): Label for the filter.
            start_date: Initial start date.
            end_date: Initial end date.
            
        Returns:
            dash.html.Div: Date range filter component.
        """
        return html.Div([
            html.Label(label),
            dcc.DatePickerRange(
                id=f"{id_prefix}-date-range",
                start_date=start_date,
                end_date=end_date,
                display_format='YYYY-MM-DD',
                className="date-range-filter"
            )
        ], className="filter-container")
    
    def create_dropdown_filter(self, id_prefix: str, 
                             options: List[Dict[str, str]],
                             label: str = "Filter",
                             multi: bool = False,
                             placeholder: str = "Select...",
                             value=None) -> html.Div:
        """
        Create a dropdown filter component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            options (list): List of dropdown options.
            label (str): Label for the filter.
            multi (bool): Whether to allow multiple selections.
            placeholder (str): Placeholder text.
            value: Initial selected value(s).
            
        Returns:
            dash.html.Div: Dropdown filter component.
        """
        return html.Div([
            html.Label(label),
            dcc.Dropdown(
                id=f"{id_prefix}-dropdown",
                options=options,
                value=value,
                multi=multi,
                placeholder=placeholder,
                className="dropdown-filter"
            )
        ], className="filter-container")
    
    def create_range_slider(self, id_prefix: str, 
                          min_val: float, 
                          max_val: float,
                          label: str = "Range",
                          step: float = None,
                          value: List[float] = None,
                          marks: Dict[float, str] = None) -> html.Div:
        """
        Create a range slider component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            min_val (float): Minimum value.
            max_val (float): Maximum value.
            label (str): Label for the slider.
            step (float, optional): Step size.
            value (list, optional): Initial selected range.
            marks (dict, optional): Marks for the slider.
            
        Returns:
            dash.html.Div: Range slider component.
        """
        # Set default value if not provided
        if value is None:
            value = [min_val, max_val]
        
        # Generate default marks if not provided
        if marks is None:
            # Create marks at regular intervals
            num_marks = 5
            interval = (max_val - min_val) / (num_marks - 1)
            marks = {min_val + i * interval: f"{min_val + i * interval:.1f}" 
                    for i in range(num_marks)}
        
        return html.Div([
            html.Label(label),
            dcc.RangeSlider(
                id=f"{id_prefix}-range-slider",
                min=min_val,
                max=max_val,
                step=step,
                value=value,
                marks=marks,
                className="range-slider"
            )
        ], className="filter-container")
    
    def create_radio_filter(self, id_prefix: str, 
                          options: List[Dict[str, str]],
                          label: str = "Options",
                          value=None) -> html.Div:
        """
        Create a radio button filter component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            options (list): List of radio options.
            label (str): Label for the filter.
            value: Initial selected value.
            
        Returns:
            dash.html.Div: Radio button filter component.
        """
        return html.Div([
            html.Label(label),
            dcc.RadioItems(
                id=f"{id_prefix}-radio",
                options=options,
                value=value,
                className="radio-filter"
            )
        ], className="filter-container")
    
    def create_checklist_filter(self, id_prefix: str, 
                              options: List[Dict[str, str]],
                              label: str = "Options",
                              value=None) -> html.Div:
        """
        Create a checklist filter component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            options (list): List of checklist options.
            label (str): Label for the filter.
            value: Initial selected value(s).
            
        Returns:
            dash.html.Div: Checklist filter component.
        """
        return html.Div([
            html.Label(label),
            dcc.Checklist(
                id=f"{id_prefix}-checklist",
                options=options,
                value=value,
                className="checklist-filter"
            )
        ], className="filter-container")
    
    def create_filter_panel(self, filters: List[html.Div], 
                          title: str = "Filters",
                          collapsible: bool = True) -> html.Div:
        """
        Create a panel containing multiple filters.
        
        Args:
            filters (list): List of filter components.
            title (str): Title for the filter panel.
            collapsible (bool): Whether the panel can be collapsed.
            
        Returns:
            dash.html.Div: Filter panel component.
        """
        if collapsible:
            return html.Div([
                html.Div([
                    html.H3(title, className="filter-panel-title"),
                    html.Button(
                        "▼", 
                        id="filter-panel-toggle",
                        className="filter-panel-toggle"
                    )
                ], className="filter-panel-header"),
                html.Div(
                    filters,
                    id="filter-panel-content",
                    className="filter-panel-content"
                )
            ], className="filter-panel")
        else:
            return html.Div([
                html.H3(title, className="filter-panel-title"),
                html.Div(filters, className="filter-panel-content")
            ], className="filter-panel")
    
    def create_tabs_container(self, tabs: List[Dict[str, Any]], 
                            id_prefix: str = "dashboard") -> html.Div:
        """
        Create a tabbed container for multiple dashboards.
        
        Args:
            tabs (list): List of tab configurations.
            id_prefix (str): Prefix for component IDs.
            
        Returns:
            dash.html.Div: Tabs container component.
        """
        tab_labels = [tab.get("label", f"Tab {i+1}") for i, tab in enumerate(tabs)]
        tab_children = [tab.get("content", html.Div()) for tab in tabs]
        
        return html.Div([
            dcc.Tabs(
                id=f"{id_prefix}-tabs",
                value=0,
                children=[
                    dcc.Tab(label=label, value=i)
                    for i, label in enumerate(tab_labels)
                ],
                className="dashboard-tabs"
            ),
            html.Div(
                id=f"{id_prefix}-tabs-content",
                className="tabs-content"
            )
        ], className="tabs-container")
    
    def create_download_button(self, id_prefix: str, 
                             label: str = "Download Data",
                             filename: str = "data.csv") -> html.Div:
        """
        Create a download button component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            label (str): Label for the button.
            filename (str): Default filename for downloads.
            
        Returns:
            dash.html.Div: Download button component.
        """
        return html.Div([
            html.Button(
                label,
                id=f"{id_prefix}-download-button",
                className="download-button"
            ),
            dcc.Download(id=f"{id_prefix}-download")
        ], className="download-container")
    
    def create_data_table(self, id_prefix: str, 
                        columns: List[Dict[str, str]],
                        data: List[Dict[str, Any]] = None,
                        page_size: int = 10,
                        sortable: bool = True,
                        filterable: bool = True) -> html.Div:
        """
        Create an interactive data table component.
        
        Args:
            id_prefix (str): Prefix for component IDs.
            columns (list): List of column configurations.
            data (list, optional): Initial data for the table.
            page_size (int): Number of rows per page.
            sortable (bool): Whether columns are sortable.
            filterable (bool): Whether columns are filterable.
            
        Returns:
            dash.html.Div: Data table component.
        """
        from dash import dash_table
        
        return html.Div([
            dash_table.DataTable(
                id=f"{id_prefix}-table",
                columns=columns,
                data=data if data is not None else [],
                page_size=page_size,
                sort_action="native" if sortable else "none",
                filter_action="native" if filterable else "none",
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'minWidth': '100px',
                    'maxWidth': '300px',
                    'whiteSpace': 'normal',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f1f3f5'
                    }
                ]
            )
        ], className="table-container")
    
    def register_callbacks(self):
        """
        Register callbacks for interactive components.
        
        Note: This method should be called after all components are added to the layout.
        """
        if self.app is None:
            logger.error("No app instance set. Call set_app() first.")
            return
        
        # Register callback for filter panel toggle
        @self.app.callback(
            Output("filter-panel-content", "style"),
            Input("filter-panel-toggle", "n_clicks"),
            State("filter-panel-content", "style"),
            prevent_initial_call=True
        )
        def toggle_filter_panel(n_clicks, current_style):
            if current_style is None:
                current_style = {}
            
            if "display" not in current_style or current_style["display"] != "none":
                return {"display": "none"}
            else:
                return {"display": "block"}
        
        # Register callback for tabs
        @self.app.callback(
            Output("dashboard-tabs-content", "children"),
            Input("dashboard-tabs", "value")
        )
        def render_tab_content(tab_index):
            # This is a placeholder callback that should be customized
            # based on the actual tab content
            return html.Div(f"Content for tab {tab_index}")
        
        # Register callback for download button
        @self.app.callback(
            Output("dashboard-download", "data"),
            Input("dashboard-download-button", "n_clicks"),
            prevent_initial_call=True
        )
        def download_data(n_clicks):
            # This is a placeholder callback that should be customized
            # based on the actual data to download
            return dcc.send_data_frame(
                pd.DataFrame({"Column": ["Data"]}),
                "data.csv",
                index=False
            )


class ResponsiveLayouts:
    """
    Provides responsive layout templates for dashboards.
    Creates grid-based and flexible layouts that adapt to different screen sizes.
    """
    
    def __init__(self):
        """Initialize the responsive layouts."""
        pass
    
    def create_grid_layout(self, items: List[html.Div], 
                         columns: int = 2,
                         gap: str = "20px") -> html.Div:
        """
        Create a responsive grid layout.
        
        Args:
            items (list): List of components to arrange in the grid.
            columns (int): Number of columns in the grid.
            gap (str): Gap between grid items.
            
        Returns:
            dash.html.Div: Grid layout component.
        """
        return html.Div(
            items,
            style={
                'display': 'grid',
                'gridTemplateColumns': f'repeat(auto-fit, minmax({100/columns}%, 1fr))',
                'gap': gap
            },
            className="grid-layout"
        )
    
    def create_flex_layout(self, items: List[html.Div], 
                         direction: str = "row",
                         wrap: bool = True,
                         justify: str = "space-between",
                         align: str = "stretch",
                         gap: str = "20px") -> html.Div:
        """
        Create a responsive flex layout.
        
        Args:
            items (list): List of components to arrange in the flex container.
            direction (str): Flex direction ('row' or 'column').
            wrap (bool): Whether items should wrap.
            justify (str): Justify content value.
            align (str): Align items value.
            gap (str): Gap between flex items.
            
        Returns:
            dash.html.Div: Flex layout component.
        """
        return html.Div(
            items,
            style={
                'display': 'flex',
                'flexDirection': direction,
                'flexWrap': 'wrap' if wrap else 'nowrap',
                'justifyContent': justify,
                'alignItems': align,
                'gap': gap
            },
            className="flex-layout"
        )
    
    def create_dashboard_layout(self, title: str,
                              description: str,
                              filters: html.Div,
                              visualizations: List[html.Div],
                              data_table: html.Div = None,
                              download_button: html.Div = None) -> html.Div:
        """
        Create a complete responsive dashboard layout.
        
        Args:
            title (str): Dashboard title.
            description (str): Dashboard description.
            filters (dash.html.Div): Filter panel component.
            visualizations (list): List of visualization components.
            data_table (dash.html.Div, optional): Data table component.
            download_button (dash.html.Div, optional): Download button component.
            
        Returns:
            dash.html.Div: Complete dashboard layout.
        """
        # Create header section
        header = html.Div([
            html.H1(title, className="dashboard-title"),
            html.P(description, className="dashboard-description")
        ], className="dashboard-header")
        
        # Create main content section with filters and visualizations
        main_content = html.Div([
            html.Div([
                filters
            ], className="dashboard-sidebar"),
            html.Div([
                self.create_grid_layout(visualizations)
            ], className="dashboard-main")
        ], className="dashboard-content")
        
        # Create footer section with data table and download button
        footer_items = []
        if data_table is not None:
            footer_items.append(data_table)
        if download_button is not None:
            footer_items.append(download_button)
        
        footer = html.Div(
            footer_items,
            className="dashboard-footer"
        ) if footer_items else None
        
        # Combine all sections
        dashboard_components = [header, main_content]
        if footer is not None:
            dashboard_components.append(footer)
        
        return html.Div(
            dashboard_components,
            className="dashboard-container"
        )
    
    def create_multi_dashboard_layout(self, dashboards: List[Dict[str, Any]]) -> html.Div:
        """
        Create a layout with multiple dashboards in tabs.
        
        Args:
            dashboards (list): List of dashboard configurations.
            
        Returns:
            dash.html.Div: Multi-dashboard layout with tabs.
        """
        tabs = []
        
        for i, dashboard in enumerate(dashboards):
            title = dashboard.get("title", f"Dashboard {i+1}")
            description = dashboard.get("description", "")
            filters = dashboard.get("filters", html.Div())
            visualizations = dashboard.get("visualizations", [])
            data_table = dashboard.get("data_table")
            download_button = dashboard.get("download_button")
            
            # Create dashboard layout
            dashboard_layout = self.create_dashboard_layout(
                title=title,
                description=description,
                filters=filters,
                visualizations=visualizations,
                data_table=data_table,
                download_button=download_button
            )
            
            # Add to tabs
            tabs.append({
                "label": title,
                "content": dashboard_layout
            })
        
        # Create interactive components for tabs
        interactive = InteractiveComponents()
        tabs_container = interactive.create_tabs_container(tabs)
        
        return html.Div([
            html.H1("Dashboard Suite", className="dashboard-suite-title"),
            tabs_container
        ], className="dashboard-suite-container")
    
    def get_css_styles(self) -> str:
        """
        Get CSS styles for responsive layouts.
        
        Returns:
            str: CSS styles as a string.
        """
        return """
        /* Base styles */
        .dashboard-container {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header styles */
        .dashboard-header {
            margin-bottom: 30px;
            text-align: center;
        }
        
        .dashboard-title {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .dashboard-description {
            color: #7f8c8d;
            font-size: 16px;
        }
        
        /* Content layout */
        .dashboard-content {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .dashboard-sidebar {
            flex: 0 0 300px;
        }
        
        .dashboard-main {
            flex: 1;
        }
        
        /* Filter panel styles */
        .filter-panel {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .filter-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .filter-panel-title {
            margin: 0;
            font-size: 18px;
        }
        
        .filter-panel-toggle {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        
        .filter-container {
            margin-bottom: 15px;
        }
        
        .filter-container label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        /* Visualization container */
        .visualization-container {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Table styles */
        .table-container {
            margin-top: 20px;
            overflow-x: auto;
        }
        
        /* Download button */
        .download-button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .download-button:hover {
            background-color: #2980b9;
        }
        
        /* Tabs styles */
        .dashboard-tabs {
            margin-bottom: 20px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .dashboard-content {
                flex-direction: column;
            }
            
            .dashboard-sidebar {
                flex: 0 0 auto;
                width: 100%;
            }
        }
        """
