import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

class VisualizationComponents:
    """
    Provides visualization components for dashboard generation.
    Creates various chart types and interactive elements.
    """
    
    def __init__(self, theme: str = "light"):
        """
        Initialize the visualization components.
        
        Args:
            theme (str): Color theme for visualizations ('light' or 'dark').
        """
        self.theme = theme
        self.color_schemes = {
            "light": ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"],
            "dark": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        }
        self.template = "plotly_white" if theme == "light" else "plotly_dark"
    
    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str = None, color_col: str = None, 
                        orientation: str = 'v') -> go.Figure:
        """
        Create a bar chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis.
            y_col (str): Column to use for y-axis.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            orientation (str): Bar orientation ('v' for vertical, 'h' for horizontal).
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure.
        """
        try:
            if orientation == 'h':
                # Swap x and y for horizontal bar chart
                x_col, y_col = y_col, x_col
            
            if color_col:
                fig = px.bar(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    orientation=orientation
                )
            else:
                fig = px.bar(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    orientation=orientation
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"{y_col} by {x_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating bar chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = None, color_col: str = None,
                         add_markers: bool = False) -> go.Figure:
        """
        Create a line chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis.
            y_col (str): Column to use for y-axis.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            add_markers (bool): Whether to add markers to the lines.
            
        Returns:
            plotly.graph_objects.Figure: Line chart figure.
        """
        try:
            # Ensure x-axis is sorted
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                df = df.sort_values(by=x_col)
            
            # Set line mode
            line_mode = 'lines+markers' if add_markers else 'lines'
            
            if color_col:
                fig = px.line(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    markers=add_markers
                )
            else:
                fig = px.line(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    markers=add_markers
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"{y_col} over {x_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating line chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_pie_chart(self, df: pd.DataFrame, names_col: str, values_col: str, 
                        title: str = None, hole: float = 0) -> go.Figure:
        """
        Create a pie chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            names_col (str): Column to use for slice names.
            values_col (str): Column to use for slice values.
            title (str, optional): Chart title.
            hole (float): Size of the hole in the center (0-1, 0 for pie, >0 for donut).
            
        Returns:
            plotly.graph_objects.Figure: Pie chart figure.
        """
        try:
            fig = px.pie(
                df, names=names_col, values=values_col, 
                title=title,
                color_discrete_sequence=self.color_schemes[self.theme],
                template=self.template,
                hole=hole
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"Distribution of {values_col} by {names_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Customize traces
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                insidetextorientation='radial'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating pie chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           title: str = None, color_col: str = None, 
                           size_col: str = None, add_trendline: bool = False) -> go.Figure:
        """
        Create a scatter plot.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis.
            y_col (str): Column to use for y-axis.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            size_col (str, optional): Column to use for marker size.
            add_trendline (bool): Whether to add a trendline.
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure.
        """
        try:
            # Set up trendline
            trendline = 'ols' if add_trendline else None
            
            if color_col and size_col:
                fig = px.scatter(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    size=size_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    trendline=trendline
                )
            elif color_col:
                fig = px.scatter(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    trendline=trendline
                )
            elif size_col:
                fig = px.scatter(
                    df, x=x_col, y=y_col, 
                    title=title,
                    size=size_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    trendline=trendline
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    trendline=trendline
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"Relationship between {x_col} and {y_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating scatter plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_heatmap(self, df: pd.DataFrame, x_col: str = None, y_col: str = None, 
                      z_col: str = None, title: str = None) -> go.Figure:
        """
        Create a heatmap.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str, optional): Column to use for x-axis.
            y_col (str, optional): Column to use for y-axis.
            z_col (str, optional): Column to use for z-axis (values).
            title (str, optional): Chart title.
            
        Returns:
            plotly.graph_objects.Figure: Heatmap figure.
        """
        try:
            # If x_col, y_col, and z_col are provided, create a pivot table
            if x_col and y_col and z_col:
                pivot_df = df.pivot_table(
                    values=z_col,
                    index=y_col,
                    columns=x_col,
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    pivot_df,
                    title=title,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    template=self.template
                )
                
            # If no columns are specified, use correlation matrix
            else:
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title=title if title else "Correlation Matrix",
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    template=self.template
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else "Heatmap",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Add value annotations
            fig.update_traces(
                text=np.around(fig.data[0].z, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating heatmap: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_area_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = None, color_col: str = None, 
                         stacked: bool = False) -> go.Figure:
        """
        Create an area chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis.
            y_col (str): Column to use for y-axis.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            stacked (bool): Whether to stack areas.
            
        Returns:
            plotly.graph_objects.Figure: Area chart figure.
        """
        try:
            # Ensure x-axis is sorted
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                df = df.sort_values(by=x_col)
            
            # Set grouping mode
            grouping = 'stack' if stacked else 'overlay'
            
            if color_col:
                fig = px.area(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template,
                    groupnorm=grouping
                )
            else:
                fig = px.area(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"{y_col} over {x_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating area chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating area chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_box_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                       title: str = None, color_col: str = None) -> go.Figure:
        """
        Create a box plot.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis (categories).
            y_col (str): Column to use for y-axis (values).
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            
        Returns:
            plotly.graph_objects.Figure: Box plot figure.
        """
        try:
            if color_col:
                fig = px.box(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color=color_col,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template
                )
            else:
                fig = px.box(
                    df, x=x_col, y=y_col, 
                    title=title,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"Distribution of {y_col} by {x_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating box plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_histogram(self, df: pd.DataFrame, x_col: str, 
                        title: str = None, color_col: str = None, 
                        nbins: int = 20) -> go.Figure:
        """
        Create a histogram.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for x-axis.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            nbins (int): Number of bins.
            
        Returns:
            plotly.graph_objects.Figure: Histogram figure.
        """
        try:
            if color_col:
                fig = px.histogram(
                    df, x=x_col, 
                    title=title,
                    color=color_col,
                    nbins=nbins,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template
                )
            else:
                fig = px.histogram(
                    df, x=x_col, 
                    title=title,
                    nbins=nbins,
                    color_discrete_sequence=self.color_schemes[self.theme],
                    template=self.template
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"Distribution of {x_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x_col,
                yaxis_title="Count",
                legend_title=color_col if color_col else None,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating histogram: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_radar_chart(self, df: pd.DataFrame, categories_col: str, values_col: str, 
                          title: str = None, color_col: str = None) -> go.Figure:
        """
        Create a radar chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            categories_col (str): Column to use for radar categories.
            values_col (str): Column to use for radar values.
            title (str, optional): Chart title.
            color_col (str, optional): Column to use for color encoding.
            
        Returns:
            plotly.graph_objects.Figure: Radar chart figure.
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # If color column is provided, create a trace for each color
            if color_col:
                for color_val in df[color_col].unique():
                    color_df = df[df[color_col] == color_val]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=color_df[values_col],
                        theta=color_df[categories_col],
                        fill='toself',
                        name=str(color_val)
                    ))
            else:
                fig.add_trace(go.Scatterpolar(
                    r=df[values_col],
                    theta=df[categories_col],
                    fill='toself'
                ))
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"{values_col} by {categories_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, df[values_col].max() * 1.1]
                    )
                ),
                showlegend=True if color_col else False,
                template=self.template,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating radar chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_funnel_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           title: str = None) -> go.Figure:
        """
        Create a funnel chart.
        
        Args:
            df (pandas.DataFrame): Data to visualize.
            x_col (str): Column to use for stage names.
            y_col (str): Column to use for stage values.
            title (str, optional): Chart title.
            
        Returns:
            plotly.graph_objects.Figure: Funnel chart figure.
        """
        try:
            # Sort data by values in descending order
            df_sorted = df.sort_values(by=y_col, ascending=False)
            
            fig = px.funnel(
                df_sorted, x=y_col, y=x_col, 
                title=title,
                color_discrete_sequence=self.color_schemes[self.theme],
                template=self.template
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else f"Funnel Chart of {x_col} by {y_col}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating funnel chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating funnel chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_gauge_chart(self, value: float, title: str = None, 
                          min_val: float = 0, max_val: float = 100, 
                          threshold_values: List[float] = None) -> go.Figure:
        """
        Create a gauge chart.
        
        Args:
            value (float): Value to display on the gauge.
            title (str, optional): Chart title.
            min_val (float): Minimum value for the gauge.
            max_val (float): Maximum value for the gauge.
            threshold_values (list, optional): List of threshold values for color changes.
            
        Returns:
            plotly.graph_objects.Figure: Gauge chart figure.
        """
        try:
            # Set default thresholds if not provided
            if threshold_values is None:
                threshold_values = [0.33 * max_val, 0.66 * max_val, max_val]
            
            # Set colors based on thresholds
            if value <= threshold_values[0]:
                color = "green"
            elif value <= threshold_values[1]:
                color = "yellow"
            else:
                color = "red"
            
            # Calculate the percentage of the value within the range
            percentage = (value - min_val) / (max_val - min_val) * 100
            
            # Create the gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title if title else "Gauge Chart"},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [min_val, threshold_values[0]], 'color': "lightgreen"},
                        {'range': [threshold_values[0], threshold_values[1]], 'color': "lightyellow"},
                        {'range': [threshold_values[1], max_val], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
            
            # Customize layout
            fig.update_layout(
                template=self.template,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating gauge chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_multi_chart_dashboard(self, charts: List[Dict[str, Any]], 
                                    title: str = None, 
                                    layout: List[List[int]] = None) -> go.Figure:
        """
        Create a dashboard with multiple charts.
        
        Args:
            charts (list): List of chart configurations.
            title (str, optional): Dashboard title.
            layout (list, optional): Grid layout specification.
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure.
        """
        try:
            # Determine layout if not provided
            if layout is None:
                n_charts = len(charts)
                if n_charts <= 2:
                    rows, cols = 1, n_charts
                elif n_charts <= 4:
                    rows, cols = 2, 2
                elif n_charts <= 6:
                    rows, cols = 2, 3
                elif n_charts <= 9:
                    rows, cols = 3, 3
                else:
                    rows, cols = (n_charts + 2) // 3, 3
                
                # Create default layout
                layout = []
                chart_idx = 0
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        if chart_idx < n_charts:
                            row.append(chart_idx)
                            chart_idx += 1
                        else:
                            row.append(None)
                    layout.append(row)
            
            # Determine subplot titles
            subplot_titles = []
            for chart in charts:
                chart_title = chart.get("title", "")
                subplot_titles.append(chart_title)
            
            # Create subplot grid
            fig = make_subplots(
                rows=len(layout),
                cols=max(len(row) for row in layout),
                subplot_titles=subplot_titles,
                specs=[[{"type": "xy"} for _ in row] for row in layout]
            )
            
            # Add charts to the grid
            for r, row in enumerate(layout):
                for c, chart_idx in enumerate(row):
                    if chart_idx is not None and chart_idx < len(charts):
                        chart = charts[chart_idx]
                        chart_fig = chart.get("figure")
                        
                        if chart_fig is not None:
                            for trace in chart_fig.data:
                                fig.add_trace(
                                    trace,
                                    row=r+1,
                                    col=c+1
                                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title if title else "Dashboard",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                height=300 * len(layout),
                width=300 * max(len(row) for row in layout),
                template=self.template,
                showlegend=True,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-chart dashboard: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating multi-chart dashboard: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_interactive_elements(self, chart_type: str, 
                                   fig: go.Figure, 
                                   options: Dict[str, Any] = None) -> go.Figure:
        """
        Add interactive elements to a chart.
        
        Args:
            chart_type (str): Type of chart.
            fig (plotly.graph_objects.Figure): Chart figure.
            options (dict, optional): Options for interactive elements.
            
        Returns:
            plotly.graph_objects.Figure: Chart figure with interactive elements.
        """
        try:
            options = options or {}
            
            # Add hover information
            if options.get("add_hover_info", True):
                if chart_type in ["bar", "line", "scatter", "area"]:
                    fig.update_traces(
                        hoverinfo="x+y+text",
                        hovertemplate="%{x}<br>%{y}<extra></extra>"
                    )
                elif chart_type == "pie":
                    fig.update_traces(
                        hoverinfo="label+percent+value",
                        hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>"
                    )
                elif chart_type == "heatmap":
                    fig.update_traces(
                        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>"
                    )
            
            # Add range slider for time series
            if options.get("add_range_slider", False) and chart_type in ["line", "area"]:
                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeslider_thickness=0.05
                )
            
            # Add zoom and pan capabilities
            if options.get("add_zoom_pan", True):
                fig.update_layout(
                    dragmode="zoom",
                    selectdirection="any"
                )
            
            # Add annotations for key points
            if options.get("add_annotations", False) and "annotation_points" in options:
                for point in options["annotation_points"]:
                    fig.add_annotation(
                        x=point["x"],
                        y=point["y"],
                        text=point.get("text", ""),
                        showarrow=True,
                        arrowhead=1
                    )
            
            # Add trend lines
            if options.get("add_trend_line", False) and chart_type in ["scatter"]:
                # Add OLS trend line
                if "x" in fig.data[0] and "y" in fig.data[0]:
                    x = fig.data[0].x
                    y = fig.data[0].y
                    
                    # Simple linear regression
                    if len(x) > 1 and len(y) > 1:
                        try:
                            import numpy as np
                            from scipy import stats
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            line_x = np.array([min(x), max(x)])
                            line_y = slope * line_x + intercept
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=line_x,
                                    y=line_y,
                                    mode="lines",
                                    name=f"Trend (r²={r_value**2:.2f})",
                                    line=dict(color="red", dash="dash")
                                )
                            )
                        except:
                            # If scipy is not available or error occurs, skip trend line
                            pass
            
            return fig
            
        except Exception as e:
            logger.error(f"Error adding interactive elements: {e}")
            return fig
