import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DesignRationaleGenerator:
    """
    Generates design rationales for dashboard visualizations.
    Explains the reasoning behind visualization choices and layout decisions.
    """
    
    def __init__(self):
        """Initialize the design rationale generator."""
        pass
    
    def generate_visualization_rationale(self, data_summary: Dict[str, Any], 
                                        visualization: Dict[str, Any]) -> str:
        """
        Generate a rationale for a specific visualization.
        
        Args:
            data_summary (dict): Summary of the data being visualized.
            visualization (dict): Visualization configuration.
            
        Returns:
            str: Rationale for the visualization choice.
        """
        try:
            chart_type = visualization.get("chart_type", "").lower()
            columns = visualization.get("data_columns", [])
            title = visualization.get("title", "")
            
            # Get data types for the columns
            data_types = {}
            if "data_types" in data_summary:
                data_types = data_summary["data_types"]
            
            # Get statistics for the columns
            statistics = {}
            if "statistics" in data_summary:
                statistics = data_summary["statistics"]
            
            # Generate rationale based on chart type and data characteristics
            rationale = ""
            
            if chart_type == "bar":
                rationale = self._generate_bar_chart_rationale(columns, data_types, statistics, title)
            elif chart_type == "line":
                rationale = self._generate_line_chart_rationale(columns, data_types, statistics, title)
            elif chart_type == "pie":
                rationale = self._generate_pie_chart_rationale(columns, data_types, statistics, title)
            elif chart_type == "scatter":
                rationale = self._generate_scatter_chart_rationale(columns, data_types, statistics, title)
            elif chart_type == "heatmap":
                rationale = self._generate_heatmap_rationale(columns, data_types, statistics, title)
            elif chart_type == "area":
                rationale = self._generate_area_chart_rationale(columns, data_types, statistics, title)
            elif chart_type == "box":
                rationale = self._generate_box_plot_rationale(columns, data_types, statistics, title)
            elif chart_type == "histogram":
                rationale = self._generate_histogram_rationale(columns, data_types, statistics, title)
            else:
                rationale = f"This {chart_type} chart was selected to visualize the relationship between the data elements."
            
            return rationale
            
        except Exception as e:
            logger.error(f"Error generating visualization rationale: {e}")
            return "This visualization was selected based on the data characteristics."
    
    def _generate_bar_chart_rationale(self, columns: List[str], 
                                     data_types: Dict[str, str], 
                                     statistics: Dict[str, Dict[str, Any]],
                                     title: str) -> str:
        """
        Generate rationale for a bar chart.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the bar chart.
        """
        if len(columns) < 2:
            return "This bar chart was selected to show categorical data distribution."
            
        x_col = columns[0]
        y_col = columns[1]
        
        x_type = data_types.get(x_col, "unknown")
        y_type = data_types.get(y_col, "unknown")
        
        if x_type in ["object", "category"] and y_type in ["int", "float"]:
            # Categorical vs numeric
            x_unique = statistics.get(x_col, {}).get("unique_values", "multiple")
            if isinstance(x_unique, int) and x_unique < 10:
                return f"This bar chart effectively compares {y_col} across different {x_col} categories. Bar charts are ideal for this comparison as they clearly show the relative values for each category, making it easy to identify patterns, outliers, and the highest/lowest values."
            else:
                return f"This bar chart shows the distribution of {y_col} across {x_col} categories. Given the number of categories, the chart helps identify key patterns and significant categories that stand out in terms of {y_col}."
        
        elif y_type in ["object", "category"] and x_type in ["int", "float"]:
            # Numeric vs categorical (horizontal bar chart)
            return f"This horizontal bar chart displays {x_col} values for each {y_col} category. This orientation is particularly effective when category names are long or there are many categories to display."
        
        elif x_type in ["int", "float"] and y_type in ["int", "float"]:
            # Numeric vs numeric (unusual for bar chart)
            return f"This bar chart represents the relationship between {x_col} and {y_col}. While scatter plots are often used for two numeric variables, a bar chart was chosen here to emphasize the discrete nature of the data points."
        
        else:
            return f"This bar chart was selected to clearly visualize and compare {y_col} values across different {x_col} categories, making it easy to identify patterns and outliers in the data."
    
    def _generate_line_chart_rationale(self, columns: List[str], 
                                      data_types: Dict[str, str], 
                                      statistics: Dict[str, Dict[str, Any]],
                                      title: str) -> str:
        """
        Generate rationale for a line chart.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the line chart.
        """
        if len(columns) < 2:
            return "This line chart was selected to show trends over a continuous variable."
            
        x_col = columns[0]
        y_col = columns[1]
        
        x_type = data_types.get(x_col, "unknown")
        
        if x_type == "datetime":
            # Time series
            return f"This line chart visualizes how {y_col} changes over time. Line charts are ideal for time series data as they emphasize trends, patterns, and changes over time, making it easy to identify seasonal patterns, growth trends, or anomalies in the data."
        
        elif x_type in ["int", "float"]:
            # Continuous numeric x-axis
            return f"This line chart shows the relationship between {x_col} and {y_col}, emphasizing the continuous nature of the data and highlighting trends as {x_col} increases. Line charts are effective for showing how one variable changes with respect to another continuous variable."
        
        else:
            return f"This line chart was selected to visualize the progression of {y_col} across different {x_col} values, highlighting trends and patterns in the data that might not be as apparent in other chart types."
    
    def _generate_pie_chart_rationale(self, columns: List[str], 
                                     data_types: Dict[str, str], 
                                     statistics: Dict[str, Dict[str, Any]],
                                     title: str) -> str:
        """
        Generate rationale for a pie chart.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the pie chart.
        """
        if len(columns) < 2:
            return "This pie chart was selected to show the proportion of categories in the data."
            
        cat_col = columns[0]
        val_col = columns[1]
        
        cat_unique = statistics.get(cat_col, {}).get("unique_values", "multiple")
        
        if isinstance(cat_unique, int) and cat_unique <= 7:
            return f"This pie chart effectively shows the proportion of {val_col} across different {cat_col} categories. With {cat_unique} distinct categories, a pie chart is ideal for showing the relative contribution of each category to the total, making it easy to identify which categories represent the largest and smallest proportions."
        else:
            return f"This pie chart shows the distribution of {val_col} across {cat_col} categories. While pie charts work best with fewer categories, this visualization focuses on the most significant categories to provide a clear view of their relative proportions."
    
    def _generate_scatter_chart_rationale(self, columns: List[str], 
                                         data_types: Dict[str, str], 
                                         statistics: Dict[str, Dict[str, Any]],
                                         title: str) -> str:
        """
        Generate rationale for a scatter chart.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the scatter chart.
        """
        if len(columns) < 2:
            return "This scatter plot was selected to show the relationship between two variables."
            
        x_col = columns[0]
        y_col = columns[1]
        
        # Check if there's a third column for bubble size or color
        size_or_color = None
        if len(columns) > 2:
            size_or_color = columns[2]
        
        # Check for correlation if available
        correlation = None
        if "correlations" in statistics:
            if x_col in statistics["correlations"] and y_col in statistics["correlations"][x_col]:
                correlation = statistics["correlations"][x_col][y_col]
            elif y_col in statistics["correlations"] and x_col in statistics["correlations"][y_col]:
                correlation = statistics["correlations"][y_col][x_col]
        
        if correlation is not None:
            corr_strength = abs(correlation)
            corr_direction = "positive" if correlation > 0 else "negative"
            
            if corr_strength > 0.7:
                strength_desc = "strong"
            elif corr_strength > 0.4:
                strength_desc = "moderate"
            else:
                strength_desc = "weak"
            
            if size_or_color:
                return f"This scatter plot reveals a {strength_desc} {corr_direction} correlation ({correlation:.2f}) between {x_col} and {y_col}, with {size_or_color} providing an additional dimension of information. Scatter plots are ideal for showing relationships between variables and identifying patterns, clusters, or outliers in the data."
            else:
                return f"This scatter plot reveals a {strength_desc} {corr_direction} correlation ({correlation:.2f}) between {x_col} and {y_col}. Scatter plots are ideal for showing relationships between variables and identifying patterns, clusters, or outliers in the data."
        
        else:
            if size_or_color:
                return f"This scatter plot visualizes the relationship between {x_col} and {y_col}, with {size_or_color} providing an additional dimension of information. Scatter plots are effective for identifying patterns, clusters, or outliers in the data, as well as potential correlations between variables."
            else:
                return f"This scatter plot visualizes the relationship between {x_col} and {y_col}. Scatter plots are effective for identifying patterns, clusters, or outliers in the data, as well as potential correlations between variables."
    
    def _generate_heatmap_rationale(self, columns: List[str], 
                                   data_types: Dict[str, str], 
                                   statistics: Dict[str, Dict[str, Any]],
                                   title: str) -> str:
        """
        Generate rationale for a heatmap.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the heatmap.
        """
        if len(columns) < 3:
            return "This heatmap was selected to show the intensity of values across two dimensions."
            
        row_col = columns[0]
        col_col = columns[1]
        val_col = columns[2]
        
        row_unique = statistics.get(row_col, {}).get("unique_values", "multiple")
        col_unique = statistics.get(col_col, {}).get("unique_values", "multiple")
        
        if isinstance(row_unique, int) and isinstance(col_unique, int):
            return f"This heatmap effectively visualizes the relationship between {row_col} ({row_unique} categories) and {col_col} ({col_unique} categories), with color intensity representing {val_col}. Heatmaps are ideal for showing patterns and variations across two categorical dimensions, making it easy to identify hotspots, trends, and outliers in the data."
        else:
            return f"This heatmap visualizes the relationship between {row_col} and {col_col}, with color intensity representing {val_col}. Heatmaps are effective for showing patterns and variations across two dimensions, making it easy to identify hotspots, trends, and outliers in the data."
    
    def _generate_area_chart_rationale(self, columns: List[str], 
                                      data_types: Dict[str, str], 
                                      statistics: Dict[str, Dict[str, Any]],
                                      title: str) -> str:
        """
        Generate rationale for an area chart.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the area chart.
        """
        if len(columns) < 2:
            return "This area chart was selected to show the volume or magnitude of data over a continuous variable."
            
        x_col = columns[0]
        y_col = columns[1]
        
        x_type = data_types.get(x_col, "unknown")
        
        if x_type == "datetime":
            # Time series
            return f"This area chart visualizes how {y_col} changes over time, with the filled area emphasizing the volume or magnitude of {y_col}. Area charts are particularly effective for time series data when you want to highlight the cumulative nature of the data or emphasize the total volume over time."
        
        elif x_type in ["int", "float"]:
            # Continuous numeric x-axis
            return f"This area chart shows the relationship between {x_col} and {y_col}, with the filled area emphasizing the volume or magnitude of {y_col}. Area charts are effective for showing how one variable accumulates or changes with respect to another continuous variable."
        
        else:
            return f"This area chart was selected to visualize the volume or magnitude of {y_col} across different {x_col} values, with the filled area providing a visual emphasis on the total amount or cumulative nature of the data."
    
    def _generate_box_plot_rationale(self, columns: List[str], 
                                    data_types: Dict[str, str], 
                                    statistics: Dict[str, Dict[str, Any]],
                                    title: str) -> str:
        """
        Generate rationale for a box plot.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the box plot.
        """
        if len(columns) < 2:
            return "This box plot was selected to show the distribution of a numeric variable."
            
        cat_col = columns[0]
        num_col = columns[1]
        
        cat_type = data_types.get(cat_col, "unknown")
        num_type = data_types.get(num_col, "unknown")
        
        if cat_type in ["object", "category"] and num_type in ["int", "float"]:
            # Categorical vs numeric
            return f"This box plot shows the distribution of {num_col} across different {cat_col} categories. Box plots are ideal for comparing distributions as they display the median, quartiles, and potential outliers, providing a comprehensive view of how {num_col} varies within each {cat_col} category."
        
        elif num_type in ["int", "float"]:
            # Single numeric variable
            return f"This box plot shows the distribution of {num_col}, displaying the median, quartiles, and potential outliers. Box plots are effective for understanding the central tendency, spread, and skewness of the data, as well as identifying any unusual values."
        
        else:
            return f"This box plot was selected to visualize the distribution of data, showing the median, quartiles, and potential outliers. Box plots provide a comprehensive view of the data's central tendency and variability."
    
    def _generate_histogram_rationale(self, columns: List[str], 
                                     data_types: Dict[str, str], 
                                     statistics: Dict[str, Dict[str, Any]],
                                     title: str) -> str:
        """
        Generate rationale for a histogram.
        
        Args:
            columns (list): Columns used in the visualization.
            data_types (dict): Data types for each column.
            statistics (dict): Statistics for each column.
            title (str): Visualization title.
            
        Returns:
            str: Rationale for the histogram.
        """
        if len(columns) < 1:
            return "This histogram was selected to show the distribution of a numeric variable."
            
        num_col = columns[0]
        num_type = data_types.get(num_col, "unknown")
        
        if num_type in ["int", "float"]:
            # Check for statistics to enhance the rationale
            if num_col in statistics:
                stats = statistics[num_col]
                min_val = stats.get("min", None)
                max_val = stats.get("max", None)
                mean_val = stats.get("mean", None)
                median_val = stats.get("median", None)
                
                if all(v is not None for v in [min_val, max_val, mean_val, median_val]):
                    skew_desc = ""
                    if mean_val > median_val:
                        skew_desc = "positively skewed (skewed to the right)"
                    elif mean_val < median_val:
                        skew_desc = "negatively skewed (skewed to the left)"
                    else:
                        skew_desc = "approximately symmetric"
                    
                    return f"This histogram shows the distribution of {num_col}, which ranges from {min_val} to {max_val} with a mean of {mean_val:.2f} and median of {median_val:.2f}. The distribution appears to be {skew_desc}. Histograms are ideal for understanding the shape, central tendency, and spread of numeric data."
                else:
                    return f"This histogram shows the distribution of {num_col}. Histograms are ideal for understanding the shape, central tendency, and spread of numeric data, revealing patterns such as normal distributions, skewness, or multiple modes."
            else:
                return f"This histogram shows the distribution of {num_col}. Histograms are ideal for understanding the shape, central tendency, and spread of numeric data, revealing patterns such as normal distributions, skewness, or multiple modes."
        else:
            return f"This histogram was selected to visualize the distribution of {num_col}, showing the frequency of values within different bins or ranges. Histograms help identify patterns, outliers, and the overall shape of the data distribution."
    
    def generate_dashboard_rationale(self, data_summary: Dict[str, Any], 
                                    dashboard_design: Dict[str, Any]) -> str:
        """
        Generate a rationale for the overall dashboard design.
        
        Args:
            data_summary (dict): Summary of the data being visualized.
            dashboard_design (dict): Dashboard design configuration.
            
        Returns:
            str: Rationale for the dashboard design.
        """
        try:
            title = dashboard_design.get("title", "Dashboard")
            purpose = dashboard_design.get("purpose", "")
            visualizations = dashboard_design.get("visualizations", [])
            layout = dashboard_design.get("layout", "")
            
            # Count visualization types
            viz_types = {}
            for viz in visualizations:
                chart_type = viz.get("chart_type", "").lower()
                viz_types[chart_type] = viz_types.get(chart_type, 0) + 1
            
            # Generate rationale
            rationale = f"This dashboard titled '{title}' is designed to {purpose.lower() if purpose else 'provide insights into the data'}. "
            
            # Add visualization composition
            if viz_types:
                rationale += "It combines "
                viz_descriptions = []
                for chart_type, count in viz_types.items():
                    if count > 1:
                        viz_descriptions.append(f"{count} {chart_type} charts")
                    else:
                        viz_descriptions.append(f"a {chart_type} chart")
                
                if len(viz_descriptions) > 1:
                    last_desc = viz_descriptions.pop()
                    rationale += ", ".join(viz_descriptions) + f", and {last_desc}"
                else:
                    rationale += viz_descriptions[0]
                
                rationale += " to provide a comprehensive view of the data. "
            
            # Add layout description
            if layout:
                rationale += f"The layout is organized as {layout.lower()}, "
            else:
                rationale += "The layout is organized to "
            
            rationale += "facilitate easy comparison and analysis of related metrics. "
            
            # Add purpose and audience
            rationale += "This design enables users to quickly identify patterns, trends, and outliers in the data, "
            rationale += "supporting data-driven decision making and providing actionable insights."
            
            return rationale
            
        except Exception as e:
            logger.error(f"Error generating dashboard rationale: {e}")
            return "This dashboard is designed to provide comprehensive insights into the data through a carefully selected set of visualizations."
