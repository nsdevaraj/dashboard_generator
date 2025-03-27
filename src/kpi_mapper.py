import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KPIMapper:
    """
    Maps KPIs to appropriate visualizations and metrics.
    Helps create dashboards focused on specific business objectives.
    """
    
    def __init__(self):
        """Initialize the KPI mapper."""
        # Common KPI categories and their typical metrics
        self.kpi_categories = {
            "sales": [
                "revenue", "sales volume", "average order value", "conversion rate", 
                "sales growth", "sales by region", "sales by product", "sales by channel"
            ],
            "financial": [
                "profit", "profit margin", "cost", "expenses", "revenue", "roi", 
                "cash flow", "budget variance", "operating margin"
            ],
            "marketing": [
                "conversion rate", "customer acquisition cost", "click-through rate", 
                "bounce rate", "engagement", "reach", "impressions", "campaign performance"
            ],
            "customer": [
                "customer satisfaction", "net promoter score", "churn rate", 
                "retention rate", "lifetime value", "acquisition cost", "support tickets"
            ],
            "operational": [
                "efficiency", "productivity", "utilization", "cycle time", 
                "throughput", "defect rate", "inventory turnover", "on-time delivery"
            ],
            "hr": [
                "employee satisfaction", "turnover rate", "time to hire", 
                "training completion", "productivity", "absenteeism", "overtime"
            ]
        }
        
        # Visualization types best suited for different KPI categories
        self.kpi_viz_mapping = {
            "sales": {
                "primary": ["bar", "line", "area"],
                "secondary": ["pie", "heatmap", "funnel"],
                "time_based": ["line", "area", "bar"]
            },
            "financial": {
                "primary": ["line", "bar", "area"],
                "secondary": ["pie", "scatter", "gauge"],
                "time_based": ["line", "area", "bar"]
            },
            "marketing": {
                "primary": ["line", "bar", "funnel"],
                "secondary": ["pie", "heatmap", "scatter"],
                "time_based": ["line", "area", "bar"]
            },
            "customer": {
                "primary": ["bar", "line", "gauge"],
                "secondary": ["pie", "radar", "heatmap"],
                "time_based": ["line", "area", "bar"]
            },
            "operational": {
                "primary": ["bar", "line", "gauge"],
                "secondary": ["scatter", "heatmap", "box"],
                "time_based": ["line", "area", "control"]
            },
            "hr": {
                "primary": ["bar", "line", "radar"],
                "secondary": ["pie", "heatmap", "gauge"],
                "time_based": ["line", "area", "bar"]
            }
        }
    
    def identify_kpi_category(self, kpi: str) -> str:
        """
        Identify the category of a KPI.
        
        Args:
            kpi (str): The KPI to categorize.
            
        Returns:
            str: The category of the KPI.
        """
        kpi_lower = kpi.lower()
        
        # Check each category for matching keywords
        for category, keywords in self.kpi_categories.items():
            for keyword in keywords:
                if keyword in kpi_lower:
                    return category
        
        # Default to "general" if no match found
        return "general"
    
    def suggest_visualizations_for_kpi(self, kpi: str, 
                                      time_based: bool = False) -> List[str]:
        """
        Suggest visualization types for a specific KPI.
        
        Args:
            kpi (str): The KPI to suggest visualizations for.
            time_based (bool): Whether the data is time-based.
            
        Returns:
            list: Suggested visualization types.
        """
        category = self.identify_kpi_category(kpi)
        
        if category == "general":
            # Default suggestions for general KPIs
            if time_based:
                return ["line", "area", "bar"]
            else:
                return ["bar", "pie", "gauge"]
        
        # Get suggestions based on category
        if time_based and "time_based" in self.kpi_viz_mapping[category]:
            return self.kpi_viz_mapping[category]["time_based"]
        else:
            return self.kpi_viz_mapping[category]["primary"] + self.kpi_viz_mapping[category]["secondary"][:2]
    
    def map_kpis_to_columns(self, kpis: List[str], 
                           data_summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Map KPIs to relevant columns in the data.
        
        Args:
            kpis (list): List of KPIs to map.
            data_summary (dict): Summary of the data.
            
        Returns:
            dict: Mapping of KPIs to relevant columns.
        """
        try:
            columns = data_summary.get("columns", [])
            data_types = data_summary.get("data_types", {})
            
            kpi_column_mapping = {}
            
            for kpi in kpis:
                kpi_lower = kpi.lower()
                relevant_columns = []
                
                # Look for direct matches in column names
                for col in columns:
                    col_lower = col.lower()
                    
                    # Check for exact matches or substring matches
                    if kpi_lower == col_lower or kpi_lower in col_lower or any(keyword in col_lower for keyword in kpi_lower.split()):
                        relevant_columns.append(col)
                
                # If no direct matches, try to infer based on KPI category
                if not relevant_columns:
                    category = self.identify_kpi_category(kpi)
                    
                    if category == "sales":
                        # Look for sales-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["sale", "revenue", "order", "product", "customer"]):
                                if data_types.get(col) in ["int", "float"]:
                                    relevant_columns.append(col)
                    
                    elif category == "financial":
                        # Look for financial-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["profit", "cost", "expense", "revenue", "margin", "budget"]):
                                if data_types.get(col) in ["int", "float"]:
                                    relevant_columns.append(col)
                    
                    elif category == "marketing":
                        # Look for marketing-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["conversion", "click", "impression", "campaign", "channel"]):
                                relevant_columns.append(col)
                    
                    elif category == "customer":
                        # Look for customer-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["customer", "satisfaction", "nps", "churn", "retention"]):
                                relevant_columns.append(col)
                    
                    elif category == "operational":
                        # Look for operational-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["efficiency", "productivity", "utilization", "cycle", "defect"]):
                                relevant_columns.append(col)
                    
                    elif category == "hr":
                        # Look for HR-related columns
                        for col in columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ["employee", "turnover", "hire", "training", "satisfaction"]):
                                relevant_columns.append(col)
                
                # Add time-related columns for trend analysis
                time_columns = []
                for col in columns:
                    col_lower = col.lower()
                    if data_types.get(col) == "datetime" or any(keyword in col_lower for keyword in ["date", "time", "year", "month", "day"]):
                        time_columns.append(col)
                
                # Combine relevant columns with time columns if available
                if time_columns and relevant_columns:
                    kpi_column_mapping[kpi] = {
                        "metric_columns": relevant_columns,
                        "time_columns": time_columns,
                        "category": self.identify_kpi_category(kpi)
                    }
                else:
                    kpi_column_mapping[kpi] = {
                        "metric_columns": relevant_columns,
                        "time_columns": time_columns,
                        "category": self.identify_kpi_category(kpi)
                    }
            
            return kpi_column_mapping
            
        except Exception as e:
            logger.error(f"Error mapping KPIs to columns: {e}")
            return {kpi: {"metric_columns": [], "time_columns": [], "category": "general"} for kpi in kpis}
    
    def generate_kpi_visualizations(self, kpis: List[str], 
                                   data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate visualization configurations for KPIs.
        
        Args:
            kpis (list): List of KPIs to visualize.
            data_summary (dict): Summary of the data.
            
        Returns:
            list: Visualization configurations for the KPIs.
        """
        try:
            # Map KPIs to columns
            kpi_column_mapping = self.map_kpis_to_columns(kpis, data_summary)
            
            visualizations = []
            
            for kpi, mapping in kpi_column_mapping.items():
                metric_columns = mapping["metric_columns"]
                time_columns = mapping["time_columns"]
                category = mapping["category"]
                
                if not metric_columns:
                    logger.warning(f"No relevant columns found for KPI: {kpi}")
                    continue
                
                # Generate time-based visualizations if time columns are available
                if time_columns:
                    time_col = time_columns[0]  # Use the first time column
                    
                    for metric_col in metric_columns[:2]:  # Limit to first 2 metric columns
                        # Suggest visualization types
                        viz_types = self.suggest_visualizations_for_kpi(kpi, time_based=True)
                        
                        # Create visualization configuration
                        visualizations.append({
                            "title": f"{kpi} Over Time",
                            "chart_type": viz_types[0],  # Use the first suggested type
                            "data_columns": [time_col, metric_col],
                            "kpi": kpi,
                            "rationale": f"This visualization shows how {metric_col} changes over time, providing insights into {kpi} trends and patterns."
                        })
                        
                        # Add a second visualization type if available
                        if len(viz_types) > 1:
                            visualizations.append({
                                "title": f"{kpi} Trend Analysis",
                                "chart_type": viz_types[1],
                                "data_columns": [time_col, metric_col],
                                "kpi": kpi,
                                "rationale": f"This visualization provides an alternative view of {kpi} trends, emphasizing the cumulative or comparative aspects of {metric_col} over time."
                            })
                
                # Generate category-based visualizations
                categorical_columns = []
                for col in data_summary.get("columns", []):
                    if data_summary.get("data_types", {}).get(col) in ["object", "category"]:
                        categorical_columns.append(col)
                
                if categorical_columns and metric_columns:
                    cat_col = categorical_columns[0]  # Use the first categorical column
                    metric_col = metric_columns[0]  # Use the first metric column
                    
                    # Suggest visualization types
                    viz_types = self.suggest_visualizations_for_kpi(kpi, time_based=False)
                    
                    # Create visualization configuration
                    visualizations.append({
                        "title": f"{kpi} by {cat_col}",
                        "chart_type": viz_types[0],  # Use the first suggested type
                        "data_columns": [cat_col, metric_col],
                        "kpi": kpi,
                        "rationale": f"This visualization compares {metric_col} across different {cat_col} categories, providing insights into how {kpi} varies by {cat_col}."
                    })
                    
                    # Add a second visualization type if available
                    if len(viz_types) > 1 and len(categorical_columns) > 1:
                        cat_col2 = categorical_columns[1]  # Use the second categorical column
                        visualizations.append({
                            "title": f"{kpi} by {cat_col2}",
                            "chart_type": viz_types[1],
                            "data_columns": [cat_col2, metric_col],
                            "kpi": kpi,
                            "rationale": f"This visualization provides an alternative perspective on {kpi}, showing how {metric_col} varies by {cat_col2}."
                        })
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating KPI visualizations: {e}")
            return []
    
    def create_kpi_dashboard(self, kpis: List[str], 
                            data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete dashboard focused on specific KPIs.
        
        Args:
            kpis (list): List of KPIs to focus on.
            data_summary (dict): Summary of the data.
            
        Returns:
            dict: Dashboard configuration focused on the KPIs.
        """
        try:
            # Generate visualizations for KPIs
            visualizations = self.generate_kpi_visualizations(kpis, data_summary)
            
            if not visualizations:
                logger.warning("No visualizations generated for KPIs")
                return None
            
            # Group visualizations by KPI
            kpi_groups = {}
            for viz in visualizations:
                kpi = viz.get("kpi", "Unknown")
                if kpi not in kpi_groups:
                    kpi_groups[kpi] = []
                kpi_groups[kpi].append(viz)
            
            # Create dashboard configuration
            dashboard = {
                "title": "KPI Performance Dashboard",
                "purpose": f"Monitor and analyze key performance indicators: {', '.join(kpis)}",
                "layout": "Grid layout with KPI sections",
                "visualizations": visualizations,
                "interactive_elements": [
                    {
                        "type": "Date Range Filter",
                        "purpose": "Filter data by time period",
                        "implementation": "Date picker component that updates all visualizations"
                    },
                    {
                        "type": "KPI Selector",
                        "purpose": "Focus on specific KPIs",
                        "implementation": "Dropdown or toggle buttons to show/hide KPI sections"
                    }
                ],
                "color_scheme": "Professional with distinct colors for each KPI",
                "design_considerations": "Organized by KPI with clear section headers and consistent visualization styles",
                "data_arrangement_strategy": "Grouped by KPI with time-based visualizations first, followed by categorical breakdowns"
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating KPI dashboard: {e}")
            return None
