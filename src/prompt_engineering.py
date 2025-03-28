import json
import logging
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PromptEngineering:
    """
    Handles prompt engineering for dashboard design generation.
    Creates specialized prompts based on data characteristics and requirements.
    """
    
    def __init__(self):
        """Initialize the prompt engineering module."""
        self.visualization_types = config["visualization"]["supported_chart_types"]
        self.color_schemes = config["visualization"]["color_schemes"]
        
    def create_dashboard_design_prompt(self, data_analysis, requirements):
        """
        Create a prompt for dashboard design generation based on data analysis and requirements.
        
        Args:
            data_analysis (dict): Analysis of the CSV data including columns, types, and statistics.
            requirements (dict): User requirements for the dashboard design.
            
        Returns:
            dict: A structured prompt for the OpenAI API.
        """
        try:
            # Extract key information from data analysis
            columns = data_analysis.get("columns", [])
            data_types = data_analysis.get("data_types", {})
            statistics = data_analysis.get("statistics", {})
            correlations = data_analysis.get("correlations", {})
            
            # Extract requirements
            theme = requirements.get("theme", "business")
            num_dashboards = requirements.get("num_dashboards", 1)
            kpis = requirements.get("kpis", [])
            audience = requirements.get("audience", "general")
            
            # Determine appropriate visualization types based on data characteristics
            suggested_visualizations = self._suggest_visualizations(data_types, correlations)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt
            user_prompt = self._create_user_prompt(
                columns, 
                data_types, 
                statistics, 
                suggested_visualizations,
                theme,
                num_dashboards,
                kpis,
                audience
            )
            
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard design prompt: {e}")
            return {
                "system_prompt": "You are a dashboard design expert.",
                "user_prompt": f"Design {num_dashboards} dashboard(s) based on the provided data."
            }
    
    def _create_system_prompt(self):
        """
        Create the system prompt for dashboard design generation.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert data visualization and dashboard design specialist with years of experience.
Your task is to create detailed, professional dashboard designs based on CSV data analysis.

For each dashboard, you must provide:
1. A clear layout description with precise visualization placements
2. Specific chart types for each visualization with detailed rationale
3. Recommendations for interactive elements and filters that enhance user experience
4. Color scheme and design considerations that align with the theme and audience
5. Data arrangement strategy to tell a compelling and insightful story

Your designs should prioritize:
- Clarity and ease of understanding
- Visual appeal and professional aesthetics
- Effective data communication
- Alignment with business objectives and KPIs
- Appropriate interactivity for the target audience

Respond in a structured JSON format that includes all required elements and can be parsed programmatically."""
    
    def _create_user_prompt(self, columns, data_types, statistics, suggested_visualizations, 
                           theme, num_dashboards, kpis, audience):
        """
        Create the user prompt for dashboard design generation.
        
        Args:
            columns (list): List of column names in the CSV data.
            data_types (dict): Data types for each column.
            statistics (dict): Statistical summary of the data.
            suggested_visualizations (dict): Suggested visualization types for the data.
            theme (str): Theme for the dashboard design.
            num_dashboards (int): Number of dashboards to generate.
            kpis (list): List of KPIs to focus on.
            audience (str): Target audience for the dashboard.
            
        Returns:
            str: The user prompt.
        """
        # Format columns and data types for better readability
        columns_info = []
        for col in columns:
            col_type = data_types.get(col, "unknown")
            col_stats = statistics.get(col, {})
            col_info = f"- {col} (Type: {col_type})"
            
            # Add statistics if available
            if col_stats:
                stats_str = ", ".join([f"{k}: {v}" for k, v in col_stats.items()])
                col_info += f" | Stats: {stats_str}"
                
            columns_info.append(col_info)
        
        # Format suggested visualizations
        vis_suggestions = []
        for data_category, vis_types in suggested_visualizations.items():
            vis_suggestions.append(f"- For {data_category} data: {', '.join(vis_types)}")
        
        # Create the prompt sections
        dashboard_plural = "s" if num_dashboards > 1 else ""
        kpis_str = ", ".join(kpis) if kpis else "Not specified"
        
        # Build the prompt in parts to avoid nested f-string issues
        prompt_parts = [
            f"I need you to design {num_dashboards} distinct dashboard{dashboard_plural} for {theme} data visualization.",
            "\nDATA SUMMARY:",
            "The CSV data contains the following columns:",
            "\n".join(columns_info),
            "\nSUGGESTED VISUALIZATIONS:",
            "Based on the data characteristics, consider these visualization types:",
            "\n".join(vis_suggestions),
            "\nDESIGN REQUIREMENTS:",
            f"- Theme: {theme}",
            f"- Target Audience: {audience}",
            f"- Number of Dashboards: {num_dashboards}",
            f"- Key Performance Indicators (KPIs): {kpis_str}",
            "\nData Arrangement Considerations:",
            "- Consider the best way to arrange the data visualizations in each dashboard to tell a clear and compelling story.",
            "- Ensure the dashboards are visually appealing and easy to understand.",
            "- Prioritize clarity and conciseness.",
            "- Consider data filtering and interactive elements where appropriate.",
            "\nFor each dashboard, please provide:",
            "1. Dashboard Title and Purpose",
            "2. Layout Description (detailed placement of visualizations)",
            "3. List of Visualizations with:",
            "   - Chart type",
            "   - Data columns used",
            "   - Rationale for this visualization choice",
            "4. Interactive Elements and Filters",
            "5. Color Scheme and Design Considerations",
            "6. Data Arrangement Strategy",
            "\nPlease format your response as a JSON object with the following structure:",
            "```json",
            "{",
            "  \"dashboards\": [",
            "    {",
            "      \"title\": \"Dashboard Title\",",
            "      \"purpose\": \"Dashboard purpose description\",",
            "      \"layout\": \"Detailed layout description\",",
            "      \"visualizations\": [",
            "        {",
            "          \"title\": \"Visualization Title\",",
            "          \"chart_type\": \"Chart type\",",
            "          \"data_columns\": [\"Column1\", \"Column2\"],",
            "          \"rationale\": \"Rationale for this visualization\"",
            "        }",
            "      ],",
            "      \"interactive_elements\": [",
            "        {",
            "          \"type\": \"Element type\",",
            "          \"purpose\": \"Purpose of this element\",",
            "          \"implementation\": \"How it should be implemented\"",
            "        }",
            "      ],",
            "      \"color_scheme\": \"Color scheme description\",",
            "      \"design_considerations\": \"Design considerations\",",
            "      \"data_arrangement_strategy\": \"Data arrangement strategy\"",
            "    }",
            "  ]",
            "}",
            "```"
        ]
        
        return "\n".join(prompt_parts)
    
    def _suggest_visualizations(self, data_types, correlations):
        """
        Suggest visualization types based on data characteristics.
        
        Args:
            data_types (dict): Data types for each column.
            correlations (dict): Correlations between columns.
            
        Returns:
            dict: Suggested visualization types for different data categories.
        """
        suggestions = {
            "categorical": [],
            "numerical": [],
            "temporal": [],
            "relationships": [],
            "distributions": [],
            "comparisons": []
        }
        
        # Count data types
        num_categorical = sum(1 for t in data_types.values() if t in ["object", "string", "category"])
        num_numerical = sum(1 for t in data_types.values() if t in ["int", "float", "integer", "number"])
        num_temporal = sum(1 for t in data_types.values() if t in ["datetime", "date", "time"])
        
        # Suggest visualizations for categorical data
        if num_categorical > 0:
            suggestions["categorical"] = ["bar", "pie", "heatmap"]
            
        # Suggest visualizations for numerical data
        if num_numerical > 0:
            suggestions["numerical"] = ["line", "area", "scatter"]
            
        # Suggest visualizations for temporal data
        if num_temporal > 0:
            suggestions["temporal"] = ["line", "area", "bar"]
            
        # Suggest visualizations for relationships
        if num_numerical >= 2:
            suggestions["relationships"] = ["scatter", "heatmap", "bubble"]
            
        # Suggest visualizations for distributions
        if num_numerical > 0:
            suggestions["distributions"] = ["histogram", "box", "violin"]
            
        # Suggest visualizations for comparisons
        if num_categorical > 0 and num_numerical > 0:
            suggestions["comparisons"] = ["bar", "radar", "funnel"]
            
        return suggestions
        
    def parse_dashboard_design_response(self, response_content):
        """
        Parse the response from the OpenAI API for dashboard design generation.
        
        Args:
            response_content (str): The content from the OpenAI API response.
            
        Returns:
            dict: The parsed dashboard design or None if parsing failed.
        """
        try:
            # Extract JSON from the response content
            # This handles cases where the response might contain markdown or other text
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in response content")
                return None
                
            json_content = response_content[json_start:json_end]
            
            # Parse the JSON content
            dashboard_design = json.loads(json_content)
            
            # Initialize the final structure
            final_design = {"dashboards": []}
            
            # Handle different response structures
            if isinstance(dashboard_design, dict):
                if "dashboards" in dashboard_design:
                    # Already in correct format
                    final_design = dashboard_design
                elif "title" in dashboard_design:
                    # Single dashboard object
                    final_design["dashboards"].append(dashboard_design)
                else:
                    # Try to find dashboard-like objects in the dictionary
                    for key, value in dashboard_design.items():
                        if isinstance(value, dict) and "title" in value:
                            final_design["dashboards"].append(value)
            elif isinstance(dashboard_design, list):
                # List of dashboard objects
                for item in dashboard_design:
                    if isinstance(item, dict) and "title" in item:
                        final_design["dashboards"].append(item)
            
            # If no dashboards were found, try to create one from the available data
            if not final_design["dashboards"]:
                logger.warning("No valid dashboard structure found, attempting to create one")
                dashboard = {
                    "title": "Data Analysis Dashboard",
                    "purpose": "Visualize and analyze the data",
                    "layout": "Grid layout with multiple visualizations",
                    "visualizations": []
                }
                
                # Try to extract visualizations from the response
                if isinstance(dashboard_design, dict):
                    for key, value in dashboard_design.items():
                        if isinstance(value, dict) and "chart_type" in value:
                            dashboard["visualizations"].append(value)
                
                final_design["dashboards"].append(dashboard)
            
            # Validate and fix each dashboard
            for dashboard in final_design["dashboards"]:
                # Ensure required fields exist with defaults
                dashboard.setdefault("title", "Untitled Dashboard")
                dashboard.setdefault("purpose", "Data visualization and analysis")
                dashboard.setdefault("layout", "Grid layout")
                dashboard.setdefault("visualizations", [])
                dashboard.setdefault("interactive_elements", [])
                dashboard.setdefault("color_scheme", "Default color scheme")
                dashboard.setdefault("design_considerations", "Standard design considerations")
                dashboard.setdefault("data_arrangement_strategy", "Standard data arrangement")
                
                # Ensure visualizations is a list
                if not isinstance(dashboard["visualizations"], list):
                    dashboard["visualizations"] = []
                
                # Fix each visualization
                for viz in dashboard["visualizations"]:
                    if not isinstance(viz, dict):
                        continue
                        
                    viz.setdefault("title", "Untitled Visualization")
                    viz.setdefault("chart_type", "bar")
                    viz.setdefault("data_columns", [])
                    viz.setdefault("rationale", "Standard visualization")
            
            return final_design
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing dashboard design response: {e}")
            return None
