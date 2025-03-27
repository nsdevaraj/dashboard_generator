import os
import logging
from openai import OpenAI
from src.api_key_manager import APIKeyManager
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for interacting with the OpenAI API.
    Handles API requests and response processing for dashboard design generation.
    """
    
    def __init__(self):
        """Initialize the OpenAI client with API key from the key manager."""
        self.key_manager = APIKeyManager()
        self.client = None
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
    def initialize(self):
        """
        Initialize the OpenAI client with the API key.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        api_key = self.key_manager.get_openai_api_key()
        
        if not api_key:
            logger.error("Failed to initialize OpenAI client: No API key available")
            return False
            
        try:
            self.client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def generate_dashboard_design(self, csv_data_summary, design_requirements):
        """
        Generate dashboard design based on CSV data summary and requirements.
        
        Args:
            csv_data_summary (dict): Summary of the CSV data including columns, data types, and statistics.
            design_requirements (dict): Requirements for the dashboard design.
            
        Returns:
            dict: The generated dashboard design or None if generation failed.
        """
        if not self.client:
            if not self.initialize():
                logger.error("Cannot generate dashboard design: OpenAI client not initialized")
                return None
        
        try:
            # Construct the prompt for dashboard design generation
            prompt = self._construct_dashboard_design_prompt(csv_data_summary, design_requirements)
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Process and return the response
            return self._process_dashboard_design_response(response)
            
        except Exception as e:
            logger.error(f"Error generating dashboard design: {e}")
            return None
    
    def _get_system_prompt(self):
        """
        Get the system prompt for dashboard design generation.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert data visualization and dashboard design specialist. 
Your task is to create detailed, professional dashboard designs based on CSV data analysis.
For each dashboard, provide:
1. A clear layout description with visualization placements
2. Specific chart types for each visualization with rationale
3. Recommendations for interactive elements and filters
4. Color scheme and design considerations
5. Data arrangement strategy to tell a compelling story

Your designs should prioritize clarity, visual appeal, and effective data communication.
Respond in a structured JSON format that can be parsed programmatically."""
    
    def _construct_dashboard_design_prompt(self, csv_data_summary, design_requirements):
        """
        Construct the prompt for dashboard design generation.
        
        Args:
            csv_data_summary (dict): Summary of the CSV data.
            design_requirements (dict): Requirements for the dashboard design.
            
        Returns:
            str: The constructed prompt.
        """
        # Extract key information from the CSV data summary
        columns = csv_data_summary.get("columns", [])
        data_types = csv_data_summary.get("data_types", {})
        statistics = csv_data_summary.get("statistics", {})
        sample_data = csv_data_summary.get("sample_data", [])
        
        # Extract design requirements
        theme = design_requirements.get("theme", "business")
        num_dashboards = design_requirements.get("num_dashboards", 1)
        kpis = design_requirements.get("kpis", [])
        audience = design_requirements.get("audience", "general")
        
        # Construct the prompt
        prompt = f"""
I need you to design {num_dashboards} distinct dashboard{'s' if num_dashboards > 1 else ''} for {theme} data visualization.

CSV DATA SUMMARY:
Columns: {', '.join(columns)}
Data Types: {data_types}
Statistics: {statistics}

Sample Data:
{sample_data[:5] if sample_data else 'No sample data available'}

DESIGN REQUIREMENTS:
Theme: {theme}
Target Audience: {audience}
Key Performance Indicators (KPIs): {', '.join(kpis) if kpis else 'Not specified'}

For each dashboard, please provide:
1. Dashboard Title and Purpose
2. Layout Description (placement of visualizations)
3. List of Visualizations with:
   - Chart type
   - Data columns used
   - Rationale for this visualization
4. Interactive Elements and Filters
5. Color Scheme and Design Considerations
6. Data Arrangement Strategy

Please format your response as a JSON object that can be parsed programmatically.
"""
        return prompt
    
    def _process_dashboard_design_response(self, response):
        """
        Process the response from the OpenAI API for dashboard design generation.
        
        Args:
            response: The response from the OpenAI API.
            
        Returns:
            dict: The processed dashboard design.
        """
        if not response or not hasattr(response, 'choices') or not response.choices:
            logger.error("Invalid response from OpenAI API")
            return None
            
        try:
            # Extract the content from the response
            content = response.choices[0].message.content
            
            # TODO: Parse the JSON content and validate the structure
            # For now, return the raw content
            return {"raw_content": content}
            
        except Exception as e:
            logger.error(f"Error processing dashboard design response: {e}")
            return None
