import os
import json
import logging
from src.api_key_manager import APIKeyManager
from src.openai_client import OpenAIClient
from src.prompt_engineering import PromptEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_connectivity():
    """
    Test the connectivity to the OpenAI API.
    
    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    logger.info("Testing OpenAI API connectivity...")
    
    # Initialize the API key manager
    key_manager = APIKeyManager()
    api_key = key_manager.get_openai_api_key()
    
    if not api_key:
        logger.error("No API key available. Please set the OPENAI_API_KEY environment variable.")
        return False
    
    # Initialize the OpenAI client
    client = OpenAIClient()
    if not client.initialize():
        logger.error("Failed to initialize OpenAI client.")
        return False
    
    logger.info("OpenAI client initialized successfully.")
    return True

def test_dashboard_design_generation():
    """
    Test the dashboard design generation functionality.
    
    Returns:
        bool: True if the test is successful, False otherwise.
    """
    logger.info("Testing dashboard design generation...")
    
    # Sample CSV data summary
    csv_data_summary = {
        "columns": ["Date", "Product", "Region", "Sales", "Profit", "Units", "Customer_Satisfaction"],
        "data_types": {
            "Date": "datetime",
            "Product": "object",
            "Region": "object",
            "Sales": "float",
            "Profit": "float",
            "Units": "int",
            "Customer_Satisfaction": "float"
        },
        "statistics": {
            "Sales": {"min": 100, "max": 5000, "mean": 1200, "median": 950},
            "Profit": {"min": -200, "max": 2000, "mean": 400, "median": 350},
            "Units": {"min": 1, "max": 100, "mean": 25, "median": 20},
            "Customer_Satisfaction": {"min": 1, "max": 5, "mean": 4.2, "median": 4.5}
        },
        "sample_data": [
            {"Date": "2025-01-01", "Product": "Widget A", "Region": "North", "Sales": 1200, "Profit": 400, "Units": 30, "Customer_Satisfaction": 4.5},
            {"Date": "2025-01-02", "Product": "Widget B", "Region": "South", "Sales": 950, "Profit": 350, "Units": 25, "Customer_Satisfaction": 4.2},
            {"Date": "2025-01-03", "Product": "Widget C", "Region": "East", "Sales": 1500, "Profit": 600, "Units": 40, "Customer_Satisfaction": 4.8}
        ]
    }
    
    # Sample design requirements
    design_requirements = {
        "theme": "sales performance",
        "num_dashboards": 2,
        "kpis": ["Total Sales", "Profit Margin", "Customer Satisfaction"],
        "audience": "executive team"
    }
    
    try:
        # Initialize the prompt engineering module
        prompt_engineering = PromptEngineering()
        
        # Create the prompt
        prompt = prompt_engineering.create_dashboard_design_prompt(csv_data_summary, design_requirements)
        
        if not prompt:
            logger.error("Failed to create dashboard design prompt.")
            return False
        
        logger.info("Successfully created dashboard design prompt.")
        
        # Initialize the OpenAI client
        client = OpenAIClient()
        if not client.initialize():
            logger.error("Failed to initialize OpenAI client.")
            return False
        
        # Generate the dashboard design
        dashboard_design = client.generate_dashboard_design(csv_data_summary, design_requirements)
        
        if not dashboard_design:
            logger.error("Failed to generate dashboard design.")
            return False
        
        # Parse the response
        parsed_design = prompt_engineering.parse_dashboard_design_response(dashboard_design.get("raw_content", ""))
        
        if not parsed_design:
            logger.error("Failed to parse dashboard design response.")
            return False
        
        # Save the result to a file for inspection
        with open("test_dashboard_design.json", "w") as f:
            json.dump(parsed_design, f, indent=2)
        
        logger.info("Successfully generated and parsed dashboard design. Result saved to test_dashboard_design.json")
        return True
        
    except Exception as e:
        logger.error(f"Error testing dashboard design generation: {e}")
        return False

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        print("Example: export OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Test API connectivity
    if test_api_connectivity():
        print("API connectivity test passed.")
    else:
        print("API connectivity test failed.")
        exit(1)
    
    # Test dashboard design generation
    if test_dashboard_design_generation():
        print("Dashboard design generation test passed.")
    else:
        print("Dashboard design generation test failed.")
        exit(1)
    
    print("All tests passed successfully!")
