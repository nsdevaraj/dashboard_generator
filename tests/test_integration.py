import os
import sys
import unittest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append('/home/ubuntu/dashboard_generator')

from src.api_key_manager import APIKeyManager
from src.csv_processor import CSVDataProcessor
from src.dashboard_generator import DashboardDesignGenerator
from src.openai_client import OpenAIClient
from src.ui import DashboardGeneratorUI


class TestIntegration(unittest.TestCase):
    """Integration tests for the Dashboard Generator application."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test CSV file
        self.test_csv_path = os.path.join(os.getcwd(), 'test_integration_data.csv')
        test_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'sales': [100, 150, 120, 200, 180],
            'profit': [20, 30, 25, 40, 35],
            'region': ['North', 'South', 'North', 'East', 'West']
        })
        test_df.to_csv(self.test_csv_path, index=False)
        
        # Create a test API key file
        self.test_key_file = os.path.join(os.getcwd(), 'test_integration_api_key.txt')
        with open(self.test_key_file, 'w') as f:
            f.write('test_openai_api_key_12345')
        
        # Create a mock OpenAI client response
        self.mock_openai_response = {
            "raw_content": json.dumps({
                "dashboards": [
                    {
                        "title": "Sales Performance Dashboard",
                        "purpose": "Monitor sales performance across regions and categories",
                        "visualizations": [
                            {
                                "title": "Sales by Region",
                                "chart_type": "bar",
                                "data_columns": ["region", "sales"],
                                "rationale": "This bar chart effectively compares sales across different regions."
                            },
                            {
                                "title": "Sales Trend",
                                "chart_type": "line",
                                "data_columns": ["date", "sales"],
                                "rationale": "This line chart visualizes how sales change over time."
                            }
                        ],
                        "interactive_elements": [
                            {
                                "type": "Date Range Filter",
                                "purpose": "Filter data by time period"
                            }
                        ],
                        "layout": "Grid layout with 2 columns"
                    }
                ]
            })
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists(self.test_key_file):
            os.remove(self.test_key_file)
        
        # Remove any generated files
        test_files = [
            'dashboard_designs.json',
            'dashboard_1_code.py'
        ]
        for file in test_files:
            file_path = os.path.join(os.getcwd(), file)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('src.openai_client.OpenAIClient.generate_dashboard_design')
    def test_end_to_end_workflow(self, mock_generate):
        """Test the end-to-end workflow of the application."""
        # Set up mock for OpenAI API call
        mock_generate.return_value = self.mock_openai_response
        
        # 1. Initialize API key manager and set key file
        api_key_manager = APIKeyManager()
        api_key_manager.key_file = self.test_key_file
        self.assertTrue(api_key_manager.is_api_key_set())
        
        # 2. Load and process CSV data
        csv_processor = CSVDataProcessor()
        self.assertTrue(csv_processor.load_csv(self.test_csv_path))
        self.assertTrue(csv_processor.clean_data())
        data_summary = csv_processor.analyze_data()
        self.assertIsNotNone(data_summary)
        
        # 3. Initialize dashboard generator
        dashboard_generator = DashboardDesignGenerator()
        dashboard_generator.openai_client = OpenAIClient()
        dashboard_generator.csv_processor = csv_processor
        dashboard_generator.data_transformer.set_dataframe(csv_processor.df)
        self.assertTrue(dashboard_generator.initialize())
        
        # 4. Generate dashboard designs
        requirements = {
            "title": "Sales Dashboard",
            "purpose": "Monitor sales performance",
            "kpis": ["Sales Growth", "Profit Margin"],
            "num_dashboards": 1
        }
        
        designs = dashboard_generator.generate_dashboard_designs(requirements)
        self.assertIsNotNone(designs)
        self.assertIn("dashboards", designs)
        
        # 5. Save dashboard designs
        designs_file_path = os.path.join(os.getcwd(), 'dashboard_designs.json')
        self.assertTrue(dashboard_generator.save_dashboard_designs(designs_file_path))
        self.assertTrue(os.path.exists(designs_file_path))
        
        # 6. Generate layout code
        code = dashboard_generator.generate_layout_code(0, framework='dash')
        self.assertIsNotNone(code)
        self.assertIn("import dash", code)
        
        # 7. Save code to file
        code_file_path = os.path.join(os.getcwd(), 'dashboard_1_code.py')
        with open(code_file_path, 'w') as f:
            f.write(code)
        self.assertTrue(os.path.exists(code_file_path))
    
    @patch('dash.Dash')
    @patch('src.openai_client.OpenAIClient.generate_dashboard_design')
    def test_ui_initialization(self, mock_generate, mock_dash):
        """Test the initialization of the UI components."""
        # Set up mock for OpenAI API call
        mock_generate.return_value = self.mock_openai_response
        
        # Initialize UI
        ui = DashboardGeneratorUI()
        
        # Check that components are initialized
        self.assertIsNotNone(ui.api_key_manager)
        self.assertIsNotNone(ui.csv_processor)
        self.assertIsNotNone(ui.dashboard_generator)
        self.assertIsNotNone(ui.viz_components)
        self.assertIsNotNone(ui.interactive)
        self.assertIsNotNone(ui.responsive)
        
        # Check that Dash app is initialized
        self.assertIsNotNone(ui.app)
        mock_dash.assert_called_once()


if __name__ == '__main__':
    unittest.main()
