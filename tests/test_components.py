import unittest
import os
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path
import sys
sys.path.append('/home/ubuntu/dashboard_generator')

from src.api_key_manager import APIKeyManager
from src.csv_processor import CSVDataProcessor
from src.data_transformer import DataTransformer
from src.openai_client import OpenAIClient
from src.prompt_engineering import PromptEngineering
from src.dashboard_generator import DashboardDesignGenerator
from src.design_rationale import DesignRationaleGenerator
from src.kpi_mapper import KPIMapper
from src.visualization_components import VisualizationComponents
from src.interactive_components import InteractiveComponents, ResponsiveLayouts


class TestAPIKeyManager(unittest.TestCase):
    """Test cases for the API Key Manager module."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key_manager = APIKeyManager()
        # Create a temporary key file path for testing
        self.test_key_file = os.path.join(os.getcwd(), 'test_api_key.txt')
        self.api_key_manager.key_file = self.test_key_file
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test key file if it exists
        if os.path.exists(self.test_key_file):
            os.remove(self.test_key_file)
    
    def test_save_and_load_api_key(self):
        """Test saving and loading an API key."""
        test_key = "test_openai_api_key_12345"
        
        # Save the key
        result = self.api_key_manager.save_api_key(test_key)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.test_key_file))
        
        # Load the key
        loaded_key = self.api_key_manager.load_api_key()
        self.assertEqual(loaded_key, test_key)
    
    def test_is_api_key_set(self):
        """Test checking if API key is set."""
        # Initially, key should not be set
        self.assertFalse(self.api_key_manager.is_api_key_set())
        
        # Set a key
        test_key = "test_openai_api_key_12345"
        self.api_key_manager.save_api_key(test_key)
        
        # Now key should be set
        self.assertTrue(self.api_key_manager.is_api_key_set())


class TestCSVProcessor(unittest.TestCase):
    """Test cases for the CSV Processor module."""
    
    def setUp(self):
        """Set up test environment."""
        self.csv_processor = CSVDataProcessor()
        
        # Create a test CSV file
        self.test_csv_path = os.path.join(os.getcwd(), 'test_data.csv')
        test_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'sales': [100, 150, 120, 200, 180],
            'profit': [20, 30, 25, 40, 35],
            'region': ['North', 'South', 'North', 'East', 'West']
        })
        test_df.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test CSV file if it exists
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    def test_load_csv(self):
        """Test loading a CSV file."""
        result = self.csv_processor.load_csv(self.test_csv_path)
        self.assertTrue(result)
        self.assertIsNotNone(self.csv_processor.df)
        self.assertEqual(len(self.csv_processor.df), 5)
        self.assertEqual(len(self.csv_processor.df.columns), 5)
    
    def test_validate_data(self):
        """Test data validation."""
        self.csv_processor.load_csv(self.test_csv_path)
        validation_results = self.csv_processor.validate_data()
        self.assertTrue(validation_results["valid"])
        self.assertEqual(len(validation_results["issues"]), 0)
    
    def test_clean_data(self):
        """Test data cleaning."""
        self.csv_processor.load_csv(self.test_csv_path)
        result = self.csv_processor.clean_data()
        self.assertTrue(result)
        # Check that date column is converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.csv_processor.df['date']))
    
    def test_analyze_data(self):
        """Test data analysis."""
        self.csv_processor.load_csv(self.test_csv_path)
        self.csv_processor.clean_data()
        analysis = self.csv_processor.analyze_data()
        
        # Check analysis results
        self.assertEqual(analysis["row_count"], 5)
        self.assertEqual(analysis["column_count"], 5)
        self.assertIn("data_types", analysis)
        self.assertIn("statistics", analysis)
        
        # Check data types
        self.assertEqual(analysis["data_types"]["category"], "category")
        self.assertEqual(analysis["data_types"]["sales"], "int")
        self.assertEqual(analysis["data_types"]["profit"], "int")
    
    def test_get_visualization_recommendations(self):
        """Test visualization recommendations."""
        self.csv_processor.load_csv(self.test_csv_path)
        self.csv_processor.clean_data()
        self.csv_processor.analyze_data()
        recommendations = self.csv_processor.get_visualization_recommendations()
        
        # Check recommendations
        self.assertIn("time_series", recommendations)
        self.assertIn("categorical", recommendations)
        self.assertIn("numeric", recommendations)
        self.assertIn("correlations", recommendations)


class TestDataTransformer(unittest.TestCase):
    """Test cases for the Data Transformer module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test dataframe
        self.test_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B'],
            'sales': [100, 150, 120, 200, 180, 110, 160, 210, 130, 170],
            'profit': [20, 30, 25, 40, 35, 22, 32, 42, 26, 34]
        })
        
        self.data_transformer = DataTransformer()
        self.data_transformer.set_dataframe(self.test_df)
    
    def test_aggregate_data(self):
        """Test data aggregation."""
        # Aggregate by category
        agg_df = self.data_transformer.aggregate_data(
            group_by='category',
            agg_columns=['sales', 'profit'],
            agg_functions=['sum', 'mean']
        )
        
        # Check results
        self.assertEqual(len(agg_df), 3)  # 3 unique categories
        self.assertIn('sales_sum', agg_df.columns)
        self.assertIn('profit_mean', agg_df.columns)
        
        # Check specific aggregation values
        category_a = agg_df[agg_df.index == 'A']
        self.assertEqual(category_a['sales_sum'].iloc[0], 460)  # Sum of sales for category A
    
    def test_filter_data(self):
        """Test data filtering."""
        # Filter by sales > 150
        filtered_df = self.data_transformer.filter_data(
            column='sales',
            operator='>',
            value=150
        )
        
        # Check results
        self.assertEqual(len(filtered_df), 4)  # 4 rows with sales > 150
        self.assertTrue(all(filtered_df['sales'] > 150))
    
    def test_pivot_data(self):
        """Test data pivoting."""
        # Pivot data
        pivot_df = self.data_transformer.pivot_data(
            index='date',
            columns='category',
            values='sales'
        )
        
        # Check results
        self.assertEqual(len(pivot_df), 10)  # 10 unique dates
        self.assertEqual(len(pivot_df.columns), 3)  # 3 unique categories
        self.assertIn('A', pivot_df.columns)
        self.assertIn('B', pivot_df.columns)
        self.assertIn('C', pivot_df.columns)
    
    def test_calculate_growth_rate(self):
        """Test growth rate calculation."""
        # Calculate growth rate for sales
        growth_df = self.data_transformer.calculate_growth_rate(
            column='sales',
            periods=1
        )
        
        # Check results
        self.assertEqual(len(growth_df), len(self.test_df) - 1)  # One less row due to diff
        self.assertIn('sales_growth', growth_df.columns)
        
        # Check specific growth rate values
        first_growth = growth_df['sales_growth'].iloc[0]
        expected_growth = (150 - 100) / 100  # (sales[1] - sales[0]) / sales[0]
        self.assertAlmostEqual(first_growth, expected_growth)


class TestDashboardGenerator(unittest.TestCase):
    """Test cases for the Dashboard Generator module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock OpenAI client
        self.mock_openai_client = MagicMock()
        self.mock_openai_client.initialize.return_value = True
        self.mock_openai_client.generate_dashboard_design.return_value = {
            "raw_content": json.dumps({
                "dashboards": [
                    {
                        "title": "Sales Performance Dashboard",
                        "purpose": "Monitor sales performance across regions and categories",
                        "visualizations": [
                            {
                                "title": "Sales by Region",
                                "chart_type": "bar",
                                "data_columns": ["region", "sales"]
                            },
                            {
                                "title": "Sales Trend",
                                "chart_type": "line",
                                "data_columns": ["date", "sales"]
                            }
                        ]
                    }
                ]
            })
        }
        
        # Create a test dataframe
        self.test_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B'],
            'sales': [100, 150, 120, 200, 180, 110, 160, 210, 130, 170],
            'profit': [20, 30, 25, 40, 35, 22, 32, 42, 26, 34],
            'region': ['North', 'South', 'North', 'East', 'West', 'North', 'South', 'East', 'West', 'North']
        })
        
        # Create a mock CSV processor
        self.mock_csv_processor = MagicMock()
        self.mock_csv_processor.df = self.test_df
        self.mock_csv_processor.analyze_data.return_value = {
            "row_count": 10,
            "column_count": 5,
            "data_types": {
                "date": "datetime",
                "category": "category",
                "sales": "int",
                "profit": "int",
                "region": "category"
            },
            "columns": ["date", "category", "sales", "profit", "region"]
        }
        self.mock_csv_processor.get_visualization_recommendations.return_value = {
            "time_series": ["date"],
            "categorical": ["category", "region"],
            "numeric": ["sales", "profit"]
        }
        
        # Create the dashboard generator with mocks
        self.dashboard_generator = DashboardDesignGenerator()
        self.dashboard_generator.openai_client = self.mock_openai_client
        self.dashboard_generator.csv_processor = self.mock_csv_processor
        self.dashboard_generator.data_transformer.set_dataframe(self.test_df)
    
    def test_initialize(self):
        """Test initialization of the dashboard generator."""
        result = self.dashboard_generator.initialize()
        self.assertTrue(result)
        self.mock_openai_client.initialize.assert_called_once()
    
    def test_generate_dashboard_designs(self):
        """Test generating dashboard designs."""
        requirements = {
            "title": "Sales Dashboard",
            "purpose": "Monitor sales performance",
            "kpis": ["Sales Growth", "Profit Margin"],
            "num_dashboards": 1
        }
        
        designs = self.dashboard_generator.generate_dashboard_designs(requirements)
        
        # Check results
        self.assertIsNotNone(designs)
        self.assertIn("dashboards", designs)
        self.assertEqual(len(designs["dashboards"]), 1)
        
        # Check dashboard content
        dashboard = designs["dashboards"][0]
        self.assertEqual(dashboard["title"], "Sales Performance Dashboard")
        self.assertIn("visualizations", dashboard)
        self.assertEqual(len(dashboard["visualizations"]), 2)
        
        # Check visualization content
        viz = dashboard["visualizations"][0]
        self.assertEqual(viz["title"], "Sales by Region")
        self.assertEqual(viz["chart_type"], "bar")
        self.assertEqual(viz["data_columns"], ["region", "sales"])
    
    def test_save_dashboard_designs(self):
        """Test saving dashboard designs to a file."""
        # Generate designs first
        requirements = {
            "title": "Sales Dashboard",
            "purpose": "Monitor sales performance",
            "kpis": ["Sales Growth", "Profit Margin"],
            "num_dashboards": 1
        }
        
        self.dashboard_generator.generate_dashboard_designs(requirements)
        
        # Save designs
        test_file_path = os.path.join(os.getcwd(), 'test_designs.json')
        result = self.dashboard_generator.save_dashboard_designs(test_file_path)
        
        # Check results
        self.assertTrue(result)
        self.assertTrue(os.path.exists(test_file_path))
        
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


class TestVisualizationComponents(unittest.TestCase):
    """Test cases for the Visualization Components module."""
    
    def setUp(self):
        """Set up test environment."""
        self.viz_components = VisualizationComponents()
        
        # Create a test dataframe
        self.test_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'category': ['A', 'B', 'A', 'C', 'B'],
            'sales': [100, 150, 120, 200, 180],
            'profit': [20, 30, 25, 40, 35]
        })
    
    def test_create_bar_chart(self):
        """Test creating a bar chart."""
        fig = self.viz_components.create_bar_chart(
            self.test_df,
            x_col='category',
            y_col='sales',
            title='Sales by Category'
        )
        
        # Check figure properties
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, 'Sales by Category')
        self.assertEqual(fig.layout.xaxis.title.text, 'category')
        self.assertEqual(fig.layout.yaxis.title.text, 'sales')
    
    def test_create_line_chart(self):
        """Test creating a line chart."""
        fig = self.viz_components.create_line_chart(
            self.test_df,
            x_col='date',
            y_col='sales',
            title='Sales Trend'
        )
        
        # Check figure properties
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, 'Sales Trend')
        self.assertEqual(fig.layout.xaxis.title.text, 'date')
        self.assertEqual(fig.layout.yaxis.title.text, 'sales')
    
    def test_create_pie_chart(self):
        """Test creating a pie chart."""
        # Aggregate data for pie chart
        agg_df = self.test_df.groupby('category')['sales'].sum().reset_index()
        
        fig = self.viz_components.create_pie_chart(
            agg_df,
            names_col='category',
            values_col='sales',
            title='Sales Distribution'
        )
        
        # Check figure properties
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, 'Sales Distribution')
    
    def test_create_scatter_plot(self):
        """Test creating a scatter plot."""
        fig = self.viz_components.create_scatter_plot(
            self.test_df,
            x_col='sales',
            y_col='profit',
            title='Sales vs Profit'
        )
        
        # Check figure properties
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, 'Sales vs Profit')
        self.assertEqual(fig.layout.xaxis.title.text, 'sales')
        self.assertEqual(fig.layout.yaxis.title.text, 'profit')


class TestKPIMapper(unittest.TestCase):
    """Test cases for the KPI Mapper module."""
    
    def setUp(self):
        """Set up test environment."""
        self.kpi_mapper = KPIMapper()
        
        # Create a test data summary
        self.data_summary = {
            "row_count": 10,
            "column_count": 5,
            "columns": ["date", "category", "sales", "profit", "region"],
            "data_types": {
                "date": "datetime",
                "category": "category",
                "sales": "int",
                "profit": "int",
                "region": "category"
            },
            "statistics": {
                "sales": {
                    "min": 100,
                    "max": 210,
                    "mean": 153,
                    "median": 155
                },
                "profit": {
                    "min": 20,
                    "max": 42,
                    "mean": 30.6,
                    "median": 31
                }
            }
        }
    
    def test_identify_kpi_category(self):
        """Test identifying KPI categories."""
        # Test sales KPI
        category = self.kpi_mapper.identify_kpi_category("Sales Growth")
        self.assertEqual(category, "sales")
        
        # Test financial KPI
        category = self.kpi_mapper.identify_kpi_category("Profit Margin")
        self.assertEqual(category, "financial")
        
        # Test customer KPI
        category = self.kpi_mapper.identify_kpi_category("Customer Retention")
        self.assertEqual(category, "customer")
        
        # Test unknown KPI
        category = self.kpi_mapper.identify_kpi_category("Unknown Metric")
        self.assertEqual(category, "general")
    
    def test_suggest_visualizations_for_kpi(self):
        """Test suggesting visualizations for KPIs."""
        # Test sales KPI with time-based data
        viz_types = self.kpi_mapper.suggest_visualizations_for_kpi("Sales Growth", time_based=True)
        self.assertIn("line", viz_types)
        self.assertIn("area", viz_types)
        
        # Test financial KPI without time-based data
        viz_types = self.kpi_mapper.suggest_visualizations_for_kpi("Profit Margin", time_based=False)
        self.assertIn("bar", viz_types)
    
    def test_map_kpis_to_columns(self):
        """Test mapping KPIs to data columns."""
        kpis = ["Sales Growth", "Profit Margin"]
        mapping = self.kpi_mapper.map_kpis_to_columns(kpis, self.data_summary)
        
        # Check mapping results
        self.assertIn("Sales Growth", mapping)
        self.assertIn("Profit Margin", mapping)
        
        # Check mapped columns
        self.assertIn("sales", mapping["Sales Growth"]["metric_columns"])
        self.assertIn("profit", mapping["Profit Margin"]["metric_columns"])
        
        # Check time columns
        self.assertIn("date", mapping["Sales Growth"]["time_columns"])
    
    def test_generate_kpi_visualizations(self):
        """Test generating visualizations for KPIs."""
        kpis = ["Sales Growth", "Profit Margin"]
        visualizations = self.kpi_mapper.generate_kpi_visualizations(kpis, self.data_summary)
        
        # Check visualization results
        self.assertGreater(len(visualizations), 0)
        
        # Check visualization properties
        viz = visualizations[0]
        self.assertIn("title", viz)
        self.assertIn("chart_type", viz)
        self.assertIn("data_columns", viz)
        self.assertIn("kpi", viz)
        self.assertIn("rationale", viz)


class TestDesignRationaleGenerator(unittest.TestCase):
    """Test cases for the Design Rationale Generator module."""
    
    def setUp(self):
        """Set up test environment."""
        self.rationale_generator = DesignRationaleGenerator()
        
        # Create a test data summary
        self.data_summary = {
            "data_types": {
                "date": "datetime",
                "category": "category",
                "sales": "int",
                "profit": "int"
            },
            "statistics": {
                "category": {
                    "unique_values": 3
                },
                "sales": {
                    "min": 100,
                    "max": 210,
                    "mean": 153,
                    "median": 155
                },
                "correlations": {
                    "sales": {
                        "profit": 0.95
                    }
                }
            }
        }
    
    def test_generate_bar_chart_rationale(self):
        """Test generating rationale for bar charts."""
        visualization = {
            "title": "Sales by Category",
            "chart_type": "bar",
            "data_columns": ["category", "sales"]
        }
        
        rationale = self.rationale_generator.generate_visualization_rationale(
            self.data_summary,
            visualization
        )
        
        # Check rationale content
        self.assertIsNotNone(rationale)
        self.assertGreater(len(rationale), 0)
        self.assertIn("bar chart", rationale.lower())
        self.assertIn("category", rationale.lower())
        self.assertIn("sales", rationale.lower())
    
    def test_generate_scatter_chart_rationale(self):
        """Test generating rationale for scatter plots."""
        visualization = {
            "title": "Sales vs Profit",
            "chart_type": "scatter",
            "data_columns": ["sales", "profit"]
        }
        
        rationale = self.rationale_generator.generate_visualization_rationale(
            self.data_summary,
            visualization
        )
        
        # Check rationale content
        self.assertIsNotNone(rationale)
        self.assertGreater(len(rationale), 0)
        self.assertIn("scatter plot", rationale.lower())
        self.assertIn("correlation", rationale.lower())
        self.assertIn("sales", rationale.lower())
        self.assertIn("profit", rationale.lower())
    
    def test_generate_dashboard_rationale(self):
        """Test generating rationale for a dashboard."""
        dashboard_design = {
            "title": "Sales Performance Dashboard",
            "purpose": "Monitor sales performance across categories",
            "visualizations": [
                {
                    "title": "Sales by Category",
                    "chart_type": "bar",
                    "data_columns": ["category", "sales"]
                },
                {
                    "title": "Sales vs Profit",
                    "chart_type": "scatter",
                    "data_columns": ["sales", "profit"]
                }
            ],
            "layout": "Grid layout"
        }
        
        rationale = self.rationale_generator.generate_dashboard_rationale(
            self.data_summary,
            dashboard_design
        )
        
        # Check rationale content
        self.assertIsNotNone(rationale)
        self.assertGreater(len(rationale), 0)
        self.assertIn("dashboard", rationale.lower())
        self.assertIn("sales performance dashboard", rationale.lower())
        self.assertIn("bar chart", rationale.lower())
        self.assertIn("scatter", rationale.lower())


if __name__ == '__main__':
    unittest.main()
