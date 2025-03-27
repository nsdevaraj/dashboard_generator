import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """
    Processes CSV data for dashboard design generation.
    Handles data loading, validation, cleaning, and analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the CSV data processor.
        
        Args:
            config (dict, optional): Configuration settings for data processing.
        """
        self.config = config or {
            "max_csv_size_mb": 50,
            "sample_rows_for_analysis": 1000,
            "auto_detect_data_types": True,
            "handle_missing_values": "auto"
        }
        self.df = None
        self.data_summary = None
        
    def load_csv(self, file_path: str) -> bool:
        """
        Load CSV data from a file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.config["max_csv_size_mb"]:
                logger.error(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.config['max_csv_size_mb']} MB)")
                return False
                
            # Load the CSV file
            self.df = pd.read_csv(file_path)
            
            logger.info(f"Successfully loaded CSV file: {file_path} with {len(self.df)} rows and {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return False
    
    def load_csv_from_string(self, csv_string: str) -> bool:
        """
        Load CSV data from a string.
        
        Args:
            csv_string (str): CSV data as a string.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            # Load the CSV data from string
            import io
            self.df = pd.read_csv(io.StringIO(csv_string))
            
            logger.info(f"Successfully loaded CSV data from string with {len(self.df)} rows and {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV data from string: {e}")
            return False
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded CSV data.
        
        Returns:
            dict: Validation results including issues found.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_csv() first.")
            return {"valid": False, "error": "No data loaded"}
            
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "info": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            }
        }
        
        # Check for empty dataframe
        if len(self.df) == 0:
            validation_results["valid"] = False
            validation_results["issues"].append("Empty dataframe (no rows)")
            
        # Check for duplicate column names
        duplicate_columns = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicate_columns:
            validation_results["warnings"].append(f"Duplicate column names found: {duplicate_columns}")
            
        # Check for missing values
        missing_values = self.df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            missing_info = {col: int(count) for col, count in columns_with_missing.items()}
            validation_results["warnings"].append(f"Missing values found in columns: {missing_info}")
            
        # Check for columns with all missing values
        all_missing = [col for col in self.df.columns if self.df[col].isnull().all()]
        if all_missing:
            validation_results["issues"].append(f"Columns with all missing values: {all_missing}")
            
        # Check for columns with all identical values
        constant_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_columns:
            validation_results["warnings"].append(f"Columns with constant values: {constant_columns}")
            
        # Check for columns with too many unique values (potential IDs)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio > 0.9 and len(self.df) > 10:
                    validation_results["warnings"].append(f"Column '{col}' may be an ID (unique ratio: {unique_ratio:.2f})")
        
        return validation_results
    
    def clean_data(self) -> bool:
        """
        Clean the loaded CSV data.
        
        Returns:
            bool: True if cleaning was successful, False otherwise.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_csv() first.")
            return False
            
        try:
            # Make a copy of the original dataframe
            df_original = self.df.copy()
            
            # Handle missing values based on configuration
            if self.config["handle_missing_values"] == "auto":
                # For numeric columns, fill with median
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                
                # For categorical columns, fill with mode
                categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                    self.df[col] = self.df[col].fillna(mode_value)
                    
                # For datetime columns, forward fill
                datetime_cols = self.df.select_dtypes(include=['datetime']).columns
                for col in datetime_cols:
                    self.df[col] = self.df[col].fillna(method='ffill')
            
            # Auto-detect and convert data types if configured
            if self.config["auto_detect_data_types"]:
                # Try to convert object columns to datetime
                for col in self.df.select_dtypes(include=['object']).columns:
                    try:
                        # Check if column might be a date
                        if self.df[col].str.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any() or \
                           self.df[col].str.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}').any():
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                            # If too many NaT values after conversion, revert
                            if self.df[col].isna().mean() > 0.3:
                                self.df[col] = df_original[col]
                    except:
                        pass
                
                # Try to convert object columns to numeric
                for col in self.df.select_dtypes(include=['object']).columns:
                    try:
                        numeric_values = pd.to_numeric(self.df[col], errors='coerce')
                        # If conversion doesn't introduce too many NaNs, keep it
                        if numeric_values.isna().mean() < 0.3:
                            self.df[col] = numeric_values
                    except:
                        pass
            
            # Remove duplicate rows
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()
            if len(self.df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(self.df)} duplicate rows")
            
            logger.info("Data cleaning completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            self.df = df_original  # Restore original data on error
            return False
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Analyze the loaded CSV data to generate a summary for dashboard design.
        
        Returns:
            dict: Data analysis summary.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_csv() first.")
            return {"error": "No data loaded"}
            
        try:
            # Initialize data summary
            self.data_summary = {
                "columns": list(self.df.columns),
                "data_types": {},
                "statistics": {},
                "correlations": {},
                "sample_data": []
            }
            
            # Get data types
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                if dtype.startswith('datetime'):
                    self.data_summary["data_types"][col] = "datetime"
                elif dtype.startswith('int'):
                    self.data_summary["data_types"][col] = "int"
                elif dtype.startswith('float'):
                    self.data_summary["data_types"][col] = "float"
                elif dtype == 'bool':
                    self.data_summary["data_types"][col] = "boolean"
                elif dtype == 'category':
                    self.data_summary["data_types"][col] = "category"
                else:
                    self.data_summary["data_types"][col] = "object"
            
            # Get statistics for numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                self.data_summary["statistics"][col] = {
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "std": float(self.df[col].std()),
                    "unique_values": int(self.df[col].nunique())
                }
            
            # Get statistics for categorical columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = self.df[col].value_counts()
                top_values = value_counts.head(5).to_dict()
                self.data_summary["statistics"][col] = {
                    "unique_values": int(self.df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in top_values.items()}
                }
            
            # Get statistics for datetime columns
            datetime_cols = self.df.select_dtypes(include=['datetime']).columns
            for col in datetime_cols:
                if not self.df[col].empty:
                    self.data_summary["statistics"][col] = {
                        "min": str(self.df[col].min()),
                        "max": str(self.df[col].max()),
                        "range_days": (self.df[col].max() - self.df[col].min()).days if not pd.isna(self.df[col].max()) and not pd.isna(self.df[col].min()) else None
                    }
            
            # Calculate correlations between numeric columns
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                # Convert to dictionary format, keeping only strong correlations
                for col1 in corr_matrix.columns:
                    self.data_summary["correlations"][col1] = {}
                    for col2 in corr_matrix.columns:
                        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.5:  # Only strong correlations
                            self.data_summary["correlations"][col1][col2] = float(corr_matrix.loc[col1, col2])
            
            # Get sample data (first few rows)
            sample_size = min(5, len(self.df))
            self.data_summary["sample_data"] = self.df.head(sample_size).to_dict(orient='records')
            
            # Make the summary JSON serializable
            self.data_summary = self._make_json_serializable(self.data_summary)
            
            logger.info("Data analysis completed successfully")
            return self.data_summary
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {"error": str(e)}
    
    def get_visualization_recommendations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate visualization recommendations based on data analysis.
        
        Returns:
            dict: Recommended visualizations for the data.
        """
        if self.data_summary is None:
            logger.error("No data analysis available. Call analyze_data() first.")
            return {"error": "No data analysis available"}
            
        try:
            recommendations = {
                "primary": [],
                "secondary": [],
                "exploratory": []
            }
            
            data_types = self.data_summary["data_types"]
            statistics = self.data_summary["statistics"]
            correlations = self.data_summary["correlations"]
            
            # Identify column categories
            numeric_cols = [col for col, dtype in data_types.items() if dtype in ["int", "float"]]
            categorical_cols = [col for col, dtype in data_types.items() if dtype in ["object", "category"]]
            datetime_cols = [col for col, dtype in data_types.items() if dtype == "datetime"]
            
            # Primary visualizations (most important for the data)
            
            # Time series visualizations if datetime columns exist
            if datetime_cols and numeric_cols:
                for date_col in datetime_cols:
                    for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        recommendations["primary"].append({
                            "type": "line",
                            "title": f"{num_col} over Time",
                            "columns": [date_col, num_col],
                            "rationale": f"Shows how {num_col} changes over time, revealing trends and patterns."
                        })
            
            # Bar charts for categorical vs numeric data
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                    for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                        if cat_col in statistics and statistics[cat_col].get("unique_values", 0) < 15:
                            recommendations["primary"].append({
                                "type": "bar",
                                "title": f"{num_col} by {cat_col}",
                                "columns": [cat_col, num_col],
                                "rationale": f"Compares {num_col} across different {cat_col} categories."
                            })
            
            # Scatter plots for correlated numeric columns
            if len(numeric_cols) >= 2:
                added_scatter = False
                for col1, corr_dict in correlations.items():
                    for col2, corr_value in corr_dict.items():
                        if abs(corr_value) > 0.7 and not added_scatter:  # Strong correlation
                            recommendations["primary"].append({
                                "type": "scatter",
                                "title": f"Correlation: {col1} vs {col2}",
                                "columns": [col1, col2],
                                "rationale": f"Shows the relationship between {col1} and {col2} (correlation: {corr_value:.2f})."
                            })
                            added_scatter = True
                            break
                    if added_scatter:
                        break
            
            # Secondary visualizations (complementary to primary)
            
            # Pie charts for categorical data with few unique values
            for cat_col in categorical_cols:
                if cat_col in statistics and statistics[cat_col].get("unique_values", 0) < 8:
                    recommendations["secondary"].append({
                        "type": "pie",
                        "title": f"Distribution of {cat_col}",
                        "columns": [cat_col],
                        "rationale": f"Shows the proportion of each {cat_col} category in the dataset."
                    })
            
            # Heatmaps for multiple categorical columns
            if len(categorical_cols) >= 2:
                for i, cat_col1 in enumerate(categorical_cols[:3]):
                    for cat_col2 in categorical_cols[i+1:min(i+3, len(categorical_cols))]:
                        if (statistics.get(cat_col1, {}).get("unique_values", 0) < 10 and 
                            statistics.get(cat_col2, {}).get("unique_values", 0) < 10):
                            recommendations["secondary"].append({
                                "type": "heatmap",
                                "title": f"Heatmap: {cat_col1} vs {cat_col2}",
                                "columns": [cat_col1, cat_col2],
                                "rationale": f"Shows the relationship between {cat_col1} and {cat_col2} categories."
                            })
            
            # Exploratory visualizations (for deeper analysis)
            
            # Histograms for numeric columns
            for num_col in numeric_cols:
                recommendations["exploratory"].append({
                    "type": "histogram",
                    "title": f"Distribution of {num_col}",
                    "columns": [num_col],
                    "rationale": f"Shows the distribution and frequency of {num_col} values."
                })
            
            # Box plots for numeric columns grouped by categories
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:2]:
                    if cat_col in statistics and statistics[cat_col].get("unique_values", 0) < 10:
                        for num_col in numeric_cols[:3]:
                            recommendations["exploratory"].append({
                                "type": "box",
                                "title": f"Box Plot: {num_col} by {cat_col}",
                                "columns": [cat_col, num_col],
                                "rationale": f"Shows the distribution of {num_col} across {cat_col} categories, highlighting outliers."
                            })
            
            # Scatter matrix for multiple numeric columns
            if len(numeric_cols) >= 3:
                recommendations["exploratory"].append({
                    "type": "scatter_matrix",
                    "title": "Scatter Matrix",
                    "columns": numeric_cols[:4],  # Limit to 4 columns
                    "rationale": "Shows relationships between multiple numeric variables simultaneously."
                })
            
            logger.info("Generated visualization recommendations successfully")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating visualization recommendations: {e}")
            return {"error": str(e)}
    
    def export_data_summary(self, file_path: str) -> bool:
        """
        Export the data summary to a JSON file.
        
        Args:
            file_path (str): Path to save the JSON file.
            
        Returns:
            bool: True if export was successful, False otherwise.
        """
        if self.data_summary is None:
            logger.error("No data analysis available. Call analyze_data() first.")
            return False
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.data_summary, f, indent=2)
                
            logger.info(f"Data summary exported successfully to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data summary: {e}")
            return False
    
    def _make_json_serializable(self, obj):
        """
        Convert an object to be JSON serializable.
        
        Args:
            obj: The object to convert.
            
        Returns:
            The JSON serializable object.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
