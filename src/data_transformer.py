import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Provides utilities for transforming CSV data for dashboard visualizations.
    Handles aggregation, filtering, pivoting, and other transformations.
    """
    
    def __init__(self, df=None):
        """
        Initialize the data transformer.
        
        Args:
            df (pandas.DataFrame, optional): DataFrame to transform.
        """
        self.df = df
        
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame to transform.
        
        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        self.df = df
        
    def aggregate(self, group_by_cols: List[str], agg_col: str, 
                 agg_func: str = 'sum') -> pd.DataFrame:
        """
        Aggregate data by grouping and applying an aggregation function.
        
        Args:
            group_by_cols (list): Columns to group by.
            agg_col (str): Column to aggregate.
            agg_func (str): Aggregation function to apply ('sum', 'mean', 'count', etc.).
            
        Returns:
            pandas.DataFrame: Aggregated DataFrame.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in group_by_cols + [agg_col]:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            # Perform aggregation
            result = self.df.groupby(group_by_cols)[agg_col].agg(agg_func).reset_index()
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return None
    
    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter data based on column conditions.
        
        Args:
            filters (dict): Dictionary of column:value pairs or column:condition pairs.
                Example: {'category': 'Electronics', 'price': ('>', 100)}
            
        Returns:
            pandas.DataFrame: Filtered DataFrame.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            filtered_df = self.df.copy()
            
            for col, condition in filters.items():
                if col not in filtered_df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    continue
                    
                if isinstance(condition, tuple) and len(condition) == 2:
                    operator, value = condition
                    
                    if operator == '>':
                        filtered_df = filtered_df[filtered_df[col] > value]
                    elif operator == '>=':
                        filtered_df = filtered_df[filtered_df[col] >= value]
                    elif operator == '<':
                        filtered_df = filtered_df[filtered_df[col] < value]
                    elif operator == '<=':
                        filtered_df = filtered_df[filtered_df[col] <= value]
                    elif operator == '==':
                        filtered_df = filtered_df[filtered_df[col] == value]
                    elif operator == '!=':
                        filtered_df = filtered_df[filtered_df[col] != value]
                    elif operator == 'in':
                        if isinstance(value, (list, tuple)):
                            filtered_df = filtered_df[filtered_df[col].isin(value)]
                        else:
                            logger.error(f"Value for 'in' operator must be a list or tuple")
                    elif operator == 'not in':
                        if isinstance(value, (list, tuple)):
                            filtered_df = filtered_df[~filtered_df[col].isin(value)]
                        else:
                            logger.error(f"Value for 'not in' operator must be a list or tuple")
                    elif operator == 'contains':
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(value), na=False)]
                    else:
                        logger.error(f"Unsupported operator: {operator}")
                else:
                    # Simple equality filter
                    filtered_df = filtered_df[filtered_df[col] == condition]
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return None
    
    def pivot_data(self, index_col: str, column_col: str, 
                  value_col: str, agg_func: str = 'sum') -> pd.DataFrame:
        """
        Create a pivot table from the data.
        
        Args:
            index_col (str): Column to use as index.
            column_col (str): Column to use as columns.
            value_col (str): Column to use as values.
            agg_func (str): Aggregation function to apply.
            
        Returns:
            pandas.DataFrame: Pivot table.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in [index_col, column_col, value_col]:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            # Create pivot table
            pivot_table = pd.pivot_table(
                self.df, 
                values=value_col,
                index=index_col,
                columns=column_col,
                aggfunc=agg_func,
                fill_value=0
            )
            
            return pivot_table
            
        except Exception as e:
            logger.error(f"Error creating pivot table: {e}")
            return None
    
    def time_series_resample(self, date_col: str, value_col: str, 
                            freq: str = 'M', agg_func: str = 'sum') -> pd.DataFrame:
        """
        Resample time series data to a specified frequency.
        
        Args:
            date_col (str): Column containing dates.
            value_col (str): Column containing values to aggregate.
            freq (str): Frequency to resample to ('D' for daily, 'W' for weekly, 'M' for monthly, etc.).
            agg_func (str): Aggregation function to apply.
            
        Returns:
            pandas.DataFrame: Resampled DataFrame.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in [date_col, value_col]:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            # Ensure date column is datetime type
            df_copy = self.df.copy()
            if df_copy[date_col].dtype != 'datetime64[ns]':
                try:
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                except:
                    logger.error(f"Could not convert column '{date_col}' to datetime")
                    return None
            
            # Set date as index
            df_copy = df_copy.set_index(date_col)
            
            # Resample and aggregate
            resampled = df_copy[value_col].resample(freq).agg(agg_func)
            
            # Convert back to DataFrame
            result = resampled.reset_index()
            
            return result
            
        except Exception as e:
            logger.error(f"Error resampling time series data: {e}")
            return None
    
    def calculate_growth(self, date_col: str, value_col: str, 
                        period: str = 'YoY') -> pd.DataFrame:
        """
        Calculate growth rates (Year-over-Year, Month-over-Month, etc.).
        
        Args:
            date_col (str): Column containing dates.
            value_col (str): Column containing values to calculate growth for.
            period (str): Period for growth calculation ('YoY', 'MoM', 'QoQ', etc.).
            
        Returns:
            pandas.DataFrame: DataFrame with growth rates.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in [date_col, value_col]:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            # Ensure date column is datetime type
            df_copy = self.df.copy()
            if df_copy[date_col].dtype != 'datetime64[ns]':
                try:
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                except:
                    logger.error(f"Could not convert column '{date_col}' to datetime")
                    return None
            
            # Sort by date
            df_copy = df_copy.sort_values(date_col)
            
            # Calculate growth based on period
            if period == 'YoY':
                # Extract year and month for grouping
                df_copy['year'] = df_copy[date_col].dt.year
                df_copy['month'] = df_copy[date_col].dt.month
                
                # Group by year and month, and calculate sum
                grouped = df_copy.groupby(['year', 'month'])[value_col].sum().reset_index()
                
                # Create a date column for easier comparison
                grouped['date'] = pd.to_datetime(grouped['year'].astype(str) + '-' + grouped['month'].astype(str) + '-01')
                
                # Calculate YoY growth
                grouped['previous_year_value'] = grouped.groupby('month')[value_col].shift(1)
                grouped['YoY_growth'] = (grouped[value_col] - grouped['previous_year_value']) / grouped['previous_year_value'] * 100
                
                result = grouped[['date', value_col, 'YoY_growth']]
                
            elif period == 'MoM':
                # Resample to monthly frequency
                df_copy = df_copy.set_index(date_col)
                monthly = df_copy[value_col].resample('M').sum().reset_index()
                
                # Calculate MoM growth
                monthly['previous_month_value'] = monthly[value_col].shift(1)
                monthly['MoM_growth'] = (monthly[value_col] - monthly['previous_month_value']) / monthly['previous_month_value'] * 100
                
                result = monthly[[date_col, value_col, 'MoM_growth']]
                
            elif period == 'QoQ':
                # Extract year and quarter for grouping
                df_copy['year'] = df_copy[date_col].dt.year
                df_copy['quarter'] = df_copy[date_col].dt.quarter
                
                # Group by year and quarter, and calculate sum
                grouped = df_copy.groupby(['year', 'quarter'])[value_col].sum().reset_index()
                
                # Create a date column for easier comparison (first day of quarter)
                grouped['date'] = grouped.apply(
                    lambda x: pd.Timestamp(year=int(x['year']), month=int((x['quarter']-1)*3+1), day=1), 
                    axis=1
                )
                
                # Calculate QoQ growth
                grouped['previous_quarter_value'] = grouped[value_col].shift(1)
                grouped['QoQ_growth'] = (grouped[value_col] - grouped['previous_quarter_value']) / grouped['previous_quarter_value'] * 100
                
                result = grouped[['date', value_col, 'QoQ_growth']]
                
            else:
                logger.error(f"Unsupported period: {period}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating growth rates: {e}")
            return None
    
    def calculate_moving_average(self, date_col: str, value_col: str, 
                               window: int = 7) -> pd.DataFrame:
        """
        Calculate moving average for time series data.
        
        Args:
            date_col (str): Column containing dates.
            value_col (str): Column containing values to calculate moving average for.
            window (int): Window size for moving average.
            
        Returns:
            pandas.DataFrame: DataFrame with moving average.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in [date_col, value_col]:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            # Ensure date column is datetime type
            df_copy = self.df.copy()
            if df_copy[date_col].dtype != 'datetime64[ns]':
                try:
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                except:
                    logger.error(f"Could not convert column '{date_col}' to datetime")
                    return None
            
            # Sort by date
            df_copy = df_copy.sort_values(date_col)
            
            # Calculate moving average
            df_copy[f'{value_col}_MA{window}'] = df_copy[value_col].rolling(window=window).mean()
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return None
    
    def normalize_data(self, columns: List[str], method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize data in specified columns.
        
        Args:
            columns (list): Columns to normalize.
            method (str): Normalization method ('minmax', 'zscore', 'robust').
            
        Returns:
            pandas.DataFrame: DataFrame with normalized columns.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in columns:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
                
                if not np.issubdtype(self.df[col].dtype, np.number):
                    logger.error(f"Column '{col}' is not numeric")
                    return None
            
            df_copy = self.df.copy()
            
            # Apply normalization method
            if method == 'minmax':
                for col in columns:
                    min_val = df_copy[col].min()
                    max_val = df_copy[col].max()
                    df_copy[f'{col}_normalized'] = (df_copy[col] - min_val) / (max_val - min_val)
                    
            elif method == 'zscore':
                for col in columns:
                    mean_val = df_copy[col].mean()
                    std_val = df_copy[col].std()
                    df_copy[f'{col}_normalized'] = (df_copy[col] - mean_val) / std_val
                    
            elif method == 'robust':
                for col in columns:
                    median_val = df_copy[col].median()
                    iqr_val = df_copy[col].quantile(0.75) - df_copy[col].quantile(0.25)
                    df_copy[f'{col}_normalized'] = (df_copy[col] - median_val) / iqr_val
                    
            else:
                logger.error(f"Unsupported normalization method: {method}")
                return None
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return None
    
    def bin_data(self, column: str, num_bins: int = 5, 
                labels: List[str] = None) -> pd.DataFrame:
        """
        Bin continuous data into discrete categories.
        
        Args:
            column (str): Column to bin.
            num_bins (int): Number of bins.
            labels (list, optional): Labels for the bins.
            
        Returns:
            pandas.DataFrame: DataFrame with binned column.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate column
            if column not in self.df.columns:
                logger.error(f"Column '{column}' not found in DataFrame")
                return None
                
            if not np.issubdtype(self.df[column].dtype, np.number):
                logger.error(f"Column '{column}' is not numeric")
                return None
            
            df_copy = self.df.copy()
            
            # Create bin labels if not provided
            if labels is None:
                labels = [f'Bin {i+1}' for i in range(num_bins)]
                
            if len(labels) != num_bins:
                logger.warning(f"Number of labels ({len(labels)}) does not match number of bins ({num_bins})")
                labels = [f'Bin {i+1}' for i in range(num_bins)]
            
            # Bin the data
            df_copy[f'{column}_binned'] = pd.cut(
                df_copy[column], 
                bins=num_bins, 
                labels=labels
            )
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error binning data: {e}")
            return None
    
    def calculate_percentiles(self, column: str, 
                             percentiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[float, float]:
        """
        Calculate percentiles for a column.
        
        Args:
            column (str): Column to calculate percentiles for.
            percentiles (list): List of percentiles to calculate.
            
        Returns:
            dict: Dictionary of percentile:value pairs.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate column
            if column not in self.df.columns:
                logger.error(f"Column '{column}' not found in DataFrame")
                return None
                
            if not np.issubdtype(self.df[column].dtype, np.number):
                logger.error(f"Column '{column}' is not numeric")
                return None
            
            # Calculate percentiles
            result = {}
            for p in percentiles:
                result[p] = float(self.df[column].quantile(p))
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating percentiles: {e}")
            return None
    
    def transform_for_visualization(self, viz_type: str, 
                                   columns: List[str], 
                                   params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform data specifically for a visualization type.
        
        Args:
            viz_type (str): Visualization type ('bar', 'line', 'pie', etc.).
            columns (list): Columns to include in the transformation.
            params (dict, optional): Additional parameters for the transformation.
            
        Returns:
            pandas.DataFrame: Transformed DataFrame ready for visualization.
        """
        if self.df is None:
            logger.error("No DataFrame set. Call set_dataframe() first.")
            return None
            
        try:
            # Validate columns
            for col in columns:
                if col not in self.df.columns:
                    logger.error(f"Column '{col}' not found in DataFrame")
                    return None
            
            params = params or {}
            df_copy = self.df.copy()
            
            # Apply transformation based on visualization type
            if viz_type == 'bar':
                # For bar charts, typically need category and value
                if len(columns) < 2:
                    logger.error("Bar chart requires at least 2 columns (category and value)")
                    return None
                    
                cat_col = columns[0]
                val_col = columns[1]
                
                # Aggregate if specified
                agg_func = params.get('agg_func', 'sum')
                result = df_copy.groupby(cat_col)[val_col].agg(agg_func).reset_index()
                
                # Sort if specified
                sort_by = params.get('sort_by', None)
                if sort_by == 'value':
                    result = result.sort_values(val_col, ascending=params.get('ascending', False))
                elif sort_by == 'category':
                    result = result.sort_values(cat_col, ascending=params.get('ascending', True))
                
                # Limit number of categories if specified
                limit = params.get('limit', None)
                if limit and isinstance(limit, int) and limit > 0:
                    result = result.head(limit)
                
                return result
                
            elif viz_type == 'line':
                # For line charts, typically need date/time and value
                if len(columns) < 2:
                    logger.error("Line chart requires at least 2 columns (date and value)")
                    return None
                    
                date_col = columns[0]
                val_col = columns[1]
                
                # Ensure date column is datetime type
                if df_copy[date_col].dtype != 'datetime64[ns]':
                    try:
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                    except:
                        logger.error(f"Could not convert column '{date_col}' to datetime")
                        return None
                
                # Resample if specified
                freq = params.get('freq', None)
                agg_func = params.get('agg_func', 'sum')
                
                if freq:
                    # Set date as index
                    df_copy = df_copy.set_index(date_col)
                    
                    # Resample and aggregate
                    resampled = df_copy[val_col].resample(freq).agg(agg_func)
                    
                    # Convert back to DataFrame
                    result = resampled.reset_index()
                else:
                    # Just aggregate by date
                    result = df_copy.groupby(date_col)[val_col].agg(agg_func).reset_index()
                
                # Sort by date
                result = result.sort_values(date_col)
                
                return result
                
            elif viz_type == 'pie':
                # For pie charts, typically need category and value
                if len(columns) < 2:
                    logger.error("Pie chart requires at least 2 columns (category and value)")
                    return None
                    
                cat_col = columns[0]
                val_col = columns[1]
                
                # Aggregate
                agg_func = params.get('agg_func', 'sum')
                result = df_copy.groupby(cat_col)[val_col].agg(agg_func).reset_index()
                
                # Sort if specified
                sort_by = params.get('sort_by', None)
                if sort_by == 'value':
                    result = result.sort_values(val_col, ascending=params.get('ascending', False))
                
                # Limit and group small categories if specified
                limit = params.get('limit', None)
                if limit and isinstance(limit, int) and limit > 0 and len(result) > limit:
                    # Keep top categories and group others
                    top_categories = result.nlargest(limit - 1, val_col)
                    other_categories = result.nsmallest(len(result) - (limit - 1), val_col)
                    
                    other_value = other_categories[val_col].sum()
                    other_row = pd.DataFrame({
                        cat_col: ['Other'],
                        val_col: [other_value]
                    })
                    
                    result = pd.concat([top_categories, other_row])
                
                return result
                
            elif viz_type == 'scatter':
                # For scatter plots, typically need two numeric columns
                if len(columns) < 2:
                    logger.error("Scatter plot requires at least 2 numeric columns")
                    return None
                    
                x_col = columns[0]
                y_col = columns[1]
                
                # Validate numeric columns
                for col in [x_col, y_col]:
                    if not np.issubdtype(df_copy[col].dtype, np.number):
                        logger.error(f"Column '{col}' is not numeric")
                        return None
                
                # Sample if specified
                sample_size = params.get('sample_size', None)
                if sample_size and isinstance(sample_size, int) and sample_size > 0:
                    if sample_size < len(df_copy):
                        result = df_copy.sample(sample_size)
                    else:
                        result = df_copy
                else:
                    result = df_copy
                
                # Select only required columns
                result = result[columns]
                
                return result
                
            elif viz_type == 'heatmap':
                # For heatmaps, typically need two categorical columns and one value column
                if len(columns) < 3:
                    logger.error("Heatmap requires at least 3 columns (row, column, value)")
                    return None
                    
                row_col = columns[0]
                col_col = columns[1]
                val_col = columns[2]
                
                # Create pivot table
                agg_func = params.get('agg_func', 'sum')
                result = pd.pivot_table(
                    df_copy, 
                    values=val_col,
                    index=row_col,
                    columns=col_col,
                    aggfunc=agg_func,
                    fill_value=0
                )
                
                return result
                
            else:
                logger.error(f"Unsupported visualization type: {viz_type}")
                return None
            
        except Exception as e:
            logger.error(f"Error transforming data for visualization: {e}")
            return None
