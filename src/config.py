# Dashboard Generator Configuration
config = {
    # OpenAI API settings
    "openai": {
        "model": "gpt-4-0125-preview",
        "temperature": 0.7,
        "max_tokens": 4000
    },
    
    # Visualization settings
    "visualization": {
        "default_theme": "light",
        "color_schemes": {
            "light": ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"],
            "dark": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        },
        "max_visualizations_per_dashboard": 10,
        "supported_chart_types": [
            "bar", "line", "scatter", "pie", "area", 
            "heatmap", "box", "histogram", "radar", "funnel"
        ]
    },
    
    # Dashboard layout settings
    "layout": {
        "default_grid": "responsive",
        "grid_options": ["fixed", "responsive", "fluid"],
        "default_aspect_ratio": "16:9"
    },
    
    # Data processing settings
    "data_processing": {
        "max_csv_size_mb": 50,
        "sample_rows_for_analysis": 1000,
        "auto_detect_data_types": True,
        "handle_missing_values": "auto"
    },
    
    # Application settings
    "app": {
        "port": 8050,
        "host": "0.0.0.0",
        "debug": False,
        "max_upload_size_mb": 100
    }
}
