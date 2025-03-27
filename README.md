# Dashboard Design Generator

A powerful tool that works with OpenAI API to generate designs for multiple distinct dashboards based on CSV data for data-related themes.

## Overview

The Dashboard Design Generator is an AI-powered application that helps you create professional dashboard designs tailored to your specific data and business requirements. By leveraging the OpenAI API and advanced data analysis techniques, this tool can:

- Analyze your CSV data to understand its structure and characteristics
- Generate multiple distinct dashboard designs based on your requirements
- Provide detailed rationales for visualization choices and layout decisions
- Create interactive dashboard previews and exportable code

## Features

- **CSV Data Processing**: Upload and analyze any CSV data file
- **OpenAI API Integration**: Leverage AI to generate intelligent dashboard designs
- **KPI Mapping**: Connect business metrics to appropriate visualizations
- **Multiple Dashboard Generation**: Create several distinct dashboards for different purposes
- **Visualization Recommendations**: Get data-driven suggestions for chart types
- **Design Rationales**: Understand the reasoning behind design choices
- **Interactive Previews**: See how your dashboards will look with your actual data
- **Code Generation**: Export ready-to-use code for implementation
- **Responsive Layouts**: Designs that work on any screen size

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dashboard-generator.git
   cd dashboard-generator
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Application

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8050
   ```

### Using the Dashboard Generator

1. **Configure API Key**: Enter your OpenAI API key in the configuration section.

2. **Upload Data**: Upload your CSV file using the file upload component.

3. **Specify Requirements**:
   - Enter a title and purpose for your dashboard
   - Specify KPIs you want to focus on
   - Choose the number of dashboards to generate
   - Add any additional requirements

4. **Generate Designs**: Click the "Generate Dashboard Designs" button.

5. **Review Designs**: Explore the generated dashboard designs and their rationales.

6. **Preview Dashboards**: Click "Preview Dashboard" to see how it looks with your data.

7. **Generate Code**: Click "Generate Code" to create implementation code.

8. **Export**: Download the designs and code for implementation.

## Data Arrangement Considerations

The Dashboard Design Generator considers several factors when arranging data visualizations:

- **Storytelling**: Visualizations are arranged to tell a clear and compelling story.
- **Visual Appeal**: Dashboards are designed to be visually appealing and easy to understand.
- **Clarity and Conciseness**: Information is presented in a clear and concise manner.
- **Interactivity**: Interactive elements are included where appropriate for data exploration.
- **Filtering**: Data filtering options are provided for deeper analysis.

## Architecture

The application is built with a modular architecture:

- **API Key Manager**: Securely handles OpenAI API credentials
- **CSV Processor**: Loads, validates, and analyzes CSV data
- **Data Transformer**: Prepares data for visualization
- **OpenAI Client**: Communicates with the OpenAI API
- **Dashboard Generator**: Creates dashboard designs based on data and requirements
- **Visualization Components**: Renders charts and graphs
- **Interactive Components**: Provides filtering and exploration capabilities
- **User Interface**: Ties everything together in a user-friendly interface

## Development

### Project Structure

```
dashboard_generator/
├── src/
│   ├── api_key_manager.py
│   ├── config.py
│   ├── csv_processor.py
│   ├── dashboard_generator.py
│   ├── data_transformer.py
│   ├── design_rationale.py
│   ├── interactive_components.py
│   ├── kpi_mapper.py
│   ├── openai_client.py
│   ├── prompt_engineering.py
│   ├── ui.py
│   └── visualization_components.py
├── tests/
│   ├── test_components.py
│   └── test_integration.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

### Running Tests

To run the unit tests:
```
python -m unittest discover tests
```

## Customization

### Modifying Visualization Types

You can customize the available visualization types by modifying the `visualization_components.py` file. Add new chart types by implementing additional methods in the `VisualizationComponents` class.

### Changing the UI Theme

The UI theme can be modified by changing the Bootstrap theme in the `ui.py` file. Look for the `external_stylesheets` parameter in the Dash app initialization.

### Extending KPI Mappings

To add support for additional KPIs, modify the `kpi_categories` and `kpi_viz_mapping` dictionaries in the `kpi_mapper.py` file.

## Troubleshooting

### API Key Issues

If you encounter issues with the OpenAI API key:
- Ensure your API key is valid and has not expired
- Check that you have sufficient credits in your OpenAI account
- Verify that the API key is correctly set in the `.env` file or UI

### CSV Upload Problems

If you have problems uploading CSV files:
- Ensure your CSV file is properly formatted
- Check for special characters or encoding issues
- Verify that column names do not contain unusual characters

### Visualization Errors

If visualizations fail to render:
- Check that your data contains the expected columns
- Ensure numeric columns contain valid numbers
- Verify that date columns are in a recognizable format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for providing the API that powers the design generation
- Plotly and Dash for the visualization and UI components
- The Python community for the excellent data processing libraries
