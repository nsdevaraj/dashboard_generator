import os
from src.ui import DashboardGeneratorUI

# Create assets directory if it doesn't exist
assets_dir = os.path.join(os.getcwd(), "assets")
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Create UI and run server
if __name__ == "__main__":
    ui = DashboardGeneratorUI()
    ui.run_server(debug=True)
