#!/bin/bash

# Create a .env file for testing
echo "Creating .env file for testing..."
cat > /home/ubuntu/dashboard_generator/.env << EOL
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=False
PORT=8050
HOST=0.0.0.0
EOL

echo "To run the API test, please update the OPENAI_API_KEY in the .env file with your actual API key."
echo "Then run: cd /home/ubuntu/dashboard_generator && source venv/bin/activate && python3 src/test_api.py"
