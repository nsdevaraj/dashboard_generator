import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """
    Manages API keys for external services, primarily OpenAI.
    Handles loading from environment variables and validation.
    """
    
    def __init__(self):
        """Initialize the API key manager and load environment variables."""
        # Load environment variables from .env file if it exists
        load_dotenv()
        self.openai_api_key = None
        
    def get_openai_api_key(self):
        """
        Get the OpenAI API key from environment variables or cached value.
        
        Returns:
            str: The OpenAI API key if available, None otherwise.
        """
        # Return cached key if already loaded
        if self.openai_api_key:
            return self.openai_api_key
            
        # Try to get key from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found in environment variables")
            return None
            
        # Validate key format (basic check)
        if not self._validate_openai_key_format(self.openai_api_key):
            logger.warning("OpenAI API key has invalid format")
            return None
            
        return self.openai_api_key
    
    def set_openai_api_key(self, api_key):
        """
        Set the OpenAI API key.
        
        Args:
            api_key (str): The OpenAI API key to set.
            
        Returns:
            bool: True if the key was set successfully, False otherwise.
        """
        if not api_key or not isinstance(api_key, str):
            logger.error("Invalid API key provided")
            return False
            
        if not self._validate_openai_key_format(api_key):
            logger.warning("OpenAI API key has invalid format")
            return False
            
        # Set the key in memory
        self.openai_api_key = api_key
        
        # Optionally, update the .env file for persistence
        self._update_env_file('OPENAI_API_KEY', api_key)
        
        return True
    
    def _validate_openai_key_format(self, api_key):
        """
        Validate the format of an OpenAI API key.
        
        Args:
            api_key (str): The API key to validate.
            
        Returns:
            bool: True if the key format is valid, False otherwise.
        """
        # Basic validation: OpenAI keys typically start with 'sk-' and are 51 characters long
        if not isinstance(api_key, str):
            return False
            
        if not api_key.startswith('sk-'):
            return False
            
        if len(api_key) < 20:  # Simplified length check
            return False
            
        return True
    
    def _update_env_file(self, key, value):
        """
        Update a key-value pair in the .env file.
        
        Args:
            key (str): The key to update.
            value (str): The value to set.
            
        Returns:
            bool: True if the file was updated successfully, False otherwise.
        """
        env_path = '.env'
        
        try:
            # Check if .env file exists
            if os.path.exists(env_path):
                # Read existing content
                with open(env_path, 'r') as file:
                    lines = file.readlines()
                
                # Check if key already exists
                key_exists = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{key}="):
                        lines[i] = f"{key}={value}\n"
                        key_exists = True
                        break
                
                # Add key if it doesn't exist
                if not key_exists:
                    lines.append(f"{key}={value}\n")
                
                # Write updated content
                with open(env_path, 'w') as file:
                    file.writelines(lines)
            else:
                # Create new .env file
                with open(env_path, 'w') as file:
                    file.write(f"{key}={value}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error updating .env file: {e}")
            return False
