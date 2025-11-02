#!/usr/bin/env python3
"""
Setup script for the Contextual Personal Assistant
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories"""
    directories = ["data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", "Upgrading pip"):
        return False

    # Install requirements from requirements.txt
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies from requirements.txt"):
        print("Failed to install some packages from requirements.txt")
        return False
    return True


def download_spacy_model():
    """Download spaCy English model if not already installed"""
    print("Checking spaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model already installed and working")
        return True
    except:
        print("spaCy model not found, downloading...")
        
        # Install the spaCy model directly from GitHub
        model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
        if run_command(f"{sys.executable} -m pip install {model_url}", "Installing spaCy English model"):
            print("spaCy model installed successfully")
            return True
        else:
            print("Failed to install spaCy model automatically")
            print("You can install it manually with:")
            print(f"pip install {model_url}")
            return False


def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        print("Important: You'll need to add your Groq API key to use full LLM features")
        print("Get a free API key at: https://console.groq.com")

        with open(".env", "w") as f:
            f.write("""# Contextual Personal Assistant Environment Variables

# Groq API key for LLaMA model access (get free at console.groq.com)
# GROQ_API_KEY=your_groq_api_key_here

# Database configuration
DATABASE_PATH=./data/assistant.db

# Groq API key for LLaMA model access
LLM_MODEL=llama-3.3-70b-versatile

# NLP Model configuration
SPACY_MODEL=en_core_web_sm

# Agent settings
MAX_CONTEXT_KEYWORDS=5
DEFAULT_ENVELOPE_THRESHOLD=0.7

# Local-only processing (set to true to disable LLM and use only local NLP)
USE_LOCAL_ONLY=false
""")
        print("Created .env file")
        print("Don't forget to add your GROQ_API_KEY to the .env file!")
    else:
        print(".env file already exists")


def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 Setup Complete!

🔑 Important: Get your FREE Groq API key at https://console.groq.com
   Add it to the .env file: GROQ_API_KEY=your_key_here

Next steps:
1. Try the demo with sample data:
   python demo.py

2. Start the web interface:
   streamlit run app.py

3. Use the CLI interface:
   python cli.py process "Call Sarah about the meeting"

4. View available commands:
   python cli.py --help

📚 Documentation:
   - README.md contains detailed information
   - Check the src/ directory for code organization
   - Visit the web interface for interactive exploration

🔧 Configuration:
   - Edit .env file to add your Groq API key
   - Adjust model settings as needed
   - System works in fallback mode without API key (limited features)

Happy organizing! ✨
""")


def main():
    """Main setup function"""
    print("🧠 Contextual Personal Assistant Setup")
    print("=====================================\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("Failed to download spaCy model. You may need to run this manually:")
        print("pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")

    # Setup environment
    setup_environment()
    print_next_steps()


if __name__ == "__main__":
    main()