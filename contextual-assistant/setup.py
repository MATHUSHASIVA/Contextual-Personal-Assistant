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
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["data", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", "Upgrading pip"):
        return False
    
    # Install base packages that need specific versions to avoid conflicts
    base_packages = [
        "numpy==1.24.3",  # Install numpy first as many packages depend on it
        "packaging==23.2",  # Required by multiple packages
        "pydantic==2.5.0",
        "sqlalchemy==2.0.23"
    ]
    
    for package in base_packages:
        # Wrap package spec in quotes to handle special characters in PowerShell
        if not run_command(f'{sys.executable} -m pip install "{package}"', f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            return False
    
    # Install huggingface packages in the correct order
    hf_packages = [
        "huggingface-hub==0.16.4",
        "tokenizers==0.14.1",
        "transformers==4.34.0"
    ]
    
    for package in hf_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
            # Continue as we might have a working version
    
    # Install remaining requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing remaining dependencies"):
        print("⚠️  Warning: Some packages might have failed to install")
        # Continue as core packages are already installed
    
    return True

def download_spacy_model():
    """Download spaCy English model if not already installed"""
    print("🔍 Checking spaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model already installed and working")
        return True
    except:
        print("📥 spaCy model not found, downloading...")
        # First ensure we have the right numpy version
        if not run_command(f"{sys.executable} -m pip install numpy==1.24.3", "Installing compatible numpy version"):
            return False
            
        # Install specific spaCy version from requirements
        if not run_command(f"{sys.executable} -m pip install spacy==3.7.2", "Installing spaCy"):
            return False
            
        # Download the English model
        return run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Downloading spaCy English model")

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        print("⚠️  Important: You'll need to add your Groq API key to use full LLM features")
        print("   Get a free API key at: https://console.groq.com")
        
        with open(".env", "w") as f:
            f.write("""# Contextual Personal Assistant Environment Variables

# Groq API key for LLaMA model access (get free at console.groq.com)
# GROQ_API_KEY=your_groq_api_key_here

# Database configuration
DATABASE_PATH=./data/assistant.db

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Groq API key for LLaMA model access
LLM_MODEL=llama-3.3-70b-versatile

# Model configurations (optional)
SPACY_MODEL=en_core_web_sm
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Agent settings
MAX_CONTEXT_KEYWORDS=5
DEFAULT_ENVELOPE_THRESHOLD=0.7

# Local-only processing (set to true to disable LLM and use only local NLP)
USE_LOCAL_ONLY=false
""")
        print("✅ Created .env file")
        print("📝 Don't forget to add your GROQ_API_KEY to the .env file!")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test core dependencies
        print("Testing core dependencies...")
        import langchain
        import pydantic
        import sqlalchemy
        print("✅ Core dependencies import successfully")
        
        # Test LLM dependencies
        print("Testing LLM dependencies...")
        try:
            from langchain_groq import ChatGroq
            print("✅ LangChain Groq integration available")
        except ImportError:
            print("⚠️  LangChain Groq not available - some features may be limited")
        
        # Test utility packages
        print("Testing utility packages...")
        import pandas
        import numpy
        import dateparser
        print("✅ Utility packages import successfully")
        
        # Test web interface packages
        print("Testing web interface packages...")
        import streamlit
        import typer
        import rich
        import plotly
        print("✅ Web interface packages import successfully")
        
        # Test project modules
        print("Testing project modules...")
        from src.agents.ingestion_agent import IngestionAgent
        from src.storage.database import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ Project modules import successfully")
        
        print("🎉 Core installation tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {str(e)}")
        print("Try running: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Installation test failed: {str(e)}")
        return False

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

Happy organizing! 🧠✨
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
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  Failed to download spaCy model. You may need to run this manually:")
        print("   python -m spacy download en_core_web_sm")
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if test_installation():
        print_next_steps()
    else:
        print("""
❌ Installation test failed. Common solutions:

1. Make sure all dependencies installed correctly:
   pip install -r requirements.txt

2. Download spaCy model manually:
   python -m spacy download en_core_web_sm

3. Check Python version (3.8+ required):
   python --version

4. Try running from project root directory

5. Check error messages above for specific issues
""")
        sys.exit(1)

if __name__ == "__main__":
    main()