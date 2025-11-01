# Contextual Personal Assistant

A sophisticated personal assistant system that transforms unstructured text notes into organized, actionable knowledge using AI agents.



### ✅ Core Requirements Met

**1. Ingestion & Organization Agent**
- ✅ Processes unstructured text notes into structured Cards
- ✅ Card Types: Task, Reminder, Idea/Note 
- ✅ Entity Extraction: description, date, assignee, context_keywords
- ✅ Automatic Envelope organization based on content and context
- ✅ Uses LangChain framework with Groq Llama 3.3 70B model

**2. Context Management**
- ✅ Dynamic User Context with projects, companies, people, themes
- ✅ Automatic context refinement from each processed note
- ✅ Context-aware Card classification and Envelope assignment

**3. Thinking Agent Architecture & Implementation**
- ✅ Scheduled analysis of Cards and Envelopes
- ✅ Generates proactive suggestions:
  - Next Steps based on completed tasks
  - Recommendations for optimization
  - Conflict/overlap detection
- ✅ Pattern recognition and cross-referencing

**4. Technical Requirements**
- ✅ Agent Framework: LangChain 
- ✅ NLP Models: Groq Llama + fallback processing
- ✅ Storage: SQLite database 
- ✅ Language: Python 


## 🏗️ System Architecture

### Core Components
1. **Ingestion & Organization Agent**: Processes raw text into structured Cards using LangChain framework
2. **Context Management System**: Maintains dynamic user context with automatic updates
3. **Thinking Agent**: Analyzes patterns and generates proactive suggestions


### Technology Stack & Justification

**Agent Framework: LangChain**
- ✅ Robust ecosystem with extensive tool support
- ✅ Easy LLM integration 
- ✅ Built-in agent patterns and prompt templates
- ✅ Active community and documentation


**Storage: SQLite**
- ✅ Zero configuration required
- ✅ Perfect for local deployment
- ✅ ACID compliance for data integrity
- ✅ Easy to backup and migrate

**Interface: Streamlit**
- ✅ Rapid development and deployment
- ✅ Interactive widgets for data exploration
- ✅ Built-in charting capabilities
- ✅ Professional appearance with minimal code

### System Flow
```
Raw Note → NLP Processing → Card Classification → Entity Extraction → Envelope Assignment → Context Update → Insights Generation
```

## 🚀 Quick Start

### Prerequisites
1. **Get Groq API Key** (Free): Visit [console.groq.com](https://console.groq.com) to get your free API key
2. **Python 3.8+** installed on your system

### Option 1: Automated Setup (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd contextual-assistant

# Run automated setup
python setup.py

# Add your Groq API key to .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs

# Create .env file with your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### Configuration
Create a `.env` file for full LLM functionality:
```env
# Groq API key for LLama model access (get free at console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Database path (defaults to ./data/assistant.db)
DATABASE_PATH=./data/assistant.db

# Logging level
LOG_LEVEL=INFO

# Model configurations
LLM_MODEL=llama-3.3-70b-versatile
SPACY_MODEL=en_core_web_sm
```

### Running the Application
```bash
# Try the demo first
python demo.py

# Start the web interface
streamlit run app.py

# Use the CLI
python cli.py process "Call Sarah about the Q3 budget next Monday"
```

## 📊 Data Models

### Card Types
- **Task**: Actionable items requiring completion
- **Reminder**: Time-based alerts and notifications  
- **Idea/Note**: General information and concept storage


### Envelopes
Logical groupings of related Cards representing:
- Projects
- Companies
- People
- Themes/Categories

## 🧠 Thinking Agent Design

The Thinking Agent analyzes the knowledge base to provide:

1. **Next Steps**: Suggests logical follow-up actions
2. **Recommendations**: Identifies patterns and optimization opportunities
3. **Conflict Detection**: Finds scheduling conflicts and resource overlaps
4. **Context Refinement**: Updates user context based on usage patterns

### Analysis Pipeline
```
Cards & Envelopes → Pattern Analysis → Cross-Reference → Generate Insights → Update Context
```



## 📁 Project Structure
```
contextual-assistant/
├── src/
│   ├── agents/           # Core agent implementations
│   ├── models/           # Data models and schemas
│   ├── nlp/             # NLP processing components
│   ├── storage/         # Database and persistence
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── data/                # Local data storage
├── app.py               # Streamlit web interface
├── cli.py               # Command-line interface
└── requirements.txt     # Dependencies
```



## 🚧 Future Enhancements

- [ ] Multi-agent collaboration 
- [ ] Voice input integration
- [ ] Mobile app interface
- [ ] Advanced analytics dashboard
- [ ] Team collaboration features
- [ ] Integration with external tools (calendar, email, etc.)

## 📝 License

MIT License - see LICENSE file for details.
