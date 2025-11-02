## Contextual Personal Assistant
<p align="center">
      <img src="Image\Screenshot 2025-11-01 170435.png" alt="App Screenshot" width="700" />
</p>
This project is a sophisticated personal assistant system designed to transform unstructured text notes into organized, actionable knowledge using advanced AI agents. The assistant automatically processes your notes, extracts key information, and organizes it for easy access and proactive insights.

### Key features include:

- **Ingestion & Organization:** The system processes raw text notes and converts them into structured "Cards" such as Tasks, Reminders, and Ideas/Notes. It extracts important entities like descriptions, dates, assignees, and context keywords, and organizes these Cards into logical groups (Envelopes) based on their content and context. This is powered by the LangChain framework and Groq Llama 3.3 70B model.

- **Context Management:** The assistant maintains a dynamic user context, tracking projects, companies, people, and themes. As new notes are processed, the context is automatically refined, ensuring that Cards are classified and grouped in a context-aware manner.

- **Thinking Agent:** The system includes a Thinking Agent that regularly analyzes Cards and Envelopes to generate proactive suggestions. It can recommend next steps, identify optimization opportunities, detect conflicts or overlaps, and recognize patterns across your knowledge base.

- **Technical Stack:** The project is built with Python, using LangChain for agent orchestration, Groq Llama for natural language processing, and SQLite for local data storage.


## 🏗️ System Architecture

### Core Components
1. **Ingestion & Organization Agent**: Processes raw text into structured Cards using LangChain framework
2. **Context Management System**: Maintains dynamic user context with automatic updates
3. **Thinking Agent**: Analyzes patterns and generates proactive suggestions

### Technology Stack 

**Agent Framework: LangChain**
-  Robust ecosystem with extensive tool support
-  Easy LLM integration 
-  Built-in agent patterns and prompt templates

**Storage: SQLite**
-  Zero configuration required
-  Perfect for local deployment
-  Easy to backup and migrate

**Interface: Streamlit**
-  Rapid development and deployment
-  Interactive widgets for data exploration
-  Built-in charting capabilities with Plotly (card trends, type distribution, status overview)
-  Professional appearance with minimal code
-  Real-time status updates with dropdown selectors for card management

## 🔄 Dual Processing Workflows

The assistant features **two intelligent processing workflows** that automatically switch based on API availability:

### 1. **LLM-Powered Workflow** (Primary Mode)
**When:** Groq API key is configured and API is accessible  
**Uses:** Groq Llama 3.3 70B model via LangChain

**Capabilities:**
- **Advanced Natural Language Understanding**: Deep semantic analysis of note content
- **Contextual Classification**: Intelligent card type prediction (Task/Reminder/Idea) based on user intent
- **Smart Entity Extraction**: Identifies people, dates, locations, organizations with high accuracy
- **Envelope Intelligence**: Suggests relevant envelope groupings based on context and relationships
- **Reasoning Generation**: Provides explanations for classification decisions

---

### 2. **NLP Fallback Workflow** (Offline Mode)
**When:** No API key configured OR API network issues  
**Uses:** SpaCy models + pattern matching algorithms

**Capabilities:**
- **Pattern-Based Classification**: Uses regex patterns to identify card types
  - Task patterns: call, email, buy, complete, schedule, etc. (50+ keywords)
  - Reminder patterns: remember, deadline, appointment, meeting, etc. (40+ keywords)  
  - Idea patterns: thought, concept, brainstorm, consider, etc. (35+ keywords)
- **spaCy NER**: Extracts named entities (people, dates, locations, organizations)
- **Entity Boost System**: Adjusts classification scores based on detected entities:
  - Dates found → +0.3 to Reminder, +0.2 to Task
  - People found → +0.2 to Task
- **Keyword Matching**: Finds envelope matches using keyword overlap scoring
- **Fallback Naming**: Generates meaningful envelope names from extracted keywords


---

### Automatic Switching
The system **automatically detects** which workflow to use:
```python
if GROQ_API_KEY exists and API is accessible:
    → Use LLM-Powered Workflow (Higher accuracy, context-aware)
else:
    → Use NLP Fallback Workflow (Fully offline, privacy-first)
```

**Benefits:**
- ✅ **No internet required**: Works completely offline with NLP mode
- ✅ **API cost savings**: Can run without paid API subscriptions
- ✅ **Reliability**: Automatically switches if API fails
- ✅ **Privacy option**: Process sensitive notes locally without cloud AI

### System Flow


```

User Input (Raw Note)
        ↓
   IngestionAgent
        ↓
   API Available? ──────┐
        ↓               │
    ┌───YES         NO──┤
    ↓                   ↓
LLM Workflow      NLP Workflow
(Groq Llama)      (spaCy + Patterns)
    │                   │
    └────────┬──────────┘
             ↓
      Card Creation
      (Task/Reminder/Idea)
             ↓
      Envelope Assignment
      (Group by Context)
             ↓
      Context Update
      (People, Projects, Companies)
             ↓
      Store to Database (SQLite)
             ↓
      ThinkingAgent Analysis (Periodic)
             ↓
      Insights Generation
      • Next Steps
      • Recommendations
      • Conflict Detection


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
├── data/                # Local data storage
├── app.py               # Streamlit web interface
└── requirements.txt     # Dependencies
```
---

## 📊 Data Models

### Card Types
- **Task**: Actionable items requiring completion (e.g., "Call John", "Buy groceries")
- **Reminder**: Time-based alerts and notifications (e.g., "Meeting at 3pm", "Deadline Friday")
- **Idea/Note**: General information and concept storage (e.g., "Blog post idea", "Design inspiration")

### Card Statuses
Cards can have four different statuses that can be updated directly from the UI:
- **Pending**: Newly created, not yet started
- **In Progress**: Currently being worked on
- **Completed**: Finished successfully
- **Cancelled**: No longer needed or relevant

### Envelopes
Logical groupings of related Cards representing:
- **Projects**: "Website Redesign", "Q4 Planning"
- **Companies**: "Microsoft", "Client ABC"
- **People**: "Sarah", "John Smith"
- **Themes/Categories**: "Shopping", "Health", "Finance"

## 🧠 Thinking Agent Design

The Thinking Agent analyzes the knowledge base to provide:

<p align="center">
      <img src="Image\Thinking_Agent_Screenshot 2025-11-02 161305.png" alt="Thinking_Agent_Screenshot" width="700" />
</p>

1. **Next Steps**: Suggests logical follow-up actions based on card history
2. **Recommendations**: Identifies patterns and optimization opportunities
3. **Conflict Detection**: Finds scheduling conflicts and resource overlaps
4. **Context Refinement**: Updates user context based on usage patterns


---

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/MATHUSHASIVA/Contextual-Personal-Assistant.git
cd Contextual-Personal-Assistant/contextual-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows (use `source venv/bin/activate` on Linux/Mac)

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Configuration

Create a `.env` file in the project root:

```env
# Required for LLM Workflow (get free at console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Database path (defaults to ./data/assistant.db)
DATABASE_PATH=./data/assistant.db

# Model configurations
LLM_MODEL=llama-3.3-70b-versatile
SPACY_MODEL=en_core_web_sm

# Optional: Force offline mode (uses only NLP workflow)
USE_LOCAL_ONLY=false
```

**Note:** If `GROQ_API_KEY` is not set, the system automatically uses NLP Fallback Workflow (offline mode).

### Running the Application

**Web Interface (Streamlit)**
```bash
# Start the Streamlit web interface
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

---


## 🚧 Future Enhancements

- [ ] Multi-agent collaboration for complex task decomposition
- [ ] Voice input integration for hands-free note taking
- [ ] Mobile app interface for on-the-go capture
- [ ] Bulk card operations (multi-select, batch status updates)
- [ ] Team collaboration features (shared envelopes, assignments)
- [ ] Integration with external tools (Google Calendar, Outlook, Slack)
- [ ] Smart notifications based on context and priority

---