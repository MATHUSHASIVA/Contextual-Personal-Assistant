"""
Streamlit Web Interface for the Contextual Personal Assistant
"""

# Standard library imports
import os
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import plotly.graph_objects as go
import streamlit as st

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="🧠 Contextual Personal Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local imports
try:
    from src.agents.ingestion_agent import IngestionAgent
    from src.agents.thinking_agent import ThinkingAgent
    from src.models.schemas import CardType, Status
    from src.storage.database import Card, DatabaseManager, Envelope, ThinkingInsight, UserContext
    from src.utils.context_manager import ContextManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory")
    st.stop()

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.agent = None
    st.session_state.thinking_agent = None
    st.session_state.context_manager = None

@st.cache_resource
def initialize_agents():
    """Initialize agents with caching"""
    try:
        database_path = os.getenv("DATABASE_PATH", "./data/assistant.db")
        
        # Ensure data directory exists
        Path("./data").mkdir(exist_ok=True)
        
        agent = IngestionAgent(database_url=f"sqlite:///{database_path}")
        thinking_agent = ThinkingAgent(database_url=f"sqlite:///{database_path}")
        context_manager = ContextManager(database_url=f"sqlite:///{database_path}")
        
        return agent, thinking_agent, context_manager
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        return None, None, None

def get_database_session():
    """Get database session"""
    database_path = os.getenv("DATABASE_PATH", "./data/assistant.db")
    db_manager = DatabaseManager(database_url=f"sqlite:///{database_path}")
    return db_manager.get_session()

# Emoji mapping constants
CARD_TYPE_EMOJI_UPPER = {"TASK": "🎯", "REMINDER": "🔔", "IDEA": "💡"}
CARD_TYPE_EMOJI_LOWER = {"task": "🎯", "reminder": "🔔", "idea": "💡"}
STATUS_EMOJI = {
    "pending": "⏳",
    "in_progress": "🔄", 
    "completed": "✅",
    "cancelled": "❌"
}
INSIGHT_TYPE_COLORS = {
    "next_step": "🟢",
    "recommendation": "🔵", 
    "conflict": "🔴",
    "optimization": "🟡"
}

def main():
    """Main Streamlit application"""
    
    # Initialize agents
    if not st.session_state.agents_initialized:
        with st.spinner("🚀 Initializing AI agents..."):
            agent, thinking_agent, context_manager = initialize_agents()
            if agent:
                st.session_state.agent = agent
                st.session_state.thinking_agent = thinking_agent
                st.session_state.context_manager = context_manager
                st.session_state.agents_initialized = True
                st.rerun()
            else:
                st.error("Failed to initialize agents. Please check your setup.")
                return
    
    # Header
    st.title("🧠 Contextual Personal Assistant")
    st.markdown("Transform your unstructured notes into organized, actionable knowledge")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Show workflow status in sidebar
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.sidebar.success("🤖 LLM Mode Active")
    else:
        st.sidebar.info("🔧 NLP Mode Active")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📝 Process Notes", "📋 View Cards", "📁 Envelopes", "🧠 Context", "💡 Insights", "📊 Analytics"]
    )
    
    # Route to appropriate page
    if page == "📝 Process Notes":
        process_notes_page()
    elif page == "📋 View Cards":
        view_cards_page()
    elif page == "📁 Envelopes":
        envelopes_page()
    elif page == "🧠 Context":
        context_page()
    elif page == "💡 Insights":
        insights_page()
    elif page == "📊 Analytics":
        analytics_page()

def process_notes_page():
    """Page for processing new notes"""
    # Show current workflow mode at the top
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.info("🤖 **Active Mode:** LLM-Powered Workflow (Groq Llama 3.3 70B)")
    else:
        st.warning("🔧 **Active Mode:** NLP Fallback Workflow (spaCy + Pattern Matching)")
    
    st.header("📝 Process New Notes")
    
    st.markdown("Enter your thoughts, ideas, or tasks below and watch them get organized automatically!")
    
    # Input form
    with st.form("note_form", clear_on_submit=True):
        note_text = st.text_area(
            "Enter your note:",
            placeholder="e.g., 'Call Sarah about the Q3 budget next Monday' or 'Idea: new logo should be blue and green'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 Process Note", type="primary")
        with col2:
            show_details = st.checkbox("Show processing details", value=False)
    
    if submitted and note_text.strip():
        with st.spinner("🔄 Processing your note..."):
            try:
                result = st.session_state.agent.process_note(note_text.strip())
                
                if result.success:
                    st.success("✅ Note processed successfully!")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Created Card")
                        card = result.card
                        
                        # Card info display
                        card_data = {
                            "Type": card.card_type.value.title(),
                            "Description": card.description,
                            "Status": card.status.value.replace('_', ' ').title()
                        }
                        
                        # Always show assignee and due date
                        card_data["Assignee"] = card.assignee.title() if card.assignee else "Not assigned"
                        card_data["Due Date"] = card.due_date.strftime('%Y-%m-%d') if card.due_date else "No due date"
                        if card.location:
                            card_data["Location"] = card.location
                        if card.keywords:
                            card_data["Keywords"] = ", ".join(card.keywords)
                        
                        for key, value in card_data.items():
                            st.write(f"**{key}:** {value}")
                    
                    with col2:
                        if result.envelope:
                            st.subheader("📁 Envelope")
                            envelope = result.envelope
                            st.write(f"**Name:** {envelope.name}")
                            st.write(f"**Type:** {envelope.envelope_type.title()}")
                            st.write(f"**Total Cards:** {envelope.card_count}")
                        else:
                            st.info("No envelope assigned")
                    
                    # Show processing details if requested
                    if show_details:
                        st.subheader("🔍 Processing Details")
                        details_col1, details_col2 = st.columns(2)
                        
                        with details_col1:
                            st.write(f"**Processing Time:** {result.processing_time_ms}ms")
                        
                        with details_col2:
                            # Extracted entities
                            if result.extracted_entities:
                                st.write("**Extracted Entities:**")
                                entities = result.extracted_entities
                                if entities.get('people'):
                                    st.write(f"  • People: {', '.join(entities['people'])}")
                                if entities.get('organizations'):
                                    st.write(f"  • Organizations: {', '.join(entities['organizations'])}")
                                if entities.get('locations'):
                                    st.write(f"  • Locations: {', '.join(entities['locations'])}")
                
                else:
                    st.error(f"❌ Processing failed: {result.error_message}")
                    
            except Exception as e:
                st.error(f"❌ Error processing note: {str(e)}")

def view_cards_page():
    """Page for viewing existing cards"""
    st.header("📋 View Cards")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        card_type_filter = st.selectbox(
            "Card Type",
            ["All", "Task", "Reminder", "Idea"],
            key="card_type_filter"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status", 
            ["All", "Pending", "In Progress", "Completed", "Cancelled"],
            key="status_filter"
        )
    
    with col3:
        limit = st.number_input("Limit", min_value=5, max_value=100, value=20)
    
    with col4:
        st.write("")  # Empty space for layout balance
    
    # Get cards from database
    session = get_database_session()
    try:
        query = session.query(Card)
        
        # Apply filters
        if card_type_filter != "All":
            query = query.filter(Card.card_type == card_type_filter.lower())
        
        if status_filter != "All":
            status_value = status_filter.lower().replace(' ', '_')
            query = query.filter(Card.status == status_value)
        
        cards = query.order_by(Card.created_at.desc()).limit(limit).all()
        
        if cards:
            # Build table data
            table_rows = []
            for card in cards:
                desc = card.description if card.description else "-"
                # Show full description without truncation
                
                keywords = ", ".join([kw.word for kw in card.keywords]) if card.keywords else "-"
                
                table_rows.append({
                    "ID": f"#{card.id}",
                    "Type": card.card_type.upper(),
                    "Description": desc,
                    "Status": card.status.replace('_', ' ').title(),
                    "Assignee": card.assignee if card.assignee else "-",
                    "Due Date": card.due_date.strftime('%Y-%m-%d') if card.due_date else "-",
                    "Envelope": card.envelope.name if card.envelope else "-",
                    "Keywords": keywords,
                    "Created": card.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            # Display using custom table layout (no pandas)
            # Header row
            st.markdown("### 📋 Cards Table")
            header_cols = st.columns([1, 2, 5, 2, 2, 2, 2, 3, 2, 2])
            headers = ["ID", "Type", "Description", "Status", "Assignee", "Due Date", "Envelope", "Keywords", "Created", "Action"]
            for col, header in zip(header_cols, headers):
                col.markdown(f"**{header}**")
            
            st.markdown("---")
            
            # Data rows with status update
            for idx, row in enumerate(table_rows):
                cols = st.columns([1, 2, 5, 2, 2, 2, 2, 3, 2, 2])
                
                # Color code the ID based on type
                emoji = CARD_TYPE_EMOJI_UPPER.get(row["Type"], "📝")
                
                cols[0].write(f"{emoji} {row['ID']}")
                cols[1].write(row["Type"])
                cols[2].write(row["Description"])
                cols[3].write(row["Status"])
                cols[4].write(row["Assignee"])
                cols[5].write(row["Due Date"])
                cols[6].write(row["Envelope"])
                cols[7].write(row["Keywords"][:30] + "..." if len(row["Keywords"]) > 30 else row["Keywords"])
                cols[8].write(row["Created"])
                
                # Status update dropdown
                card_id = int(row["ID"].replace("#", ""))
                status_options = ["Pending", "In Progress", "Completed", "Cancelled"]
                current_status = row["Status"]
                
                new_status = cols[9].selectbox(
                    "Update",
                    options=status_options,
                    index=status_options.index(current_status) if current_status in status_options else 0,
                    key=f"status_{card_id}_{idx}",
                    label_visibility="collapsed"
                )
                
                # Update status if changed
                if new_status != current_status:
                    card_to_update = session.query(Card).filter(Card.id == card_id).first()
                    if card_to_update:
                        card_to_update.status = new_status.lower().replace(' ', '_')
                        session.commit()
                        st.success(f"✅ Card #{card_id} status updated to {new_status}")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No cards found matching the selected criteria.")
    
    finally:
        session.close()

def envelopes_page():
    """Page for viewing envelopes"""
    st.header("📁 Envelopes")
    
    session = get_database_session()
    try:
        envelopes = session.query(Envelope).filter(
            Envelope.is_active == True
        ).order_by(Envelope.card_count.desc()).all()
        
        if envelopes:
            # Display envelope cards
            for envelope in envelopes:
                with st.expander(f"📁 {envelope.name} ({envelope.card_count} cards)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {envelope.envelope_type.title()}")
                        if envelope.description:
                            st.write(f"**Description:** {envelope.description}")
                        st.write(f"**Created:** {envelope.created_at.strftime('%Y-%m-%d')}")
                        
                        # Keywords
                        if envelope.keywords:
                            keywords = [kw.word for kw in envelope.keywords]
                            st.write(f"**Keywords:** {', '.join(keywords)}")
                    
                    with col2:
                        # Get cards in this envelope
                        envelope_cards = session.query(Card).filter(
                            Card.envelope_id == envelope.id
                        ).order_by(Card.created_at.desc()).limit(5).all()
                        
                        if envelope_cards:
                            st.write("**Recent Cards:**")
                            for card in envelope_cards:
                                emoji = STATUS_EMOJI.get(card.status, "📋")
                                # Show full description without truncation
                                st.write(f"{emoji} {card.description}")
        else:
            st.info("No active envelopes found.")
    
    finally:
        session.close()

def context_page():
    """Page for viewing user context"""
    st.header("🧠 User Context")
    
    # Context analysis toggle
    if st.button("🔍 Run Context Analysis"):
        with st.spinner("Analyzing context patterns..."):
            try:
                analysis = st.session_state.context_manager.analyze_context_patterns()
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["👥 People", "🎯 Themes", "🚀 Projects", "📊 Evolution"])
                
                with tab1:
                    if analysis["most_active_people"]:
                        st.write("**Most Active People:**")
                        for item in analysis["most_active_people"]:
                            cols = st.columns([3, 2, 2])
                            cols[0].write(str(item.get("name", "-")))
                            cols[1].write(f"Cards: {item.get('card_count', 0)}")
                            cols[2].write(f"Relevance: {item.get('relevance_score', 0):.2f}")
                            st.markdown("---")
                    else:
                        st.info("No active people found in context.")
                
                with tab2:
                    if analysis["trending_themes"]:
                        st.write("**Trending Themes:**")
                        for item in analysis["trending_themes"]:
                            cols = st.columns([3, 2])
                            cols[0].write(str(item.get("theme", "-")))
                            cols[1].write(f"Frequency: {item.get('frequency', 0)}")
                            st.markdown("---")
                    else:
                        st.info("No trending themes found.")
                
                with tab3:
                    if analysis["active_projects"]:
                        st.write("**Active Projects:**")
                        for item in analysis["active_projects"]:
                            cols = st.columns([3, 2, 2])
                            cols[0].write(str(item.get("name", "-")))
                            cols[1].write(f"Activity: {item.get('recent_activity', 0)}")
                            cols[2].write(f"Relevance: {item.get('relevance_score', 0):.2f}")
                            st.markdown("---")
                    else:
                        st.info("No active projects found in context.")
                
                with tab4:
                    evolution = analysis["context_evolution"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Contexts", evolution.get("total_contexts", 0))
                    with col2:
                        monthly = sum(evolution.get("contexts_by_type", {}).get("monthly", {}).values())
                        st.metric("Monthly New", monthly)
                    with col3:
                        recent = sum(evolution.get("contexts_by_type", {}).get("recent", {}).values())
                        st.metric("Recent (7d)", recent)
                    
                    # Show trending categories if available
                    if evolution.get("trending_categories"):
                        st.write("**📈 Trending Categories:**")
                        for cat in evolution["trending_categories"][:3]:
                            cols = st.columns([3, 2, 2])
                            cols[0].write(cat.get("type", "-").title())
                            cols[1].write(f"Growth: {cat.get('growth_rate', 0):.1f}%")
                            cols[2].write(f"Score: {cat.get('trend_score', 0):.1f}")
                        
            except Exception as e:
                st.error(f"Error analyzing context: {str(e)}")
    
    # Current context display
    st.subheader("Current Context")
    
    try:
        current_context = st.session_state.context_manager.get_current_context(limit_per_type=10)
        
        for context_type, contexts in current_context.items():
            if contexts:
                st.write(f"**{str(context_type).title()}:**")
                
                # Header
                cols = st.columns([4, 2, 3])
                cols[0].markdown("**Name**")
                cols[1].markdown("**Relevance**")
                cols[2].markdown("**Last Referenced**")
                st.markdown("---")
                
                # Data rows
                for context in contexts:
                    cols = st.columns([4, 2, 3])
                    cols[0].write(str(context.name).title())
                    cols[1].write(f"{float(context.relevance_score):.2f}")
                    cols[2].write(context.last_referenced.strftime('%Y-%m-%d'))
                
                st.write("")
    
    except Exception as e:
        st.error(f"Error loading context: {str(e)}")

def insights_page():
    """Page for viewing thinking agent insights"""
    st.header("💡 Thinking Agent Insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🧠 Run New Analysis", type="primary"):
            with st.spinner("Running thinking agent analysis..."):
                try:
                    results = st.session_state.thinking_agent.run_analysis()
                    
                    st.success(f"✅ Generated {results['insights_generated']} insights in {results['analysis_duration_ms']}ms")
                    
                    # Show category breakdown
                    categories = results["categories"]
                    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
                    
                    with cat_col1:
                        st.metric("➡️ Next Steps", categories["next_step"])
                    with cat_col2:
                        st.metric("💡 Recommendations", categories["recommendation"])
                    with cat_col3:
                        st.metric("⚠️ Conflicts", categories["conflict"])
                    with cat_col4:
                        st.metric("⚡ Optimizations", categories["optimization"])
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
    
    with col2:
        limit = st.number_input("Max insights to show", min_value=5, max_value=50, value=10)
    
    # Display current insights
    try:
        insights = st.session_state.thinking_agent.get_active_insights(limit=limit)
        
        if insights:
            for insight in insights:
                # Handle both enum and string types
                insight_type_str = insight.insight_type.value if hasattr(insight.insight_type, 'value') else insight.insight_type
                color = INSIGHT_TYPE_COLORS.get(insight_type_str, "⚫")
                
                # Create insight card
                with st.container():
                    st.markdown(f"### {color} {insight.title}")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(insight.description)
                        if insight.suggested_action:
                            st.info(f"💡 **Suggested Action:** {insight.suggested_action}")
                        
                        # Show related cards with their actual content
                        related_card_ids = insight.get_related_card_ids()
                        
                        if related_card_ids:
                            st.markdown("**📝 Related Cards:**")
                            session = get_database_session()
                            try:
                                for card_id in related_card_ids[:5]:  # Limit to 5 cards
                                    card = session.query(Card).filter(Card.id == card_id).first()
                                    if card:
                                        # Show card with emoji based on type
                                        emoji = CARD_TYPE_EMOJI_LOWER.get(card.card_type, "📝")
                                        st.markdown(f"- {emoji} {card.description}")
                            finally:
                                session.close()
                    
                    with col2:
                        # Handle both enum and string types  
                        insight_type_display = insight.insight_type.value if hasattr(insight.insight_type, 'value') else insight.insight_type
                        st.write(f"**Type:** {insight_type_display.replace('_', ' ').title()}")
                    
                    with col3:
                        if st.button(f"✅ Mark as Acted Upon", key=f"act_{insight.id}"):
                            st.session_state.thinking_agent.mark_insight_acted_upon(insight.id)
                            st.success("Marked as acted upon!")
                            st.rerun()
                        
                        if st.button(f"❌ Dismiss", key=f"dismiss_{insight.id}"):
                            st.session_state.thinking_agent.dismiss_insight(insight.id)
                            st.success("Insight dismissed!")
                            st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("No active insights found. Try running a new analysis!")
    
    except Exception as e:
        st.error(f"Error loading insights: {str(e)}")

def analytics_page():
    """Page for system analytics and statistics"""
    st.header("📊 System Analytics")
    
    session = get_database_session()
    try:
        # Get basic statistics
        from sqlalchemy import func, extract
        
        total_cards = session.query(Card).count()
        total_envelopes = session.query(Envelope).filter(Envelope.is_active == True).count()
        active_contexts = session.query(UserContext).filter(UserContext.is_active == True).count()
        recent_insights = session.query(ThinkingInsight).filter(
            ThinkingInsight.is_dismissed == False,
            ThinkingInsight.is_acted_upon == False
        ).count()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Total Cards", total_cards)
        with col2:
            st.metric("📁 Active Envelopes", total_envelopes)  
        with col3:
            st.metric("🧠 Active Contexts", active_contexts)
        with col4:
            st.metric("💡 Pending Insights", recent_insights)
        
        st.markdown("---")
        
        # Charts and visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Card Trends", "🎯 Type Distribution", "⚡ Status Overview", "📅 Timeline"])
        
        with tab1:
            # Card creation over time
            cards_over_time = session.query(
                func.date(Card.created_at).label('date'),
                func.count(Card.id).label('count')
            ).group_by(func.date(Card.created_at)).order_by('date').all()
            
            if cards_over_time:
                dates = [str(item.date) for item in cards_over_time]
                counts = [int(item.count) for item in cards_over_time]
                
                fig = go.Figure(data=go.Scatter(x=dates, y=counts, mode='lines+markers'))
                fig.update_layout(
                    title="Cards Created Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cards Created"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No card creation data available.")
        
        with tab2:
            # Card type distribution
            card_types = session.query(
                Card.card_type,
                func.count(Card.id).label('count')
            ).group_by(Card.card_type).all()
            
            if card_types:
                types = [str(item.card_type).title() for item in card_types]
                counts = [int(item.count) for item in card_types]
                
                fig = go.Figure(data=go.Pie(labels=types, values=counts))
                fig.update_layout(title="Card Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No card type data available.")
        
        with tab3:
            # Status distribution
            statuses = session.query(
                Card.status,
                func.count(Card.id).label('count')
            ).group_by(Card.status).all()
            
            if statuses:
                status_names = [str(item.status).replace('_', ' ').title() for item in statuses]
                status_counts = [int(item.count) for item in statuses]
                
                fig = go.Figure(data=go.Bar(x=status_names, y=status_counts))
                fig.update_layout(
                    title="Card Status Distribution",
                    xaxis_title="Status",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No status data available.")
        
        with tab4:
            # Activity timeline
            recent_cards = session.query(Card).order_by(Card.created_at.desc()).limit(20).all()
            
            if recent_cards:
                st.write("**Recent Activity Timeline:**")
                
                # Header
                cols = st.columns([3, 2, 7])
                cols[0].markdown("**Date**")
                cols[1].markdown("**Type**")
                cols[2].markdown("**Description**")
                st.markdown("---")
                
                # Data rows
                for card in recent_cards:
                    cols = st.columns([3, 2, 7])
                    cols[0].write(card.created_at.strftime('%Y-%m-%d %H:%M'))
                    
                    # Add emoji for type
                    emoji = CARD_TYPE_EMOJI_LOWER.get(card.card_type, "📝")
                    cols[1].write(f"{emoji} {card.card_type.title()}")
                    
                    # Show full description
                    cols[2].write(card.description)
            else:
                st.info("No recent activity data available.")
    
    finally:
        session.close()

if __name__ == "__main__":
    main()