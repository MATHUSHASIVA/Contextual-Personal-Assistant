"""
Streamlit Web Interface for the Contextual Personal Assistant
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="🧠 Contextual Personal Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our components
try:
    from src.agents.ingestion_agent import IngestionAgent
    from src.agents.thinking_agent import ThinkingAgent
    from src.utils.context_manager import ContextManager
    from src.storage.database import DatabaseManager, Card, Envelope, UserContext, ThinkingInsight
    from src.models.schemas import CardType, Status
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
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📝 Process Notes", "📋 View Cards", "📁 Envelopes", "🧠 Context", "💡 Insights", "📊 Analytics"]
    )
    
    # Route to appropriate page
    if page == "📝 Process Notes":
        process_notes_page()
    elif page == "📋 View Cards":
        view_cards_page()
    elif page == "📁 View Envelopes":
        envelopes_page()
    elif page == "🧠 Context Management":
        context_page()
    elif page == "💡 Insights":
        insights_page()
    elif page == "📊 Analytics":
        analytics_page()

def process_notes_page():
    """Page for processing new notes"""
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
    
    # Quick examples
    st.markdown("---")
    st.subheader("💡 Try these examples:")
    
    examples = [
        "Call Sarah about the Q3 budget next Monday",
        "Remember to pick up milk on the way home",
        "Idea: new logo should be blue and green",
        "Schedule meeting with the design team for project Alpha",
        "Buy groceries: bread, eggs, coffee",
        "Research competitors for the marketing campaign"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        col = cols[i % 2]
        with col:
            if st.button(f"📝 {example}", key=f"example_{i}"):
                st.rerun()

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
            # Convert to DataFrame for display
            card_data = []
            for card in cards:
                card_data.append({
                    "ID": card.id,
                    "Type": card.card_type.title(),
                    "Description": card.description,
                    "Status": card.status.replace('_', ' ').title(),
                    "Assignee": card.assignee.title() if card.assignee else "-",
                    "Due Date": card.due_date.strftime('%Y-%m-%d') if card.due_date else "-",
                    "Created": card.created_at.strftime('%Y-%m-%d %H:%M'),
                    "Keywords": ", ".join([kw.word for kw in card.keywords]) if card.keywords else "-"
                })
            
            df = pd.DataFrame(card_data)
            
            # Display cards
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Description": st.column_config.TextColumn(width="large"),
                    "Keywords": st.column_config.TextColumn(width="medium")
                }
            )
            
            # Card details modal
            if st.button("🔍 View Card Details"):
                card_id = st.number_input("Enter Card ID", min_value=1, value=cards[0].id)
                selected_card = session.query(Card).get(card_id)
                
                if selected_card:
                    st.json({
                        "id": selected_card.id,
                        "content": selected_card.content,
                        "description": selected_card.description,
                        "card_type": selected_card.card_type,
                        "status": selected_card.status,
                        "assignee": selected_card.assignee,
                        "due_date": selected_card.due_date.isoformat() if selected_card.due_date else None,
                        "location": selected_card.location,
                        "additional_entities": selected_card.get_additional_entities()
                    })
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
                                status_emoji = {
                                    "pending": "⏳",
                                    "in_progress": "🔄", 
                                    "completed": "✅",
                                    "cancelled": "❌"
                                }.get(card.status, "📋")
                                
                                st.write(f"{status_emoji} {card.description[:50]}...")
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
                        people_df = pd.DataFrame(analysis["most_active_people"])
                        st.dataframe(people_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No active people found in context.")
                
                with tab2:
                    if analysis["trending_themes"]:
                        themes_df = pd.DataFrame(analysis["trending_themes"])
                        fig = px.bar(themes_df, x="theme", y="frequency", title="Trending Themes")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trending themes found.")
                
                with tab3:
                    if analysis["active_projects"]:
                        projects_df = pd.DataFrame(analysis["active_projects"])
                        st.dataframe(projects_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No active projects found in context.")
                
                with tab4:
                    evolution = analysis["context_evolution"]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Contexts", evolution["total_contexts"])
                    with col2:
                        st.metric("Monthly New", evolution["monthly_new"])
                    with col3:
                        st.metric("Weekly New", evolution["weekly_new"])
                    with col4:
                        st.metric("Growth Rate", f"{evolution['growth_rate']:.1f}%")
                        
            except Exception as e:
                st.error(f"Error analyzing context: {str(e)}")
    
    # Current context display
    st.subheader("Current Context")
    
    try:
        current_context = st.session_state.context_manager.get_current_context(limit_per_type=10)
        
        for context_type, contexts in current_context.items():
            if contexts:
                st.write(f"**{context_type.title()}:**")
                
                context_data = []
                for context in contexts:
                    context_data.append({
                        "Name": context.name.title(),
                        "Relevance": f"{context.relevance_score:.2f}",
                        "Last Referenced": context.last_referenced.strftime('%Y-%m-%d')
                    })
                
                df = pd.DataFrame(context_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
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
                # Choose styling based on insight type
                type_colors = {
                    "next_step": "🟢",
                    "recommendation": "🔵", 
                    "conflict": "🔴",
                    "optimization": "🟡"
                }
                # Handle both enum and string types
                insight_type_str = insight.insight_type.value if hasattr(insight.insight_type, 'value') else insight.insight_type
                color = type_colors.get(insight_type_str, "⚫")
                
                # Create insight card
                with st.container():
                    st.markdown(f"### {color} {insight.title}")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(insight.description)
                        if insight.suggested_action:
                            st.info(f"💡 **Suggested Action:** {insight.suggested_action}")
                    
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
                dates = [item.date for item in cards_over_time]
                counts = [item.count for item in cards_over_time]
                
                # Create DataFrame for plotly
                df_timeline = pd.DataFrame({'Date': dates, 'Count': counts})
                
                fig = px.line(df_timeline, x='Date', y='Count', title="Cards Created Over Time")
                fig.update_layout(xaxis_title="Date", yaxis_title="Cards Created")
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
                types = [item.card_type.title() for item in card_types]
                counts = [item.count for item in card_types]
                
                fig = px.pie(values=counts, names=types, title="Card Type Distribution")
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
                status_names = [item.status.replace('_', ' ').title() for item in statuses]
                status_counts = [item.count for item in statuses]
                
                # Create DataFrame for plotly
                df_status = pd.DataFrame({'Status': status_names, 'Count': status_counts})
                
                fig = px.bar(df_status, x='Status', y='Count', title="Card Status Distribution")
                fig.update_layout(xaxis_title="Status", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No status data available.")
        
        with tab4:
            # Activity timeline
            recent_cards = session.query(Card).order_by(Card.created_at.desc()).limit(20).all()
            
            if recent_cards:
                timeline_data = []
                for card in recent_cards:
                    timeline_data.append({
                        "Date": card.created_at.strftime('%Y-%m-%d %H:%M'),
                        "Type": card.card_type.title(),
                        "Description": card.description[:50] + "..." if len(card.description) > 50 else card.description
                    })
                
                df = pd.DataFrame(timeline_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No recent activity data available.")
    
    finally:
        session.close()

if __name__ == "__main__":
    main()