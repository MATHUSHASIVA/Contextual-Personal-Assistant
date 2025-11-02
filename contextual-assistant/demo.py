"""
Demo script for Contextual Personal Assistant
Demonstrates the full capabilities of the system with sample notes
"""

# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Local imports
from src.agents.ingestion_agent import IngestionAgent
from src.storage.database import DatabaseManager

console = Console()

# Sample notes demonstrating different card types and scenarios
DEMO_NOTES = [
    # Tasks
    "Call Sarah about the Q3 budget review next Monday at 2pm",
    "Email the client proposal to John by Friday",
    "Buy groceries - milk, eggs, bread, and coffee",
    "Schedule team meeting to discuss project timeline",
    "Complete the design mockups for the new website",
    
    # Reminders
    "Remember to pick up the package from post office today",
    "Don't forget: doctor's appointment tomorrow at 10am",
    "Renew car insurance before it expires on November 15th",
    "Meeting with Microsoft team on Thursday afternoon",
    
    # Ideas
    "Blog post idea: How to improve productivity with AI tools",
    "Maybe we should consider using a different color scheme for the app",
    "Interesting concept: gamification for employee onboarding",
    "Design inspiration: minimalist dashboard with dark mode",
    
    # Mixed scenarios
    "Follow up with Sarah about the budget and send her the updated spreadsheet",
    "Research best practices for API design and document findings",
]


def display_banner():
    """Display demo banner"""
    console.print("\n" + "=" * 63, style="bold cyan")
    console.print("[bold cyan]  CONTEXTUAL PERSONAL ASSISTANT - DEMO[/bold cyan]".center(63))
    console.print("=" * 63, style="bold cyan")
    console.print("\n[cyan]Demonstrating Dual Processing Workflows:[/cyan]")
    console.print("  [green]*[/green] LLM-Powered Processing (Groq Llama 3.3)")
    console.print("  [green]*[/green] NLP Fallback Processing (spaCy + Patterns)\n")


def display_processing_mode(agent):
    """Display which processing mode is active"""
    has_llm = agent.llm is not None
    
    if has_llm:
        mode_panel = Panel.fit(
            "[bold green]🌐 LLM-Powered Mode Active[/bold green]\n"
            "Using Groq Llama 3.3 70B for intelligent processing",
            border_style="green",
            title="Processing Mode"
        )
    else:
        mode_panel = Panel.fit(
            "[bold yellow]🔌 NLP Fallback Mode Active[/bold yellow]\n"
            "Using spaCy + Pattern Matching (Offline Processing)",
            border_style="yellow",
            title="Processing Mode"
        )
    
    console.print(mode_panel)
    console.print()


def process_demo_notes():
    """Process all demo notes"""
    console.print("[bold]Initializing Contextual Personal Assistant...[/bold]\n")
    
    # Initialize agent
    agent = IngestionAgent()
    
    # Display processing mode
    display_processing_mode(agent)
    
    # Statistics
    total_notes = len(DEMO_NOTES)
    successful = 0
    failed = 0
    total_time = 0
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for i, note in enumerate(DEMO_NOTES, 1):
            task = progress.add_task(f"Processing note {i}/{total_notes}...", total=None)
            
            try:
                result = agent.process_note(note)
                
                if result.success:
                    successful += 1
                    total_time += result.processing_time_ms
                    results.append({
                        'note': note,
                        'card': result.card,
                        'envelope': result.envelope,
                        'time': result.processing_time_ms
                    })
                else:
                    failed += 1
                    error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown error"
                    console.print(f"[red]✗ Failed:[/red] {note[:50]}...")
                    console.print(f"   [dim red]Error: {error_msg}[/dim red]")
                
                progress.update(task, completed=True)
                
            except Exception as e:
                failed += 1
                console.print(f"[red]✗ Error:[/red] {note[:50]}...")
                console.print(f"   [dim red]Exception: {str(e)}[/dim red]")
                progress.update(task, completed=True)
    return results, successful, failed, total_time


def display_results(results, successful, failed, total_time):
    """Display processing results"""
    console.print("\n")
    console.print("═" * 70, style="cyan")
    console.print("[bold cyan]PROCESSING RESULTS[/bold cyan]")
    console.print("═" * 70, style="cyan")
    console.print()
    
    # Summary statistics
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="white", justify="right")
    
    summary_table.add_row("Total Notes Processed", str(successful + failed))
    summary_table.add_row("Successful", f"[green]{successful}[/green]")
    summary_table.add_row("Failed", f"[red]{failed}[/red]")
    summary_table.add_row("Average Processing Time", f"{total_time / successful:.0f}ms" if successful > 0 else "N/A")
    summary_table.add_row("Total Processing Time", f"{total_time:.0f}ms")
    
    console.print(summary_table)
    console.print()
    
    # Detailed results
    console.print("[bold]Processed Cards:[/bold]\n")
    
    results_table = Table(show_header=True, header_style="bold magenta", title="Card Details")
    results_table.add_column("#", style="cyan", width=3)
    results_table.add_column("Type", style="yellow", width=10)
    results_table.add_column("Note Preview", style="white", width=35)
    results_table.add_column("Envelope", style="green", width=15)
    results_table.add_column("Time", style="blue", width=8)
    
    for i, result in enumerate(results, 1):
        note_preview = result['note'][:32] + "..." if len(result['note']) > 35 else result['note']
        envelope_name = result['envelope'].name if result['envelope'] else "None"
        
        results_table.add_row(
            str(i),
            result['card'].card_type.upper(),
            note_preview,
            envelope_name,
            f"{result['time']}ms"
        )
    
    console.print(results_table)
    console.print()


def display_card_type_distribution(results):
    """Display distribution of card types"""
    from collections import Counter
    
    card_types = [r['card'].card_type for r in results]
    type_counts = Counter(card_types)
    
    console.print("[bold]Card Type Distribution:[/bold]\n")
    
    dist_table = Table(show_header=True, header_style="bold magenta")
    dist_table.add_column("Card Type", style="yellow", width=15)
    dist_table.add_column("Count", style="white", justify="right", width=10)
    dist_table.add_column("Percentage", style="cyan", justify="right", width=15)
    
    total = len(results)
    for card_type, count in sorted(type_counts.items()):
        percentage = (count / total) * 100
        dist_table.add_row(
            card_type.upper(),
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(dist_table)
    console.print()


def display_envelope_summary(results):
    """Display envelope grouping summary"""
    from collections import Counter
    
    envelopes = [r['envelope'].name for r in results if r['envelope']]
    envelope_counts = Counter(envelopes)
    
    if envelope_counts:
        console.print("[bold]Envelope Grouping:[/bold]\n")
        
        env_table = Table(show_header=True, header_style="bold magenta")
        env_table.add_column("Envelope Name", style="green", width=30)
        env_table.add_column("Cards", style="white", justify="right", width=10)
        
        for envelope, count in sorted(envelope_counts.items(), key=lambda x: x[1], reverse=True):
            env_table.add_row(envelope, str(count))
        
        console.print(env_table)
        console.print()


def display_sample_cards(results):
    """Display sample detailed cards"""
    console.print("[bold]Sample Card Details:[/bold]\n")
    
    # Show first 3 cards in detail
    for i, result in enumerate(results[:3], 1):
        card = result['card']
        envelope = result['envelope']
        
        card_details = f"""[bold]Note:[/bold] {result['note']}

[cyan]Card Type:[/cyan] {card.card_type.upper()}
[cyan]Description:[/cyan] {card.description or 'N/A'}
[cyan]Assignee:[/cyan] {card.assignee or 'None'}
[cyan]Due Date:[/cyan] {card.due_date or 'None'}
[cyan]Status:[/cyan] {card.status.upper()}
[cyan]Envelope:[/cyan] {envelope.name if envelope else 'None'} ({envelope.envelope_type if envelope else 'N/A'})
[cyan]Processing Time:[/cyan] {result['time']}ms"""
        
        panel = Panel.fit(
            card_details,
            border_style="cyan",
            title=f"Card #{i}"
        )
        console.print(panel)
        console.print()


def display_feature_highlights():
    """Display system feature highlights"""
    console.print("═" * 70, style="cyan")
    console.print("[bold cyan]SYSTEM FEATURES DEMONSTRATED[/bold cyan]")
    console.print("═" * 70, style="cyan")
    console.print()
    
    features = [
        ("🎯", "Pattern-Based Classification", "50+ task, 40+ reminder, 35+ idea patterns"),
        ("🏷️", "Entity Extraction", "People, dates, organizations, locations detected"),
        ("📊", "Entity Boost System", "Date/people entities adjust classification scores"),
        ("📁", "Smart Envelope Matching", "Keyword overlap + semantic grouping"),
        ("💡", "Intelligent Naming", "Meaningful envelope names from note content"),
        ("⚡", "Fast Processing", "Average <100ms per note with NLP mode"),
        ("🔄", "Automatic Fallback", "Seamless switch between LLM and NLP modes"),
        ("🗄️", "Persistent Storage", "SQLite database with full context tracking"),
    ]
    
    for emoji, feature, description in features:
        console.print(f"{emoji} [bold]{feature}[/bold]: [dim]{description}[/dim]")
    console.print()


def main():
    """Main demo function"""
    try:
        # Clear screen and display banner
        console.clear()
        display_banner()
        console.print()
        
        # Process demo notes
        results, successful, failed, total_time = process_demo_notes()
        
        if not results:
            console.print("[red]No results to display. Demo failed.[/red]")
            return
        
        # Display various summaries
        display_results(results, successful, failed, total_time)
        display_card_type_distribution(results)
        display_envelope_summary(results)
        display_sample_cards(results)
        display_feature_highlights()
        
        # Final message
        console.print("═" * 70, style="green")
        console.print("[bold green]✓ DEMO COMPLETED SUCCESSFULLY[/bold green]")
        console.print("═" * 70, style="green")
        console.print()
        console.print("[dim]Run 'python cli.py list-cards' to see all processed cards[/dim]")
        console.print("[dim]Run 'python cli.py stats' to see database statistics[/dim]")
        console.print("[dim]Run 'streamlit run app.py' to open the web interface[/dim]")
        console.print()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Demo error:[/bold red] {str(e)}")
        console.print_exception()


if __name__ == "__main__":
    main()
