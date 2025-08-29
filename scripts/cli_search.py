#!/usr/bin/env python3
"""
CLI Search Interface for Hybrid Search Engine

Beautiful command-line interface for natural language search queries.
Displays Twitter content in a compact, colorful format with relevance scores.
"""

import asyncio
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.rule import Rule
from rich.markdown import Markdown


console = Console()

# API Configuration
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_SIZE = 8


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp to relative time."""
    try:
        tweet_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(tweet_time.tzinfo)
        diff = now - tweet_time
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "Just now"
    except:
        return "Unknown time"


def format_engagement_score(score: int) -> str:
    """Format engagement score with appropriate suffix."""
    if score >= 1000000:
        return f"{score/1000000:.1f}M"
    elif score >= 1000:
        return f"{score/1000:.1f}K"
    else:
        return str(score)


def format_follower_count(count: int) -> str:
    """Format follower count with appropriate suffix."""
    if count >= 1000000:
        return f"{count/1000000:.1f}M"
    elif count >= 1000:
        return f"{count/1000:.1f}K"
    else:
        return str(count)


def get_market_impact_emoji(impact: str) -> str:
    """Get emoji for market impact."""
    impact_emojis = {
        "bullish": "💎",
        "bearish": "🔻", 
        "neutral": "➖"
    }
    return impact_emojis.get(impact.lower(), "❓")


def get_rank_emoji(rank: int) -> str:
    """Get emoji for result ranking."""
    rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}
    return rank_emojis.get(rank, f"{rank}.")


def truncate_content(content: str, max_length: int = 100) -> str:
    """Truncate content while preserving words."""
    if len(content) <= max_length:
        return content
    
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Only truncate at word boundary if reasonably close
        truncated = truncated[:last_space]
    
    return truncated + "..."


def extract_entities_text(entities: List[Dict]) -> str:
    """Extract entity names for display."""
    if not entities:
        return ""
    
    entity_names = [entity.get("name", entity.get("entity_id", "")) for entity in entities]
    return ", ".join(entity_names[:3])  # Show max 3 entities


def display_search_result(result: Dict[str, Any], rank: int):
    """Display a single search result in a compact, beautiful format."""
    
    # Extract data
    content = result.get("content", "")
    source_handle = result.get("source_handle", "unknown")
    source_followers = result.get("source_followers", 0)
    engagement_score = result.get("engagement_score", 0)
    created_at = result.get("created_at", "")
    market_impact = result.get("market_impact", "neutral")
    entities = result.get("entities", [])
    
    # Scores
    final_score = result.get("final_score", 0.0)
    bm25_score = result.get("bm25_score", 0.0)
    vector_score = result.get("vector_score", 0.0)
    
    # Format components
    rank_emoji = get_rank_emoji(rank)
    impact_emoji = get_market_impact_emoji(market_impact)
    truncated_content = truncate_content(content, 100)
    entities_text = extract_entities_text(entities)
    relative_time = format_timestamp(created_at)
    
    # Create score text
    score_text = f"Score: {final_score:.3f}"
    if bm25_score > 0 or vector_score > 0:
        score_text += f" (BM25: {bm25_score:.2f}, Vector: {vector_score:.2f})"
    
    # Create content line
    content_line = Text()
    content_line.append(f"{rank_emoji} ", style="bold")
    content_line.append(score_text, style="cyan")
    content_line.append(f"  {impact_emoji} ", style="bold")
    content_line.append(market_impact.upper(), style="bold magenta")
    
    # Display content
    console.print(content_line)
    
    # Tweet content
    tweet_text = Text()
    tweet_text.append("📱 ", style="bold blue")
    tweet_text.append(truncated_content, style="white")
    
    if entities_text:
        tweet_text.append(f" [ENTITIES: {entities_text}]", style="dim yellow")
    
    console.print(tweet_text)
    
    # Source info
    source_text = Text()
    source_text.append("👤 ", style="bold green")
    source_text.append(f"@{source_handle}", style="bold green")
    source_text.append(f" ({format_follower_count(source_followers)} followers)", style="dim")
    source_text.append(" | ⚡ ", style="bold yellow")
    source_text.append(f"{format_engagement_score(engagement_score)} engagement", style="yellow")
    source_text.append(f" | 🕐 {relative_time}", style="dim")
    
    console.print(source_text)
    console.print()  # Empty line for spacing


def display_clustered_results(response: Dict[str, Any]):
    """Display clustered search results."""
    clusters = response.get("clusters", [])
    
    if not clusters:
        console.print("❌ No clusters found in results", style="red")
        return
        
    for i, cluster in enumerate(clusters):
        theme = cluster.get("theme", f"Cluster {i+1}")
        confidence = cluster.get("confidence", 0.0)
        results = cluster.get("results", [])
        
        # Cluster header
        header = Text()
        header.append(f"📂 Theme: {theme}", style="bold blue")
        header.append(f" (Confidence: {confidence:.2f})", style="dim")
        
        console.print(Panel(header, expand=False, style="blue"))
        
        # Display results in cluster
        for j, result in enumerate(results, 1):
            display_search_result(result, j)
        
        if i < len(clusters) - 1:  # Add separator between clusters
            console.print(Rule(style="dim"))


def call_search_api(query: str, api_base: str, size: int = DEFAULT_SIZE, clustered: bool = False) -> Optional[Dict[str, Any]]:
    """Call the search API with the given query."""
    
    endpoint = "/search/clustered" if clustered else "/ask"
    url = f"{api_base}{endpoint}"
    
    payload = {
        "query": query,
        "size": size
    }
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"🔍 Searching: '{query}'...", total=None)
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
        return response.json()
        
    except requests.exceptions.ConnectionError:
        console.print(f"❌ Cannot connect to API at {api_base}", style="red")
        console.print("💡 Make sure the API server is running: uvicorn src.api.main:app --reload --port 8000", style="yellow")
        return None
    except requests.exceptions.Timeout:
        console.print("⏱️  Request timed out", style="red")
        return None
    except requests.exceptions.RequestException as e:
        console.print(f"❌ API error: {e}", style="red")
        return None
    except json.JSONDecodeError:
        console.print("❌ Invalid JSON response from API", style="red")
        return None


@click.command()
@click.argument("query", required=False)
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--clustered", "-c", is_flag=True, help="Show clustered results for thematic organization")
@click.option("--size", "-s", default=DEFAULT_SIZE, help=f"Number of results to return (default: {DEFAULT_SIZE})")
@click.option("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
def search(query: Optional[str], interactive: bool, clustered: bool, size: int, api_base: str):
    """
    CLI Search Interface for Hybrid Search Engine
    
    Search Twitter content using natural language queries with beautiful formatting.
    
    Examples:
        cli_search.py "What's happening with Bitcoin?"
        cli_search.py "DeFi protocols" --clustered --size 10
        cli_search.py --interactive
    """
    
    # Display banner
    console.print()
    console.print("🚀 [bold blue]Hybrid Search Engine CLI[/bold blue]", justify="center")
    console.print("💎 [dim]Financial/Crypto Content Search[/dim]", justify="center")
    console.print()
    
    if interactive:
        # Interactive mode
        console.print("🔍 [bold green]Interactive Search Mode[/bold green]")
        console.print("💡 Type your natural language queries. Press Ctrl+C or type 'quit' to exit.")
        console.print()
        
        try:
            while True:
                query = Prompt.ask("🔎 [bold cyan]Search query[/bold cyan]")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!", style="green")
                    break
                
                if not query.strip():
                    console.print("⚠️  Please enter a search query", style="yellow")
                    continue
                
                console.print()
                perform_search(query, api_base, size, clustered)
                console.print()
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="green")
            
    elif query:
        # Single query mode
        perform_search(query, api_base, size, clustered)
        
    else:
        # No query provided
        console.print("❌ Please provide a query or use --interactive mode", style="red")
        console.print("💡 Example: python cli_search.py \"Bitcoin price analysis\"", style="yellow")
        sys.exit(1)


def perform_search(query: str, api_base: str, size: int, clustered: bool):
    """Perform a single search and display results."""
    
    # Call API
    response = call_search_api(query, api_base, size, clustered)
    
    if not response:
        return
    
    # Extract results
    if clustered:
        # Clustered response
        total_hits = response.get("total_results", 0)
        execution_time = response.get("search_time_ms", 0)
        clustering_time = response.get("clustering_time_ms", 0)
        
        # Results header
        header_text = f"📊 Found {total_hits} results in {execution_time}ms (+ {clustering_time}ms clustering)"
        console.print(header_text, style="bold green")
        console.print(Rule("─", style="green"))
        console.print()
        
        if total_hits > 0:
            display_clustered_results(response)
        else:
            console.print("🔍 No results found. Try a different query.", style="yellow")
            
    else:
        # Regular response
        results = response.get("results", [])
        total_hits = response.get("total_hits", len(results))
        execution_time = response.get("execution_time_ms", 0)
        
        # Results header
        header_text = f"📊 Found {total_hits} results in {execution_time}ms"
        console.print(header_text, style="bold green")
        console.print(Rule("─", style="green"))
        console.print()
        
        if results:
            # Display each result
            for i, result in enumerate(results, 1):
                display_search_result(result, i)
        else:
            console.print("🔍 No results found. Try a different query.", style="yellow")
    
    # Query suggestions for no results
    if (not clustered and len(response.get("results", [])) == 0) or \
       (clustered and response.get("total_results", 0) == 0):
        console.print("💡 [bold]Suggestions:[/bold]", style="cyan")
        console.print("   • Try broader terms: 'crypto' instead of specific tokens", style="dim")
        console.print("   • Use different keywords: 'DeFi', 'Bitcoin', 'market sentiment'", style="dim")
        console.print("   • Check spelling and try synonyms", style="dim")


if __name__ == "__main__":
    search()