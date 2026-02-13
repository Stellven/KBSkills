"""CLI entry point for KBSkills."""

import click
from rich.console import Console
from rich.table import Table

from kbskills.config import load_config, save_config, Config, CONFIG_FILE

console = Console()


@click.group()
@click.pass_context
def cli(ctx):
    """KBSkills - Local knowledge base with GraphRAG and Skills-driven outline generation."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()


# ─── init ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--api-key", prompt="Gemini API Key", hide_input=True, help="Your Google Gemini API key")
@click.pass_context
def init(ctx, api_key: str):
    """Initialize KBSkills configuration."""
    config = ctx.obj["config"]
    config.gemini_api_key = api_key
    config.ensure_dirs()
    save_config(config)
    console.print(f"[green]Configuration saved to {CONFIG_FILE}[/green]")
    console.print(f"[green]Data directory: {config.data_dir}[/green]")
    console.print(f"[green]Skills directory: {config.skills_dir}[/green]")
    console.print(f"[green]Output directory: {config.output_dir}[/green]")


# ─── ingest ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dir", "source_dir", type=click.Path(exists=True), help="Local directory to ingest")
@click.option("--urls", "urls_file", type=click.Path(exists=True), help="Text file containing URLs (one per line)")
@click.pass_context
def ingest(ctx, source_dir: str | None, urls_file: str | None):
    """Ingest data sources into the knowledge base."""
    if not source_dir and not urls_file:
        console.print("[red]Error: Provide at least one of --dir or --urls[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    config.ensure_dirs()

    if not config.gemini_api_key:
        console.print("[red]Error: Gemini API key not configured. Run 'kbskills init' first.[/red]")
        raise SystemExit(1)

    from kbskills.ingestion import pipeline
    pipeline.run_ingestion(config, source_dir=source_dir, urls_file=urls_file)


# ─── status ──────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def status(ctx):
    """Show knowledge base status."""
    config = ctx.obj["config"]
    from kbskills.knowledge.store import get_kb_status
    info = get_kb_status(config)

    table = Table(title="Knowledge Base Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in info.items():
        table.add_row(k, str(v))
    console.print(table)


# ─── skills ──────────────────────────────────────────────────────────────────

@cli.group()
def skills():
    """Manage thinking skills."""
    pass


@skills.command("list")
@click.pass_context
def skills_list(ctx):
    """List all available skills."""
    config = ctx.obj["config"]
    from kbskills.skills.loader import load_all_skills

    all_skills = load_all_skills(config.skills_dir)
    if not all_skills:
        console.print("[yellow]No skills found. Add YAML files to the skills/ directory.[/yellow]")
        return

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Domains")

    for skill in all_skills:
        table.add_row(
            skill.metadata.name,
            skill.metadata.display_name,
            skill.metadata.version,
            skill.metadata.description,
            ", ".join(skill.metadata.trigger.domains[:3]),
        )
    console.print(table)


@skills.command("show")
@click.argument("name")
@click.pass_context
def skills_show(ctx, name: str):
    """Show details of a specific skill."""
    config = ctx.obj["config"]
    from kbskills.skills.loader import load_all_skills

    all_skills = load_all_skills(config.skills_dir)
    skill = next((s for s in all_skills if s.metadata.name == name), None)
    if not skill:
        console.print(f"[red]Skill '{name}' not found.[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]{skill.metadata.display_name}[/bold cyan] (v{skill.metadata.version})")
    console.print(f"[dim]{skill.metadata.description}[/dim]\n")

    console.print("[bold]Trigger Domains:[/bold]", ", ".join(skill.metadata.trigger.domains))
    console.print("[bold]Keywords:[/bold]", ", ".join(skill.metadata.trigger.keywords))
    console.print("[bold]Threshold:[/bold]", skill.metadata.trigger.threshold)
    console.print()

    console.print("[bold]Thinking Framework:[/bold]")
    console.print(skill.thinking_framework.description)

    if skill.thinking_framework.steps:
        console.print("\n[bold]Steps:[/bold]")
        for i, step in enumerate(skill.thinking_framework.steps, 1):
            console.print(f"  {i}. {step.name}")


@skills.command("match")
@click.argument("topic")
@click.pass_context
def skills_match(ctx, topic: str):
    """Test which skills match a given topic (without executing query)."""
    config = ctx.obj["config"]

    if not config.gemini_api_key:
        console.print("[red]Error: Gemini API key not configured. Run 'kbskills init' first.[/red]")
        raise SystemExit(1)

    from kbskills.skills.loader import load_all_skills
    from kbskills.skills.matcher import SkillMatcher

    all_skills = load_all_skills(config.skills_dir)
    if not all_skills:
        console.print("[yellow]No skills found.[/yellow]")
        return

    matcher = SkillMatcher(config)
    matches = matcher.match(topic, all_skills)

    if not matches:
        console.print("[yellow]No skills matched for this topic.[/yellow]")
        return

    table = Table(title=f"Skill Matches for: {topic}")
    table.add_column("Skill", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Matched Domains")
    table.add_column("Matched Keywords")

    for m in matches:
        table.add_row(
            m.skill.metadata.display_name,
            f"{m.score:.2f}",
            ", ".join(m.matched_domains),
            ", ".join(m.matched_keywords),
        )
    console.print(table)


# ─── query ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("topic")
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: auto-generated)")
@click.option("--mode", type=click.Choice(["naive", "local", "global", "hybrid"]), default=None,
              help="Search mode (default: from config)")
@click.pass_context
def query(ctx, topic: str, output: str | None, mode: str | None):
    """Generate an outline for a topic based on knowledge base."""
    config = ctx.obj["config"]

    if not config.gemini_api_key:
        console.print("[red]Error: Gemini API key not configured. Run 'kbskills init' first.[/red]")
        raise SystemExit(1)

    search_mode = mode or config.default_search_mode

    from kbskills.agent.topic_agent import TopicAgent
    agent = TopicAgent(config)
    result_path = agent.run(topic, search_mode=search_mode, output_path=output)

    console.print(f"\n[green]Outline generated: {result_path}[/green]")


if __name__ == "__main__":
    cli()
