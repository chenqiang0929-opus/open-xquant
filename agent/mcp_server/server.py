from __future__ import annotations

import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from oxq.tools import registry

mcp = FastMCP("open-xquant")

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Auto-register all SDK tools from the registry
for tool_def in registry.all_tools():
    mcp.tool(name=tool_def.name, description=tool_def.description)(tool_def.fn)


# MCP-only tool (not part of oxq SDK)
@mcp.tool(
    name="get_current_date",
    description=(
        "Get today's date. Call this FIRST when the user mentions "
        "relative dates like 'last 6 months', 'recent year', etc."
    ),
)
def get_current_date() -> dict[str, str]:
    from datetime import date

    today = date.today()
    return {"today": today.isoformat(), "weekday": today.strftime("%A")}


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from a markdown file."""
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


@mcp.tool(
    name="skill_list",
    description=(
        "List all available agent skills with their names and descriptions. "
        "Call this at the start of a conversation to discover which skill "
        "best matches the user's intent."
    ),
)
def skill_list() -> dict[str, object]:
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.md")):
        if f.stat().st_size <= 1:
            continue
        fm = _parse_frontmatter(f.read_text())
        if fm.get("redirect"):
            continue
        skills.append({
            "name": fm.get("name", f.stem),
            "description": fm.get("description", ""),
        })
    return {"skills": skills}


@mcp.tool(
    name="skill_load",
    description=(
        "Load a skill by name and return its full instructions. "
        "Call this after skill_list to get detailed guidance for the task."
    ),
)
def skill_load(name: str) -> dict[str, str]:
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return {"error": f"Skill '{name}' not found"}
    content = path.read_text()
    fm = _parse_frontmatter(content)
    if fm.get("redirect"):
        redirect = fm["redirect"]
        path = SKILLS_DIR / f"{redirect}.md"
        if not path.exists():
            return {"error": f"Redirected skill '{redirect}' not found"}
        content = path.read_text()
    return {"name": name, "content": content}


if __name__ == "__main__":
    mcp.run(transport="stdio")
