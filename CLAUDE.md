# CLAUDE.md

## Project Overview

open-xquant is an Agent First quantitative trading framework. See `README.md` for motivation and `docs/architecture.md` for full design.

## Project Structure

- `src/oxq/` — main Python package (pip install open-xquant)
- `mcp_server/` — MCP protocol server exposing oxq as tools
- `skills/` — Agent skill definitions (markdown)
- `examples/` — example strategies, demo apps, tutorials
- `tests/` — mirrors src/oxq/ structure
- `docs/` — documentation

## Bug Fixing

Follow this strict TDD protocol for every bug fix:

1. **Write a failing test first** — describe the expected behavior with hand-calculated values, not values copied from buggy code.
2. **Run the test, confirm it fails for the right reason** — if it passes, your understanding of the bug is wrong.
3. **Implement the smallest possible fix** — no refactoring, no drive-by improvements.
4. **Run the new test** — confirm it passes.
5. **Run the full test suite** (`uv run pytest`) — confirm no regressions.
6. **Grep for the same pattern across the entire codebase** — fix ALL instances before declaring done. E.g., if `prices = {symbol: price}` is wrong in `risk.py`, check `entry.py`, `rebalance.py`, etc.

Never guess root causes — provide concrete evidence (specific line numbers, variable values) before proposing a fix.

## Cross-File Sync

When fixing a bug or updating logic in one module, always check and update all related modules that share the same pattern. Use `grep` to find all affected locations before editing any of them.

## Conventions

- Code and comments in English
- User-facing docs may be in Chinese
- Follow ruff defaults (E, F, I, N, W, UP rules)
- Prefer Protocol over ABC for interfaces
- Keep Indicator/Signal compute functions pure (no side effects)
- **Do not add new top-level directories** — this is an open-source project; keep the root structure stable. New apps, demos, and tutorials go under `examples/`.
