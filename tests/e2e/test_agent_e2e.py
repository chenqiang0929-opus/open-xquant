"""End-to-end Playwright tests for strategy-builder skill pipeline."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import API_KEY

USER_MESSAGE = (
    "我要构建一个 SMA 均线交叉策略。\n"
    "约束：初始资金 100000，品种 AAPL，日频。\n"
    "目标：total_return min 达到 -0.5, max_drawdown max 达到 0.0。\n"
    "假设：当 SMA(10) 从下方穿越 SMA(50) 时买入 AAPL，"
    "当 SMA(10) 回落到 SMA(50) 以下时卖出。\n"
    "本地已有 AAPL 数据，请直接构建策略并执行回测，"
    "回测区间 2024-01-01 到 2024-12-31。"
)


@pytest.mark.e2e
class TestStrategyBuilderE2E:
    """Test the strategy-builder skill through the agent demo UI."""

    def _setup_skill(self, page: Page) -> None:
        """Switch to Manual mode, select strategy-builder skill, then fill API key."""
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Switch to Manual skill mode first (triggers Streamlit rerun)
        sidebar.get_by_text("Manual").click()
        page.wait_for_selector('[data-testid="stChatInput"]', timeout=10000)

        # Select strategy-builder skill from dropdown (triggers another rerun)
        skill_select = sidebar.locator('[data-testid="stSelectbox"]').first
        skill_select.click()
        page.locator('[data-testid="stSelectboxVirtualDropdown"]').get_by_text(
            "strategy-builder"
        ).click()

        # Verify skill loaded
        expect(sidebar.get_by_text("Loaded Skill: strategy-builder")).to_be_visible(
            timeout=5000,
        )

        # Fill API key last so no subsequent rerun clears it
        sidebar.get_by_label("API Key").fill(API_KEY)

    def test_sma_crossover_full_pipeline(self, app_page: Page) -> None:
        """Send a comprehensive message and verify the agent uses strategy + engine tools."""
        self._setup_skill(app_page)

        # Send user message
        chat_input = app_page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill(USER_MESSAGE)
        chat_input.press("Enter")

        page_content = app_page.locator("main")

        # Wait for strategy_create tool call
        expect(
            page_content.locator('[data-testid="stAlert"]:has-text("strategy_create")').first,
        ).to_be_visible(timeout=120_000)

        # Wait for engine_run tool call
        expect(
            page_content.locator('[data-testid="stAlert"]:has-text("engine_run")').first,
        ).to_be_visible(timeout=120_000)

        # Wait for results (total_return metric in tool result)
        expect(
            page_content.locator('[data-testid="stAlert"]:has-text("total_return")').first,
        ).to_be_visible(timeout=120_000)

        # No Streamlit exceptions
        error_alerts = page_content.locator('[data-testid="stException"]')
        expect(error_alerts).to_have_count(0)
