"""
reva view — interactive Textual TUI for watching agent activity.

Rendering is driven by the ATIF trajectory, so backend-specific parsing lives
in `reva.translators` and visual style lives in `reva.render`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Markdown,
    RichLog,
    Select,
    TabbedContent,
    TabPane,
)

from reva.render import render_step_textual
from reva.session import SessionContext
from reva.tmux import list_sessions


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


class RevaViewer(App):
    TITLE = "reva viewer"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_agents", "Refresh"),
    ]
    CSS = """
    #toolbar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
    }
    #agent-select {
        width: 1fr;
    }
    #refresh-btn {
        width: 12;
        margin-left: 1;
    }
    #output-log {
        scrollbar-gutter: stable;
    }
    #prompt-scroll {
        padding: 1 2;
    }
    #agent-table {
        height: 1fr;
    }
    TabbedContent {
        height: 1fr;
    }
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self._current_agent: str | None = None
        self._tail_running = False
        self._known_agents: list[str] = []
        self._session: SessionContext | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar"):
            yield Label("Agent: ", classes="label")
            yield Select([], id="agent-select", prompt="— pick an agent —")
            yield Button("Refresh", id="refresh-btn", variant="primary")
        with TabbedContent():
            with TabPane("Output", id="tab-output"):
                yield RichLog(id="output-log", highlight=False, markup=False, wrap=True)
            with TabPane("System Prompt", id="tab-prompt"):
                with VerticalScroll(id="prompt-scroll"):
                    yield Markdown("", id="system-prompt")
            with TabPane("Agent Info", id="tab-info"):
                yield DataTable(id="agent-table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#agent-table", DataTable)
        table.add_columns("Field", "Value")
        self._populate_agent_list()
        self.set_interval(5, self._populate_agent_list)

    # ------------------------------------------------------------------ #
    # agent list
    # ------------------------------------------------------------------ #

    def _get_agent_names(self) -> list[str]:
        """All agents with a config.json (running or not)."""
        running = {s.agent_name for s in list_sessions()}
        names = set()
        if self.cfg.agents_dir.exists():
            for d in self.cfg.agents_dir.iterdir():
                if d.is_dir() and (d / "config.json").exists():
                    names.add(d.name)
        return sorted(running, key=str) + sorted(names - running, key=str)

    def _populate_agent_list(self) -> None:
        names = self._get_agent_names()
        if names == self._known_agents:
            return
        self._known_agents = names
        sel = self.query_one("#agent-select", Select)
        sel.set_options([(name, name) for name in names])
        if self._current_agent and self._current_agent in names:
            sel.value = self._current_agent

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        name = str(event.value)
        if name == self._current_agent:
            return
        self._current_agent = name
        self._load_agent(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-btn":
            self._populate_agent_list()

    def action_refresh_agents(self) -> None:
        self._populate_agent_list()

    # ------------------------------------------------------------------ #
    # agent loading
    # ------------------------------------------------------------------ #

    def _load_agent(self, name: str) -> None:
        agent_dir = self.cfg.agents_dir / name

        # output log
        log_widget = self.query_one("#output-log", RichLog)
        log_widget.clear()
        self._tail_running = False
        time.sleep(0.05)
        self._session = SessionContext.for_agent(agent_dir)
        log_path = agent_dir / "agent.log"
        if log_path.exists():
            self._tail_log(log_path)
        else:
            log_widget.write(Text("(no log yet — agent not yet launched)", style="dim"))

        # Read config.json once — used for both the prompt file selection
        # and the agent info table below.
        config_path = agent_dir / "config.json"
        cfg_data: dict | None = None
        if config_path.exists():
            try:
                cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg_data = None

        # system prompt — pick the right file for this backend
        prompt_widget = self.query_one("#system-prompt", Markdown)
        prompt_file = agent_dir / "prompt.md"  # fallback
        if cfg_data is not None:
            try:
                from reva.backends import get_backend
                backend_file = get_backend(cfg_data["backend"]).prompt_filename
                candidate = agent_dir / backend_file
                if candidate.exists():
                    prompt_file = candidate
            except Exception:
                pass
        if prompt_file.exists():
            self.call_later(prompt_widget.update, prompt_file.read_text(encoding="utf-8"))
        else:
            self.call_later(prompt_widget.update, "_No system prompt found._")

        # agent info
        table = self.query_one("#agent-table", DataTable)
        table.clear()
        if cfg_data is not None:
            for key, val in cfg_data.items():
                if isinstance(val, str) and "/" in val:
                    val = Path(val).name
                table.add_row(key, str(val))
        running = {s.agent_name for s in list_sessions()}
        table.add_row("status", "running" if name in running else "stopped")

    # ------------------------------------------------------------------ #
    # log tailing
    # ------------------------------------------------------------------ #

    @work(thread=True)
    def _tail_log(self, log_path: Path) -> None:
        self._tail_running = True
        log_widget = self.query_one("#output-log", RichLog)
        session = self._session
        last_flush = time.time()
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            while self._tail_running:
                line = fh.readline()
                if not line:
                    if session and time.time() - last_flush > 1.0:
                        try:
                            session.flush()
                        except Exception:
                            pass
                        last_flush = time.time()
                    time.sleep(0.2)
                    continue
                if session is None:
                    continue
                for step in session.consume_lines([line]):
                    for text_obj in render_step_textual(step):
                        self.call_from_thread(log_widget.write, text_obj)
        if session is not None:
            try:
                session.flush()
            except Exception:
                pass
