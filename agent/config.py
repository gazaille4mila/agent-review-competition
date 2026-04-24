"""
Configuration loader for the Koala Science review agent.

References
----------
- Koala Science competition: https://koala.science/competition
- GitHub Models API (model names): https://github.com/marketplace/models
- GitHub Copilot CLI: https://github.com/github/copilot-cli
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Holds all runtime configuration loaded from environment variables."""

    # Koala Science credentials
    koala_api_key: str = field(default_factory=lambda: os.environ["KOALA_API_KEY"])
    koala_agent_id: str = field(default_factory=lambda: os.environ["KOALA_AGENT_ID"])

    # GitHub Models
    gh_model: str = field(
        default_factory=lambda: os.getenv("GH_MODEL", "gpt-4o")
    )
    max_papers_per_run: int = field(
        default_factory=lambda: int(os.getenv("MAX_PAPERS_PER_RUN", "5"))
    )
    loop_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("LOOP_INTERVAL_SECONDS", "900"))
    )
    min_karma_threshold: float = field(
        default_factory=lambda: float(os.getenv("MIN_KARMA_THRESHOLD", "10"))
    )

    # MCP endpoint
    mcp_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "MCP_ENDPOINT", "https://koala.science/mcp"
        )
    )

    # State persistence
    state_file: str = field(
        default_factory=lambda: os.getenv("STATE_FILE", "agent_state.json")
    )

    # Trajectory log file (required for prize eligibility)
    trajectory_log_file: str = field(
        default_factory=lambda: os.getenv("TRAJECTORY_LOG_FILE", "trajectory.log")
    )


def load_config() -> Config:
    """Load and return configuration, raising on missing required variables."""
    required = ["KOALA_API_KEY", "KOALA_AGENT_ID"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    return Config()
