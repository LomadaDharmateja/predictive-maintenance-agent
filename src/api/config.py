"""Service configuration. Environment variables only.

Milestone 7 item 4. Nothing here reads a config file, and nothing here has a
value baked in that an operator would need to edit code to change. The one
thing that is *not* configuration is the credential: this module records which
provider is selected and never touches `ANTHROPIC_API_KEY`, which the SDK
resolves for itself (`src/agent/providers.py`).

`.env.example` carries the names of every variable below and no values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.agent.loop import ModelConfig

#: Every variable the service reads. Kept as data so `.env.example` and the
#: tests can both be checked against it rather than against a comment.
ENV_VARS = (
    "PDM_PROVIDER",
    "PDM_MODEL",
    "PDM_DATABASE",
    "PDM_RUN_STORE",
    "PDM_MAX_ITERATIONS",
    "PDM_LOG_LEVEL",
    "PDM_ENABLE_DOCS",
    "PDM_OLLAMA_BASE_URL",
    # Demo mode replays recorded transcripts instead of calling a model.
    # Defaults to on: the page has to work for a visitor with no key, and a
    # default that spends money on first click is the wrong default.
    "PDM_DEMO_MODE",
    "PDM_TRANSCRIPTS",
    "PDM_SCENARIOS",
    # Resolved by the Anthropic SDK, never read by this repository. Named here
    # so `.env.example` documents it; nothing below ever loads its value.
    "ANTHROPIC_API_KEY",
)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model_name: str = "qwen3:4b-q4_K_M"
    database: str = "data/pdm.db"
    run_store: str = "data/runs"
    max_iterations: int = 6
    log_level: str = "INFO"
    enable_docs: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    #: Replay recorded transcripts rather than calling a provider. On by
    #: default so the demo page costs nothing and needs no credential.
    demo_mode: bool = True
    transcripts: str = "evals/transcripts"
    scenarios: str = "evals/scenarios.yaml"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.environ.get("PDM_PROVIDER", "ollama"),
            model_name=os.environ.get("PDM_MODEL", "qwen3:4b-q4_K_M"),
            database=os.environ.get("PDM_DATABASE", "data/pdm.db"),
            run_store=os.environ.get("PDM_RUN_STORE", "data/runs"),
            max_iterations=int(os.environ.get("PDM_MAX_ITERATIONS", "6")),
            log_level=os.environ.get("PDM_LOG_LEVEL", "INFO"),
            enable_docs=_flag("PDM_ENABLE_DOCS", True),
            ollama_base_url=os.environ.get(
                "PDM_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            demo_mode=_flag("PDM_DEMO_MODE", True),
            transcripts=os.environ.get("PDM_TRANSCRIPTS", "evals/transcripts"),
            scenarios=os.environ.get("PDM_SCENARIOS", "evals/scenarios.yaml"),
        )

    def model_config_for_agent(self) -> ModelConfig:
        from dataclasses import replace

        config = ModelConfig(provider=self.provider)
        if self.provider == "anthropic":
            return replace(config, model=self.model_name)
        return replace(
            config, local_model=self.model_name, local_base_url=self.ollama_base_url
        )

    def build_client(self):
        """The provider adapter, or None if the provider is unknown.

        Returns rather than raises so the caller maps it to a 503 instead of a
        traceback: an unknown provider is a deployment mistake, and the service
        should say so in a status code.
        """
        from src.agent.providers import ProviderError, build_client

        try:
            return build_client(self.model_config_for_agent())
        except (ProviderError, Exception):  # noqa: BLE001 - reported as 503
            return None
