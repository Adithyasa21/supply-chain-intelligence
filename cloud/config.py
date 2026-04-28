"""
Azure configuration loader.

Priority order for secrets:
  1. Environment variables (CI/CD pipelines, GitHub Actions, local .env)
  2. Azure Key Vault (production — secrets never touch environment)
  3. Graceful local-only fallback (no Azure credentials = local mode)

Authentication uses DefaultAzureCredential which chains:
  EnvironmentCredential → ManagedIdentityCredential → AzureCliCredential
  → AzurePowerShellCredential → InteractiveBrowserCredential

This means the SAME code works in:
  - Local dev    (az login)
  - GitHub CI    (AZURE_CLIENT_ID / SECRET / TENANT env vars)
  - Azure VMs    (Managed Identity — zero secrets in config)
  - AKS pods     (Workload Identity)
  - Local demo   (AZURE_USE_EMULATOR=true → Azurite)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Azurite well-known development account (public, safe to commit)
AZURITE_ACCOUNT = "devstoreaccount1"
AZURITE_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq"
    "/K1SZFPTOtr/KBHBeksoGMGw=="
)
AZURITE_URL = "http://127.0.0.1:10000/devstoreaccount1"


@dataclass
class AzureConfig:
    # ── Storage ───────────────────────────────────────────────────────────────
    storage_account_name: str = ""
    storage_filesystem: str = "supply-chain"

    # ── Emulator flag ─────────────────────────────────────────────────────────
    use_emulator: bool = False   # True = Azurite (local demo mode)

    # ── Key Vault ─────────────────────────────────────────────────────────────
    key_vault_url: str = ""

    # ── Azure ML ──────────────────────────────────────────────────────────────
    ml_subscription_id: str = ""
    ml_resource_group: str = ""
    ml_workspace_name: str = ""
    ml_experiment_name: str = "supply-chain-demand-forecasting"

    # ── Derived flags ─────────────────────────────────────────────────────────
    enabled: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.use_emulator:
            self.storage_account_name = AZURITE_ACCOUNT
        self.enabled = bool(self.storage_account_name)

    @property
    def storage_url(self) -> str:
        if self.use_emulator:
            return AZURITE_URL
        return f"https://{self.storage_account_name}.dfs.core.windows.net"

    @property
    def ml_configured(self) -> bool:
        return bool(self.ml_subscription_id and self.ml_resource_group and self.ml_workspace_name)

    @property
    def keyvault_configured(self) -> bool:
        return bool(self.key_vault_url)

    @property
    def mode_label(self) -> str:
        if self.use_emulator:
            return "Azurite (local emulator)"
        if self.enabled:
            return "Azure Cloud"
        return "Local only"


def load_config() -> AzureConfig:
    use_emulator = os.getenv("AZURE_USE_EMULATOR", "").lower() in ("1", "true", "yes")

    cfg = AzureConfig(
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME", ""),
        storage_filesystem=os.getenv("AZURE_STORAGE_FILESYSTEM", "supply-chain"),
        use_emulator=use_emulator,
        key_vault_url=os.getenv("AZURE_KEY_VAULT_URL", ""),
        ml_subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        ml_resource_group=os.getenv("AZURE_ML_RESOURCE_GROUP", ""),
        ml_workspace_name=os.getenv("AZURE_ML_WORKSPACE_NAME", ""),
        ml_experiment_name=os.getenv(
            "AZURE_ML_EXPERIMENT_NAME", "supply-chain-demand-forecasting"
        ),
    )

    # Enrich from Key Vault if configured
    if cfg.keyvault_configured and not cfg.storage_account_name:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            kv = SecretClient(
                vault_url=cfg.key_vault_url,
                credential=DefaultAzureCredential(),
            )

            def _kv(name: str, fallback: str = "") -> str:
                try:
                    return kv.get_secret(name).value or fallback
                except Exception:
                    return fallback

            cfg.storage_account_name = _kv("storage-account-name")
            cfg.ml_subscription_id = _kv("subscription-id", cfg.ml_subscription_id)
            cfg.ml_resource_group = _kv("ml-resource-group", cfg.ml_resource_group)
            cfg.ml_workspace_name = _kv("ml-workspace-name", cfg.ml_workspace_name)
            cfg.__post_init__()
        except Exception:
            pass

    return cfg


azure_config = load_config()
