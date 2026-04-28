#!/usr/bin/env bash
# =============================================================================
# Supply Chain Intelligence Platform — Azure Resource Provisioning
#
# Provisions the complete Azure data platform using the Azure CLI.
# Run once to set up the environment; safe to re-run (idempotent where possible).
#
# Prerequisites
# -------------
#   az login                     (or use service principal env vars)
#   az account set -s <sub_id>
#   az extension add --name ml   (Azure ML CLI v2)
#
# What this creates
# -----------------
#   Resource Group
#   ├── ADLS Gen2 Storage Account  (Hierarchical Namespace enabled)
#   │   └── Filesystem: supply-chain
#   │       ├── bronze/raw/supply_chain/         ← raw ingestion
#   │       ├── silver/processed/supply_chain/   ← cleaned data
#   │       └── gold/serving/supply_chain/       ← ML outputs + serving
#   ├── Azure Key Vault            (secrets — no credentials in code)
#   ├── Azure ML Workspace         (experiment tracking + model registry)
#   │   └── Experiment: supply-chain-demand-forecasting
#   └── Service Principal          (CI/CD identity — least-privilege RBAC)
#
# After running
# -------------
#   1. Copy the printed env vars into your .env file
#   2. Run: python run_pipeline.py
#   3. View runs: https://ml.azure.com
# =============================================================================
set -euo pipefail

# ── Configuration — edit these ────────────────────────────────────────────────
RESOURCE_GROUP="rg-supply-chain-intelligence"
LOCATION="eastus2"
STORAGE_ACCOUNT="scsupplychain$(openssl rand -hex 4)"   # must be globally unique
STORAGE_FILESYSTEM="supply-chain"
KEY_VAULT_NAME="kv-sc-$(openssl rand -hex 4)"           # must be globally unique
ML_WORKSPACE="aml-supply-chain"
SP_NAME="sp-supply-chain-pipeline"

echo "============================================================"
echo "  Supply Chain Intelligence Platform — Azure Provisioning"
echo "============================================================"

# ── 1. Resource Group ────────────────────────────────────────────────────────
echo ""
echo "[1/7] Creating resource group: $RESOURCE_GROUP in $LOCATION..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

# ── 2. ADLS Gen2 Storage Account ─────────────────────────────────────────────
echo ""
echo "[2/7] Creating ADLS Gen2 storage account: $STORAGE_ACCOUNT..."
echo "      (--enable-hierarchical-namespace true = ADLS Gen2, not plain Blob)"
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --enable-hierarchical-namespace true \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false \
  --output table

# Create the filesystem (ADLS Gen2 container)
echo "      Creating filesystem: $STORAGE_FILESYSTEM..."
az storage fs create \
  --name "$STORAGE_FILESYSTEM" \
  --account-name "$STORAGE_ACCOUNT" \
  --auth-mode login \
  --output table

# Create medallion directories
echo "      Creating Bronze / Silver / Gold directories..."
for LAYER_PATH in \
  "bronze/raw/supply_chain" \
  "silver/processed/supply_chain" \
  "gold/serving/supply_chain" \
  "gold/serving/supply_chain/models" \
  "gold/serving/supply_chain/manifests"; do
  az storage fs directory create \
    --file-system "$STORAGE_FILESYSTEM" \
    --name "$LAYER_PATH" \
    --account-name "$STORAGE_ACCOUNT" \
    --auth-mode login \
    --output none
  echo "      ✓ $LAYER_PATH"
done

# ── 3. Azure Key Vault ────────────────────────────────────────────────────────
echo ""
echo "[3/7] Creating Key Vault: $KEY_VAULT_NAME..."
az keyvault create \
  --name "$KEY_VAULT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --enable-rbac-authorization true \
  --output table

# Store storage account name as a secret
az keyvault secret set \
  --vault-name "$KEY_VAULT_NAME" \
  --name "storage-account-name" \
  --value "$STORAGE_ACCOUNT" \
  --output none
echo "      ✓ Secret: storage-account-name"

# ── 4. Azure ML Workspace ─────────────────────────────────────────────────────
echo ""
echo "[4/7] Creating Azure ML Workspace: $ML_WORKSPACE..."
echo "      (This creates associated Storage, ACR, and App Insights automatically)"
az ml workspace create \
  --name "$ML_WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

# Store ML workspace name in Key Vault
az keyvault secret set \
  --vault-name "$KEY_VAULT_NAME" \
  --name "ml-workspace-name" \
  --value "$ML_WORKSPACE" \
  --output none
az keyvault secret set \
  --vault-name "$KEY_VAULT_NAME" \
  --name "ml-resource-group" \
  --value "$RESOURCE_GROUP" \
  --output none

# ── 5. Service Principal (for CI/CD) ─────────────────────────────────────────
echo ""
echo "[5/7] Creating service principal: $SP_NAME..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

SP_JSON=$(az ad sp create-for-rbac \
  --name "$SP_NAME" \
  --role "Storage Blob Data Contributor" \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --output json)

SP_CLIENT_ID=$(echo "$SP_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['appId'])")
SP_CLIENT_SECRET=$(echo "$SP_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['password'])")
SP_TENANT_ID=$(echo "$SP_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['tenant'])")

# Also grant ML Contributor role
az role assignment create \
  --assignee "$SP_CLIENT_ID" \
  --role "AzureML Data Scientist" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --output none

# Grant Key Vault Secrets Officer to SP
KV_RESOURCE_ID=$(az keyvault show --name "$KEY_VAULT_NAME" --query id -o tsv)
az role assignment create \
  --assignee "$SP_CLIENT_ID" \
  --role "Key Vault Secrets Officer" \
  --scope "$KV_RESOURCE_ID" \
  --output none

echo "      ✓ Service principal created with Storage + ML + Key Vault roles"

# ── 6. RBAC — grant current user access ──────────────────────────────────────
echo ""
echo "[6/7] Granting current user Storage Blob Data Contributor..."
CURRENT_USER=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || echo "")
if [ -n "$CURRENT_USER" ]; then
  STORAGE_RESOURCE_ID=$(az storage account show \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv)
  az role assignment create \
    --assignee "$CURRENT_USER" \
    --role "Storage Blob Data Contributor" \
    --scope "$STORAGE_RESOURCE_ID" \
    --output none
  echo "      ✓ Current user has Storage Blob Data Contributor"
else
  echo "      ⚠ Could not detect signed-in user. Assign Storage Blob Data Contributor manually."
fi

# ── 7. Print environment configuration ───────────────────────────────────────
echo ""
echo "[7/7] Provisioning complete."
echo ""
echo "============================================================"
echo "  Copy the following into your .env file:"
echo "============================================================"
echo ""
echo "AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT"
echo "AZURE_STORAGE_FILESYSTEM=$STORAGE_FILESYSTEM"
echo "AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net/"
echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
echo "AZURE_ML_RESOURCE_GROUP=$RESOURCE_GROUP"
echo "AZURE_ML_WORKSPACE_NAME=$ML_WORKSPACE"
echo ""
echo "# Service principal (for CI/CD only — use 'az login' locally)"
echo "AZURE_TENANT_ID=$SP_TENANT_ID"
echo "AZURE_CLIENT_ID=$SP_CLIENT_ID"
echo "AZURE_CLIENT_SECRET=$SP_CLIENT_SECRET"
echo ""
echo "============================================================"
echo "  Next steps:"
echo "    1. Copy env vars above into .env"
echo "    2. python run_pipeline.py"
echo "    3. View ML runs: https://ml.azure.com"
echo "    4. View storage: https://portal.azure.com → $STORAGE_ACCOUNT"
echo "============================================================"
