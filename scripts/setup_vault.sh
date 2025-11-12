#!/bin/bash

# CPAS5 Vault Setup Script
# Initializes HashiCorp Vault with CPAS5 secrets
#
# Usage:
#   ./scripts/setup_vault.sh

set -e

echo "========================================="
echo "CPAS5 Vault Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Start Vault container
echo "Starting Vault container..."
docker-compose -f docker-compose.vault.yml up -d vault

# Wait for Vault to be ready
echo "Waiting for Vault to be ready..."
sleep 5

# Check Vault health
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec cpas5-vault vault status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Vault is ready${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Vault... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Vault failed to start${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "Configuring Vault"
echo "========================================="
echo ""

# Set Vault address and token
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN='cpas5-dev-token'

# Enable KV secrets engine v2
echo "Enabling KV secrets engine..."
docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault secrets enable -version=2 -path=secret kv || echo "KV engine already enabled"

echo -e "${GREEN}✅ KV secrets engine enabled${NC}"
echo ""

# Generate secrets
echo "Generating secrets..."
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
JWT_REFRESH_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo -e "${GREEN}✅ Secrets generated${NC}"
echo ""

# Store secrets in Vault
echo "Storing secrets in Vault..."

docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv put secret/cpas5/jwt_secret value="$JWT_SECRET"

docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv put secret/cpas5/jwt_refresh_secret value="$JWT_REFRESH_SECRET"

docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv put secret/cpas5/encryption_key value="$ENCRYPTION_KEY"

docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv put secret/cpas5/api_key value="$API_KEY"

echo -e "${GREEN}✅ Secrets stored in Vault${NC}"
echo ""

# Verify secrets
echo "Verifying secrets..."
docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv list secret/cpas5

echo -e "${GREEN}✅ Secrets verified${NC}"
echo ""

# Create .env.vault file
echo "Creating .env.vault file..."
cat > .env.vault << EOF
# CPAS5 Vault Configuration
# Generated: $(date)

# Vault Configuration
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=cpas5-dev-token
VAULT_MOUNT_POINT=secret

# Secrets Backend
SECRETS_BACKEND=vault

# Redis Configuration (for JWT blacklisting)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=cpas5-redis-password
REDIS_DB=0

# MongoDB Configuration (if using Docker)
# MONGODB_URL=mongodb://cpas5:cpas5-mongo-password@localhost:27017/greta_pai?authSource=admin
EOF

echo -e "${GREEN}✅ .env.vault file created${NC}"
echo ""

echo "========================================="
echo "Vault Setup Complete!"
echo "========================================="
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo ""
echo "1. Add Vault configuration to your .env file:"
echo "   source .env.vault"
echo ""
echo "2. Or manually add these lines to backend/.env:"
echo "   VAULT_ADDR=http://localhost:8200"
echo "   VAULT_TOKEN=cpas5-dev-token"
echo "   SECRETS_BACKEND=vault"
echo ""
echo "3. Install Python dependencies:"
echo "   pip install hvac redis"
echo ""
echo "4. Start Redis for JWT blacklisting:"
echo "   docker-compose -f docker-compose.vault.yml up -d redis"
echo ""
echo "5. Test the secrets manager:"
echo "   python scripts/test_secrets_manager.py"
echo ""
echo -e "${GREEN}✅ Vault is running at: http://localhost:8200${NC}"
echo -e "${GREEN}✅ Vault UI Token: cpas5-dev-token${NC}"
echo ""
echo "========================================="

