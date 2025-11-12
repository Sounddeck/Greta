# CPAS5 Secrets Management Setup Guide

**Date:** 2025-11-10  
**Version:** 1.0  
**Phase:** Phase 1 - Security Hardening

---

## Overview

CPAS5 now supports secure secrets management using industry-standard solutions:

- **HashiCorp Vault** (recommended for development and self-hosted deployments)
- **AWS Secrets Manager** (recommended for AWS cloud deployments)
- **Environment Variables** (fallback for local development only)

This guide will walk you through setting up secrets management for CPAS5.

---

## Quick Start (Development with Vault)

### 1. Install Dependencies

```bash
cd /Users/macone/Desktop/CPAS5
pip install hvac redis
```

### 2. Start Vault and Redis

```bash
# Make setup script executable
chmod +x scripts/setup_vault.sh

# Run setup script
./scripts/setup_vault.sh
```

This script will:
- Start HashiCorp Vault in development mode
- Start Redis for JWT token blacklisting
- Generate secure secrets
- Store secrets in Vault
- Create `.env.vault` configuration file

### 3. Configure Backend

Add Vault configuration to `backend/.env`:

```bash
# Secrets Management
SECRETS_BACKEND=vault
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=cpas5-dev-token

# Redis (for JWT blacklisting)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=cpas5-redis-password
```

### 4. Test Secrets Manager

```bash
python scripts/test_secrets_manager.py
```

You should see:
```
✅ Secrets Manager initialized
   Backend: VaultBackend

✅ jwt_secret: abc123...xyz789
✅ jwt_refresh_secret: def456...uvw012
✅ encryption_key: ghi789...rst345
✅ api_key: jkl012...opq678
```

### 5. Restart Backend

```bash
cd backend
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Architecture

### Secrets Flow

```
┌─────────────────┐
│   Application   │
│   (CPAS5)       │
└────────┬────────┘
         │
         │ get_secret("jwt_secret")
         ▼
┌─────────────────┐
│ Secrets Manager │
│   (Abstraction) │
└────────┬────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Vault      │ │     AWS      │ │  Environment │ │    Redis     │
│   Backend    │ │   Secrets    │ │  Variables   │ │   (Cache)    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Backend Selection

The secrets manager automatically selects the best available backend:

1. **Vault** - If `VAULT_ADDR` and `VAULT_TOKEN` are set
2. **AWS Secrets Manager** - If `AWS_REGION` and AWS credentials are set
3. **Environment Variables** - Fallback (development only)

---

## Production Setup

### Option 1: HashiCorp Vault (Self-Hosted)

#### 1. Install Vault

```bash
# macOS
brew install vault

# Linux
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
```

#### 2. Configure Vault

Create `vault-config.hcl`:

```hcl
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/opt/vault/tls/vault.crt"
  tls_key_file  = "/opt/vault/tls/vault.key"
}

api_addr = "https://vault.yourdomain.com:8200"
cluster_addr = "https://vault.yourdomain.com:8201"
ui = true
```

#### 3. Initialize Vault

```bash
# Start Vault
vault server -config=vault-config.hcl

# Initialize (in another terminal)
export VAULT_ADDR='https://vault.yourdomain.com:8200'
vault operator init

# Save the unseal keys and root token securely!
# Unseal Vault
vault operator unseal <unseal-key-1>
vault operator unseal <unseal-key-2>
vault operator unseal <unseal-key-3>

# Login
vault login <root-token>

# Enable KV secrets engine
vault secrets enable -version=2 -path=secret kv
```

#### 4. Store Secrets

```bash
# Generate secrets
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
JWT_REFRESH_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Store in Vault
vault kv put secret/cpas5/jwt_secret value="$JWT_SECRET"
vault kv put secret/cpas5/jwt_refresh_secret value="$JWT_REFRESH_SECRET"
vault kv put secret/cpas5/encryption_key value="$ENCRYPTION_KEY"
```

#### 5. Configure CPAS5

```bash
# Production .env
SECRETS_BACKEND=vault
VAULT_ADDR=https://vault.yourdomain.com:8200
VAULT_TOKEN=<app-token>  # Use AppRole or Kubernetes auth in production
```

---

### Option 2: AWS Secrets Manager

#### 1. Install AWS CLI and boto3

```bash
pip install boto3 awscli
aws configure
```

#### 2. Create Secrets

```bash
# Generate secrets
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
JWT_REFRESH_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Store in AWS Secrets Manager
aws secretsmanager create-secret \
    --name cpas5/jwt_secret \
    --secret-string "{\"value\":\"$JWT_SECRET\"}"

aws secretsmanager create-secret \
    --name cpas5/jwt_refresh_secret \
    --secret-string "{\"value\":\"$JWT_REFRESH_SECRET\"}"

aws secretsmanager create-secret \
    --name cpas5/encryption_key \
    --secret-string "{\"value\":\"$ENCRYPTION_KEY\"}"
```

#### 3. Configure CPAS5

```bash
# Production .env
SECRETS_BACKEND=aws
AWS_REGION=us-east-1
# AWS credentials from IAM role or environment variables
```

---

## Secret Rotation

### Manual Rotation

```python
from services.secrets_manager import get_secrets_manager
import secrets

async def rotate_jwt_secret():
    manager = get_secrets_manager()
    
    # Generate new secret
    new_secret = secrets.token_urlsafe(64)
    
    # Rotate (keeps old version for grace period in Vault)
    await manager.rotate_secret("cpas5/jwt_secret", new_secret)
    
    print("✅ JWT secret rotated")
```

### Automated Rotation (Vault)

Create rotation policy in Vault:

```bash
vault write sys/policies/rotation/cpas5-jwt \
    rotation_period="30d" \
    rotation_window="24h"
```

---

## Security Best Practices

### 1. Never Commit Secrets

```bash
# Add to .gitignore
.env
.env.local
.env.*.local
.env.secrets
.env.vault
vault-config.hcl
```

### 2. Use Strong Secrets

```python
# Generate strong secrets
import secrets

jwt_secret = secrets.token_urlsafe(64)  # 64 bytes = 512 bits
encryption_key = secrets.token_urlsafe(32)  # 32 bytes = 256 bits
```

### 3. Rotate Secrets Regularly

- **JWT Secrets:** Every 90 days
- **API Keys:** Every 180 days
- **Encryption Keys:** Every 365 days

### 4. Use Least Privilege

- Vault: Use AppRole or Kubernetes auth instead of root token
- AWS: Use IAM roles with minimal permissions

### 5. Enable Audit Logging

```bash
# Vault audit logging
vault audit enable file file_path=/var/log/vault/audit.log
```

---

## Troubleshooting

### Vault Connection Failed

```bash
# Check Vault status
docker ps | grep vault

# Check Vault logs
docker logs cpas5-vault

# Verify Vault address
curl http://localhost:8200/v1/sys/health
```

### Secrets Not Found

```bash
# List secrets in Vault
docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv list secret/cpas5

# Get specific secret
docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv get secret/cpas5/jwt_secret
```

### Backend Not Initialized

```python
# Check backend selection
from services.secrets_manager import get_secrets_manager

manager = get_secrets_manager()
print(f"Backend: {manager.backend.__class__.__name__}")
```

---

## Migration from Environment Variables

### 1. Export Current Secrets

```bash
# Save current secrets
echo "JWT_SECRET=$(grep JWT_SECRET backend/.env | cut -d'=' -f2)" > .env.backup
echo "JWT_REFRESH_SECRET=$(grep JWT_REFRESH_SECRET backend/.env | cut -d'=' -f2)" >> .env.backup
echo "ENCRYPTION_KEY=$(grep ENCRYPTION_KEY backend/.env | cut -d'=' -f2)" >> .env.backup
```

### 2. Import to Vault

```bash
source .env.backup

docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv put secret/cpas5/jwt_secret value="$JWT_SECRET"

docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv put secret/cpas5/jwt_refresh_secret value="$JWT_REFRESH_SECRET"

docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv put secret/cpas5/encryption_key value="$ENCRYPTION_KEY"
```

### 3. Update Configuration

```bash
# Update backend/.env
echo "SECRETS_BACKEND=vault" >> backend/.env
echo "VAULT_ADDR=http://localhost:8200" >> backend/.env
echo "VAULT_TOKEN=cpas5-dev-token" >> backend/.env
```

### 4. Remove Secrets from .env

```bash
# Comment out secrets in backend/.env
sed -i.bak 's/^JWT_SECRET=/#JWT_SECRET=/' backend/.env
sed -i.bak 's/^JWT_REFRESH_SECRET=/#JWT_REFRESH_SECRET=/' backend/.env
sed -i.bak 's/^ENCRYPTION_KEY=/#ENCRYPTION_KEY=/' backend/.env
```

---

## Next Steps

After setting up secrets management:

1. ✅ **Phase 1.2:** Set up Google Cloud OAuth credentials
2. ✅ **Phase 1.3:** Implement API key encryption in database
3. ✅ **Phase 1.4:** Add JWT token blacklisting with Redis
4. ✅ **Phase 1.5:** Configure production CORS settings

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-10  
**Next Review:** Weekly during implementation  
**Owner:** Security Team

