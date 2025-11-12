# CPAS5 Secrets Management - Quick Reference Card

**Last Updated:** 2025-11-10

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install hvac redis boto3

# 2. Start Vault and Redis
chmod +x scripts/setup_vault.sh
./scripts/setup_vault.sh

# 3. Configure backend
echo "SECRETS_BACKEND=vault" >> backend/.env
echo "VAULT_ADDR=http://localhost:8200" >> backend/.env
echo "VAULT_TOKEN=cpas5-dev-token" >> backend/.env

# 4. Test
python scripts/test_secrets_manager.py

# 5. Restart backend
cd backend && python -m uvicorn main:app --reload
```

---

## 📝 Common Commands

### Start Services
```bash
# Start Vault and Redis
docker-compose -f docker-compose.vault.yml up -d

# Stop services
docker-compose -f docker-compose.vault.yml down

# View logs
docker logs cpas5-vault
docker logs cpas5-redis
```

### Vault CLI
```bash
# Set environment
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN='cpas5-dev-token'

# List secrets
docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv list secret/cpas5

# Get secret
docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv get secret/cpas5/jwt_secret

# Set secret
docker exec -e VAULT_ADDR=$VAULT_ADDR -e VAULT_TOKEN=$VAULT_TOKEN cpas5-vault \
    vault kv put secret/cpas5/my_secret value="my_value"
```

---

## 💻 Code Examples

### Get a Secret
```python
from services.secrets_manager import get_secrets_manager

async def example():
    manager = get_secrets_manager()
    secret = await manager.get_secret("cpas5/jwt_secret")
    return secret
```

### Set a Secret
```python
from services.secrets_manager import get_secrets_manager

async def example():
    manager = get_secrets_manager()
    await manager.set_secret("cpas5/api_key", "new_value")
```

### Rotate a Secret
```python
from services.secrets_manager import get_secrets_manager
import secrets

async def example():
    manager = get_secrets_manager()
    new_secret = secrets.token_urlsafe(64)
    await manager.rotate_secret("cpas5/jwt_secret", new_secret)
```

---

## 🔧 Configuration

### Development (.env)
```bash
SECRETS_BACKEND=vault
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=cpas5-dev-token
```

### Production (Vault)
```bash
SECRETS_BACKEND=vault
VAULT_ADDR=https://vault.yourdomain.com:8200
VAULT_TOKEN=<app-token>  # Use AppRole in production
```

### Production (AWS)
```bash
SECRETS_BACKEND=aws
AWS_REGION=us-east-1
# AWS credentials from IAM role
```

---

## 🐛 Troubleshooting

### Vault Not Starting
```bash
# Check Docker
docker ps | grep vault

# Check logs
docker logs cpas5-vault

# Restart
docker-compose -f docker-compose.vault.yml restart vault
```

### Secrets Not Found
```bash
# Verify secret exists
docker exec -e VAULT_ADDR=http://localhost:8200 -e VAULT_TOKEN=cpas5-dev-token cpas5-vault \
    vault kv get secret/cpas5/jwt_secret

# Re-run setup
./scripts/setup_vault.sh
```

### Backend Not Initialized
```python
# Check backend
from services.secrets_manager import get_secrets_manager
manager = get_secrets_manager()
print(f"Backend: {manager.backend.__class__.__name__}")
```

---

## 📊 Secret Naming Convention

```
cpas5/
├── jwt_secret              # JWT signing secret
├── jwt_refresh_secret      # JWT refresh token secret
├── encryption_key          # Data encryption key
├── api_key                 # General API key
├── google/
│   ├── client_id          # Google OAuth client ID
│   ├── client_secret      # Google OAuth client secret
│   └── api_key            # Google API key
├── aws/
│   ├── access_key_id      # AWS access key
│   └── secret_access_key  # AWS secret key
└── external/
    ├── openai_api_key     # OpenAI API key
    ├── anthropic_api_key  # Anthropic API key
    └── jina_api_key       # Jina API key
```

---

## 🔐 Security Checklist

- [ ] Vault running with TLS in production
- [ ] Using AppRole or Kubernetes auth (not root token)
- [ ] Secrets rotated regularly (90 days for JWT)
- [ ] Audit logging enabled
- [ ] `.env.secrets` in `.gitignore`
- [ ] No secrets in code or version control
- [ ] Backup and disaster recovery plan

---

## 📚 Documentation

- **Full Setup Guide:** `docs/SECRETS_MANAGEMENT_SETUP.md`
- **Implementation Details:** `docs/PHASE1_ITEM1_SECRETS_MANAGEMENT_COMPLETE.md`
- **Code:** `backend/services/secrets_manager.py`

---

## 🆘 Support

**Issues?** Check:
1. Docker is running
2. Vault container is healthy: `docker ps`
3. Environment variables are set
4. Test script passes: `python scripts/test_secrets_manager.py`

**Still stuck?** Review full documentation in `docs/SECRETS_MANAGEMENT_SETUP.md`

