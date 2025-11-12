"""
CPAS5 Secrets Management Service
Provides secure secret storage and retrieval using HashiCorp Vault or AWS Secrets Manager

This service abstracts secret management to support multiple backends:
- HashiCorp Vault (recommended for development and self-hosted)
- AWS Secrets Manager (recommended for AWS deployments)
- Environment variables (fallback for development)

Usage:
    from services.secrets_manager import get_secrets_manager
    
    secrets = get_secrets_manager()
    jwt_secret = await secrets.get_secret("jwt_secret")
    await secrets.set_secret("api_key", "new_value")
"""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import asyncio

logger = logging.getLogger(__name__)


class SecretsBackend(ABC):
    """Abstract base class for secrets backends"""
    
    @abstractmethod
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value"""
        pass
    
    @abstractmethod
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret value"""
        pass
    
    @abstractmethod
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> list[str]:
        """List all secret keys"""
        pass
    
    @abstractmethod
    async def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret (keep old version for grace period)"""
        pass


class VaultBackend(SecretsBackend):
    """HashiCorp Vault backend for secrets management"""
    
    def __init__(self, vault_addr: str, vault_token: str, mount_point: str = "secret"):
        self.vault_addr = vault_addr
        self.vault_token = vault_token
        self.mount_point = mount_point
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Vault client"""
        try:
            import hvac
            self.client = hvac.Client(
                url=self.vault_addr,
                token=self.vault_token
            )
            
            if not self.client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            logger.info(f"✅ Connected to HashiCorp Vault at {self.vault_addr}")
        except ImportError:
            logger.error("hvac library not installed. Install with: pip install hvac")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")
            raise
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from Vault"""
        try:
            # Vault operations are synchronous, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.secrets.kv.v2.read_secret_version(
                    path=key,
                    mount_point=self.mount_point
                )
            )
            
            if response and 'data' in response and 'data' in response['data']:
                return response['data']['data'].get('value')
            
            return None
        except Exception as e:
            logger.error(f"Failed to get secret '{key}' from Vault: {e}")
            return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in Vault"""
        try:
            secret_data = {
                'value': value,
                'created_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.secrets.kv.v2.create_or_update_secret(
                    path=key,
                    secret=secret_data,
                    mount_point=self.mount_point
                )
            )
            
            logger.info(f"✅ Secret '{key}' stored in Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret '{key}' in Vault: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from Vault"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=key,
                    mount_point=self.mount_point
                )
            )
            
            logger.info(f"✅ Secret '{key}' deleted from Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}' from Vault: {e}")
            return False
    
    async def list_secrets(self) -> list[str]:
        """List all secrets in Vault"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.secrets.kv.v2.list_secrets(
                    path='',
                    mount_point=self.mount_point
                )
            )
            
            if response and 'data' in response and 'keys' in response['data']:
                return response['data']['keys']
            
            return []
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []
    
    async def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret (Vault keeps version history automatically)"""
        return await self.set_secret(key, new_value, {'rotated_at': datetime.utcnow().isoformat()})


class AWSSecretsBackend(SecretsBackend):
    """AWS Secrets Manager backend"""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS Secrets Manager client"""
        try:
            import boto3
            self.client = boto3.client('secretsmanager', region_name=self.region_name)
            logger.info(f"✅ Connected to AWS Secrets Manager in {self.region_name}")
        except ImportError:
            logger.error("boto3 library not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager client: {e}")
            raise
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from AWS Secrets Manager"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_secret_value(SecretId=key)
            )
            
            if 'SecretString' in response:
                secret_data = json.loads(response['SecretString'])
                return secret_data.get('value')
            
            return None
        except Exception as e:
            logger.error(f"Failed to get secret '{key}' from AWS: {e}")
            return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in AWS Secrets Manager"""
        try:
            secret_data = {
                'value': value,
                'created_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            loop = asyncio.get_event_loop()
            
            # Try to update existing secret first
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self.client.update_secret(
                        SecretId=key,
                        SecretString=json.dumps(secret_data)
                    )
                )
            except self.client.exceptions.ResourceNotFoundException:
                # Secret doesn't exist, create it
                await loop.run_in_executor(
                    None,
                    lambda: self.client.create_secret(
                        Name=key,
                        SecretString=json.dumps(secret_data)
                    )
                )
            
            logger.info(f"✅ Secret '{key}' stored in AWS Secrets Manager")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret '{key}' in AWS: {e}")
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from AWS Secrets Manager"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_secret(
                    SecretId=key,
                    ForceDeleteWithoutRecovery=True
                )
            )
            
            logger.info(f"✅ Secret '{key}' deleted from AWS Secrets Manager")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}' from AWS: {e}")
            return False
    
    async def list_secrets(self) -> list[str]:
        """List all secrets in AWS Secrets Manager"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.list_secrets()
            )
            
            if 'SecretList' in response:
                return [secret['Name'] for secret in response['SecretList']]
            
            return []
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []
    
    async def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret in AWS Secrets Manager"""
        return await self.set_secret(key, new_value, {'rotated_at': datetime.utcnow().isoformat()})


class EnvironmentBackend(SecretsBackend):
    """Environment variables backend (fallback for development)"""
    
    def __init__(self):
        logger.warning("⚠️ Using environment variables for secrets (NOT recommended for production)")
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from environment variables"""
        return os.getenv(key.upper())
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in environment variables (not persistent)"""
        os.environ[key.upper()] = value
        logger.warning(f"⚠️ Secret '{key}' set in environment (not persistent)")
        return True
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from environment variables"""
        if key.upper() in os.environ:
            del os.environ[key.upper()]
            return True
        return False
    
    async def list_secrets(self) -> list[str]:
        """List all environment variables (filtered for secrets)"""
        secret_keys = ['JWT_SECRET', 'JWT_REFRESH_SECRET', 'ENCRYPTION_KEY', 'API_KEY']
        return [key for key in secret_keys if key in os.environ]
    
    async def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret in environment variables"""
        return await self.set_secret(key, new_value)


class SecretsManager:
    """
    Unified secrets management interface
    Automatically selects the appropriate backend based on configuration
    """
    
    def __init__(self, backend: Optional[SecretsBackend] = None):
        if backend:
            self.backend = backend
        else:
            self.backend = self._auto_select_backend()
    
    def _auto_select_backend(self) -> SecretsBackend:
        """Automatically select the best available backend"""
        # Check for Vault configuration
        vault_addr = os.getenv("VAULT_ADDR")
        vault_token = os.getenv("VAULT_TOKEN")
        
        if vault_addr and vault_token:
            try:
                return VaultBackend(vault_addr, vault_token)
            except Exception as e:
                logger.warning(f"Failed to initialize Vault backend: {e}")
        
        # Check for AWS configuration
        aws_region = os.getenv("AWS_REGION")
        if aws_region and os.getenv("AWS_ACCESS_KEY_ID"):
            try:
                return AWSSecretsBackend(aws_region)
            except Exception as e:
                logger.warning(f"Failed to initialize AWS backend: {e}")
        
        # Fallback to environment variables
        logger.warning("⚠️ No secrets backend configured, using environment variables")
        return EnvironmentBackend()
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret"""
        return await self.backend.get_secret(key)
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret"""
        return await self.backend.set_secret(key, value, metadata)
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        return await self.backend.delete_secret(key)
    
    async def list_secrets(self) -> list[str]:
        """List all secrets"""
        return await self.backend.list_secrets()
    
    async def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret"""
        return await self.backend.rotate_secret(key, new_value)


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

