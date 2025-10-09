"""
API Key Service for CPAS4
Service layer for API key generation, verification, and management
"""

import logging
import uuid
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from database import Database
from models.api_key_models import *

logger = logging.getLogger(__name__)


class APIKeyService:
    """API key management service"""
    
    def __init__(self, db: Database):
        self.db = db
        self.api_keys_collection = "api_keys"
        self.api_key_usage_collection = "api_key_usage"
    
    # Key Generation
    def generate_api_key(self) -> str:
        """Generate new API key"""
        # Generate random part (35 characters)
        random_part = secrets.token_urlsafe(26)[:API_KEY_RANDOM_LENGTH]
        
        # Combine with prefix
        api_key = f"{API_KEY_PREFIX}{random_part}"
        
        return api_key
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _get_key_prefix(self, api_key: str) -> str:
        """Get key prefix (first 8 characters)"""
        return api_key[:8] if len(api_key) >= 8 else api_key
    
    # Key Management
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[APIKeyScope] = None,
        expires_at: Optional[datetime] = None
    ) -> APIKeyResponse:
        """Create new API key"""
        # Generate API key
        api_key = self.generate_api_key()
        key_hash = self._hash_api_key(api_key)
        prefix = self._get_key_prefix(api_key)
        
        # Create key record
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        
        if scopes is None:
            scopes = [APIKeyScope.READ]
        
        key_record = APIKey(
            key_id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            scopes=scopes,
            enabled=True,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        await self.db.insert_one(self.api_keys_collection, key_record.model_dump())
        
        logger.info(f"API key created: {key_id} for user {user_id}")
        
        return APIKeyResponse(
            key_id=key_id,
            user_id=user_id,
            name=name,
            key=api_key,  # Full key (only shown once)
            prefix=prefix,
            scopes=scopes,
            expires_at=expires_at,
            created_at=key_record.created_at
        )
    
    async def verify_api_key(self, api_key: str) -> APIKeyVerificationResult:
        """Verify API key and return user info"""
        # Hash the provided key
        key_hash = self._hash_api_key(api_key)
        
        # Find key in database
        key_record = await self.db.find_one(
            self.api_keys_collection,
            {"key_hash": key_hash}
        )
        
        if not key_record:
            return APIKeyVerificationResult(
                valid=False,
                error="Invalid API key"
            )
        
        key = APIKey(**key_record)
        
        # Check if key is enabled
        if not key.enabled:
            return APIKeyVerificationResult(
                valid=False,
                error="API key is disabled"
            )
        
        # Check if key is revoked
        if key.revoked_at:
            return APIKeyVerificationResult(
                valid=False,
                error="API key has been revoked"
            )
        
        # Check if key is expired
        if key.expires_at and datetime.utcnow() > key.expires_at:
            return APIKeyVerificationResult(
                valid=False,
                error="API key has expired"
            )
        
        # Update last used timestamp
        await self.update_last_used(key.key_id)
        
        return APIKeyVerificationResult(
            valid=True,
            key_id=key.key_id,
            user_id=key.user_id,
            scopes=key.scopes
        )
    
    async def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        doc = await self.db.find_one(self.api_keys_collection, {"key_id": key_id})
        
        if doc:
            return APIKey(**doc)
        
        return None
    
    async def list_api_keys(
        self,
        user_id: str,
        include_revoked: bool = False
    ) -> List[APIKeyList]:
        """List user's API keys"""
        query = {"user_id": user_id}
        
        if not include_revoked:
            query["revoked_at"] = None
        
        docs = await self.db.find_many(self.api_keys_collection, query, limit=1000)
        
        keys = []
        for doc in docs:
            key = APIKey(**doc)
            
            # Determine status
            status = APIKeyStatus.ACTIVE
            if key.revoked_at:
                status = APIKeyStatus.REVOKED
            elif key.expires_at and datetime.utcnow() > key.expires_at:
                status = APIKeyStatus.EXPIRED
            
            keys.append(APIKeyList(
                key_id=key.key_id,
                user_id=key.user_id,
                name=key.name,
                prefix=key.prefix,
                scopes=key.scopes,
                enabled=key.enabled,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
                expires_at=key.expires_at,
                revoked_at=key.revoked_at,
                status=status
            ))
        
        return keys
    
    async def update_api_key(
        self,
        key_id: str,
        update: APIKeyUpdate
    ) -> Optional[APIKey]:
        """Update API key"""
        update_data = update.model_dump(exclude_unset=True)
        
        success = await self.db.update_one(
            self.api_keys_collection,
            {"key_id": key_id},
            {"$set": update_data}
        )
        
        if success:
            logger.info(f"API key updated: {key_id}")
            return await self.get_api_key(key_id)
        
        return None
    
    async def revoke_api_key(
        self,
        key_id: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None
    ) -> bool:
        """Revoke API key"""
        success = await self.db.update_one(
            self.api_keys_collection,
            {"key_id": key_id},
            {"$set": {
                "enabled": False,
                "revoked_at": datetime.utcnow().isoformat(),
                "revoked_by": revoked_by,
                "revoke_reason": reason
            }}
        )
        
        if success:
            logger.info(f"API key revoked: {key_id}")
        
        return success
    
    async def delete_api_key(self, key_id: str) -> bool:
        """Delete API key"""
        success = await self.db.delete_one(self.api_keys_collection, {"key_id": key_id})
        
        if success:
            logger.info(f"API key deleted: {key_id}")
        
        return success
    
    async def rotate_api_key(
        self,
        key_id: str,
        revoked_by: Optional[str] = None
    ) -> Optional[APIKeyRotateResponse]:
        """Rotate API key (create new, revoke old)"""
        # Get old key
        old_key = await self.get_api_key(key_id)
        
        if not old_key:
            return None
        
        # Create new key with same settings
        new_key = await self.create_api_key(
            user_id=old_key.user_id,
            name=f"{old_key.name} (rotated)",
            scopes=old_key.scopes,
            expires_at=old_key.expires_at
        )
        
        # Revoke old key
        await self.revoke_api_key(
            key_id=key_id,
            reason="Key rotated",
            revoked_by=revoked_by
        )
        
        logger.info(f"API key rotated: {key_id} -> {new_key.key_id}")
        
        return APIKeyRotateResponse(
            old_key_id=key_id,
            new_key_id=new_key.key_id,
            new_key=new_key.key,
            prefix=new_key.prefix
        )
    
    async def update_last_used(self, key_id: str):
        """Update last used timestamp"""
        await self.db.update_one(
            self.api_keys_collection,
            {"key_id": key_id},
            {"$set": {"last_used_at": datetime.utcnow().isoformat()}}
        )
    
    # Usage Tracking
    async def record_usage(
        self,
        key_id: str,
        user_id: str,
        endpoint: str,
        method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        response_status: Optional[int] = None
    ):
        """Record API key usage"""
        usage_id = f"usage_{uuid.uuid4().hex[:12]}"
        
        usage = APIKeyUsageRecord(
            usage_id=usage_id,
            key_id=key_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            response_status=response_status
        )
        
        await self.db.insert_one(self.api_key_usage_collection, usage.model_dump())

    async def get_usage_stats(self, key_id: str) -> APIKeyUsageStats:
        """Get API key usage statistics"""
        # Get all usage records for this key
        usage_records = await self.db.find_many(
            self.api_key_usage_collection,
            {"key_id": key_id},
            limit=10000
        )

        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(weeks=1)

        total_requests = len(usage_records)
        requests_last_hour = 0
        requests_last_day = 0
        requests_last_week = 0
        last_used = None
        endpoint_counts = {}

        for record in usage_records:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # Count by time period
            if timestamp > one_hour_ago:
                requests_last_hour += 1
            if timestamp > one_day_ago:
                requests_last_day += 1
            if timestamp > one_week_ago:
                requests_last_week += 1

            # Track last used
            if last_used is None or timestamp > last_used:
                last_used = timestamp

            # Count by endpoint
            endpoint = record.get("endpoint", "unknown")
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        # Get top endpoints
        most_used_endpoints = [
            {"endpoint": k, "count": v}
            for k, v in sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        return APIKeyUsageStats(
            key_id=key_id,
            total_requests=total_requests,
            requests_last_hour=requests_last_hour,
            requests_last_day=requests_last_day,
            requests_last_week=requests_last_week,
            last_used_at=last_used,
            most_used_endpoints=most_used_endpoints
        )

