"""
Rate Limiting Service for CPAS4
Service layer for rate limiting rules, enforcement, and tracking
"""

import logging
import uuid
import fnmatch
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from database import Database
from models.rate_limit_models import *

logger = logging.getLogger(__name__)


class RateLimitService:
    """Rate limiting service"""
    
    def __init__(self, db: Database):
        self.db = db
        self.rules_collection = "rate_limit_rules"
        self.violations_collection = "rate_limit_violations"
        self.requests_collection = "rate_limit_requests"
        
        # In-memory cache for fast lookups
        self._request_cache: Dict[str, List[datetime]] = defaultdict(list)
        self._rules_cache: Optional[List[RateLimitRule]] = None
        self._cache_updated_at: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh cache every 60 seconds
    
    # Rule Management
    async def create_rule(
        self,
        rule: RateLimitRuleCreate,
        created_by: Optional[str] = None
    ) -> RateLimitRule:
        """Create new rate limit rule"""
        rule_id = f"rl_{uuid.uuid4().hex[:12]}"
        
        new_rule = RateLimitRule(
            rule_id=rule_id,
            name=rule.name,
            endpoint_pattern=rule.endpoint_pattern,
            max_requests=rule.max_requests,
            window_seconds=rule.window_seconds,
            scope=rule.scope,
            action=rule.action,
            enabled=rule.enabled,
            created_at=datetime.utcnow(),
            created_by=created_by
        )
        
        await self.db.insert_one(self.rules_collection, new_rule.model_dump())
        
        # Invalidate cache
        self._rules_cache = None
        
        logger.info(f"Rate limit rule created: {rule_id} - {rule.name}")
        
        return new_rule
    
    async def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """Get rate limit rule by ID"""
        doc = await self.db.find_one(self.rules_collection, {"rule_id": rule_id})
        
        if doc:
            return RateLimitRule(**doc)
        
        return None
    
    async def list_rules(self, enabled_only: bool = False) -> List[RateLimitRule]:
        """List all rate limit rules"""
        query = {"enabled": True} if enabled_only else {}
        docs = await self.db.find_many(self.rules_collection, query, limit=1000)
        
        return [RateLimitRule(**doc) for doc in docs]
    
    async def update_rule(
        self,
        rule_id: str,
        update: RateLimitRuleUpdate
    ) -> Optional[RateLimitRule]:
        """Update rate limit rule"""
        update_data = update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        success = await self.db.update_one(
            self.rules_collection,
            {"rule_id": rule_id},
            {"$set": update_data}
        )
        
        if success:
            # Invalidate cache
            self._rules_cache = None
            logger.info(f"Rate limit rule updated: {rule_id}")
            return await self.get_rule(rule_id)
        
        return None
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete rate limit rule"""
        success = await self.db.delete_one(self.rules_collection, {"rule_id": rule_id})
        
        if success:
            # Invalidate cache
            self._rules_cache = None
            logger.info(f"Rate limit rule deleted: {rule_id}")
        
        return success
    
    # Rate Limiting Enforcement
    async def _get_cached_rules(self) -> List[RateLimitRule]:
        """Get rules from cache or database"""
        now = datetime.utcnow()
        
        # Check if cache is valid
        if (self._rules_cache is not None and 
            self._cache_updated_at is not None and
            (now - self._cache_updated_at).total_seconds() < self._cache_ttl_seconds):
            return self._rules_cache
        
        # Refresh cache
        self._rules_cache = await self.list_rules(enabled_only=True)
        self._cache_updated_at = now
        
        return self._rules_cache
    
    def _match_endpoint(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern (glob-style)"""
        return fnmatch.fnmatch(endpoint, pattern)
    
    def _get_cache_key(self, identifier: str, rule_id: str) -> str:
        """Get cache key for request tracking"""
        return f"{identifier}:{rule_id}"
    
    async def check_rate_limit(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[RateLimitRule], Optional[RateLimitStatus]]:
        """
        Check if request should be rate limited
        
        Returns:
            (is_allowed, violated_rule, status)
        """
        # Get applicable rules
        rules = await self._get_cached_rules()
        
        now = datetime.utcnow()
        
        for rule in rules:
            # Check if endpoint matches
            if not self._match_endpoint(endpoint, rule.endpoint_pattern):
                continue
            
            # Determine identifier based on scope
            identifier = None
            if rule.scope == RateLimitScope.USER and user_id:
                identifier = f"user:{user_id}"
            elif rule.scope == RateLimitScope.IP and ip_address:
                identifier = f"ip:{ip_address}"
            elif rule.scope == RateLimitScope.GLOBAL:
                identifier = "global"
            elif rule.scope == RateLimitScope.ENDPOINT:
                identifier = f"endpoint:{endpoint}"
            
            if not identifier:
                continue
            
            # Get request history from cache
            cache_key = self._get_cache_key(identifier, rule.rule_id)
            request_times = self._request_cache[cache_key]
            
            # Remove old requests outside the window
            cutoff_time = now - timedelta(seconds=rule.window_seconds)
            request_times = [t for t in request_times if t > cutoff_time]
            self._request_cache[cache_key] = request_times
            
            # Check if limit exceeded
            current_requests = len(request_times)
            
            if current_requests >= rule.max_requests:
                # Rate limit exceeded
                reset_at = request_times[0] + timedelta(seconds=rule.window_seconds)
                
                status = RateLimitStatus(
                    identifier=identifier,
                    scope=rule.scope,
                    endpoint=endpoint,
                    current_requests=current_requests,
                    max_requests=rule.max_requests,
                    window_seconds=rule.window_seconds,
                    reset_at=reset_at,
                    is_limited=True,
                    remaining_requests=0
                )
                
                # Log violation
                if rule.action == RateLimitAction.BLOCK:
                    await self._log_violation(
                        user_id=user_id,
                        ip_address=ip_address,
                        endpoint=endpoint,
                        rule=rule,
                        requests_count=current_requests
                    )
                
                return (False, rule, status)
        
        # No rate limit exceeded
        return (True, None, None)
    
    async def record_request(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Record request for rate limiting"""
        rules = await self._get_cached_rules()
        now = datetime.utcnow()
        
        for rule in rules:
            # Check if endpoint matches
            if not self._match_endpoint(endpoint, rule.endpoint_pattern):
                continue
            
            # Determine identifier based on scope
            identifier = None
            if rule.scope == RateLimitScope.USER and user_id:
                identifier = f"user:{user_id}"
            elif rule.scope == RateLimitScope.IP and ip_address:
                identifier = f"ip:{ip_address}"
            elif rule.scope == RateLimitScope.GLOBAL:
                identifier = "global"
            elif rule.scope == RateLimitScope.ENDPOINT:
                identifier = f"endpoint:{endpoint}"
            
            if not identifier:
                continue
            
            # Add request to cache
            cache_key = self._get_cache_key(identifier, rule.rule_id)
            self._request_cache[cache_key].append(now)
    
    async def get_rate_limit_status(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[RateLimitStatus]:
        """Get current rate limit status for all applicable rules"""
        rules = await self._get_cached_rules()
        statuses = []
        now = datetime.utcnow()
        
        for rule in rules:
            # Check if endpoint matches
            if not self._match_endpoint(endpoint, rule.endpoint_pattern):
                continue
            
            # Determine identifier based on scope
            identifier = None
            if rule.scope == RateLimitScope.USER and user_id:
                identifier = f"user:{user_id}"
            elif rule.scope == RateLimitScope.IP and ip_address:
                identifier = f"ip:{ip_address}"
            elif rule.scope == RateLimitScope.GLOBAL:
                identifier = "global"
            elif rule.scope == RateLimitScope.ENDPOINT:
                identifier = f"endpoint:{endpoint}"
            
            if not identifier:
                continue
            
            # Get request history from cache
            cache_key = self._get_cache_key(identifier, rule.rule_id)
            request_times = self._request_cache[cache_key]
            
            # Remove old requests
            cutoff_time = now - timedelta(seconds=rule.window_seconds)
            request_times = [t for t in request_times if t > cutoff_time]
            
            current_requests = len(request_times)
            remaining = max(0, rule.max_requests - current_requests)
            is_limited = current_requests >= rule.max_requests
            
            reset_at = now + timedelta(seconds=rule.window_seconds)
            if request_times:
                reset_at = request_times[0] + timedelta(seconds=rule.window_seconds)
            
            status = RateLimitStatus(
                identifier=identifier,
                scope=rule.scope,
                endpoint=endpoint,
                current_requests=current_requests,
                max_requests=rule.max_requests,
                window_seconds=rule.window_seconds,
                reset_at=reset_at,
                is_limited=is_limited,
                remaining_requests=remaining
            )
            
            statuses.append(status)
        
        return statuses

    # Violation Tracking
    async def _log_violation(
        self,
        user_id: Optional[str],
        ip_address: Optional[str],
        endpoint: str,
        rule: RateLimitRule,
        requests_count: int
    ):
        """Log rate limit violation"""
        violation_id = f"violation_{uuid.uuid4().hex[:12]}"

        violation = RateLimitViolation(
            violation_id=violation_id,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            timestamp=datetime.utcnow(),
            requests_count=requests_count,
            max_requests=rule.max_requests,
            window_seconds=rule.window_seconds
        )

        await self.db.insert_one(self.violations_collection, violation.model_dump())

        logger.warning(
            f"Rate limit violation: {endpoint} - "
            f"user={user_id}, ip={ip_address}, "
            f"requests={requests_count}/{rule.max_requests}"
        )

    async def get_violations(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        limit: int = 100
    ) -> List[RateLimitViolation]:
        """Get rate limit violations"""
        query = {}
        if user_id:
            query["user_id"] = user_id
        if ip_address:
            query["ip_address"] = ip_address

        docs = await self.db.find_many(self.violations_collection, query, limit=limit)

        return [RateLimitViolation(**doc) for doc in docs]

    # Statistics
    async def get_stats(self) -> RateLimitStats:
        """Get rate limit statistics"""
        # Get all rules
        all_rules = await self.list_rules()
        enabled_rules = [r for r in all_rules if r.enabled]

        # Get all violations
        all_violations = await self.db.find_many(self.violations_collection, {}, limit=10000)

        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        violations_last_hour = 0
        violations_last_day = 0

        # Count violations by time
        for v in all_violations:
            timestamp = v.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            if timestamp > one_hour_ago:
                violations_last_hour += 1
            if timestamp > one_day_ago:
                violations_last_day += 1

        # Get top violators
        violator_counts = defaultdict(int)
        for v in all_violations:
            identifier = v.get("user_id") or v.get("ip_address") or "unknown"
            violator_counts[identifier] += 1

        top_violators = [
            {"identifier": k, "violations": v}
            for k, v in sorted(violator_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        return RateLimitStats(
            total_rules=len(all_rules),
            enabled_rules=len(enabled_rules),
            total_violations=len(all_violations),
            violations_last_hour=violations_last_hour,
            violations_last_day=violations_last_day,
            top_violators=top_violators
        )

    # Initialization
    async def initialize_default_rules(self) -> List[RateLimitRule]:
        """Initialize default rate limit rules"""
        created_rules = []

        for rule_data in DEFAULT_RATE_LIMIT_RULES:
            # Check if rule already exists
            existing = await self.db.find_one(
                self.rules_collection,
                {"name": rule_data["name"]}
            )

            if not existing:
                rule = RateLimitRuleCreate(**rule_data)
                created_rule = await self.create_rule(rule, created_by="system")
                created_rules.append(created_rule)
                logger.info(f"Created default rate limit rule: {rule_data['name']}")

        return created_rules

