"""
Data Extraction Service

Intelligently parses natural language requests to extract data requirements
for file generation, including:
- MongoDB collection names
- Date ranges and filters
- Column/field specifications
- Aggregation requirements
- Sorting and limits
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class DataExtractionService:
    """Service for extracting data requirements from natural language requests"""
    
    def __init__(self):
        # Collection name mappings (common terms -> actual collection names)
        self.collection_mappings = {
            # CRM
            'contact': 'crm_contacts',
            'contacts': 'crm_contacts',
            'customer': 'crm_contacts',
            'customers': 'crm_contacts',
            'company': 'crm_companies',
            'companies': 'crm_companies',
            'deal': 'crm_deals',
            'deals': 'crm_deals',
            'activity': 'crm_activities',
            'activities': 'crm_activities',
            
            # ATS
            'job': 'ats_jobs',
            'jobs': 'ats_jobs',
            'candidate': 'ats_candidates',
            'candidates': 'ats_candidates',
            'application': 'ats_applications',
            'applications': 'ats_applications',
            'interview': 'ats_interviews',
            'interviews': 'ats_interviews',
            
            # Finance
            'transaction': 'finance_transactions',
            'transactions': 'finance_transactions',
            'invoice': 'finance_invoices',
            'invoices': 'finance_invoices',
            'account': 'finance_accounts',
            'accounts': 'finance_accounts',
            'payment': 'stripe_payments',
            'payments': 'stripe_payments',
            
            # Projects
            'project': 'projects_projects',
            'projects': 'projects_projects',
            'task': 'projects_tasks',
            'tasks': 'projects_tasks',
            'milestone': 'projects_milestones',
            'milestones': 'projects_milestones',
            'time entry': 'projects_time_entries',
            'time entries': 'projects_time_entries',
            
            # Inventory
            'product': 'inventory_products',
            'products': 'inventory_products',
            'stock': 'inventory_stock',
            'warehouse': 'inventory_warehouses',
            'warehouses': 'inventory_warehouses',
            'purchase order': 'inventory_purchase_orders',
            'purchase orders': 'inventory_purchase_orders',
            
            # Leads
            'lead': 'leads',
            'leads': 'leads',
            'lead score': 'lead_scores',
            'lead scores': 'lead_scores',
            
            # Agents
            'agent': 'custom_agents',
            'agents': 'custom_agents',
            'agent template': 'agent_templates',
            'agent templates': 'agent_templates',
            
            # Analytics
            'report': 'analytics_reports',
            'reports': 'analytics_reports',
            'dashboard': 'analytics_dashboards',
            'dashboards': 'analytics_dashboards',
            'kpi': 'analytics_kpis',
            'kpis': 'analytics_kpis',
            
            # System
            'conversation': 'conversations',
            'conversations': 'conversations',
            'audit log': 'audit_logs',
            'audit logs': 'audit_logs',
            'user': 'users',
            'users': 'users',
        }
        
        # Date range patterns
        self.date_patterns = {
            'today': lambda: (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), datetime.now()),
            'yesterday': lambda: (
                (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                (datetime.now() - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            ),
            'this week': lambda: (
                datetime.now() - timedelta(days=datetime.now().weekday()),
                datetime.now()
            ),
            'last week': lambda: (
                datetime.now() - timedelta(days=datetime.now().weekday() + 7),
                datetime.now() - timedelta(days=datetime.now().weekday())
            ),
            'this month': lambda: (
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now()
            ),
            'last month': lambda: self._get_last_month_range(),
            'this year': lambda: (
                datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now()
            ),
        }
        
        # Aggregation keywords
        self.aggregation_keywords = {
            'total': 'sum',
            'sum': 'sum',
            'average': 'avg',
            'avg': 'avg',
            'mean': 'avg',
            'count': 'count',
            'minimum': 'min',
            'min': 'min',
            'maximum': 'max',
            'max': 'max',
        }
        
        logger.info("✅ Data Extraction Service initialized")
    
    def _get_last_month_range(self) -> Tuple[datetime, datetime]:
        """Get date range for last month"""
        today = datetime.now()
        first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return (first_day_last_month, last_day_last_month.replace(hour=23, minute=59, second=59, microsecond=999999))
    
    def extract_data_requirements(self, request_text: str) -> Dict[str, Any]:
        """
        Extract data requirements from natural language request
        
        Args:
            request_text: Natural language request from user
            
        Returns:
            Dictionary containing:
            - collection: MongoDB collection name
            - filters: Query filters
            - date_range: Start and end dates
            - columns: List of columns/fields to include
            - aggregations: Aggregation operations
            - sort: Sort field and direction
            - limit: Maximum number of records
        """
        request_lower = request_text.lower()
        
        result = {
            'collection': None,
            'filters': {},
            'date_range': None,
            'columns': [],
            'aggregations': [],
            'sort': None,
            'limit': 1000,  # Default limit
            'has_real_data_requirement': False
        }
        
        # Extract collection name
        result['collection'] = self._extract_collection(request_lower)
        
        # Extract date range
        result['date_range'] = self._extract_date_range(request_lower)
        
        # Extract filters
        result['filters'] = self._extract_filters(request_lower, request_text)
        
        # Extract columns
        result['columns'] = self._extract_columns(request_lower, request_text)
        
        # Extract aggregations
        result['aggregations'] = self._extract_aggregations(request_lower, request_text)
        
        # Extract sort
        result['sort'] = self._extract_sort(request_lower)
        
        # Extract limit
        result['limit'] = self._extract_limit(request_lower)
        
        # Determine if this request requires real data
        result['has_real_data_requirement'] = bool(
            result['collection'] or 
            result['date_range'] or 
            result['filters'] or 
            result['aggregations']
        )
        
        logger.info(f"📊 Extracted data requirements: {result}")
        return result
    
    def _extract_collection(self, request_lower: str) -> Optional[str]:
        """Extract MongoDB collection name from request"""
        # Check for explicit collection mentions
        for term, collection in self.collection_mappings.items():
            # Look for patterns like "with contacts", "from contacts", "contacts data"
            patterns = [
                f"with {term}",
                f"from {term}",
                f"{term} data",
                f"{term} list",
                f"all {term}",
                f"{term} report",
                f"{term} export",
            ]
            
            for pattern in patterns:
                if pattern in request_lower:
                    logger.info(f"🎯 Detected collection: {collection} (from pattern: '{pattern}')")
                    return collection
        
        return None
    
    def _extract_date_range(self, request_lower: str) -> Optional[Dict[str, datetime]]:
        """Extract date range from request"""
        # Check for relative date patterns
        for pattern, date_func in self.date_patterns.items():
            if pattern in request_lower:
                start_date, end_date = date_func()
                logger.info(f"📅 Detected date range: {pattern} ({start_date} to {end_date})")
                return {'start': start_date, 'end': end_date}
        
        # Check for "last N days/weeks/months"
        last_n_pattern = r'last (\d+) (day|week|month)s?'
        match = re.search(last_n_pattern, request_lower)
        if match:
            n = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'day':
                start_date = datetime.now() - timedelta(days=n)
            elif unit == 'week':
                start_date = datetime.now() - timedelta(weeks=n)
            elif unit == 'month':
                start_date = datetime.now() - timedelta(days=n*30)  # Approximate
            
            end_date = datetime.now()
            logger.info(f"📅 Detected date range: last {n} {unit}(s) ({start_date} to {end_date})")
            return {'start': start_date, 'end': end_date}
        
        return None
    
    def _extract_filters(self, request_lower: str, request_text: str) -> Dict[str, Any]:
        """Extract query filters from request"""
        filters = {}
        
        # Status filters
        status_pattern = r'(?:where|with|filter by|filtered by) status (?:is |equals? )?["\']?(\w+)["\']?'
        match = re.search(status_pattern, request_lower)
        if match:
            filters['status'] = match.group(1)
            logger.info(f"🔍 Detected status filter: {match.group(1)}")
        
        # Priority filters
        priority_pattern = r'(?:where|with|filter by) priority (?:is |equals? )?["\']?(\w+)["\']?'
        match = re.search(priority_pattern, request_lower)
        if match:
            filters['priority'] = match.group(1)
            logger.info(f"🔍 Detected priority filter: {match.group(1)}")
        
        # Industry filters
        industry_pattern = r'(?:where|with|filter by|filtered by) industry (?:is |equals? )?["\']?([^"\']+)["\']?'
        match = re.search(industry_pattern, request_lower)
        if match:
            filters['industry'] = match.group(1).strip()
            logger.info(f"🔍 Detected industry filter: {match.group(1)}")
        
        # Active/inactive filters
        if 'active' in request_lower and 'inactive' not in request_lower:
            filters['status'] = 'active'
            logger.info("🔍 Detected active filter")
        elif 'inactive' in request_lower:
            filters['status'] = 'inactive'
            logger.info("🔍 Detected inactive filter")
        
        return filters
    
    def _extract_columns(self, request_lower: str, request_text: str) -> List[str]:
        """Extract column/field specifications from request"""
        columns = []
        
        # Look for "with columns X, Y, Z" or "include X, Y, and Z"
        column_patterns = [
            r'with columns? ([^.]+)',
            r'include (?:columns? )?([^.]+)',
            r'showing ([^.]+)',
            r'fields? ([^.]+)',
        ]
        
        for pattern in column_patterns:
            match = re.search(pattern, request_lower)
            if match:
                # Extract column names
                column_text = match.group(1)
                # Split by commas and 'and'
                column_list = re.split(r',|\sand\s', column_text)
                columns = [col.strip() for col in column_list if col.strip()]
                logger.info(f"📋 Detected columns: {columns}")
                break
        
        return columns
    
    def _extract_aggregations(self, request_lower: str, request_text: str) -> List[Dict[str, Any]]:
        """Extract aggregation requirements from request"""
        aggregations = []
        
        for keyword, agg_type in self.aggregation_keywords.items():
            if keyword in request_lower:
                # Try to find what field to aggregate
                pattern = f'{keyword}\\s+(?:of\\s+)?([\\w_]+)'
                match = re.search(pattern, request_lower)
                if match:
                    field = match.group(1)
                    aggregations.append({
                        'type': agg_type,
                        'field': field
                    })
                    logger.info(f"📊 Detected aggregation: {agg_type} of {field}")
        
        # Group by detection
        group_by_pattern = r'group(?:ed)? by ([\\w_]+)'
        match = re.search(group_by_pattern, request_lower)
        if match:
            field = match.group(1)
            aggregations.append({
                'type': 'group_by',
                'field': field
            })
            logger.info(f"📊 Detected group by: {field}")
        
        return aggregations
    
    def _extract_sort(self, request_lower: str) -> Optional[Dict[str, Any]]:
        """Extract sort requirements from request"""
        # Look for "sort by X" or "order by X"
        sort_pattern = r'(?:sort|order)(?:ed)? by ([\\w_]+)(?: (asc|desc|ascending|descending))?'
        match = re.search(sort_pattern, request_lower)
        if match:
            field = match.group(1)
            direction = match.group(2) if match.group(2) else 'asc'
            # Normalize direction
            if direction in ['desc', 'descending']:
                direction = 'desc'
            else:
                direction = 'asc'
            
            logger.info(f"🔄 Detected sort: {field} {direction}")
            return {'field': field, 'direction': direction}
        
        return None
    
    def _extract_limit(self, request_lower: str) -> int:
        """Extract limit from request"""
        # Look for "limit N" or "top N" or "first N"
        limit_patterns = [
            r'limit (\d+)',
            r'top (\d+)',
            r'first (\d+)',
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, request_lower)
            if match:
                limit = int(match.group(1))
                logger.info(f"🔢 Detected limit: {limit}")
                return limit
        
        return 1000  # Default limit

