"""
MongoDB Query Service

Handles querying MongoDB collections based on extracted data requirements
and returns formatted data for file generation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Import database instance
try:
    from database import database as db
except ImportError:
    db = None
    logger.warning("⚠️ Database not available for MongoDB Query Service")


class MongoDBQueryService:
    """Service for querying MongoDB based on data requirements"""
    
    def __init__(self):
        logger.info("✅ MongoDB Query Service initialized")
    
    async def query_data(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query MongoDB based on data requirements
        
        Args:
            requirements: Data requirements from DataExtractionService
            
        Returns:
            Dictionary containing:
            - success: Whether query was successful
            - data: List of documents
            - headers: List of column headers
            - count: Number of documents returned
            - collection: Collection name queried
            - error: Error message if failed
        """
        try:
            collection_name = requirements.get('collection')
            
            if not collection_name:
                return {
                    'success': False,
                    'error': 'No collection specified',
                    'data': [],
                    'headers': [],
                    'count': 0
                }
            
            # Check if database is available and connected
            if db is None or not db.connected or db.db is None:
                logger.warning("⚠️ Database not available or not connected, cannot query real data")
                return {
                    'success': False,
                    'error': 'Database not available or not connected',
                    'data': [],
                    'headers': [],
                    'count': 0
                }
            
            # Build MongoDB query
            query = self._build_query(requirements)
            
            # Get collection
            collection = db.db[collection_name]
            
            # Build sort criteria
            sort_criteria = self._build_sort(requirements)
            
            # Get limit
            limit = requirements.get('limit', 1000)
            
            # Execute query
            logger.info(f"🔍 Querying collection '{collection_name}' with query: {query}")
            logger.info(f"🔍 Sort: {sort_criteria}, Limit: {limit}")
            
            cursor = collection.find(query)
            
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)
            
            cursor = cursor.limit(limit)
            
            # Fetch documents
            documents = []
            async for doc in cursor:
                # Convert ObjectId to string
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            logger.info(f"✅ Retrieved {len(documents)} documents from '{collection_name}'")
            
            # Extract headers
            headers = self._extract_headers(documents, requirements.get('columns', []))
            
            # Apply aggregations if needed
            if requirements.get('aggregations'):
                documents = await self._apply_aggregations(
                    collection,
                    query,
                    requirements.get('aggregations', [])
                )
            
            return {
                'success': True,
                'data': documents,
                'headers': headers,
                'count': len(documents),
                'collection': collection_name,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"❌ Error querying MongoDB: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'headers': [],
                'count': 0
            }
    
    def _build_query(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Build MongoDB query from requirements"""
        query = {}
        
        # Add filters
        filters = requirements.get('filters', {})
        query.update(filters)
        
        # Add date range filter
        date_range = requirements.get('date_range')
        if date_range:
            # Try common date field names
            date_fields = ['created_at', 'updated_at', 'timestamp', 'date', 'created', 'modified']
            
            # Use the first date field we find (could be made smarter)
            date_field = 'created_at'  # Default
            
            query[date_field] = {
                '$gte': date_range['start'].isoformat(),
                '$lte': date_range['end'].isoformat()
            }
            
            logger.info(f"📅 Added date range filter on '{date_field}': {date_range['start']} to {date_range['end']}")
        
        return query
    
    def _build_sort(self, requirements: Dict[str, Any]) -> Optional[List[tuple]]:
        """Build MongoDB sort criteria from requirements"""
        sort_spec = requirements.get('sort')
        if not sort_spec:
            return None
        
        field = sort_spec.get('field')
        direction = sort_spec.get('direction', 'asc')
        
        # Convert to MongoDB sort format
        sort_direction = 1 if direction == 'asc' else -1
        
        return [(field, sort_direction)]
    
    def _extract_headers(self, documents: List[Dict[str, Any]], requested_columns: List[str]) -> List[str]:
        """Extract column headers from documents"""
        if requested_columns:
            # Use requested columns
            return requested_columns
        
        if not documents:
            return []
        
        # Extract all unique keys from documents
        headers = set()
        for doc in documents:
            headers.update(doc.keys())
        
        # Remove internal fields
        headers.discard('_id')
        
        # Convert to sorted list
        return sorted(list(headers))
    
    async def _apply_aggregations(
        self,
        collection,
        base_query: Dict[str, Any],
        aggregations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply aggregation operations"""
        try:
            pipeline = []
            
            # Add match stage if there's a base query
            if base_query:
                pipeline.append({'$match': base_query})
            
            # Process aggregations
            group_by_field = None
            agg_operations = {}
            
            for agg in aggregations:
                agg_type = agg.get('type')
                field = agg.get('field')
                
                if agg_type == 'group_by':
                    group_by_field = field
                elif agg_type in ['sum', 'avg', 'min', 'max', 'count']:
                    # Map to MongoDB aggregation operators
                    mongo_op = f'${agg_type}'
                    if agg_type == 'count':
                        agg_operations[f'{field}_count'] = {'$sum': 1}
                    else:
                        agg_operations[f'{field}_{agg_type}'] = {mongo_op: f'${field}'}
            
            # Add group stage if needed
            if group_by_field or agg_operations:
                group_stage = {
                    '_id': f'${group_by_field}' if group_by_field else None
                }
                group_stage.update(agg_operations)
                pipeline.append({'$group': group_stage})
            
            # Execute aggregation
            if pipeline:
                logger.info(f"📊 Executing aggregation pipeline: {pipeline}")
                cursor = collection.aggregate(pipeline)
                results = []
                async for doc in cursor:
                    if '_id' in doc and doc['_id'] is not None:
                        doc[group_by_field or 'group'] = doc['_id']
                    if '_id' in doc:
                        del doc['_id']
                    results.append(doc)
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error applying aggregations: {e}")
            return []
    
    def format_data_for_file(
        self,
        data: List[Dict[str, Any]],
        headers: List[str],
        file_type: str
    ) -> Dict[str, Any]:
        """
        Format queried data for file generation
        
        Args:
            data: List of documents from MongoDB
            headers: List of column headers
            file_type: Type of file (pdf, excel, csv, chart)
            
        Returns:
            Formatted data structure for file generation
        """
        if file_type == 'csv':
            # Format for CSV: headers + rows
            rows = []
            for doc in data:
                row = [str(doc.get(header, '')) for header in headers]
                rows.append(row)
            
            return {
                'headers': headers,
                'data': rows
            }
        
        elif file_type == 'excel':
            # Format for Excel: sheets with headers and data
            rows = []
            for doc in data:
                row = [doc.get(header, '') for header in headers]
                rows.append(row)
            
            return {
                'sheets': [{
                    'name': 'Data',
                    'headers': headers,
                    'data': rows
                }]
            }
        
        elif file_type == 'pdf':
            # Format for PDF: content sections
            content = []
            
            # Add summary
            content.append({
                'type': 'paragraph',
                'data': f'Total records: {len(data)}'
            })
            
            content.append({'type': 'spacer', 'data': 0.2})
            
            # Add table
            if data:
                table_data = [headers]  # Header row
                for doc in data[:100]:  # Limit to first 100 for PDF
                    row = [str(doc.get(header, '')) for header in headers]
                    table_data.append(row)
                
                content.append({
                    'type': 'table',
                    'data': table_data
                })
            
            return {
                'content': content
            }
        
        elif file_type == 'chart':
            # Format for chart: x and y data
            if not data or len(headers) < 2:
                return {
                    'chart_type': 'bar',
                    'data': {
                        'x': ['No Data'],
                        'y': [0],
                        'x_label': 'Category',
                        'y_label': 'Value'
                    }
                }
            
            # Use first column as x-axis, second as y-axis
            x_field = headers[0]
            y_field = headers[1]
            
            x_data = [str(doc.get(x_field, '')) for doc in data[:50]]  # Limit to 50 points
            y_data = []
            
            for doc in data[:50]:
                y_value = doc.get(y_field, 0)
                # Try to convert to number
                try:
                    y_data.append(float(y_value))
                except (ValueError, TypeError):
                    y_data.append(0)
            
            return {
                'chart_type': 'bar',
                'data': {
                    'x': x_data,
                    'y': y_data,
                    'x_label': x_field,
                    'y_label': y_field
                }
            }
        
        return {}

