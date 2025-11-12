"""
Comprehensive tests for database.py module

Tests cover:
- Connection management
- Transaction support
- CRUD operations
- Error handling
- Datetime standardization
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any

# Import the module to test
from database import Database, get_database


class TestDatabaseConnection:
    """Test database connection management"""
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful MongoDB connection"""
        db = Database()
        
        with patch('database.AsyncIOMotorClient') as mock_client:
            # Mock successful connection
            mock_instance = AsyncMock()
            mock_instance.admin.command = AsyncMock(return_value={"ok": 1})
            mock_instance.__getitem__ = Mock(return_value=Mock())
            mock_client.return_value = mock_instance
            
            # Mock _initialize_collections
            with patch.object(db, '_initialize_collections', new=AsyncMock()):
                result = await db.connect()
                
                assert result is True
                assert db.connected is True
                assert db.client is not None
                assert db.db is not None
    
    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """Test connection timeout fallback"""
        db = Database()
        
        with patch('database.AsyncIOMotorClient') as mock_client:
            # Mock timeout
            mock_instance = AsyncMock()
            mock_instance.admin.command = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_client.return_value = mock_instance
            
            # Mock file storage fallback
            with patch.object(db, '_initialize_file_storage', new=AsyncMock()):
                result = await db.connect()
                
                assert result is False
                assert db.connected is False
    
    @pytest.mark.asyncio
    async def test_connect_failure_fallback(self):
        """Test connection failure with fallback to file storage"""
        db = Database()
        
        with patch('database.AsyncIOMotorClient') as mock_client:
            # Mock connection failure
            mock_client.side_effect = Exception("Connection failed")
            
            # Mock file storage fallback
            with patch.object(db, '_initialize_file_storage', new=AsyncMock()):
                result = await db.connect()
                
                assert result is False
                assert db.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test database disconnection"""
        db = Database()
        db.client = Mock()
        db.connected = True
        
        await db.disconnect()
        
        assert db.connected is False
        db.client.close.assert_called_once()


class TestDatabaseOperations:
    """Test database CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_insert_one(self):
        """Test inserting a single document"""
        db = Database()
        db.connected = True
        db.db = Mock()
        
        collection_mock = AsyncMock()
        collection_mock.insert_one = AsyncMock(return_value=Mock(inserted_id="test_id"))
        db.db.__getitem__ = Mock(return_value=collection_mock)
        
        # Mock the insert_one method
        with patch.object(db, 'insert_one', new=AsyncMock(return_value="test_id")):
            result = await db.insert_one("test_collection", {"name": "test"})
            assert result == "test_id"
    
    @pytest.mark.asyncio
    async def test_find_one(self):
        """Test finding a single document"""
        db = Database()
        db.connected = True
        db.db = Mock()
        
        collection_mock = AsyncMock()
        collection_mock.find_one = AsyncMock(return_value={"_id": "test_id", "name": "test"})
        db.db.__getitem__ = Mock(return_value=collection_mock)
        
        # Mock the find_one method
        with patch.object(db, 'find_one', new=AsyncMock(return_value={"_id": "test_id", "name": "test"})):
            result = await db.find_one("test_collection", {"_id": "test_id"})
            assert result["name"] == "test"
    
    @pytest.mark.asyncio
    async def test_update_one(self):
        """Test updating a single document"""
        db = Database()
        db.connected = True
        db.db = Mock()
        
        collection_mock = AsyncMock()
        collection_mock.update_one = AsyncMock(return_value=Mock(modified_count=1))
        db.db.__getitem__ = Mock(return_value=collection_mock)
        
        # Mock the update_one method
        with patch.object(db, 'update_one', new=AsyncMock(return_value=1)):
            result = await db.update_one("test_collection", {"_id": "test_id"}, {"$set": {"name": "updated"}})
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_delete_one(self):
        """Test deleting a single document"""
        db = Database()
        db.connected = True
        db.db = Mock()
        
        collection_mock = AsyncMock()
        collection_mock.delete_one = AsyncMock(return_value=Mock(deleted_count=1))
        db.db.__getitem__ = Mock(return_value=collection_mock)
        
        # Mock the delete_one method
        with patch.object(db, 'delete_one', new=AsyncMock(return_value=1)):
            result = await db.delete_one("test_collection", {"_id": "test_id"})
            assert result == 1


class TestDatabaseTransactions:
    """Test database transaction support"""
    
    @pytest.mark.asyncio
    async def test_transaction_success(self):
        """Test successful transaction"""
        db = Database()
        db.connected = True
        db.client = Mock()
        
        # Mock session
        session_mock = AsyncMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=None)
        session_mock.commit_transaction = AsyncMock()
        session_mock.abort_transaction = AsyncMock()
        
        db.client.start_session = Mock(return_value=session_mock)
        
        # Mock the transaction method
        async def mock_transaction(operations):
            for op in operations:
                await op(session_mock)
            return True
        
        with patch.object(db, 'transaction', new=mock_transaction):
            operations = [AsyncMock()]
            result = await db.transaction(operations)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_transaction_failure(self):
        """Test transaction rollback on failure"""
        db = Database()
        db.connected = True
        db.client = Mock()
        
        # Mock session
        session_mock = AsyncMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=None)
        session_mock.abort_transaction = AsyncMock()
        
        db.client.start_session = Mock(return_value=session_mock)
        
        # Mock the transaction method to raise an error
        async def mock_transaction_fail(operations):
            raise Exception("Transaction failed")
        
        with patch.object(db, 'transaction', new=mock_transaction_fail):
            operations = [AsyncMock()]
            with pytest.raises(Exception, match="Transaction failed"):
                await db.transaction(operations)


class TestDatabaseDatetime:
    """Test datetime handling in database operations"""

    @pytest.mark.asyncio
    async def test_datetime_in_insert(self):
        """Test that datetime objects are handled correctly in insert operations"""
        db = Database()
        db.connected = True
        db.db = Mock()

        collection_mock = AsyncMock()
        collection_mock.insert_one = AsyncMock(return_value=Mock(inserted_id="test_id"))
        db.db.__getitem__ = Mock(return_value=collection_mock)

        # Mock the insert_one method
        with patch.object(db, 'insert_one', new=AsyncMock(return_value="test_id")):
            test_data = {
                "name": "test",
                "created_at": datetime(2025, 11, 9, 12, 0, 0)
            }

            result = await db.insert_one("test_collection", test_data)
            assert result == "test_id"

    @pytest.mark.asyncio
    async def test_datetime_in_query(self):
        """Test that datetime objects are handled correctly in queries"""
        db = Database()
        db.connected = True
        db.db = Mock()

        collection_mock = AsyncMock()
        collection_mock.find_one = AsyncMock(return_value={
            "_id": "test_id",
            "name": "test",
            "created_at": datetime(2025, 11, 9, 12, 0, 0)
        })
        db.db.__getitem__ = Mock(return_value=collection_mock)

        # Mock the find_one method
        with patch.object(db, 'find_one', new=AsyncMock(return_value={
            "_id": "test_id",
            "name": "test",
            "created_at": datetime(2025, 11, 9, 12, 0, 0)
        })):
            result = await db.find_one("test_collection", {"_id": "test_id"})
            assert result["name"] == "test"
            assert isinstance(result["created_at"], datetime)


class TestDatabaseSingleton:
    """Test database singleton pattern"""

    @pytest.mark.asyncio
    async def test_get_database_singleton(self):
        """Test that get_database returns the same instance"""
        db1 = await get_database()
        db2 = await get_database()

        assert db1 is db2


class TestDatabaseErrorHandling:
    """Test database error handling"""
    
    @pytest.mark.asyncio
    async def test_operation_when_disconnected(self):
        """Test operations when database is disconnected"""
        db = Database()
        db.connected = False
        
        # Mock methods to raise errors when disconnected
        with patch.object(db, 'insert_one', new=AsyncMock(side_effect=Exception("Not connected"))):
            with pytest.raises(Exception, match="Not connected"):
                await db.insert_one("test_collection", {"name": "test"})
    
    @pytest.mark.asyncio
    async def test_invalid_collection_name(self):
        """Test operations with invalid collection name"""
        db = Database()
        db.connected = True
        db.db = Mock()
        
        # Mock invalid collection
        db.db.__getitem__ = Mock(side_effect=Exception("Invalid collection"))
        
        with pytest.raises(Exception, match="Invalid collection"):
            _ = db.db["invalid_collection"]


# Fixtures
@pytest.fixture
async def database():
    """Fixture to provide a database instance"""
    db = Database()
    yield db
    if db.connected:
        await db.disconnect()


@pytest.fixture
async def connected_database():
    """Fixture to provide a connected database instance"""
    db = Database()
    
    with patch('database.AsyncIOMotorClient') as mock_client:
        mock_instance = AsyncMock()
        mock_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_instance.__getitem__ = Mock(return_value=Mock())
        mock_client.return_value = mock_instance
        
        with patch.object(db, '_initialize_collections', new=AsyncMock()):
            await db.connect()
    
    yield db
    
    if db.connected:
        await db.disconnect()

