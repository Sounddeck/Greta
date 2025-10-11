"""
Portability Service - Week 15-16 Implementation

Cross-platform data portability, backup/restore, cloud sync, and encryption.
Enables export/import of all Greta data across platforms and devices.
"""

import asyncio
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet
from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================


class ExportMetadata(BaseModel):
    """Metadata for exported data"""

    export_id: str
    export_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    greta_version: str = "1.0.0"
    data_version: str = "1.0.0"
    export_type: str = "full"  # full, incremental, selective
    data_types: List[str] = Field(default_factory=list)
    total_items: int = 0
    encrypted: bool = False
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImportResult(BaseModel):
    """Result of import operation"""

    success: bool
    imported_items: int = 0
    skipped_items: int = 0
    failed_items: int = 0
    conflicts: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BackupInfo(BaseModel):
    """Backup information"""

    backup_id: str
    backup_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    backup_type: str = "full"  # full, incremental
    file_path: str
    file_size: int = 0
    encrypted: bool = False
    checksum: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CloudSyncStatus(BaseModel):
    """Cloud sync status"""

    provider: str  # s3, gdrive, dropbox, local
    last_sync: Optional[str] = None
    sync_enabled: bool = False
    auto_sync: bool = False
    sync_interval: int = 3600  # seconds
    last_sync_status: str = "never"  # never, success, failed
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Portability Service
# ============================================================================


class PortabilityService:
    """
    Portability Service

    Features:
    - Data export/import for all services
    - Cloud storage integration
    - Backup and restore
    - Data encryption
    - Data validation
    - Incremental backups
    - Migration tools
    """

    def __init__(
        self,
        exports_dir: str = "./data/exports",
        backups_dir: str = "./data/backups",
        encryption_key: Optional[bytes] = None,
    ):
        self.exports_dir = Path(exports_dir)
        self.backups_dir = Path(backups_dir)

        # Create directories
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        # Encryption
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

        # Cloud sync status
        self.cloud_sync_status = CloudSyncStatus(provider="local")

        # Backup history
        self.backup_history: List[BackupInfo] = []

        logger.info(f"Portability Service initialized")
        logger.info(f"  Exports directory: {self.exports_dir}")
        logger.info(f"  Backups directory: {self.backups_dir}")
        logger.info(f"  Encryption: {'enabled' if encryption_key else 'default key'}")

    # ========================================================================
    # Encryption
    # ========================================================================

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt data using Fernet"""
        return self.cipher.encrypt(data.encode())

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data using Fernet"""
        return self.cipher.decrypt(encrypted_data).decode()

    def generate_checksum(self, data: str) -> str:
        """Generate SHA256 checksum for data"""
        return hashlib.sha256(data.encode()).hexdigest()

    # ========================================================================
    # Data Export
    # ========================================================================

    async def export_conversations(self) -> Dict[str, Any]:
        """Export all conversations from conversation service"""
        try:
            # Import here to avoid circular dependency
            from backend.services.conversation_service import conversation_service

            conversations = []
            # Get all conversations (simplified - in production, paginate)
            all_convs = await conversation_service.list_conversations(limit=1000)

            for conv in all_convs:
                conversations.append(conv.model_dump())

            logger.info(f"Exported {len(conversations)} conversations")
            return {"conversations": conversations, "count": len(conversations)}

        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
            return {"conversations": [], "count": 0, "error": str(e)}

    async def export_agent_templates(self) -> Dict[str, Any]:
        """Export all agent templates"""
        try:
            from backend.services.agent_template_service import agent_template_service

            templates = []
            all_templates = await agent_template_service.list_templates(limit=1000)

            for template in all_templates:
                templates.append(template.model_dump())

            logger.info(f"Exported {len(templates)} agent templates")
            return {"templates": templates, "count": len(templates)}

        except Exception as e:
            logger.error(f"Error exporting agent templates: {e}")
            return {"templates": [], "count": 0, "error": str(e)}

    async def export_contexts(self) -> Dict[str, Any]:
        """Export all context windows"""
        try:
            from backend.services.context_management_service import (
                context_management_service,
            )

            contexts = []
            # Export all context windows
            for window_id, window in context_management_service.contexts.items():
                contexts.append(window.model_dump())

            logger.info(f"Exported {len(contexts)} context windows")
            return {"contexts": contexts, "count": len(contexts)}

        except Exception as e:
            logger.error(f"Error exporting contexts: {e}")
            return {"contexts": [], "count": 0, "error": str(e)}

    async def export_rag_documents(self) -> Dict[str, Any]:
        """Export RAG documents and collections"""
        try:
            from backend.services.rag_service import rag_service

            collections = []
            # Get all collections
            all_collections = await rag_service.list_collections()

            for collection_name in all_collections:
                # Get documents from collection
                docs = await rag_service.get_all_documents(collection_name)
                collections.append(
                    {
                        "collection_name": collection_name,
                        "documents": docs,
                        "count": len(docs),
                    }
                )

            logger.info(f"Exported {len(collections)} RAG collections")
            return {"collections": collections, "count": len(collections)}

        except Exception as e:
            logger.error(f"Error exporting RAG documents: {e}")
            return {"collections": [], "count": 0, "error": str(e)}

    async def export_models(self) -> Dict[str, Any]:
        """Export model configurations"""
        try:
            from backend.services.model_service import model_service

            models = []
            all_models = await model_service.list_models()

            for model in all_models:
                models.append(model.model_dump())

            logger.info(f"Exported {len(models)} model configurations")
            return {"models": models, "count": len(models)}

        except Exception as e:
            logger.error(f"Error exporting models: {e}")
            return {"models": [], "count": 0, "error": str(e)}

    async def export_lora_adapters(self) -> Dict[str, Any]:
        """Export LoRA adapters"""
        try:
            from backend.services.adapter_registry import adapter_registry

            adapters = []
            all_adapters = await adapter_registry.list_adapters()

            for adapter in all_adapters:
                adapters.append(adapter.model_dump())

            logger.info(f"Exported {len(adapters)} LoRA adapters")
            return {"adapters": adapters, "count": len(adapters)}

        except Exception as e:
            logger.error(f"Error exporting LoRA adapters: {e}")
            return {"adapters": [], "count": 0, "error": str(e)}

    async def export_all_data(
        self, export_id: Optional[str] = None, encrypt: bool = False
    ) -> Optional[str]:
        """
        Export all Greta data to a portable format

        Returns path to export file (ZIP)
        """
        export_id = export_id or f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting full data export: {export_id}")

        # Collect all data
        export_data = {
            "conversations": await self.export_conversations(),
            "agent_templates": await self.export_agent_templates(),
            "contexts": await self.export_contexts(),
            "rag_documents": await self.export_rag_documents(),
            "models": await self.export_models(),
            "lora_adapters": await self.export_lora_adapters(),
        }

        # Calculate total items
        total_items = sum(
            data.get("count", 0) for data in export_data.values() if isinstance(data, dict)
        )

        # Create metadata
        metadata = ExportMetadata(
            export_id=export_id,
            export_type="full",
            data_types=list(export_data.keys()),
            total_items=total_items,
            encrypted=encrypt,
        )

        # Convert to JSON
        export_json = json.dumps(
            {"metadata": metadata.model_dump(), "data": export_data}, indent=2
        )

        # Generate checksum
        metadata.checksum = self.generate_checksum(export_json)

        # Encrypt if requested
        if encrypt:
            export_json = self.encrypt_data(export_json).decode("latin-1")

        # Create ZIP file
        zip_path = self.exports_dir / f"{export_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("export.json", export_json)
            zf.writestr("metadata.json", metadata.model_dump_json(indent=2))

        logger.info(f"Export complete: {zip_path} ({total_items} items)")
        return str(zip_path)

    # ========================================================================
    # Data Import
    # ========================================================================

    async def import_conversations(
        self, conversations: List[Dict[str, Any]], conflict_resolution: str = "skip"
    ) -> Tuple[int, int, List[str]]:
        """
        Import conversations

        conflict_resolution: skip, overwrite, merge
        Returns: (imported, skipped, conflicts)
        """
        try:
            from backend.services.conversation_service import (
                conversation_service,
                Conversation,
            )

            imported = 0
            skipped = 0
            conflicts = []

            for conv_data in conversations:
                try:
                    # Check if conversation exists
                    existing = await conversation_service.get_conversation(
                        conv_data.get("conversation_id")
                    )

                    if existing:
                        if conflict_resolution == "skip":
                            skipped += 1
                            conflicts.append(
                                f"Conversation {conv_data.get('conversation_id')} already exists"
                            )
                            continue
                        elif conflict_resolution == "overwrite":
                            # Delete existing
                            await conversation_service.delete_conversation(
                                conv_data.get("conversation_id")
                            )

                    # Create conversation
                    conv = Conversation(**conv_data)
                    # Save to database (simplified)
                    imported += 1

                except Exception as e:
                    logger.error(f"Error importing conversation: {e}")
                    skipped += 1

            logger.info(f"Imported {imported} conversations, skipped {skipped}")
            return imported, skipped, conflicts

        except Exception as e:
            logger.error(f"Error in import_conversations: {e}")
            return 0, 0, [str(e)]

    async def import_agent_templates(
        self, templates: List[Dict[str, Any]], conflict_resolution: str = "skip"
    ) -> Tuple[int, int, List[str]]:
        """Import agent templates"""
        try:
            from backend.services.agent_template_service import (
                agent_template_service,
                AgentTemplate,
            )

            imported = 0
            skipped = 0
            conflicts = []

            for template_data in templates:
                try:
                    # Check if template exists
                    existing = await agent_template_service.get_template(
                        template_data.get("template_id")
                    )

                    if existing:
                        if conflict_resolution == "skip":
                            skipped += 1
                            conflicts.append(
                                f"Template {template_data.get('template_id')} already exists"
                            )
                            continue
                        elif conflict_resolution == "overwrite":
                            await agent_template_service.delete_template(
                                template_data.get("template_id")
                            )

                    # Create template
                    template = AgentTemplate(**template_data)
                    await agent_template_service.create_template(
                        name=template.name,
                        description=template.description,
                        agent_type=template.agent_type,
                        system_message=template.system_message,
                        category=template.category,
                    )
                    imported += 1

                except Exception as e:
                    logger.error(f"Error importing template: {e}")
                    skipped += 1

            logger.info(f"Imported {imported} templates, skipped {skipped}")
            return imported, skipped, conflicts

        except Exception as e:
            logger.error(f"Error in import_agent_templates: {e}")
            return 0, 0, [str(e)]

    async def import_all_data(
        self, import_file: str, conflict_resolution: str = "skip", decrypt: bool = False
    ) -> ImportResult:
        """
        Import all data from export file

        conflict_resolution: skip, overwrite, merge
        """
        logger.info(f"Starting data import from: {import_file}")

        try:
            # Extract ZIP
            with zipfile.ZipFile(import_file, "r") as zf:
                # Read metadata
                metadata_json = zf.read("metadata.json").decode()
                metadata = ExportMetadata(**json.loads(metadata_json))

                # Read export data
                export_json = zf.read("export.json").decode()

                # Decrypt if needed
                if decrypt or metadata.encrypted:
                    export_json = self.decrypt_data(export_json.encode("latin-1"))

                # Verify checksum
                if metadata.checksum:
                    current_checksum = self.generate_checksum(export_json)
                    if current_checksum != metadata.checksum:
                        logger.warning("Checksum mismatch - data may be corrupted")

                # Parse data
                export_data = json.loads(export_json)
                data = export_data.get("data", {})

                # Import each data type
                total_imported = 0
                total_skipped = 0
                all_conflicts = []

                # Import conversations
                if "conversations" in data:
                    convs = data["conversations"].get("conversations", [])
                    imported, skipped, conflicts = await self.import_conversations(
                        convs, conflict_resolution
                    )
                    total_imported += imported
                    total_skipped += skipped
                    all_conflicts.extend(conflicts)

                # Import agent templates
                if "agent_templates" in data:
                    templates = data["agent_templates"].get("templates", [])
                    imported, skipped, conflicts = await self.import_agent_templates(
                        templates, conflict_resolution
                    )
                    total_imported += imported
                    total_skipped += skipped
                    all_conflicts.extend(conflicts)

                # TODO: Import other data types (contexts, RAG, models, adapters)

                logger.info(
                    f"Import complete: {total_imported} imported, {total_skipped} skipped"
                )

                return ImportResult(
                    success=True,
                    imported_items=total_imported,
                    skipped_items=total_skipped,
                    conflicts=all_conflicts,
                )

        except Exception as e:
            logger.error(f"Error importing data: {e}")
            return ImportResult(
                success=False, errors=[str(e)], metadata={"import_file": import_file}
            )

    # ========================================================================
    # Backup and Restore
    # ========================================================================

    async def create_backup(
        self, backup_type: str = "full", encrypt: bool = True
    ) -> Optional[BackupInfo]:
        """Create a backup of all data"""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Creating {backup_type} backup: {backup_id}")

        # Export all data
        export_path = await self.export_all_data(export_id=backup_id, encrypt=encrypt)

        if not export_path:
            logger.error("Failed to create backup")
            return None

        # Move to backups directory
        export_file = Path(export_path)
        backup_file = self.backups_dir / export_file.name
        export_file.rename(backup_file)

        # Get file size
        file_size = backup_file.stat().st_size

        # Generate checksum
        with open(backup_file, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=backup_type,
            file_path=str(backup_file),
            file_size=file_size,
            encrypted=encrypt,
            checksum=checksum,
        )

        # Add to history
        self.backup_history.append(backup_info)

        logger.info(f"Backup created: {backup_file} ({file_size} bytes)")
        return backup_info

    async def restore_backup(
        self, backup_file: str, conflict_resolution: str = "skip"
    ) -> ImportResult:
        """Restore from backup file"""
        logger.info(f"Restoring from backup: {backup_file}")

        # Find backup info
        backup_info = None
        for info in self.backup_history:
            if info.file_path == backup_file:
                backup_info = info
                break

        # Verify checksum if available
        if backup_info and backup_info.checksum:
            with open(backup_file, "rb") as f:
                current_checksum = hashlib.sha256(f.read()).hexdigest()
                if current_checksum != backup_info.checksum:
                    logger.error("Backup file checksum mismatch - file may be corrupted")
                    return ImportResult(
                        success=False, errors=["Checksum verification failed"]
                    )

        # Import data
        decrypt = backup_info.encrypted if backup_info else True
        return await self.import_all_data(
            backup_file, conflict_resolution=conflict_resolution, decrypt=decrypt
        )

    async def list_backups(self) -> List[BackupInfo]:
        """List all available backups"""
        return self.backup_history

    # ========================================================================
    # Cloud Storage Integration
    # ========================================================================

    async def configure_cloud_storage(
        self,
        provider: str,
        credentials: Dict[str, Any],
        auto_sync: bool = False,
        sync_interval: int = 3600,
    ) -> bool:
        """
        Configure cloud storage provider

        provider: s3, gdrive, dropbox, local
        credentials: provider-specific credentials
        """
        logger.info(f"Configuring cloud storage: {provider}")

        self.cloud_sync_status.provider = provider
        self.cloud_sync_status.sync_enabled = True
        self.cloud_sync_status.auto_sync = auto_sync
        self.cloud_sync_status.sync_interval = sync_interval

        # Store credentials securely (simplified - in production, use secure storage)
        self.cloud_sync_status.metadata["credentials"] = credentials

        logger.info(f"Cloud storage configured: {provider}")
        return True

    async def sync_to_cloud(self, file_path: str) -> bool:
        """Sync file to cloud storage"""
        if not self.cloud_sync_status.sync_enabled:
            logger.warning("Cloud sync not enabled")
            return False

        provider = self.cloud_sync_status.provider

        logger.info(f"Syncing to {provider}: {file_path}")

        try:
            if provider == "local":
                # Local sync (copy to another directory)
                # In production, implement actual cloud providers
                logger.info("Local sync - file already available")
                success = True

            elif provider == "s3":
                # TODO: Implement S3 upload using boto3
                logger.warning("S3 sync not yet implemented")
                success = False

            elif provider == "gdrive":
                # TODO: Implement Google Drive upload
                logger.warning("Google Drive sync not yet implemented")
                success = False

            elif provider == "dropbox":
                # TODO: Implement Dropbox upload
                logger.warning("Dropbox sync not yet implemented")
                success = False

            else:
                logger.error(f"Unknown provider: {provider}")
                success = False

            # Update sync status
            self.cloud_sync_status.last_sync = datetime.utcnow().isoformat()
            self.cloud_sync_status.last_sync_status = "success" if success else "failed"

            return success

        except Exception as e:
            logger.error(f"Error syncing to cloud: {e}")
            self.cloud_sync_status.last_sync_status = "failed"
            return False

    async def sync_from_cloud(self, file_name: str, destination: str) -> bool:
        """Sync file from cloud storage"""
        if not self.cloud_sync_status.sync_enabled:
            logger.warning("Cloud sync not enabled")
            return False

        provider = self.cloud_sync_status.provider

        logger.info(f"Syncing from {provider}: {file_name}")

        try:
            if provider == "local":
                logger.info("Local sync - file already available")
                return True

            # TODO: Implement cloud download for other providers

            return False

        except Exception as e:
            logger.error(f"Error syncing from cloud: {e}")
            return False

    async def get_cloud_sync_status(self) -> CloudSyncStatus:
        """Get current cloud sync status"""
        return self.cloud_sync_status

    # ========================================================================
    # Data Validation
    # ========================================================================

    async def validate_export(self, export_file: str) -> Tuple[bool, List[str]]:
        """
        Validate export file integrity

        Returns: (is_valid, errors)
        """
        errors = []

        try:
            # Check file exists
            if not Path(export_file).exists():
                errors.append(f"Export file not found: {export_file}")
                return False, errors

            # Extract and validate ZIP
            with zipfile.ZipFile(export_file, "r") as zf:
                # Check required files
                if "metadata.json" not in zf.namelist():
                    errors.append("Missing metadata.json")

                if "export.json" not in zf.namelist():
                    errors.append("Missing export.json")

                if errors:
                    return False, errors

                # Validate metadata
                metadata_json = zf.read("metadata.json").decode()
                metadata = ExportMetadata(**json.loads(metadata_json))

                # Validate export data
                export_json = zf.read("export.json").decode()

                # Decrypt if encrypted
                if metadata.encrypted:
                    try:
                        export_json = self.decrypt_data(export_json.encode("latin-1"))
                    except Exception as e:
                        errors.append(f"Decryption failed: {e}")
                        return False, errors

                # Verify checksum
                if metadata.checksum:
                    current_checksum = self.generate_checksum(export_json)
                    if current_checksum != metadata.checksum:
                        errors.append("Checksum mismatch - data may be corrupted")

                # Validate JSON structure
                try:
                    export_data = json.loads(export_json)
                    if "metadata" not in export_data or "data" not in export_data:
                        errors.append("Invalid export data structure")
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON: {e}")

            if errors:
                return False, errors

            logger.info(f"Export validation successful: {export_file}")
            return True, []

        except Exception as e:
            logger.error(f"Error validating export: {e}")
            return False, [str(e)]

    async def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of all data in the system

        Returns validation report
        """
        logger.info("Validating data integrity")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversations": {"valid": 0, "invalid": 0, "errors": []},
            "templates": {"valid": 0, "invalid": 0, "errors": []},
            "contexts": {"valid": 0, "invalid": 0, "errors": []},
            "models": {"valid": 0, "invalid": 0, "errors": []},
        }

        # Validate conversations
        try:
            from backend.services.conversation_service import conversation_service

            convs = await conversation_service.list_conversations(limit=1000)
            report["conversations"]["valid"] = len(convs)
        except Exception as e:
            report["conversations"]["errors"].append(str(e))

        # Validate templates
        try:
            from backend.services.agent_template_service import agent_template_service

            templates = await agent_template_service.list_templates(limit=1000)
            report["templates"]["valid"] = len(templates)
        except Exception as e:
            report["templates"]["errors"].append(str(e))

        # Validate contexts
        try:
            from backend.services.context_management_service import (
                context_management_service,
            )

            contexts = context_management_service.contexts
            report["contexts"]["valid"] = len(contexts)
        except Exception as e:
            report["contexts"]["errors"].append(str(e))

        logger.info("Data integrity validation complete")
        return report

    # ========================================================================
    # Migration Tools
    # ========================================================================

    async def migrate_from_platform(
        self, platform: str, data_file: str
    ) -> ImportResult:
        """
        Migrate data from other AI platforms

        platform: chatgpt, claude, other
        """
        logger.info(f"Migrating from {platform}: {data_file}")

        # TODO: Implement platform-specific migration logic
        # For now, return placeholder

        return ImportResult(
            success=False,
            errors=[f"Migration from {platform} not yet implemented"],
            metadata={"platform": platform, "data_file": data_file},
        )

    # ========================================================================
    # Incremental Backup
    # ========================================================================

    async def create_incremental_backup(
        self, since_date: Optional[str] = None
    ) -> Optional[BackupInfo]:
        """
        Create incremental backup (only changed data since last backup)

        since_date: ISO format date string
        """
        logger.info(f"Creating incremental backup since: {since_date}")

        # TODO: Implement incremental backup logic
        # For now, create full backup
        return await self.create_backup(backup_type="incremental", encrypt=True)


# ============================================================================
# Singleton Instance
# ============================================================================

portability_service = PortabilityService()

