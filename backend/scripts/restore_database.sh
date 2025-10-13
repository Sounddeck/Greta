#!/bin/bash

################################################################################
# Greta Database Restore Script
# 
# This script restores the greta_pai MongoDB database from a backup.
#
# Usage:
#   ./restore_database.sh [options] <backup_file>
#
# Options:
#   --collection NAME   Restore specific collection only
#   --drop              Drop existing database/collection before restore
#   --verbose           Verbose output
#   --dry-run           Show what would be restored without actually restoring
#
# Examples:
#   ./restore_database.sh ~/greta_backups/greta_20251012_140000.tar.gz
#   ./restore_database.sh --drop ~/greta_backups/greta_20251012_140000.tar.gz
#   ./restore_database.sh --collection interactions backup.tar.gz
################################################################################

set -euo pipefail

# Configuration
DB_NAME="greta_pai"
LOG_FILE="$HOME/greta_backups/restore.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
COLLECTION=""
DROP_EXISTING=false
VERBOSE=false
DRY_RUN=false
BACKUP_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --collection)
            COLLECTION="$2"
            shift 2
            ;;
        --drop)
            DROP_EXISTING=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            BACKUP_FILE="$1"
            shift
            ;;
    esac
done

# Functions
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case $level in
        ERROR)
            echo -e "${RED}❌ $message${NC}"
            ;;
        SUCCESS)
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        WARNING)
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        INFO)
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        log INFO "$@"
    fi
}

check_prerequisites() {
    if ! command -v mongorestore &> /dev/null; then
        log ERROR "mongorestore not found. Please install MongoDB tools."
        log INFO "Install with: brew install mongodb-database-tools"
        exit 1
    fi
    
    if [ -z "$BACKUP_FILE" ]; then
        log ERROR "No backup file specified"
        echo ""
        echo "Usage: $0 [options] <backup_file>"
        exit 1
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        log ERROR "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
}

extract_backup() {
    local temp_dir=$(mktemp -d)

    log INFO "Extracting backup..." >&2
    verbose_log "Temporary directory: $temp_dir" >&2

    if [[ "$BACKUP_FILE" == *.tar.gz ]]; then
        if tar -xzf "$BACKUP_FILE" -C "$temp_dir" 2>&1 >> "$LOG_FILE"; then
            log SUCCESS "Backup extracted" >&2
        else
            log ERROR "Failed to extract backup" >&2
            rm -rf "$temp_dir"
            exit 1
        fi
    else
        log ERROR "Unsupported backup format. Expected .tar.gz file" >&2
        rm -rf "$temp_dir"
        exit 1
    fi

    # Only output the temp directory path to stdout
    echo "$temp_dir"
}

show_backup_info() {
    local temp_dir=$1

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Backup Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Backup File:     $(basename "$BACKUP_FILE")"
    echo "  Backup Size:     $(du -sh "$BACKUP_FILE" | cut -f1)"
    echo "  Database:        $DB_NAME"

    # List collections in backup
    local db_path=$(find "$temp_dir" -type d -name "$DB_NAME" 2>/dev/null | head -1)
    if [ -n "$db_path" ] && [ -d "$db_path" ]; then
        echo "  Collections:"
        find "$db_path" -name "*.bson" 2>/dev/null | while read -r bson_file; do
            local coll_name=$(basename "$bson_file" .bson)
            local coll_size=$(du -sh "$bson_file" | cut -f1)
            echo "    - $coll_name ($coll_size)"
        done
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

confirm_restore() {
    if [ "$DRY_RUN" = true ]; then
        log INFO "DRY RUN - No changes will be made"
        return 0
    fi
    
    echo ""
    if [ "$DROP_EXISTING" = true ]; then
        log WARNING "This will DROP the existing database/collection before restoring!"
    else
        log WARNING "This will merge the backup with existing data"
    fi
    
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log INFO "Restore cancelled by user"
        exit 0
    fi
}

perform_restore() {
    local temp_dir=$1

    # Find the database directory (it's inside the extracted backup folder)
    local db_path=$(find "$temp_dir" -type d -name "$DB_NAME" 2>/dev/null | head -1)

    if [ -z "$db_path" ] || [ ! -d "$db_path" ]; then
        log ERROR "Database directory not found in backup"
        verbose_log "Searched in: $temp_dir"
        verbose_log "Looking for directory named: $DB_NAME"
        verbose_log "Contents of temp_dir:"
        ls -la "$temp_dir" 2>&1 | tee -a "$LOG_FILE"
        return 1
    fi

    verbose_log "Found database at: $db_path"

    if [ "$DRY_RUN" = true ]; then
        log INFO "DRY RUN - Would restore from: $db_path"
        return 0
    fi

    log INFO "Starting restore..."

    local restore_cmd="mongorestore --db=$DB_NAME"

    if [ "$DROP_EXISTING" = true ]; then
        restore_cmd="$restore_cmd --drop"
        verbose_log "Will drop existing data before restore"
    fi

    if [ -n "$COLLECTION" ]; then
        restore_cmd="$restore_cmd --collection=$COLLECTION"
        local collection_file="$db_path/${COLLECTION}.bson"
        if [ ! -f "$collection_file" ]; then
            log ERROR "Collection not found in backup: $COLLECTION"
            return 1
        fi
        restore_cmd="$restore_cmd $collection_file"
        verbose_log "Restoring collection: $COLLECTION"
    else
        restore_cmd="$restore_cmd $db_path"
        verbose_log "Restoring entire database"
    fi

    verbose_log "Restore command: $restore_cmd"

    if eval "$restore_cmd" 2>&1 >> "$LOG_FILE"; then
        log SUCCESS "Restore completed successfully"
        return 0
    else
        log ERROR "Restore failed"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    echo "🔄 Greta Database Restore Script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check prerequisites
    check_prerequisites
    
    # Extract backup
    temp_dir=$(extract_backup)
    
    # Show backup info
    show_backup_info "$temp_dir"
    
    # Confirm restore
    confirm_restore
    
    # Perform restore
    if perform_restore "$temp_dir"; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log SUCCESS "Database restored successfully! 🎉"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    else
        log ERROR "Restore failed"
        exit 1
    fi
    
    # Cleanup temp directory
    rm -rf "$temp_dir"
    verbose_log "Cleaned up temporary files"
}

# Run main function
main

