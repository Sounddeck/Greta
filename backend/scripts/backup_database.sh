#!/bin/bash

################################################################################
# Greta Database Backup Script
# 
# This script creates automated backups of the greta_pai MongoDB database
# with compression, retention management, and error handling.
#
# Usage:
#   ./backup_database.sh [options]
#
# Options:
#   --full              Backup entire database (default)
#   --collection NAME   Backup specific collection only
#   --keep N            Keep last N backups (default: 7)
#   --dir PATH          Backup directory (default: ~/greta_backups)
#   --no-compress       Skip compression
#   --verbose           Verbose output
#
# Examples:
#   ./backup_database.sh
#   ./backup_database.sh --collection interactions --keep 14
#   ./backup_database.sh --dir /path/to/backups --verbose
################################################################################

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Configuration
DB_NAME="greta_pai"
DEFAULT_BACKUP_DIR="$HOME/greta_backups"
DEFAULT_KEEP_COUNT=7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$HOME/greta_backups/backup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
BACKUP_DIR="$DEFAULT_BACKUP_DIR"
KEEP_COUNT="$DEFAULT_KEEP_COUNT"
COLLECTION=""
COMPRESS=true
VERBOSE=false
BACKUP_TYPE="full"

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            BACKUP_TYPE="full"
            shift
            ;;
        --collection)
            COLLECTION="$2"
            BACKUP_TYPE="collection"
            shift 2
            ;;
        --keep)
            KEEP_COUNT="$2"
            shift 2
            ;;
        --dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
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

check_mongodb() {
    if ! command -v mongodump &> /dev/null; then
        log ERROR "mongodump not found. Please install MongoDB tools."
        log INFO "Install with: brew install mongodb-database-tools"
        exit 1
    fi
    
    if ! command -v mongorestore &> /dev/null; then
        log ERROR "mongorestore not found. Please install MongoDB tools."
        exit 1
    fi
    
    # Check if MongoDB is running
    if ! pgrep -x "mongod" > /dev/null; then
        log WARNING "MongoDB may not be running. Backup might fail."
    fi
}

create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    if [ ! -d "$BACKUP_DIR" ]; then
        log ERROR "Failed to create backup directory: $BACKUP_DIR"
        exit 1
    fi
    verbose_log "Backup directory: $BACKUP_DIR"
}

perform_backup() {
    local backup_name
    if [ "$BACKUP_TYPE" = "collection" ]; then
        backup_name="greta_${COLLECTION}_${TIMESTAMP}"
    else
        backup_name="greta_${TIMESTAMP}"
    fi
    
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log INFO "Starting backup: $backup_name"
    verbose_log "Backup path: $backup_path"
    
    # Perform backup
    if [ "$BACKUP_TYPE" = "collection" ]; then
        verbose_log "Backing up collection: $COLLECTION"
        if mongodump --db="$DB_NAME" --collection="$COLLECTION" --out="$backup_path" 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "Collection backup completed: $COLLECTION"
        else
            log ERROR "Collection backup failed: $COLLECTION"
            return 1
        fi
    else
        verbose_log "Backing up entire database: $DB_NAME"
        if mongodump --db="$DB_NAME" --out="$backup_path" 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "Full database backup completed"
        else
            log ERROR "Full database backup failed"
            return 1
        fi
    fi
    
    # Get backup size
    local backup_size=$(du -sh "$backup_path" | cut -f1)
    verbose_log "Backup size: $backup_size"
    
    # Compress backup
    if [ "$COMPRESS" = true ]; then
        log INFO "Compressing backup..."
        if tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "$backup_name" 2>&1 | tee -a "$LOG_FILE"; then
            local compressed_size=$(du -sh "$backup_path.tar.gz" | cut -f1)
            log SUCCESS "Backup compressed: $compressed_size"
            
            # Remove uncompressed backup
            rm -rf "$backup_path"
            verbose_log "Removed uncompressed backup"
            
            echo "$backup_path.tar.gz"
        else
            log ERROR "Compression failed"
            return 1
        fi
    else
        echo "$backup_path"
    fi
}

cleanup_old_backups() {
    log INFO "Cleaning up old backups (keeping last $KEEP_COUNT)..."
    
    local backup_pattern
    if [ "$BACKUP_TYPE" = "collection" ]; then
        backup_pattern="greta_${COLLECTION}_*.tar.gz"
    else
        backup_pattern="greta_*.tar.gz"
    fi
    
    # Count existing backups
    local backup_count=$(ls -1 "$BACKUP_DIR"/$backup_pattern 2>/dev/null | wc -l | tr -d ' ')
    verbose_log "Found $backup_count existing backups"
    
    if [ "$backup_count" -gt "$KEEP_COUNT" ]; then
        local to_delete=$((backup_count - KEEP_COUNT))
        verbose_log "Deleting $to_delete old backup(s)"
        
        ls -t "$BACKUP_DIR"/$backup_pattern | tail -n +$((KEEP_COUNT + 1)) | while read -r old_backup; do
            log INFO "Deleting old backup: $(basename "$old_backup")"
            rm -f "$old_backup"
        done
        
        log SUCCESS "Cleanup completed"
    else
        verbose_log "No cleanup needed"
    fi
}

verify_backup() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log ERROR "Backup file not found: $backup_file"
        return 1
    fi
    
    # Check if file is not empty
    if [ ! -s "$backup_file" ]; then
        log ERROR "Backup file is empty: $backup_file"
        return 1
    fi
    
    # Verify tar.gz integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" > /dev/null 2>&1; then
            verbose_log "Backup integrity verified"
            return 0
        else
            log ERROR "Backup file is corrupted: $backup_file"
            return 1
        fi
    fi
    
    return 0
}

print_summary() {
    local backup_file=$1
    local backup_size=$(du -sh "$backup_file" | cut -f1)
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log SUCCESS "Backup Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Database:        $DB_NAME"
    echo "  Backup Type:     $BACKUP_TYPE"
    if [ "$BACKUP_TYPE" = "collection" ]; then
        echo "  Collection:      $COLLECTION"
    fi
    echo "  Backup File:     $(basename "$backup_file")"
    echo "  Backup Size:     $backup_size"
    echo "  Backup Location: $backup_file"
    echo "  Timestamp:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Retention:       Last $KEEP_COUNT backups"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "🚀 Greta Database Backup Script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check prerequisites
    check_mongodb
    
    # Create backup directory
    create_backup_dir
    
    # Perform backup
    backup_file=$(perform_backup)
    if [ $? -ne 0 ]; then
        log ERROR "Backup failed"
        exit 1
    fi
    
    # Verify backup
    verify_backup "$backup_file"
    if [ $? -ne 0 ]; then
        log ERROR "Backup verification failed"
        exit 1
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Print summary
    print_summary "$backup_file"
    
    log SUCCESS "Backup completed successfully! 🎉"
}

# Run main function
main

