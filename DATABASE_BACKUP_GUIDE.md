# Greta Database Backup & Restore Guide

**Database:** greta_pai (MongoDB)  
**Backup Location:** `~/greta_backups/`  
**Retention Policy:** Last 7 backups (configurable)  
**Schedule:** Daily at 2:00 AM (automated)  

---

## Quick Start

### Manual Backup (Right Now)
```bash
cd /Users/macone/CPAS/Desktop/CPAS5/backend/scripts
./backup_database.sh
```

### Restore from Backup
```bash
cd /Users/macone/CPAS/Desktop/CPAS5/backend/scripts
./restore_database.sh ~/greta_backups/greta_20251012_140000.tar.gz
```

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Manual Backup](#manual-backup)
3. [Automated Backup](#automated-backup)
4. [Restore Procedures](#restore-procedures)
5. [Backup Verification](#backup-verification)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Installation & Setup

### Prerequisites

1. **MongoDB Tools** (required)
```bash
# Check if installed
which mongodump mongorestore

# If not installed, install with Homebrew
brew install mongodb-database-tools
```

2. **Make Scripts Executable**
```bash
cd /Users/macone/CPAS/Desktop/CPAS5/backend/scripts
chmod +x backup_database.sh
chmod +x restore_database.sh
```

3. **Create Backup Directory**
```bash
mkdir -p ~/greta_backups
```

### Verify Installation
```bash
# Test backup script
./backup_database.sh --verbose

# Check backup was created
ls -lh ~/greta_backups/
```

---

## Manual Backup

### Full Database Backup
```bash
cd /Users/macone/CPAS/Desktop/CPAS5/backend/scripts

# Basic backup
./backup_database.sh

# Verbose output
./backup_database.sh --verbose

# Custom backup directory
./backup_database.sh --dir /path/to/backups

# Keep last 14 backups instead of 7
./backup_database.sh --keep 14

# No compression (faster but larger)
./backup_database.sh --no-compress
```

### Collection-Specific Backup
```bash
# Backup only interactions collection
./backup_database.sh --collection interactions

# Backup agents collection with verbose output
./backup_database.sh --collection agents --verbose

# Backup knowledge_base collection, keep last 30
./backup_database.sh --collection knowledge_base --keep 30
```

### Backup Output
```
🚀 Greta Database Backup Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ️  Starting backup: greta_20251012_140530
✅ Full database backup completed
ℹ️  Compressing backup...
✅ Backup compressed: 2.3M
ℹ️  Cleaning up old backups (keeping last 7)...
✅ Cleanup completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Backup Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Database:        greta_pai
  Backup Type:     full
  Backup File:     greta_20251012_140530.tar.gz
  Backup Size:     2.3M
  Backup Location: /Users/macone/greta_backups/greta_20251012_140530.tar.gz
  Timestamp:       2025-10-12 14:05:30
  Retention:       Last 7 backups
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Backup completed successfully! 🎉
```

---

## Automated Backup

### Setup Automated Daily Backups

**Step 1: Copy launchd Configuration**
```bash
# Copy plist to LaunchAgents directory
cp /Users/macone/CPAS/Desktop/CPAS5/backend/scripts/com.greta.backup.plist \
   ~/Library/LaunchAgents/

# Verify it was copied
ls -l ~/Library/LaunchAgents/com.greta.backup.plist
```

**Step 2: Load the Service**
```bash
# Load the backup service
launchctl load ~/Library/LaunchAgents/com.greta.backup.plist

# Verify it's loaded
launchctl list | grep greta
```

**Step 3: Verify Schedule**
```bash
# Check service status
launchctl list com.greta.backup
```

### Managing Automated Backups

**Check Service Status:**
```bash
launchctl list | grep greta
```

**Unload Service (Stop Automated Backups):**
```bash
launchctl unload ~/Library/LaunchAgents/com.greta.backup.plist
```

**Reload Service (After Editing):**
```bash
launchctl unload ~/Library/LaunchAgents/com.greta.backup.plist
launchctl load ~/Library/LaunchAgents/com.greta.backup.plist
```

**View Automated Backup Logs:**
```bash
# View stdout log
tail -f ~/greta_backups/launchd_stdout.log

# View stderr log
tail -f ~/greta_backups/launchd_stderr.log

# View backup log
tail -f ~/greta_backups/backup.log
```

**Test Automated Backup (Run Now):**
```bash
# Trigger backup immediately (without waiting for 2 AM)
launchctl start com.greta.backup
```

### Customizing Schedule

Edit `~/Library/LaunchAgents/com.greta.backup.plist`:

**Daily at Different Time:**
```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>14</integer>  <!-- 2:00 PM -->
    <key>Minute</key>
    <integer>30</integer>
</dict>
```

**Multiple Times Per Day:**
```xml
<key>StartCalendarInterval</key>
<array>
    <dict>
        <key>Hour</key>
        <integer>2</integer>  <!-- 2:00 AM -->
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <dict>
        <key>Hour</key>
        <integer>14</integer>  <!-- 2:00 PM -->
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</array>
```

**Weekly (Every Monday at 2 AM):**
```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Weekday</key>
    <integer>1</integer>  <!-- Monday -->
    <key>Hour</key>
    <integer>2</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

After editing, reload the service:
```bash
launchctl unload ~/Library/LaunchAgents/com.greta.backup.plist
launchctl load ~/Library/LaunchAgents/com.greta.backup.plist
```

---

## Restore Procedures

### Full Database Restore

**⚠️ WARNING:** This will replace your entire database!

```bash
cd /Users/macone/CPAS/Desktop/CPAS5/backend/scripts

# List available backups
ls -lht ~/greta_backups/*.tar.gz | head -10

# Restore (with confirmation prompt)
./restore_database.sh ~/greta_backups/greta_20251012_140000.tar.gz

# Restore and drop existing data first (clean restore)
./restore_database.sh --drop ~/greta_backups/greta_20251012_140000.tar.gz

# Dry run (see what would be restored)
./restore_database.sh --dry-run ~/greta_backups/greta_20251012_140000.tar.gz
```

### Collection-Specific Restore

```bash
# Restore only interactions collection
./restore_database.sh --collection interactions ~/greta_backups/greta_20251012_140000.tar.gz

# Restore agents collection and drop existing
./restore_database.sh --collection agents --drop ~/greta_backups/greta_20251012_140000.tar.gz
```

### Restore Output
```
🔄 Greta Database Restore Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ️  Extracting backup...
✅ Backup extracted

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  Backup Information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Backup File:     greta_20251012_140000.tar.gz
  Backup Size:     2.3M
  Database:        greta_pai
  Collections:
    - interactions (1.2M)
    - agents (512K)
    - knowledge_base (384K)
    - mcp_servers (128K)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  This will merge the backup with existing data

Are you sure you want to continue? (yes/no): yes

ℹ️  Starting restore...
✅ Restore completed successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Database restored successfully! 🎉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Backup Verification

### Verify Backup Integrity
```bash
# Check if backup file is valid
tar -tzf ~/greta_backups/greta_20251012_140000.tar.gz

# List contents
tar -tzf ~/greta_backups/greta_20251012_140000.tar.gz | head -20

# Extract to temporary location for inspection
mkdir -p /tmp/backup_test
tar -xzf ~/greta_backups/greta_20251012_140000.tar.gz -C /tmp/backup_test
ls -lR /tmp/backup_test
rm -rf /tmp/backup_test
```

### Verify Backup Contains Data
```bash
# Extract and check BSON file sizes
tar -xzf ~/greta_backups/greta_20251012_140000.tar.gz -C /tmp
find /tmp -name "*.bson" -exec ls -lh {} \;
rm -rf /tmp/greta_*
```

### Test Restore to Different Database
```bash
# Restore to test database (doesn't affect production)
mongorestore --db=greta_pai_test ~/greta_backups/greta_20251012_140000/greta_pai

# Verify test database
mongosh greta_pai_test --eval "db.getCollectionNames()"

# Drop test database when done
mongosh greta_pai_test --eval "db.dropDatabase()"
```

---

## Troubleshooting

### Issue: "mongodump not found"

**Solution:**
```bash
# Install MongoDB tools
brew install mongodb-database-tools

# Verify installation
which mongodump mongorestore
```

### Issue: "Permission denied"

**Solution:**
```bash
# Make scripts executable
chmod +x /Users/macone/CPAS/Desktop/CPAS5/backend/scripts/*.sh

# Check permissions
ls -l /Users/macone/CPAS/Desktop/CPAS5/backend/scripts/
```

### Issue: "MongoDB not running"

**Solution:**
```bash
# Check if MongoDB is running
pgrep -x mongod

# Start MongoDB (if using Homebrew)
brew services start mongodb-community

# Or start manually
mongod --config /usr/local/etc/mongod.conf
```

### Issue: "Backup file corrupted"

**Solution:**
```bash
# Verify tar.gz integrity
tar -tzf ~/greta_backups/greta_20251012_140000.tar.gz

# If corrupted, use previous backup
ls -lt ~/greta_backups/*.tar.gz | head -5
```

### Issue: "launchd service not running"

**Solution:**
```bash
# Check service status
launchctl list | grep greta

# If not listed, load it
launchctl load ~/Library/LaunchAgents/com.greta.backup.plist

# Check for errors
tail -50 ~/greta_backups/launchd_stderr.log
```

### Issue: "Backup directory full"

**Solution:**
```bash
# Check disk space
df -h ~/greta_backups

# Reduce retention count
./backup_database.sh --keep 3

# Or change backup location
./backup_database.sh --dir /path/to/larger/disk
```

---

## Advanced Usage

### Backup Specific Collections

**Critical Collections:**
```bash
# Backup user interactions (conversation history)
./backup_database.sh --collection interactions

# Backup agents (custom AI agents)
./backup_database.sh --collection agents

# Backup knowledge base
./backup_database.sh --collection knowledge_base

# Backup MCP servers
./backup_database.sh --collection mcp_servers
```

**All Collections in Database:**
- `interactions` - Chat history and conversations
- `agents` - Custom AI agents
- `knowledge_base` - Uploaded documents and knowledge
- `mcp_servers` - MCP server configurations
- `mcp_tools` - MCP tool definitions
- `mcp_integrations` - MCP integration settings
- `oauth_tokens` - OAuth authentication tokens
- `pai_interactions` - PAI learning data
- `prompt_evolutions` - Prompt evolution history
- `osint_investigations` - OSINT investigation records
- `training_courses` - Training platform data
- `voice_profiles` - Voice configuration profiles
- `hooks` - Hook system configurations

### Backup Before Major Changes

**Before Updating Code:**
```bash
# Create backup with descriptive name
./backup_database.sh --verbose
# Rename for clarity
mv ~/greta_backups/greta_$(date +%Y%m%d)_*.tar.gz \
   ~/greta_backups/before_update_$(date +%Y%m%d).tar.gz
```

**Before Database Migration:**
```bash
# Create backup before schema changes
./backup_database.sh --keep 30 --verbose
```

### Scheduled Backups at Different Intervals

**Hourly Backups (for critical periods):**

Edit `com.greta.backup.plist`:
```xml
<key>StartInterval</key>
<integer>3600</integer>  <!-- Every hour -->
```

**Weekly Backups:**
```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Weekday</key>
    <integer>0</integer>  <!-- Sunday -->
    <key>Hour</key>
    <integer>2</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

### Remote Backup (Cloud Storage)

**Sync to Cloud After Backup:**

Add to end of `backup_database.sh`:
```bash
# Sync to iCloud Drive
rsync -av ~/greta_backups/ ~/Library/Mobile\ Documents/com~apple~CloudDocs/greta_backups/

# Or sync to Dropbox
rsync -av ~/greta_backups/ ~/Dropbox/greta_backups/

# Or use rclone for other cloud providers
rclone sync ~/greta_backups/ remote:greta_backups/
```

---

## Backup Best Practices

### 1. Regular Testing
```bash
# Test restore monthly to ensure backups are valid
./restore_database.sh --dry-run ~/greta_backups/latest_backup.tar.gz
```

### 2. Multiple Backup Locations
```bash
# Keep local backups
./backup_database.sh --dir ~/greta_backups

# Also backup to external drive
./backup_database.sh --dir /Volumes/ExternalDrive/greta_backups
```

### 3. Monitor Backup Size
```bash
# Check total backup size
du -sh ~/greta_backups

# List backups by size
ls -lhS ~/greta_backups/*.tar.gz
```

### 4. Retention Strategy

**Recommended retention:**
- Daily backups: Keep last 7 days
- Weekly backups: Keep last 4 weeks
- Monthly backups: Keep last 12 months

**Implementation:**
```bash
# Daily backups (keep 7)
./backup_database.sh --keep 7

# Weekly backups (keep 4)
# Run on Sundays, keep 4
./backup_database.sh --dir ~/greta_backups/weekly --keep 4

# Monthly backups (keep 12)
# Run on 1st of month, keep 12
./backup_database.sh --dir ~/greta_backups/monthly --keep 12
```

---

## Emergency Recovery

### Scenario 1: Database Corruption

```bash
# 1. Stop the application
# 2. Restore from most recent backup
./restore_database.sh --drop ~/greta_backups/greta_latest.tar.gz

# 3. Restart application
```

### Scenario 2: Accidental Data Deletion

```bash
# 1. Find backup from before deletion
ls -lt ~/greta_backups/*.tar.gz

# 2. Restore specific collection (doesn't drop other data)
./restore_database.sh --collection interactions ~/greta_backups/greta_20251012_100000.tar.gz
```

### Scenario 3: Migration to New Machine

```bash
# On old machine: Create backup
./backup_database.sh --verbose

# Copy backup to new machine
scp ~/greta_backups/greta_*.tar.gz newmachine:~/

# On new machine: Restore
./restore_database.sh ~/greta_20251012_140000.tar.gz
```

---

## Monitoring & Maintenance

### Check Backup Health

**Daily Check:**
```bash
# Verify latest backup exists and is recent
ls -lt ~/greta_backups/*.tar.gz | head -1

# Check backup log for errors
tail -20 ~/greta_backups/backup.log | grep ERROR
```

**Weekly Check:**
```bash
# Verify backup count
ls -1 ~/greta_backups/*.tar.gz | wc -l

# Check total backup size
du -sh ~/greta_backups

# Test restore (dry run)
./restore_database.sh --dry-run $(ls -t ~/greta_backups/*.tar.gz | head -1)
```

### Backup Alerts

**Get Notified of Backup Failures:**

Add to `backup_database.sh` (after main function):
```bash
# Send notification on failure
if [ $? -ne 0 ]; then
    osascript -e 'display notification "Greta database backup failed!" with title "Backup Error"'
fi
```

---

## Quick Reference

### Common Commands

```bash
# Manual backup now
./backup_database.sh

# Restore latest backup
./restore_database.sh $(ls -t ~/greta_backups/*.tar.gz | head -1)

# List all backups
ls -lht ~/greta_backups/*.tar.gz

# Check backup logs
tail -50 ~/greta_backups/backup.log

# Check automated backup status
launchctl list | grep greta

# Trigger automated backup now
launchctl start com.greta.backup
```

### File Locations

```
Backup Scripts:
  /Users/macone/CPAS/Desktop/CPAS5/backend/scripts/backup_database.sh
  /Users/macone/CPAS/Desktop/CPAS5/backend/scripts/restore_database.sh

Launchd Configuration:
  ~/Library/LaunchAgents/com.greta.backup.plist

Backups:
  ~/greta_backups/greta_YYYYMMDD_HHMMSS.tar.gz

Logs:
  ~/greta_backups/backup.log
  ~/greta_backups/restore.log
  ~/greta_backups/launchd_stdout.log
  ~/greta_backups/launchd_stderr.log
```

---

## Next Steps

1. ✅ Run your first manual backup
2. ✅ Verify backup was created successfully
3. ✅ Set up automated daily backups with launchd
4. ✅ Test restore procedure (dry run)
5. ✅ Monitor backup logs for the first week

**Your data is now protected! 🛡️**

