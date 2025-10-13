#!/bin/bash

################################################################################
# Setup Automated Backup for Greta Database
#
# This script sets up automated daily backups using launchd
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "🔧 Greta Automated Backup Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if launchd plist exists
PLIST_SOURCE="/Users/macone/CPAS/Desktop/CPAS5/backend/scripts/com.greta.backup.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.greta.backup.plist"

if [ ! -f "$PLIST_SOURCE" ]; then
    echo -e "${YELLOW}⚠️  Plist file not found: $PLIST_SOURCE${NC}"
    exit 1
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$HOME/Library/LaunchAgents"

# Copy plist file
echo -e "${BLUE}ℹ️  Copying launchd configuration...${NC}"
cp "$PLIST_SOURCE" "$PLIST_DEST"
echo -e "${GREEN}✅ Copied to: $PLIST_DEST${NC}"

# Unload if already loaded (ignore errors)
echo ""
echo -e "${BLUE}ℹ️  Unloading existing service (if any)...${NC}"
launchctl unload "$PLIST_DEST" 2>/dev/null || true

# Load the service
echo -e "${BLUE}ℹ️  Loading backup service...${NC}"
if launchctl load "$PLIST_DEST"; then
    echo -e "${GREEN}✅ Backup service loaded successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Failed to load service${NC}"
    exit 1
fi

# Verify service is loaded
echo ""
echo -e "${BLUE}ℹ️  Verifying service status...${NC}"
if launchctl list | grep -q "com.greta.backup"; then
    echo -e "${GREEN}✅ Service is running${NC}"
else
    echo -e "${YELLOW}⚠️  Service not found in launchctl list${NC}"
fi

# Create backup directory
echo ""
echo -e "${BLUE}ℹ️  Creating backup directory...${NC}"
mkdir -p "$HOME/greta_backups"
echo -e "${GREEN}✅ Backup directory ready: $HOME/greta_backups${NC}"

# Show configuration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Automated Backup Configuration${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Schedule:        Daily at 2:00 AM"
echo "  Backup Location: $HOME/greta_backups/"
echo "  Retention:       Last 7 backups"
echo "  Database:        greta_pai"
echo "  Service:         com.greta.backup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show useful commands
echo "📚 Useful Commands:"
echo ""
echo "  # Check service status"
echo "  launchctl list | grep greta"
echo ""
echo "  # Trigger backup now (without waiting for 2 AM)"
echo "  launchctl start com.greta.backup"
echo ""
echo "  # View backup logs"
echo "  tail -f ~/greta_backups/backup.log"
echo ""
echo "  # View launchd logs"
echo "  tail -f ~/greta_backups/launchd_stdout.log"
echo ""
echo "  # Stop automated backups"
echo "  launchctl unload ~/Library/LaunchAgents/com.greta.backup.plist"
echo ""
echo "  # Restart automated backups"
echo "  launchctl load ~/Library/LaunchAgents/com.greta.backup.plist"
echo ""

# Ask if user wants to test backup now
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Would you like to run a test backup now? (yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${BLUE}ℹ️  Triggering test backup...${NC}"
    echo ""
    launchctl start com.greta.backup
    
    echo ""
    echo -e "${GREEN}✅ Test backup triggered!${NC}"
    echo ""
    echo "Check the backup log for results:"
    echo "  tail -f ~/greta_backups/backup.log"
    echo ""
    echo "Or check launchd output:"
    echo "  tail -f ~/greta_backups/launchd_stdout.log"
    echo ""
else
    echo -e "${BLUE}ℹ️  Skipping test backup${NC}"
    echo ""
    echo "The first automated backup will run tomorrow at 2:00 AM"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Setup complete! Your database will be backed up daily at 2:00 AM${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

