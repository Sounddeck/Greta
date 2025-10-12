# Bug Fixing & Maintenance Plan - CPAS4/Greta

**Project:** CPAS4/Greta - Personal AI Assistant Desktop App  
**Location:** `/Users/macone/Desktop/Greta`  
**Repository:** https://github.com/Sounddeck/Greta.git  
**Type:** Personal project (not open source)  
**Last Updated:** October 12, 2025  

---

## 1. Bug Reporting Workflow

### How to Report Bugs

Since this is a personal project, use a **simple text-based tracking system**:

**Create a `BUGS.md` file in the project root:**

```markdown
# Bug Log - CPAS4/Greta

## Active Bugs

### [BUG-001] Short descriptive title
- **Severity:** Critical/High/Medium/Low
- **Component:** Frontend/Backend/Desktop/Database
- **Reported:** 2025-10-12
- **Status:** Open/In Progress/Fixed/Closed
- **Description:** What went wrong
- **Steps to Reproduce:**
  1. Step 1
  2. Step 2
  3. Expected vs. Actual result
- **Error Messages:** (if any)
- **Screenshots:** (path to screenshot file)
- **Logs:** (relevant log excerpts)
- **Environment:** Dev/Production, macOS version, Greta version
- **Fix Commit:** (commit hash when fixed)

## Fixed Bugs
(Move bugs here when resolved)

## Known Issues
(Non-critical issues to fix later)
```

### Information to Provide

**Essential Information:**
1. **Bug Title:** Short, descriptive
2. **Severity:** Critical/High/Medium/Low
3. **Component:** Frontend/Backend/Desktop/Database/API
4. **Steps to Reproduce:** Exact steps
5. **Expected vs. Actual Behavior**
6. **Error Messages:** Full error text
7. **Environment:** Greta version, macOS version, Dev/Production
8. **Screenshots:** If UI-related
9. **Logs:** Relevant excerpts

---

## 2. Development Workflow for Bug Fixes

### Git Branching Strategy

**Simple Workflow (Recommended for Personal Project):**

- **Small fixes:** Direct to `main` branch
- **Complex fixes:** Feature branch `fix/bug-description`
- **Critical fixes:** Hotfix branch `hotfix/critical-issue`

### Bug Fix Workflow

```bash
# 1. Create bug report in BUGS.md

# 2. Create branch (for complex fixes)
cd /Users/macone/Desktop/Greta
git checkout -b fix/bug-description

# 3. Make changes and test

# 4. Commit with proper message
git commit -m "Fix: Bug description

- Root cause: Explanation
- Solution: What was changed
- Testing: How it was verified
- Fixes: BUG-XXX"

# 5. Merge to main
git checkout main
git merge fix/bug-description
git push origin main

# 6. Update BUGS.md - mark as Fixed
# 7. Rebuild production if needed
```

### Commit Message Format

```
Fix: Short description (50 chars max)

- Root cause: Why the bug occurred
- Solution: What was changed to fix it
- Testing: How the fix was verified
- Fixes: BUG-XXX

Files modified:
- path/to/file1.tsx
- path/to/file2.ts
```

---

## 3. Code Change Process

### Safe Code Change Workflow

**Before Making Changes:**
1. Understand the bug - reproduce it locally
2. Identify root cause using debugging tools
3. Plan the fix - which files need changes
4. Create backup: `git tag backup-$(date +%Y%m%d)`

**Making Changes:**
1. Make minimal, focused changes
2. Follow TypeScript best practices (strict mode, no `any`)
3. Test incrementally

**After Making Changes:**

```bash
# 1. Type checking
cd /Users/macone/Desktop/Greta/cpas4-ui
npm run build

# 2. Test in dev mode
npm run tauri:dev

# 3. Test affected features

# 4. Test in production build
npm run tauri:build
open src-tauri/target/release/bundle/macos/Greta.app
```

### Handling Frontend + Backend Changes

```bash
# 1. Fix backend first
cd cpas4-backend
# Make changes, test with curl
python main.py

# 2. Fix frontend second
cd ../cpas4-ui
# Make changes
npm run dev

# 3. Test integration
# Backend on Terminal 699, frontend in Tauri

# 4. Commit both together
git add cpas4-backend/ cpas4-ui/
git commit -m "Fix: Bug affecting frontend and backend..."
```

### Changelog Maintenance

Create `CHANGELOG.md`:

```markdown
# Changelog - CPAS4/Greta

## [Unreleased]
### Fixed
- Voice settings now persist (BUG-001)

## [0.1.1] - 2025-10-15
### Fixed
- System tray icon (BUG-002)
- MCP connection timeout (BUG-003)

## [0.1.0] - 2025-10-12
### Added
- Initial desktop app release
```

---

## 4. Debugging Strategy

### Frontend Debugging (TypeScript/React)

**Development Mode:**

```bash
# Tauri DevTools
npm run tauri:dev
# Right-click → Inspect Element

# Console logging
console.log('[DEBUG] Variable:', variable);
console.error('[ERROR] Error:', error);

# Type checking
npx tsc --noEmit
```

**Zustand Store Debugging:**

```typescript
import { devtools } from 'zustand/middleware';

export const useVoiceStore = create<VoiceStore>()(
  devtools(
    (set, get) => ({ /* ... */ }),
    { name: 'VoiceStore' }
  )
);
```

### Backend Debugging (Python/FastAPI)

```python
# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")

# Debugger
import pdb; pdb.set_trace()

# API testing
curl -X GET http://localhost:8000/api/voices
```

### Desktop/Tauri Debugging (Rust)

```rust
// Logging
println!("[DEBUG] Variable: {:?}", variable);
eprintln!("[ERROR] Error: {:?}", error);
```

### Production Build Debugging

**Check logs:**

```bash
# macOS log location
~/Library/Logs/com.greta.assistant/Greta.log

# View in real-time
tail -f ~/Library/Logs/com.greta.assistant/Greta.log

# Search for errors
grep -i error ~/Library/Logs/com.greta.assistant/Greta.log
```

**Build with debug symbols:**

```bash
npm run tauri build -- --debug
```

### Log Locations

- **Dev mode:** Terminal output, Browser/Tauri DevTools
- **Production:** `~/Library/Logs/com.greta.assistant/Greta.log`
- **Backend:** Terminal 699 output
- **App data:** `~/Library/Application Support/com.greta.assistant/`

---

## 5. Rollback Strategy

### Quick Rollback Methods

**Method 1: Git Revert (Safest)**

```bash
# Revert last commit
git revert HEAD
git push origin main

# Revert specific commit
git revert <commit-hash>
git push origin main
```

**Method 2: Git Reset (Use with Caution)**

```bash
# Reset to previous commit (destructive)
git reset --hard HEAD~1
git push origin main --force

# Reset to specific commit
git reset --hard <commit-hash>
git push origin main --force
```

**Method 3: Restore from Tag**

```bash
# List tags
git tag -l

# Restore to tagged version
git checkout v0.1.0
git checkout -b rollback-to-v0.1.0
```

### Rollback Workflow

```bash
# 1. Identify bad commit
git log --oneline -10

# 2. Revert
git revert <bad-commit-hash>

# 3. Update BUGS.md
# Mark fix as "Reverted - caused issue XYZ"

# 4. Push
git push origin main

# 5. Rebuild production
cd cpas4-ui
npm run tauri:build

# 6. Test
open src-tauri/target/release/bundle/macos/Greta.app
```

### Tagging Stable Releases

```bash
# Before risky changes
git tag -a v0.1.0-stable -m "Stable version"
git push origin v0.1.0-stable

# Rollback to stable
git checkout v0.1.0-stable
```

### Handling Database Changes

```bash
# Backup before migration
cp ~/Library/Application\ Support/com.greta.assistant/greta.db \
   ~/Library/Application\ Support/com.greta.assistant/greta.db.backup

# Restore from backup
cp ~/Library/Application\ Support/com.greta.assistant/greta.db.backup \
   ~/Library/Application\ Support/com.greta.assistant/greta.db
```

---

## 6. Priority Classification

### Bug Severity Levels

**Critical (Fix Immediately)**
- App crashes on launch
- Data loss or corruption
- Security vulnerabilities
- Core functionality completely broken

**Examples:** "App won't launch", "All data deleted"

**Action:** Drop everything, hotfix branch

---

**High (Fix Within 24-48 Hours)**
- Major feature not working
- Frequent crashes
- Performance severely degraded

**Examples:** "MCP servers won't connect", "Voice settings don't save"

**Action:** Prioritize over new features

---

**Medium (Fix Within 1 Week)**
- Minor feature not working
- Workaround available
- UI glitches
- Occasional crashes

**Examples:** "System tray icon doesn't update", "UI layout breaks"

**Action:** Schedule for next maintenance session

---

**Low (Fix When Convenient)**
- Cosmetic issues
- Nice-to-have improvements
- Rare edge cases

**Examples:** "Button color off", "Tooltip typo"

**Action:** Add to backlog

---

### Balancing Bug Fixes vs. Features

1. **Critical/High bugs:** Always fix before new features
2. **Medium bugs:** Fix if affecting daily usage
3. **Low bugs:** Batch fix during maintenance windows
4. Schedule regular "bug fix days" (e.g., every 2 weeks)

---

## 7. Testing Before Deployment

### Pre-Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Code Quality
- [ ] TypeScript compilation passes
- [ ] No errors or warnings
- [ ] No console errors

### Functionality
- [ ] Bug fix verified
- [ ] No regressions
- [ ] Edge cases tested

### Desktop Features
- [ ] System tray menu works
- [ ] Notifications appear
- [ ] Keyboard shortcuts work
- [ ] Auto-launch persists
- [ ] Window state persists

### Core Features
- [ ] Voice settings work
- [ ] MCP servers connect
- [ ] Hooks execute
- [ ] Agent builder functional
- [ ] Chat works

### Production Build
- [ ] Build completes
- [ ] .app launches
- [ ] .dmg installer works

### Data Persistence
- [ ] Settings persist
- [ ] Data not lost
- [ ] localStorage working

### Performance
- [ ] Launches quickly (< 3s)
- [ ] UI responsive
- [ ] No memory leaks
- [ ] CPU usage reasonable

### Logs
- [ ] No errors in logs
- [ ] Logs in correct location
```

### Testing Workflow

```bash
# 1. Clean build
cd /Users/macone/Desktop/Greta/cpas4-ui
rm -rf dist node_modules/.vite
npm run build

# 2. Test in dev mode
npm run tauri:dev

# 3. Build production
npm run tauri:build

# 4. Test production build
open src-tauri/target/release/bundle/macos/Greta.app

# 5. Test .dmg installer
open src-tauri/target/release/bundle/dmg/Greta_*.dmg

# 6. Check logs
tail -f ~/Library/Logs/com.greta.assistant/Greta.log

# 7. Monitor performance (Activity Monitor)
```

### Manual Testing (Current Approach)

For personal project, manual testing is sufficient:
- Use checklist for consistency
- Focus on critical paths
- Test bug-specific scenarios

**Future:** Add automated tests if bugs become frequent

---

## 8. Version Management

### Semantic Versioning

**Format:** `MAJOR.MINOR.PATCH` (e.g., `0.1.0`)

**When to Increment:**

- **MAJOR (X.0.0):** Breaking changes
  - Example: `0.x.x` → `1.0.0` (first stable)
  
- **MINOR (x.X.0):** New features
  - Example: `0.1.0` → `0.2.0` (new agent type)
  
- **PATCH (x.x.X):** Bug fixes
  - Example: `0.1.0` → `0.1.1` (fixed bug)

### Version Update Workflow

```bash
cd /Users/macone/Desktop/Greta/cpas4-ui

# Update package.json
npm version patch  # 0.1.0 → 0.1.1

# Update Cargo.toml
sed -i '' 's/version = "0.1.0"/version = "0.1.1"/' src-tauri/Cargo.toml

# Update tauri.conf.json
sed -i '' 's/"version": "0.1.0"/"version": "0.1.1"/' src-tauri/tauri.conf.json

# Commit
git add src-tauri/Cargo.toml src-tauri/tauri.conf.json
git commit --amend --no-edit

# Push with tags
git push origin main --tags
```

### Version Tracking

**Update CHANGELOG.md:**

```markdown
## [0.1.1] - 2025-10-15

### Fixed
- Voice settings persist (BUG-001)
- System tray icon (BUG-002)

### Changed
- Improved error messages
```

**Git Tags:**

```bash
# List versions
git tag -l

# View version
git show v0.1.1

# Checkout version
git checkout v0.1.0
```

---

## Quick Reference

### Common Commands

```bash
# Report bug
echo "### [BUG-XXX] Title" >> BUGS.md

# Fix bug
git checkout -b fix/bug-description
# Make changes
git commit -m "Fix: Description..."
git checkout main && git merge fix/bug-description
git push origin main

# Test
cd cpas4-ui
npm run build
npm run tauri:dev
npm run tauri:build

# Check logs
tail -f ~/Library/Logs/com.greta.assistant/Greta.log

# Rollback
git revert HEAD
git push origin main

# Version
npm version patch
# Update Cargo.toml and tauri.conf.json
git push origin main --tags
```

### File Locations

- **Bug log:** `/Users/macone/Desktop/Greta/BUGS.md`
- **Changelog:** `/Users/macone/Desktop/Greta/CHANGELOG.md`
- **Production logs:** `~/Library/Logs/com.greta.assistant/Greta.log`
- **App data:** `~/Library/Application Support/com.greta.assistant/`
- **Production build:** `cpas4-ui/src-tauri/target/release/bundle/macos/Greta.app`
- **Installer:** `cpas4-ui/src-tauri/target/release/bundle/dmg/Greta_*.dmg`

---

**Last Updated:** October 12, 2025  
**Next Review:** After first 10 bugs fixed
