# CPAS4/Greta Implementation Roadmap
**Updated:** October 11, 2025 at 3:15 PM

---

## 🎯 Current Status

✅ **Phase 1:** Frontend Integration - COMPLETE  
✅ **Phase 2:** Critical Security Fixes - COMPLETE  
🔄 **Phase 3:** High Priority Improvements - 33% COMPLETE (Task 3.1 done)  
⏸️ **Phase 4:** Desktop App with Tauri - PENDING  
⏸️ **Phase 5:** Production Deployment - PENDING

---

## 📋 Detailed Implementation Plan

### Phase 3: High Priority Improvements (Remaining: 2.5-4.5 hours)

#### ✅ Task 3.1: Rate Limiting (COMPLETE - 1.5 hours)
- [x] Install slowapi package
- [x] Create rate_limit.py middleware
- [x] Configure rate limits per endpoint type
- [x] Update main.py with rate limiter integration
- [x] Update auth_router.py with rate limiting
- [x] Update voice_router.py with rate limiting
- [x] Test rate limiting functionality

**Files Created:** 1 (95 lines)  
**Files Modified:** 3  
**Dependencies:** slowapi, limits, deprecated

---

#### 🔄 Task 3.2: Zustand State Management (2-3 hours)

**Step 1: Create Zustand Stores** (1 hour)
- [ ] Create `Greta/src/stores/voiceStore.ts`
  - Voice profiles state
  - Languages state
  - Voices state
  - Actions: fetchProfiles, fetchLanguages, fetchVoices, createProfile
  - Persist to localStorage

- [ ] Create `Greta/src/stores/mcpStore.ts`
  - MCP servers state
  - MCP tools state
  - Actions: fetchServers, fetchTools, createServer, deleteServer, testConnection
  - Persist to localStorage

- [ ] Create `Greta/src/stores/hookStore.ts`
  - Hooks state
  - Plugins state
  - Event types state
  - Actions: fetchHooks, fetchPlugins, fetchEvents, registerHook, deleteHook
  - Persist to localStorage

**Step 2: Update Components** (1-2 hours)
- [ ] Update `VoiceControl.tsx`
  - Replace useState with useVoiceStore
  - Implement optimistic updates
  - Remove local API calls (use store actions)

- [ ] Update `VoiceSettings.tsx`
  - Replace useState with useVoiceStore
  - Implement optimistic updates
  - Remove local API calls (use store actions)

- [ ] Update `MCPManager.tsx`
  - Replace useState with useMCPStore
  - Implement optimistic updates
  - Remove local API calls (use store actions)

- [ ] Update `HookManager.tsx`
  - Replace useState with useHookStore
  - Implement optimistic updates
  - Remove local API calls (use store actions)

**Expected Files:**
- Created: 3 store files (~300 lines each)
- Modified: 4 component files

---

#### ⏸️ Task 3.3: Loading States & Skeletons (1 hour)

**Step 1: Create Skeleton Components** (30 minutes)
- [ ] Create `Greta/src/components/skeletons/VoiceProfileSkeleton.tsx`
- [ ] Create `Greta/src/components/skeletons/MCPServerSkeleton.tsx`
- [ ] Create `Greta/src/components/skeletons/HookSkeleton.tsx`
- [ ] Create `Greta/src/components/skeletons/TableSkeleton.tsx`

**Step 2: Add Loading Indicators** (30 minutes)
- [ ] Add skeleton loaders to VoiceSettings during initial load
- [ ] Add skeleton loaders to MCPManager during initial load
- [ ] Add skeleton loaders to HookManager during initial load
- [ ] Add progress bars for long-running operations (STT, TTS)
- [ ] Add spinner for button actions (save, delete, test)

**Expected Files:**
- Created: 4 skeleton components (~50 lines each)
- Modified: 3 main components

---

### Phase 4: Desktop App with Tauri (6-8 hours)

#### Task 4.1: Tauri Setup (2 hours)

**Step 1: Install Tauri CLI** (30 minutes)
```bash
cd /Users/macone/Desktop/Greta/Greta
npm install --save-dev @tauri-apps/cli
npm install @tauri-apps/api
```

**Step 2: Initialize Tauri** (30 minutes)
```bash
npx tauri init
```
- Configure app name: "GRETA PAI"
- Configure window title: "GRETA Personal AI Assistant"
- Configure dev URL: http://localhost:5174
- Configure dist dir: dist

**Step 3: Configure tauri.conf.json** (1 hour)
- [ ] Set app identifier: `com.greta.pai`
- [ ] Configure window properties (size, resizable, decorations)
- [ ] Set up app icon
- [ ] Configure permissions (filesystem, network, notifications)
- [ ] Set up build configuration

**Expected Files:**
- Created: `src-tauri/` directory with Rust project
- Created: `tauri.conf.json`
- Modified: `package.json` (add tauri scripts)

---

#### Task 4.2: Native Features (2-3 hours)

**Step 1: System Tray** (1 hour)
- [ ] Create system tray icon
- [ ] Add tray menu (Show, Hide, Quit)
- [ ] Implement tray click handlers
- [ ] Add minimize to tray functionality

**Step 2: Native Notifications** (30 minutes)
- [ ] Replace browser notifications with Tauri notifications
- [ ] Add notification permissions
- [ ] Test notification display

**Step 3: Auto-Start** (30 minutes)
- [ ] Add auto-start configuration
- [ ] Create settings toggle for auto-start
- [ ] Test auto-start on login

**Step 4: Window Management** (1 hour)
- [ ] Implement window state persistence (size, position)
- [ ] Add keyboard shortcuts (Cmd+Q, Cmd+W, Cmd+M)
- [ ] Implement always-on-top option
- [ ] Add fullscreen support

**Expected Files:**
- Modified: `src-tauri/src/main.rs` (add native features)
- Created: `src/utils/tauri.ts` (Tauri API helpers)
- Modified: Components to use Tauri APIs

---

#### Task 4.3: Build & Package (2-3 hours)

**Step 1: Development Build** (30 minutes)
```bash
npm run tauri dev
```
- [ ] Test all features in development mode
- [ ] Fix any Tauri-specific issues
- [ ] Verify backend connectivity

**Step 2: Production Build** (1 hour)
```bash
npm run tauri build
```
- [ ] Configure build settings
- [ ] Build for macOS (Apple Silicon)
- [ ] Test production build
- [ ] Verify app bundle structure

**Step 3: Create Installer** (1-1.5 hours)
- [ ] Configure DMG settings
- [ ] Add custom background image
- [ ] Set up drag-to-Applications
- [ ] Test installation process
- [ ] Verify app launches after installation

**Expected Output:**
- `src-tauri/target/release/bundle/dmg/GRETA PAI_*.dmg`
- `src-tauri/target/release/bundle/macos/GRETA PAI.app`

---

### Phase 5: Production Deployment (15+ hours)

#### Task 5.1: Code Signing (4-6 hours)

**Prerequisites:**
- Apple Developer Account ($99/year)
- Developer ID Application certificate

**Step 1: Obtain Certificate** (2-3 hours)
- [ ] Enroll in Apple Developer Program
- [ ] Create Developer ID Application certificate
- [ ] Download and install certificate in Keychain
- [ ] Export certificate for CI/CD (if needed)

**Step 2: Configure Code Signing** (1 hour)
- [ ] Update `tauri.conf.json` with signing identity
- [ ] Configure entitlements
- [ ] Set up provisioning profile (if needed)

**Step 3: Sign Application** (1-2 hours)
```bash
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name" \
  "GRETA PAI.app"
```
- [ ] Sign app bundle
- [ ] Verify signature
- [ ] Test signed app

---

#### Task 5.2: Notarization (3-4 hours)

**Step 1: Prepare for Notarization** (1 hour)
- [ ] Create app-specific password for Apple ID
- [ ] Configure notarization credentials
- [ ] Create ZIP archive of app bundle

**Step 2: Submit for Notarization** (1-2 hours)
```bash
xcrun notarytool submit "GRETA PAI.zip" \
  --apple-id "your@email.com" \
  --password "app-specific-password" \
  --team-id "TEAM_ID" \
  --wait
```
- [ ] Submit app to Apple
- [ ] Wait for notarization (can take 15 minutes to 2 hours)
- [ ] Check notarization status

**Step 3: Staple Notarization** (30 minutes)
```bash
xcrun stapler staple "GRETA PAI.app"
```
- [ ] Staple notarization ticket to app
- [ ] Verify stapling
- [ ] Test notarized app

**Step 4: Verify Notarization** (30 minutes)
```bash
spctl --assess --verbose=4 --type execute "GRETA PAI.app"
```
- [ ] Verify Gatekeeper acceptance
- [ ] Test on fresh macOS system
- [ ] Confirm no security warnings

---

#### Task 5.3: DMG Installer (2-3 hours)

**Step 1: Create DMG** (1 hour)
- [ ] Install `create-dmg` tool
- [ ] Configure DMG settings (size, background, icon positions)
- [ ] Create custom background image
- [ ] Set up Applications symlink

**Step 2: Build DMG** (1 hour)
```bash
create-dmg \
  --volname "GRETA PAI" \
  --background "dmg-background.png" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "GRETA PAI.app" 200 190 \
  --hide-extension "GRETA PAI.app" \
  --app-drop-link 600 185 \
  "GRETA-PAI-Installer.dmg" \
  "GRETA PAI.app"
```
- [ ] Build DMG installer
- [ ] Test installation process
- [ ] Verify drag-to-Applications works

**Step 3: Sign and Notarize DMG** (1 hour)
- [ ] Sign DMG file
- [ ] Submit DMG for notarization
- [ ] Staple notarization to DMG
- [ ] Final testing

---

#### Task 5.4: Production Server Deployment (6-8 hours)

**Step 1: Server Setup** (2-3 hours)
- [ ] Choose hosting provider (AWS, DigitalOcean, etc.)
- [ ] Provision server (Ubuntu 22.04 LTS recommended)
- [ ] Configure firewall (ports 80, 443, 8000)
- [ ] Set up SSH access
- [ ] Install dependencies (Python 3.13, MongoDB, nginx)

**Step 2: Application Deployment** (2-3 hours)
- [ ] Clone repository to server
- [ ] Set up Python virtual environment
- [ ] Install Python dependencies
- [ ] Configure environment variables
- [ ] Set up MongoDB database
- [ ] Test backend startup

**Step 3: Web Server Configuration** (1-2 hours)
- [ ] Install and configure nginx
- [ ] Set up reverse proxy for backend
- [ ] Configure SSL/TLS with Let's Encrypt
- [ ] Set up automatic certificate renewal
- [ ] Configure CORS for production domain

**Step 4: Process Management** (1 hour)
- [ ] Set up systemd service for backend
- [ ] Configure auto-restart on failure
- [ ] Set up log rotation
- [ ] Test service start/stop/restart

**Step 5: Monitoring & Logging** (1 hour)
- [ ] Set up application logging
- [ ] Configure log aggregation
- [ ] Set up monitoring (CPU, memory, disk)
- [ ] Configure alerts for errors
- [ ] Set up uptime monitoring

---

## 📊 Time Estimates Summary

| Phase | Task | Estimated Time | Status |
|-------|------|----------------|--------|
| 3 | Rate Limiting | 1-2 hours | ✅ COMPLETE (1.5h) |
| 3 | Zustand Stores | 2-3 hours | 🔄 IN PROGRESS |
| 3 | Loading States | 1 hour | ⏸️ PENDING |
| **3 Total** | | **4-6 hours** | **33% Complete** |
| 4 | Tauri Setup | 2 hours | ⏸️ PENDING |
| 4 | Native Features | 2-3 hours | ⏸️ PENDING |
| 4 | Build & Package | 2-3 hours | ⏸️ PENDING |
| **4 Total** | | **6-8 hours** | **0% Complete** |
| 5 | Code Signing | 4-6 hours | ⏸️ PENDING |
| 5 | Notarization | 3-4 hours | ⏸️ PENDING |
| 5 | DMG Installer | 2-3 hours | ⏸️ PENDING |
| 5 | Server Deployment | 6-8 hours | ⏸️ PENDING |
| **5 Total** | | **15-21 hours** | **0% Complete** |
| **GRAND TOTAL** | | **25-35 hours** | **~6% Complete** |

---

## 🚀 Next Steps

### Immediate (Next Session):
1. Complete Task 3.2: Zustand State Management (2-3 hours)
2. Complete Task 3.3: Loading States & Skeletons (1 hour)
3. Commit Phase 3 completion

### Short Term (This Week):
1. Start Phase 4: Tauri Setup (2 hours)
2. Implement Native Features (2-3 hours)
3. Create Development Build (30 minutes)

### Medium Term (Next Week):
1. Complete Phase 4: Build & Package (2-3 hours)
2. Start Phase 5: Code Signing (4-6 hours)
3. Notarization (3-4 hours)

### Long Term (Following Weeks):
1. DMG Installer Creation (2-3 hours)
2. Production Server Deployment (6-8 hours)
3. Final Testing & Launch

---

## 📝 Notes

- **Phase 3** can be completed in one 4-6 hour session
- **Phase 4** requires macOS for Tauri development
- **Phase 5** requires Apple Developer Account ($99/year)
- **Production deployment** can be done in parallel with desktop app work
- Consider breaking Phase 5 into multiple sessions due to waiting times (notarization)

---

**Last Updated:** October 11, 2025 at 3:15 PM  
**Next Update:** After completing Phase 3

