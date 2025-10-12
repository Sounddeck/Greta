# Changelog - CPAS4/Greta

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Bug Fixing & Maintenance Plan documentation
- Bug tracking system (BUGS.md)
- Changelog (this file)

---

## [0.1.0] - 2025-10-12

### Added
- **Phase 4: Desktop App with Tauri**
  - System tray integration with 4 menu items
  - Native notifications (5 Tauri commands)
  - Global keyboard shortcuts (Cmd+Shift+V, G, Q)
  - Auto-launch on startup
  - Window state persistence
  - Production .dmg installer for macOS

- **Phase 3: High Priority Improvements**
  - Rate limiting on all backend endpoints
  - Zustand state management (voiceStore, mcpStore, hookStore)
  - Optimistic updates with rollback
  - localStorage persistence
  - 4 skeleton components for loading states

- **Phase 2: Critical Security Fixes**
  - Secure API key management
  - Environment variable protection

- **Phase 1: Frontend Integration**
  - React + TypeScript frontend
  - Voice settings management
  - MCP server management
  - Hook management
  - Agent builder with React Flow

### Changed
- **Phase 5 Tasks 5.1 & 5.2:**
  - Re-enabled TypeScript strict mode
  - Fixed all type errors (0 errors)
  - Removed `any` type assertions
  - Changed bundle identifier to `com.greta.assistant`

### Technical Details
- **Frontend:** React 18, TypeScript, Vite 7.1.9, Zustand, React Flow
- **Desktop:** Tauri 2.8.5, Rust
- **Backend:** Python, FastAPI
- **Bundle Size:** 683.05 kB (gzipped: 214.38 kB)

---

## Version History

- **v0.1.0** (2025-10-12) - Initial desktop app release
  - Commit: ce9f396
  - Repository: https://github.com/Sounddeck/Greta.git

---

**Maintained by:** Pieter Vossen (@Sounddeck)  
**Last Updated:** October 12, 2025
