# Desktop Application Conversion Guide
## CPAS4/Greta - macOS Native App Implementation

**Created**: 2025-10-11  
**Target Platform**: macOS (Apple Silicon & Intel)  
**Recommended Framework**: Tauri 2.0

---

## 📊 Framework Comparison

### Option 1: Tauri 2.0 ⭐ **RECOMMENDED**

**Pros**:
- ✅ Smallest bundle size (~3-5 MB vs Electron's 50-100 MB)
- ✅ Better performance (Rust backend)
- ✅ Lower memory usage
- ✅ Better security (sandboxed by default)
- ✅ Native macOS integration
- ✅ Auto-updates built-in
- ✅ Code signing support
- ✅ Active development

**Cons**:
- ⚠️ Smaller ecosystem than Electron
- ⚠️ Requires Rust knowledge for advanced features

**Bundle Size**: ~3-5 MB  
**Memory Usage**: ~50-100 MB  
**Startup Time**: <1 second

---

### Option 2: Electron

**Pros**:
- ✅ Mature ecosystem
- ✅ Extensive documentation
- ✅ Large community
- ✅ Many examples

**Cons**:
- ❌ Large bundle size (50-100 MB)
- ❌ High memory usage (200-500 MB)
- ❌ Slower startup
- ❌ Security concerns

**Bundle Size**: ~50-100 MB  
**Memory Usage**: ~200-500 MB  
**Startup Time**: 2-5 seconds

---

### Option 3: Neutralino

**Pros**:
- ✅ Very small bundle size
- ✅ Low memory usage

**Cons**:
- ❌ Less mature
- ❌ Limited features
- ❌ Smaller community

---

## 🎯 Recommended Approach: Tauri 2.0

---

## 🚀 Step-by-Step Implementation

### Step 1: Install Prerequisites

**1.1 Install Rust**:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**1.2 Install Tauri CLI**:
```bash
cargo install tauri-cli
```

**1.3 Install Xcode Command Line Tools** (if not already installed):
```bash
xcode-select --install
```

**1.4 Verify installations**:
```bash
rustc --version
cargo --version
tauri --version
```

---

### Step 2: Initialize Tauri in Existing Project

**2.1 Navigate to frontend directory**:
```bash
cd /Users/macone/Desktop/Greta/cpas4-ui
```

**2.2 Install Tauri dependencies**:
```bash
npm install --save-dev @tauri-apps/cli @tauri-apps/api
```

**2.3 Initialize Tauri**:
```bash
npm run tauri init
```

**Answer prompts**:
- App name: `CPAS4 Greta`
- Window title: `CPAS4 - Personal AI System`
- Web assets location: `../dist`
- Dev server URL: `http://localhost:5173`
- Frontend dev command: `npm run dev`
- Frontend build command: `npm run build`

This creates:
- `src-tauri/` directory
- `src-tauri/tauri.conf.json` - Configuration
- `src-tauri/Cargo.toml` - Rust dependencies
- `src-tauri/src/main.rs` - Rust backend

---

### Step 3: Configure Tauri

**3.1 Edit `src-tauri/tauri.conf.json`**:

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist",
    "withGlobalTauri": true
  },
  "package": {
    "productName": "CPAS4 Greta",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "createDir": true,
        "scope": ["$APPDATA/*", "$RESOURCE/*"]
      },
      "http": {
        "all": false,
        "request": true,
        "scope": ["http://localhost:8000/*"]
      },
      "notification": {
        "all": true
      },
      "dialog": {
        "all": true
      },
      "clipboard": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "targets": ["dmg", "app"],
      "identifier": "com.greta.cpas4",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "resources": [],
      "externalBin": [],
      "copyright": "Copyright © 2025 CPAS4 Greta",
      "category": "DeveloperTool",
      "shortDescription": "Personal AI System",
      "longDescription": "CPAS4 Greta - A comprehensive personal AI assistant platform with voice, MCP, and hook integrations.",
      "macOS": {
        "frameworks": [],
        "minimumSystemVersion": "10.13",
        "exceptionDomain": "",
        "signingIdentity": null,
        "providerShortName": null,
        "entitlements": null
      }
    },
    "security": {
      "csp": "default-src 'self'; connect-src 'self' http://localhost:8000 ws://localhost:8000; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:;"
    },
    "windows": [
      {
        "fullscreen": false,
        "resizable": true,
        "title": "CPAS4 Greta",
        "width": 1400,
        "height": 900,
        "minWidth": 1200,
        "minHeight": 700,
        "center": true,
        "decorations": true,
        "transparent": false,
        "alwaysOnTop": false,
        "skipTaskbar": false
      }
    ],
    "systemTray": {
      "iconPath": "icons/icon.png",
      "iconAsTemplate": true,
      "menuOnLeftClick": false,
      "tooltip": "CPAS4 Greta"
    }
  }
}
```

---

### Step 4: Create App Icons

**4.1 Create icons directory**:
```bash
mkdir -p src-tauri/icons
```

**4.2 Generate icons**:

Use an icon generator or create manually:
- `icon.icns` - macOS icon (512x512)
- `icon.ico` - Windows icon
- `32x32.png`, `128x128.png`, `128x128@2x.png` - Various sizes

**Online tool**: https://www.appicon.co/

---

### Step 5: Implement Tauri Backend Features

**5.1 Edit `src-tauri/src/main.rs`**:

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::{CustomMenuItem, SystemTray, SystemTrayMenu, SystemTrayEvent, Manager};
use tauri::api::notification::Notification;

// Custom commands
#[tauri::command]
fn show_notification(title: String, body: String) {
    Notification::new("com.greta.cpas4")
        .title(&title)
        .body(&body)
        .show()
        .unwrap();
}

#[tauri::command]
async fn check_backend_status() -> Result<bool, String> {
    // Check if backend is running
    let client = reqwest::Client::new();
    match client.get("http://localhost:8000/health").send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

#[tauri::command]
fn open_backend_logs() {
    // Open backend logs in default text editor
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg("/Users/macone/Desktop/Greta/backend/logs")
        .spawn()
        .unwrap();
}

fn main() {
    // System tray menu
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let show = CustomMenuItem::new("show".to_string(), "Show");
    let backend = CustomMenuItem::new("backend".to_string(), "Backend Status");
    
    let tray_menu = SystemTrayMenu::new()
        .add_item(show)
        .add_item(backend)
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(quit);
    
    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick { .. } => {
                let window = app.get_window("main").unwrap();
                window.show().unwrap();
                window.set_focus().unwrap();
            }
            SystemTrayEvent::MenuItemClick { id, .. } => {
                match id.as_str() {
                    "quit" => {
                        std::process::exit(0);
                    }
                    "show" => {
                        let window = app.get_window("main").unwrap();
                        window.show().unwrap();
                        window.set_focus().unwrap();
                    }
                    "backend" => {
                        // Show backend status notification
                        Notification::new("com.greta.cpas4")
                            .title("Backend Status")
                            .body("Checking backend...")
                            .show()
                            .unwrap();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            show_notification,
            check_backend_status,
            open_backend_logs
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**5.2 Update `src-tauri/Cargo.toml`**:

```toml
[package]
name = "cpas4-greta"
version = "1.0.0"
description = "CPAS4 Greta - Personal AI System"
authors = ["Your Name"]
license = "MIT"
repository = ""
edition = "2021"

[build-dependencies]
tauri-build = { version = "2.0.0-beta", features = [] }

[dependencies]
tauri = { version = "2.0.0-beta", features = ["shell-open", "notification-all", "dialog-all", "fs-all", "http-all", "clipboard-all", "system-tray"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

---

### Step 6: Update Frontend for Tauri

**6.1 Create Tauri API wrapper** (`src/utils/tauri.ts`):

```typescript
import { invoke } from '@tauri-apps/api/tauri';
import { sendNotification } from '@tauri-apps/api/notification';
import { open } from '@tauri-apps/api/shell';

export const isTauri = () => {
  return '__TAURI__' in window;
};

export const showNotification = async (title: string, body: string) => {
  if (isTauri()) {
    await invoke('show_notification', { title, body });
  } else {
    // Fallback to browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, { body });
    }
  }
};

export const checkBackendStatus = async (): Promise<boolean> => {
  if (isTauri()) {
    return await invoke('check_backend_status');
  } else {
    // Fallback to fetch
    try {
      const response = await fetch('http://localhost:8000/health');
      return response.ok;
    } catch {
      return false;
    }
  }
};

export const openBackendLogs = async () => {
  if (isTauri()) {
    await invoke('open_backend_logs');
  } else {
    window.open('http://localhost:8000/logs', '_blank');
  }
};

export const openExternal = async (url: string) => {
  if (isTauri()) {
    await open(url);
  } else {
    window.open(url, '_blank');
  }
};
```

**6.2 Update components to use Tauri API**:

```typescript
// In VoiceControl.tsx
import { showNotification } from '../../utils/tauri';

// Replace toast with Tauri notification
await showNotification('Voice Input', 'Listening started');
```

---

### Step 7: Build and Package

**7.1 Development mode**:
```bash
npm run tauri dev
```

This will:
1. Start Vite dev server (localhost:5173)
2. Launch Tauri app window
3. Enable hot reload

**7.2 Production build**:
```bash
npm run tauri build
```

This creates:
- `src-tauri/target/release/bundle/dmg/CPAS4 Greta_1.0.0_aarch64.dmg` (Apple Silicon)
- `src-tauri/target/release/bundle/dmg/CPAS4 Greta_1.0.0_x64.dmg` (Intel)
- `src-tauri/target/release/bundle/macos/CPAS4 Greta.app`

**7.3 Universal binary** (both architectures):
```bash
npm run tauri build -- --target universal-apple-darwin
```

---

### Step 8: Code Signing (for distribution)

**8.1 Get Apple Developer certificate**:
1. Join Apple Developer Program ($99/year)
2. Create Developer ID Application certificate
3. Download and install in Keychain

**8.2 Update `tauri.conf.json`**:
```json
{
  "tauri": {
    "bundle": {
      "macOS": {
        "signingIdentity": "Developer ID Application: Your Name (TEAM_ID)"
      }
    }
  }
}
```

**8.3 Sign the app**:
```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" "CPAS4 Greta.app"
```

**8.4 Notarize** (required for macOS 10.15+):
```bash
xcrun notarytool submit "CPAS4 Greta.dmg" --apple-id "your@email.com" --password "app-specific-password" --team-id "TEAM_ID"
```

---

### Step 9: Auto-Updates

**9.1 Install Tauri updater**:
```bash
cargo add tauri-plugin-updater
```

**9.2 Configure updater** in `tauri.conf.json`:
```json
{
  "tauri": {
    "updater": {
      "active": true,
      "endpoints": [
        "https://your-server.com/updates/{{target}}/{{current_version}}"
      ],
      "dialog": true,
      "pubkey": "YOUR_PUBLIC_KEY"
    }
  }
}
```

**9.3 Implement update check**:
```typescript
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';

async function checkForUpdates() {
  try {
    const { shouldUpdate, manifest } = await checkUpdate();
    
    if (shouldUpdate) {
      const confirmed = confirm(
        `Update to ${manifest?.version} available. Install now?`
      );
      
      if (confirmed) {
        await installUpdate();
        await relaunch();
      }
    }
  } catch (error) {
    console.error('Update check failed:', error);
  }
}

// Check on app start
checkForUpdates();
```

---

### Step 10: Distribution

**10.1 Create DMG installer**:

The build process creates a DMG automatically. Customize it:

```bash
# Install create-dmg
brew install create-dmg

# Create custom DMG
create-dmg \
  --volname "CPAS4 Greta" \
  --volicon "icon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "CPAS4 Greta.app" 200 190 \
  --hide-extension "CPAS4 Greta.app" \
  --app-drop-link 600 185 \
  "CPAS4-Greta-Installer.dmg" \
  "src-tauri/target/release/bundle/macos/"
```

**10.2 Distribution options**:

1. **Direct Download**: Host DMG on your website
2. **Mac App Store**: Submit through App Store Connect
3. **Homebrew Cask**: Create formula for `brew install --cask cpas4-greta`
4. **GitHub Releases**: Attach DMG to releases

---

## 📁 Final Project Structure

```
cpas4-ui/
├── src/                          # React source code
│   ├── components/
│   ├── api/
│   ├── utils/
│   │   └── tauri.ts             # Tauri API wrapper
│   └── ...
├── src-tauri/                    # Tauri backend
│   ├── src/
│   │   └── main.rs              # Rust backend
│   ├── icons/                    # App icons
│   ├── Cargo.toml               # Rust dependencies
│   └── tauri.conf.json          # Tauri configuration
├── dist/                         # Build output
├── package.json
└── vite.config.ts
```

---

## 🧪 Testing

**Test checklist**:
- [ ] App launches successfully
- [ ] Window size and position correct
- [ ] System tray icon appears
- [ ] Menu items work
- [ ] Backend communication works
- [ ] Voice features work
- [ ] MCP features work
- [ ] Hook features work
- [ ] Notifications work
- [ ] File system access works
- [ ] Auto-update works
- [ ] App quits cleanly

---

## 📊 Comparison: Before vs After

| Metric | Web App | Desktop App |
|--------|---------|-------------|
| Bundle Size | N/A | ~3-5 MB |
| Memory Usage | ~100 MB | ~50-100 MB |
| Startup Time | Instant | <1 second |
| Offline Support | No | Yes |
| System Integration | Limited | Full |
| Security | Browser | Sandboxed |
| Distribution | URL | DMG/App Store |
| Auto-Updates | Manual | Automatic |

---

## 🚀 Next Steps

1. ✅ Complete backend integration (BACKEND_INTEGRATION_GUIDE.md)
2. ✅ Fix any navigation issues (NAVIGATION_VERIFICATION.md)
3. ✅ Address codebase issues (CODEBASE_ANALYSIS.md)
4. ⏭️ Implement Tauri desktop app (this guide)
5. 🎉 Deploy and distribute!

---

**End of Desktop App Guide**

