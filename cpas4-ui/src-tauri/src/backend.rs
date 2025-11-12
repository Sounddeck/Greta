// Backend Sidecar Management Module
// Handles starting, stopping, and monitoring the Python FastAPI backend

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::ShellExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendStatus {
    Starting,
    Running,
    Stopped,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct BackendState {
    pub status: BackendStatus,
    pub port: u16,
    pub pid: Option<u32>,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            status: BackendStatus::Stopped,
            port: 8000,
            pid: None,
        }
    }
}

// Global backend state
pub type BackendStateHandle = Arc<Mutex<BackendState>>;

/// Start the backend sidecar process
#[tauri::command]
pub async fn start_backend(app: AppHandle) -> Result<String, String> {
    log::info!("🚀 Starting backend sidecar...");

    let state = app.state::<BackendStateHandle>();

    // Check if already running
    {
        let current_state = state.lock()
            .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
        if matches!(current_state.status, BackendStatus::Running) {
            log::info!("✅ Backend already running");
            return Ok("Backend already running".to_string());
        }
    }

    // Update status to Starting
    {
        let mut current_state = state.lock()
            .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
        current_state.status = BackendStatus::Starting;
    }
    
    // Get the backend executable path
    // The backend is located in the MacOS directory alongside the main app executable
    use std::env;
    let current_exe = env::current_exe()
        .map_err(|e| format!("Failed to get current executable path: {}", e))?;

    let macos_dir = current_exe.parent()
        .ok_or_else(|| "Failed to get MacOS directory".to_string())?;

    let backend_dir = macos_dir.join("cpas5-backend");
    let backend_exe = backend_dir.join("cpas5-backend");

    if !backend_exe.exists() {
        return Err(format!("Backend executable not found at: {}", backend_exe.display()));
    }

    log::info!("📂 Backend directory: {}", backend_dir.display());
    log::info!("📦 Backend executable: {}", backend_exe.display());

    // Create command to run the backend
    let backend_command = app
        .shell()
        .command(backend_exe.to_string_lossy().to_string())
        .current_dir(backend_dir); // Set working directory to backend directory

    // Spawn the backend process
    let (mut rx, child) = backend_command
        .spawn()
        .map_err(|e| format!("Failed to spawn backend: {}", e))?;
    
    // Get PID
    let pid = child.pid();
    log::info!("📦 Backend process started with PID: {}", pid);

    // Update state with PID
    {
        let mut current_state = state.lock()
            .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
        current_state.pid = Some(pid);
    }
    
    // Spawn a task to monitor backend output
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                    log::info!("[Backend] {}", String::from_utf8_lossy(&line));
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                    log::error!("[Backend] {}", String::from_utf8_lossy(&line));
                }
                tauri_plugin_shell::process::CommandEvent::Error(err) => {
                    log::error!("[Backend] Error: {}", err);
                    {
                        let state = app_clone.state::<BackendStateHandle>();
                        if let Ok(mut current_state) = state.lock() {
                            current_state.status = BackendStatus::Error(err);
                        } else {
                            log::error!("Failed to acquire lock to update error status");
                        };
                    }
                }
                tauri_plugin_shell::process::CommandEvent::Terminated(payload) => {
                    log::warn!("[Backend] Process terminated with code: {:?}", payload.code);
                    {
                        let state = app_clone.state::<BackendStateHandle>();
                        if let Ok(mut current_state) = state.lock() {
                            current_state.status = BackendStatus::Stopped;
                            current_state.pid = None;
                        } else {
                            log::error!("Failed to acquire lock to update terminated status");
                        };
                    }
                }
                _ => {}
            }
        }
    });
    
    // Wait for backend to be ready (poll health endpoint)
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        let max_attempts = 30; // 30 seconds
        let mut attempts = 0;

        // Create HTTP client with timeout
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build() {
            Ok(c) => c,
            Err(e) => {
                log::error!("Failed to create HTTP client: {}", e);
                return;
            }
        };

        // Get backend port from state
        let port = {
            let state = app_clone.state::<BackendStateHandle>();
            let port_value = match state.lock() {
                Ok(current_state) => current_state.port,
                Err(e) => {
                    log::error!("Failed to acquire lock to get port: {}", e);
                    8000 // fallback to default
                }
            };
            port_value
        };

        let health_url = format!("http://localhost:{}/api/v1/workflows/health", port);

        while attempts < max_attempts {
            tokio::time::sleep(Duration::from_secs(1)).await;
            attempts += 1;

            // Try to connect to health endpoint
            match client.get(&health_url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        log::info!("✅ Backend health check passed");
                        let state = app_clone.state::<BackendStateHandle>();
                        if let Ok(mut current_state) = state.lock() {
                            current_state.status = BackendStatus::Running;

                            // Emit event to frontend
                            let _ = app_clone.emit("backend-status", BackendStatus::Running);
                        } else {
                            log::error!("Failed to acquire lock to update running status");
                        }
                        return;
                    }
                }
                Err(e) => {
                    log::debug!("⏳ Backend not ready yet (attempt {}/{}): {}", attempts, max_attempts, e);
                }
            }
        }

        // Timeout
        log::error!("❌ Backend failed to start within 30 seconds");
        {
            let state = app_clone.state::<BackendStateHandle>();
            if let Ok(mut current_state) = state.lock() {
                current_state.status = BackendStatus::Error("Startup timeout".to_string());

                // Emit error event to frontend
                let _ = app_clone.emit("backend-status", BackendStatus::Error("Startup timeout".to_string()));
            } else {
                log::error!("Failed to acquire lock to update timeout status");
            };
        }
    });
    
    Ok(format!("Backend starting with PID: {}", pid))
}

/// Stop the backend sidecar process
#[tauri::command]
pub async fn stop_backend(app: AppHandle) -> Result<String, String> {
    log::info!("🛑 Stopping backend sidecar...");

    let state = app.state::<BackendStateHandle>();
    let pid = {
        let current_state = state.lock()
            .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
        current_state.pid
    };

    if let Some(pid) = pid {
        // Kill the process
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let output = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .output()
                .map_err(|e| format!("Failed to kill backend: {}", e))?;

            if output.status.success() {
                log::info!("✅ Backend stopped (PID: {})", pid);
            } else {
                log::error!("❌ Failed to stop backend: {}", String::from_utf8_lossy(&output.stderr));
            }
        }

        // Update state
        let mut current_state = state.lock()
            .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
        current_state.status = BackendStatus::Stopped;
        current_state.pid = None;

        // Emit event to frontend
        let _ = app.emit("backend-status", BackendStatus::Stopped);

        Ok(format!("Backend stopped (PID: {})", pid))
    } else {
        Ok("Backend not running".to_string())
    }
}

/// Restart the backend sidecar process
#[tauri::command]
pub async fn restart_backend(app: AppHandle) -> Result<String, String> {
    log::info!("🔄 Restarting backend sidecar...");
    
    // Stop backend
    stop_backend(app.clone()).await?;
    
    // Wait a bit
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Start backend
    start_backend(app).await
}

/// Get backend status
#[tauri::command]
pub async fn get_backend_status(app: AppHandle) -> Result<BackendStatus, String> {
    let state = app.state::<BackendStateHandle>();
    let current_state = state.lock()
        .map_err(|e| format!("Failed to acquire backend state lock: {}", e))?;
    Ok(current_state.status.clone())
}

/// Check backend health
#[tauri::command]
pub async fn check_backend_health(app: AppHandle) -> Result<bool, String> {
    // Create HTTP client with timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    // Get backend port from state
    let port = {
        let state = app.state::<BackendStateHandle>();
        let port_value = match state.lock() {
            Ok(current_state) => current_state.port,
            Err(_) => 8000, // fallback to default
        };
        port_value
    };

    let health_url = format!("http://localhost:{}/api/v1/workflows/health", port);

    match client.get(&health_url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// Initialize backend state
pub fn init_backend_state() -> BackendStateHandle {
    Arc::new(Mutex::new(BackendState::default()))
}

/// Setup backend auto-start
pub async fn setup_backend_autostart(app: &AppHandle) -> Result<(), String> {
    log::info!("🔧 Setting up backend auto-start...");
    
    // Start backend automatically
    match start_backend(app.clone()).await {
        Ok(msg) => {
            log::info!("✅ {}", msg);
            Ok(())
        }
        Err(e) => {
            log::error!("❌ Failed to auto-start backend: {}", e);
            Err(e)
        }
    }
}

/// Cleanup backend on app exit
pub async fn cleanup_backend(app: &AppHandle) -> Result<(), String> {
    log::info!("🧹 Cleaning up backend...");
    stop_backend(app.clone()).await?;
    Ok(())
}

/// Force kill backend process (used for emergency cleanup)
pub fn force_kill_backend(pid: u32) {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        log::warn!("⚠️ Force killing backend process (PID: {})", pid);
        let _ = Command::new("kill")
            .arg("-9")  // SIGKILL
            .arg(pid.to_string())
            .output();
    }
}

/// Setup panic handler to ensure backend cleanup on crash
pub fn setup_panic_handler(app: AppHandle) {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        log::error!("🚨 Application panic detected: {:?}", panic_info);

        // Try to get backend PID and kill it
        let state = app.state::<BackendStateHandle>();
        if let Ok(current_state) = state.lock() {
            if let Some(pid) = current_state.pid {
                force_kill_backend(pid);
            }
        }

        // Call original panic hook
        original_hook(panic_info);
    }));
}

