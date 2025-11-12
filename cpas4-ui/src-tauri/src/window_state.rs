use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WindowState {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
    pub visible: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 1200,
            height: 800,
            maximized: false,
            visible: true,
        }
    }
}

fn get_state_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    let app_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;
    
    fs::create_dir_all(&app_dir)
        .map_err(|e| format!("Failed to create app data dir: {}", e))?;
    
    Ok(app_dir.join("window_state.json"))
}

#[tauri::command]
pub async fn save_window_state(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        let position = window
            .outer_position()
            .map_err(|e| format!("Failed to get window position: {}", e))?;
        
        let size = window
            .outer_size()
            .map_err(|e| format!("Failed to get window size: {}", e))?;
        
        let maximized = window
            .is_maximized()
            .map_err(|e| format!("Failed to check if maximized: {}", e))?;
        
        let visible = window
            .is_visible()
            .map_err(|e| format!("Failed to check visibility: {}", e))?;
        
        let state = WindowState {
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
            maximized,
            visible,
        };
        
        let state_file = get_state_file_path(&app)?;
        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize window state: {}", e))?;
        
        fs::write(state_file, json)
            .map_err(|e| format!("Failed to write window state: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
pub async fn load_window_state(app: AppHandle) -> Result<WindowState, String> {
    let state_file = get_state_file_path(&app)?;
    
    if !state_file.exists() {
        return Ok(WindowState::default());
    }
    
    let json = fs::read_to_string(state_file)
        .map_err(|e| format!("Failed to read window state: {}", e))?;
    
    let state: WindowState = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse window state: {}", e))?;
    
    Ok(state)
}

pub fn apply_window_state(app: &AppHandle, state: &WindowState) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        // Check if position is valid (not off-screen)
        // For simplicity, we'll apply the position as-is
        // In production, you'd want to check screen bounds
        
        let position = PhysicalPosition::new(state.x, state.y);
        let size = PhysicalSize::new(state.width, state.height);
        
        window
            .set_position(position)
            .map_err(|e| format!("Failed to set window position: {}", e))?;
        
        window
            .set_size(size)
            .map_err(|e| format!("Failed to set window size: {}", e))?;
        
        if state.maximized {
            window
                .maximize()
                .map_err(|e| format!("Failed to maximize window: {}", e))?;
        }
        
        if !state.visible {
            window
                .hide()
                .map_err(|e| format!("Failed to hide window: {}", e))?;
        }
    }
    
    Ok(())
}
