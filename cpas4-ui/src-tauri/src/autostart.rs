use tauri::AppHandle;
use tauri_plugin_autostart::ManagerExt;

#[tauri::command]
pub async fn enable_autostart(app: AppHandle) -> Result<bool, String> {
    let autolaunch = app.autolaunch();
    
    autolaunch
        .enable()
        .map_err(|e| format!("Failed to enable autostart: {}", e))?;
    
    Ok(true)
}

#[tauri::command]
pub async fn disable_autostart(app: AppHandle) -> Result<bool, String> {
    let autolaunch = app.autolaunch();
    
    autolaunch
        .disable()
        .map_err(|e| format!("Failed to disable autostart: {}", e))?;
    
    Ok(true)
}

#[tauri::command]
pub async fn is_autostart_enabled(app: AppHandle) -> Result<bool, String> {
    let autolaunch = app.autolaunch();
    
    autolaunch
        .is_enabled()
        .map_err(|e| format!("Failed to check autostart status: {}", e))
}
