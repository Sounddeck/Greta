use tauri::AppHandle;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NotificationPayload {
    pub title: String,
    pub body: String,
    pub icon: Option<String>,
}

#[tauri::command]
pub async fn send_notification(
    app: AppHandle,
    payload: NotificationPayload,
) -> Result<(), String> {
    use tauri_plugin_notification::NotificationExt;
    
    let mut builder = app.notification()
        .builder()
        .title(&payload.title)
        .body(&payload.body);
    
    if let Some(icon) = payload.icon {
        builder = builder.icon(&icon);
    }
    
    builder
        .show()
        .map_err(|e| format!("Failed to send notification: {}", e))?;
    
    Ok(())
}

#[tauri::command]
pub async fn request_notification_permission(app: AppHandle) -> Result<bool, String> {
    use tauri_plugin_notification::NotificationExt;
    use tauri_plugin_notification::PermissionState;
    
    // On macOS, notifications are allowed by default
    // This function is here for cross-platform compatibility
    let permission = app.notification().permission_state()
        .map_err(|e| format!("Failed to check permission: {}", e))?;
    
    Ok(matches!(permission, PermissionState::Granted))
}

#[tauri::command]
pub async fn notify_voice_complete(app: AppHandle, text: String) -> Result<(), String> {
    send_notification(
        app,
        NotificationPayload {
            title: "Voice Recognition Complete".to_string(),
            body: format!("Recognized: {}", text),
            icon: None,
        },
    ).await
}

#[tauri::command]
pub async fn notify_mcp_status(
    app: AppHandle,
    server_name: String,
    status: String,
) -> Result<(), String> {
    send_notification(
        app,
        NotificationPayload {
            title: format!("MCP Server: {}", server_name),
            body: format!("Status: {}", status),
            icon: None,
        },
    ).await
}

#[tauri::command]
pub async fn notify_hook_executed(
    app: AppHandle,
    hook_name: String,
    success: bool,
) -> Result<(), String> {
    send_notification(
        app,
        NotificationPayload {
            title: format!("Hook: {}", hook_name),
            body: if success {
                "Executed successfully".to_string()
            } else {
                "Execution failed".to_string()
            },
            icon: None,
        },
    ).await
}
