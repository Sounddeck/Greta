mod notifications;
mod autostart;
mod window_state;

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, AppHandle,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

// Tauri command to show window and navigate to voice control
#[tauri::command]
async fn activate_voice_control(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
        // Navigate to voice control page
        let _ = window.eval("window.location.hash = '#/voice'");
    }
    Ok(())
}

// Tauri command to toggle window visibility
#[tauri::command]
async fn toggle_window(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
        }
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        .invoke_handler(tauri::generate_handler![
            notifications::send_notification,
            notifications::request_notification_permission,
            notifications::notify_voice_complete,
            notifications::notify_mcp_status,
            notifications::notify_hook_executed,
            activate_voice_control,
            toggle_window,
            autostart::enable_autostart,
            autostart::disable_autostart,
            autostart::is_autostart_enabled,
            window_state::save_window_state,
            window_state::load_window_state,
        ])
        .setup(|app| {
            // Initialize logging only in debug mode
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Load and apply window state
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Ok(state) = window_state::load_window_state(app_handle.clone()).await {
                    let _ = window_state::apply_window_state(&app_handle, &state);
                }
            });

            // Register global keyboard shortcuts
            // Cmd+Shift+V (macOS) / Ctrl+Shift+V (Windows/Linux) - Activate voice control
            app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+V", move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    let app_clone = app.clone();
                    tauri::async_runtime::spawn(async move {
                        let _ = activate_voice_control(app_clone).await;
                    });
                }
            })?;

            // Cmd+Shift+G (macOS) / Ctrl+Shift+G (Windows/Linux) - Toggle window
            app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+G", move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    let app_clone = app.clone();
                    tauri::async_runtime::spawn(async move {
                        let _ = toggle_window(app_clone).await;
                    });
                }
            })?;

            // Cmd+Shift+Q (macOS) / Ctrl+Shift+Q (Windows/Linux) - Quit application
            app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+Q", move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    // Save window state before quitting
                    let app_clone = app.clone();
                    tauri::async_runtime::spawn(async move {
                        let _ = window_state::save_window_state(app_clone.clone()).await;
                        app_clone.exit(0);
                    });
                }
            })?;

            // Create system tray menu
            let show_hide = MenuItem::with_id(app, "show_hide", "Show/Hide Window", true, None::<&str>)?;
            let voice_control = MenuItem::with_id(app, "voice_control", "Voice Control", true, None::<&str>)?;
            let settings = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let menu = Menu::with_items(
                app,
                &[&show_hide, &voice_control, &settings, &quit],
            )?;

            // Build system tray
            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "show_hide" => {
                        if let Some(window) = app.get_webview_window("main") {
                            if window.is_visible().unwrap_or(false) {
                                let _ = window.hide();
                            } else {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    }
                    "voice_control" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                            // Navigate to voice control page
                            let _ = window.eval("window.location.hash = '#/voice'");
                        }
                    }
                    "settings" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                            // Navigate to settings page
                            let _ = window.eval("window.location.hash = '#/settings'");
                        }
                    }
                    "quit" => {
                        // Save window state before quitting
                        let app_clone = app.clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = window_state::save_window_state(app_clone.clone()).await;
                            app_clone.exit(0);
                        });
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            if window.is_visible().unwrap_or(false) {
                                let _ = window.hide();
                            } else {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    }
                })
                .build(app)?;

            // Register window close event to save state
            if let Some(window) = app.get_webview_window("main") {
                let app_handle = app.handle().clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { .. } = event {
                        let app_clone = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = window_state::save_window_state(app_clone).await;
                        });
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
