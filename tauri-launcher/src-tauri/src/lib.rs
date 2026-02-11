use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, UNIX_EPOCH};
use std::fs;

use tauri::Manager;

#[derive(Default)]
struct BackendState {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
}

fn pick_port() -> std::io::Result<u16> {
    for port in 8787u16..=8887u16 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn health_ok(port: u16) -> bool {
    let mut stream = match TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_millis(500),
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };

    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

    let req = format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }

    let mut buf = [0u8; 512];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return false,
    };

    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

fn wait_for_health(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if health_ok(port) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    false
}

fn ensure_executable(path: &PathBuf) {
    #[cfg(unix)]
    {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            let mode = perms.mode();
            if (mode & 0o111) == 0 {
                perms.set_mode(mode | 0o111);
                let _ = fs::set_permissions(path, perms);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn clear_quarantine(path: &PathBuf) {
    // If the extracted backend retains the quarantine attribute, macOS can
    // perform slow Gatekeeper verification on every launch.
    // Best-effort: remove it recursively. Ignore failures (xattr may be absent).
    let _ = Command::new("/usr/bin/xattr")
        .arg("-dr")
        .arg("com.apple.quarantine")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(target_os = "macos"))]
fn clear_quarantine(_path: &PathBuf) {}

fn kill_stale_backends(backend_path: &PathBuf) {
    // When the app is force-quit, the backend process can be orphaned.
    // Multiple concurrent PyInstaller onefile instances can also behave poorly.
    // Best-effort: kill any existing processes that match the exact backend binary path.
    #[cfg(unix)]
    {
        let _ = Command::new("/usr/bin/pkill")
            .arg("-f")
            .arg(backend_path.to_string_lossy().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn backend_zip_signature(zip_path: &PathBuf) -> Result<String, String> {
    let meta = fs::metadata(zip_path)
        .map_err(|e| format!("Failed to stat backend zip {}: {e}", zip_path.display()))?;
    let len = meta.len();
    let modified_s = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(format!("{len}:{modified_s}"))
}

fn ensure_backend_extracted(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let zip_path = app
        .path()
        .resolve(
            "resources/backend/pbrain-web-backend.zip",
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| format!("Failed to resolve backend zip resource: {e}"))?;

    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;

    let backend_root = app_data_dir.join("backend");
    let extracted_dir = backend_root.join("pbrain-web-backend");

    let exe_name = if cfg!(windows) {
        "pbrain-web-backend.exe"
    } else {
        "pbrain-web-backend"
    };
    let exe_path = extracted_dir.join(exe_name);

    let sig_path = backend_root.join("pbrain-web-backend.signature");
    let sig = backend_zip_signature(&zip_path)?;
    let existing_sig = fs::read_to_string(&sig_path).ok();
    let needs_extract = !exe_path.exists() || existing_sig.as_deref().map(|s| s.trim()) != Some(sig.as_str());

    if !needs_extract {
        return Ok(exe_path);
    }

    let _ = fs::remove_dir_all(&extracted_dir);
    fs::create_dir_all(&backend_root)
        .map_err(|e| format!("Failed to create backend directory {}: {e}", backend_root.display()))?;

    #[cfg(windows)]
    {
        let status = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg(format!(
                "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                zip_path.display(),
                backend_root.display()
            ))
            .status()
            .map_err(|e| format!("Failed to run PowerShell Expand-Archive: {e}"))?;
        if !status.success() {
            return Err(format!(
                "Failed to extract backend zip (exit: {status}). Zip: {}",
                zip_path.display()
            ));
        }
    }

    #[cfg(not(windows))]
    {
        let unzip_candidates = ["/usr/bin/unzip", "unzip"];
        let mut extracted = false;
        for unzip in unzip_candidates {
            let status = Command::new(unzip)
                .arg("-o")
                .arg("-qq")
                .arg(&zip_path)
                .arg("-d")
                .arg(&backend_root)
                .status();
            if let Ok(status) = status {
                if status.success() {
                    extracted = true;
                    break;
                }
            }
        }
        if !extracted {
            return Err(format!(
                "Failed to extract backend zip using unzip. Zip: {}",
                zip_path.display()
            ));
        }
    }

    if !exe_path.exists() {
        return Err(format!(
            "Backend executable missing after extraction (expected {}).",
            exe_path.display()
        ));
    }

    let _ = fs::write(&sig_path, format!("{sig}\n"));

    Ok(exe_path)
}

fn start_backend(app: &tauri::AppHandle) -> Result<u16, String> {
    let backend_path = ensure_backend_extracted(app)?;

    ensure_executable(&backend_path);
    // Remove macOS quarantine if present to avoid slow verification.
    // Apply to the whole extracted directory (contains shared libs in onedir mode).
    if let Some(parent) = backend_path.parent() {
        clear_quarantine(&parent.to_path_buf());
    }

    kill_stale_backends(&backend_path);

    let port = pick_port().map_err(|e| format!("Failed to pick port: {e}"))?;

    let log_path = app
        .path()
        .app_data_dir()
        .ok()
        .and_then(|dir| {
            let _ = fs::create_dir_all(&dir);
            Some(dir.join("backend.log"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("pbrain-backend.log"));

    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    let mut cmd = Command::new(&backend_path);
    cmd.env("PBRAIN_HOST", "127.0.0.1")
        .env("PBRAIN_PORT", port.to_string())
        .env("PBRAIN_LOG_LEVEL", "info")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(
            log_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map_or(Stdio::null(), Stdio::from),
        )
        .stderr(
            log_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map_or(Stdio::null(), Stdio::from),
        );

    let child = cmd
        .spawn()
        .map_err(|e| {
            format!(
                "Failed to start backend: {e} (path: {})",
                backend_path.display()
            )
        })?;

    let state = app.state::<BackendState>();
    *state.child.lock().unwrap() = Some(child);
    *state.port.lock().unwrap() = Some(port);

    // PyInstaller onefile cold start can be slow (unpacking + heavy imports).
    // Also, the bootloader may transiently spawn intermediate processes.
    // Be patient and surface useful diagnostics if it fails.
    let timeout = Duration::from_secs(120);
    let start = Instant::now();
    loop {
        if health_ok(port) {
            break;
        }

        // If the backend process already exited, fail fast with exit code.
        if let Some(mut child) = state.child.lock().unwrap().take() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // Ensure we don't leave an orphan behind.
                    kill_stale_backends(&backend_path);
                    return Err(format!(
                        "Backend exited before becoming ready (status: {status}). Log: {}",
                        log_path.display()
                    ));
                }
                Ok(None) => {
                    *state.child.lock().unwrap() = Some(child);
                }
                Err(_) => {
                    *state.child.lock().unwrap() = Some(child);
                }
            }
        }

        if start.elapsed() >= timeout {
            // Best-effort: kill any lingering backend processes by path.
            kill_stale_backends(&backend_path);
            return Err(format!(
                "Backend did not become ready (health check timeout after {}s). Log: {}",
                timeout.as_secs(),
                log_path.display()
            ));
        }

        std::thread::sleep(Duration::from_millis(250));
    }

    Ok(port)
}

fn stop_backend(app: &tauri::AppHandle) {
    let state = app.state::<BackendState>();
    let mut guard = state.child.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
    }
}

fn eval_with_retry(window: &tauri::WebviewWindow, js: &str, timeout: Duration) {
    let start = Instant::now();
    loop {
        if window.eval(js).is_ok() {
            return;
        }
        if start.elapsed() >= timeout {
            let _ = window.eval(js);
            return;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

#[tauri::command]
fn pick_folder() -> Option<String> {
    rfd::FileDialog::new()
        .pick_folder()
        .map(|p| p.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(BackendState::default())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![pick_folder])
        .setup(|app| {
            // Create the main window immediately. We do this in code to avoid any
            // config/label mismatches preventing window creation.
            let main_window = if let Some(window) = app.get_webview_window("main") {
                window
            } else {
                tauri::WebviewWindowBuilder::new(
                    app,
                    "main",
                    tauri::WebviewUrl::App("index.html".into()),
                )
                .title("p-brain")
                .inner_size(1200.0, 800.0)
                .build()?
            };

            let _ = main_window.show();
            let _ = main_window.set_focus();

            // Start backend without blocking window creation.
            let handle = app.handle().clone();
            std::thread::spawn(move || match start_backend(&handle) {
                Ok(port) => {
                    let url = format!("http://127.0.0.1:{port}");
                    if let Some(window) = handle.get_webview_window("main") {
                        let js = format!(
                            "window.__PBRAIN_BACKEND_URL = {}; window.dispatchEvent(new Event('pbrain-backend-ready'));",
                            serde_json::to_string(&url).unwrap()
                        );
                        eval_with_retry(&window, &js, Duration::from_secs(10));
                    }
                }
                Err(err) => {
                    eprintln!("Backend startup failed: {err}");
                    if let Some(window) = handle.get_webview_window("main") {
                        let js = format!(
                            "window.__PBRAIN_BACKEND_ERROR = {}; window.dispatchEvent(new Event('pbrain-backend-error'));",
                            serde_json::to_string(&err).unwrap()
                        );
                        eval_with_retry(&window, &js, Duration::from_secs(10));
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            stop_backend(app_handle);
        }
    });
}
