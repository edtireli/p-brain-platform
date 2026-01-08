# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/edt/p-brain-web/backend/launcher_entry.py'],
    pathex=['/Users/edt/p-brain-web/backend'],
    binaries=[],
    datas=[],
    hiddenimports=['anyio._backends._asyncio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow', 'tensorflow_macos', 'keras', 'onnx', 'onnxruntime', 'cv2', 'opencv', 'opencv-python', 'PIL', 'Pillow', 'matplotlib', 'pandas', 'sklearn', 'scikit-learn'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pbrain-web-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
