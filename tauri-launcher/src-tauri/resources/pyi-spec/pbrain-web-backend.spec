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
    [],
    exclude_binaries=True,
    name='pbrain-web-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pbrain-web-backend',
)
