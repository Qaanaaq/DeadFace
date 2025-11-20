# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# List all the extra data files we want to ship next to the exe
datas = [
    ("Dead_Marks/DeadFace.task", "."),
    ("Dead_Marks/sky_dark_theme.json", "."),
    ("Dead_Marks/multipliers.json", "."),
    ("Dead_Marks/neutral_pose.json", "."),
    ("Dead_Marks/deadface.png", "."),
    ("Dead_Marks/deadface.ico", "."),
    ("Dead_Marks/Commands.txt", "."),
]

a = Analysis(
    ["Dead_Marks/dual_app.py"],   # main app
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DeadFace",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="Dead_Marks/deadface.ico",  # exe icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DeadFace",
)
