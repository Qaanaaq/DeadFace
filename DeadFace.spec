# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

# Use current working directory as project root (works fine for pyinstaller DeadFace.spec)
project_root = Path.cwd()
dead_marks = project_root / "Dead_Marks"

# Data files to include next to the executable.
datas = [
    # Mediapipe model
    (str(dead_marks / "DeadFace.task"), "."),
    # CustomTkinter theme (if you have it in Dead_Marks)
    (str(dead_marks / "sky_dark_theme.json"), "."),
    # If you later want to bundle these, uncomment:
    # (str(dead_marks / "neutral_pose.json"), "."),
    # (str(dead_marks / "multipliers.json"), "."),
]

a = Analysis(
    ['Dead_Marks/dual_app.py'],   # <-- main entry script
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
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
    [],
    exclude_binaries=True,
    name='DeadFace',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # set True if you want a console window for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(dead_marks / "deadface.ico"),   # <-- ICON ADDED HERE
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeadFace',
)
