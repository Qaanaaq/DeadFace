# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

# Project paths
# In PyInstaller spec execution, __file__ is not always defined the way we expect,
# but pyinstaller runs from the project root, so Path.cwd() is fine here.
project_root = Path.cwd()
dead_marks = project_root / "Dead_Marks"

# Data files to include next to the executable.
# These are accessed at runtime by simple filenames like "DeadFace.task" or
# "sky_dark_theme.json", so we copy them to "." in the bundle.
datas = [
    (str(dead_marks / "DeadFace.task"), "."),          # Mediapipe model
    (str(dead_marks / "sky_dark_theme.json"), "."),    # CustomTkinter theme (if present)
    # If you have other JSONs you want pre-bundled, uncomment/add here:
    # (str(dead_marks / "neutral_pose.json"), "."),
    # (str(dead_marks / "multipliers.json"), "."),
]

a = Analysis(
    ['Dead_Marks/dual_app.py'],   # main entry script
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
    console=False,      # set to True if you want a console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
