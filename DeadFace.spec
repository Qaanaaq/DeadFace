# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

# Use the current working directory as project root.
# In CI and locally you run `pyinstaller DeadFace.spec` from the repo root,
# so this will be correct.
project_root = Path.cwd()
dead_marks = project_root / "Dead_Marks"

# All runtime data files you want shipped with the app
datas = [
    (str(dead_marks / "DeadFace.task"), "."),
    (str(dead_marks / "sky_dark_theme.json"), "."),
    (str(dead_marks / "multipliers.json"), "."),
    (str(dead_marks / "neutral_pose.json"), "."),
    (str(dead_marks / "deadface.png"), "."),
    (str(dead_marks / "deadface.ico"), "."),   # optional, as file in dist
    (str(dead_marks / "commands.txt"), "."),
]

a = Analysis(
    ['Dead_Marks/dual_app.py'],   # <-- main app
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
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DeadFace',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(dead_marks / "deadface.ico"),  # this sets the exe icon
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
