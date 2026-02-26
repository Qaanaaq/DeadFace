# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

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
    ['Dead_Marks/dual_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('Dead_Marks/DeadFace.task', '.'),
        ('Dead_Marks/multipliers.json', '.'),
        ('Dead_Marks/sky_dark_theme.json', '.'),
        ('Dead_Marks/neutral_pose.json', '.'),
        ('Dead_Marks/deadface.png', '.'),
    ],
    hiddenimports=[
        'mediapipe.python.solutions.face_mesh',
        'mediapipe.python.solutions.face_detection',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.framework.formats.landmark_pb2',
        'mediapipe.tasks.python.vision',
        'transforms3d',
        'numpy',
        'main',
        'main_stream'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# This EXE is ONLY used inside COLLECT — not written to dist/
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,    # <--- THIS PREVENTS the outer EXE file
    name='DeadFaceApp',
    console=False,
    icon='Dead_Marks/deadface.ico',
)

# This is the ONLY output you care about
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    name='DeadFace'
)
