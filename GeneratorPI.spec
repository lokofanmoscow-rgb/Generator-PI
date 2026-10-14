# -*- mode: python ; coding: utf-8 -*-
# Сборка: pyinstaller --noconfirm GeneratorPI.spec  ->  dist/GeneratorPI.exe

import os
import sys

# Библиотеки Visual C++ кладём в exe, чтобы программа запускалась и на ПК
# без установленного «Microsoft Visual C++ Redistributable».
VC_DLLS = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll',
           'msvcp140_2.dll', 'concrt140.dll', 'vcomp140.dll']
vc_binaries = []
for dll in VC_DLLS:
    for d in (sys.prefix, os.path.join(sys.prefix, 'Scripts'), os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')):
        path = os.path.join(d, dll)
        if os.path.exists(path):
            vc_binaries.append((path, '.'))
            break

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=vc_binaries,
    datas=[('generator_pi/models/digits_rec.onnx', 'generator_pi/models')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'pandas', 'IPython', 'pytest', 'setuptools', 'cv2',
              'PIL._avif', 'PIL.AvifImagePlugin'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GeneratorPI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon='assets/icon.ico',
)
