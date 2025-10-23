# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, get_package_paths
import os
import sys
import site
import pathlib

# Add the project root to Python path
block_cipher = None

# Get package paths
import sys
from pathlib import Path

# Get absolute paths
SPEC_DIR = Path(SPECPATH)
VENV_DIR = SPEC_DIR / '.venv'
SITE_PACKAGES = VENV_DIR / 'Lib' / 'site-packages'
ARVIZ_DIR = SITE_PACKAGES / 'arviz'
STATIC_DIR = ARVIZ_DIR / 'static'

# Convert to strings and normalize paths
venv_site_packages = str(SITE_PACKAGES.resolve())
sys.path.insert(0, venv_site_packages)

# Explicitly collect arviz static files with proper directory structure
arviz_static_files = []
if STATIC_DIR.exists():
    print(f"\nFound arviz static directory at: {STATIC_DIR}")
    for root, dirs, files in os.walk(str(STATIC_DIR)):
        for file in files:
            file_path = Path(root) / file
            # Calculate relative path from static dir
            rel_path = file_path.relative_to(STATIC_DIR)
            # Keep the exact same directory structure
            dest_dir = os.path.join('arviz', 'static', str(rel_path.parent))
            print(f"Adding arviz file: {file_path} -> {dest_dir}")
            arviz_static_files.append((str(file_path), dest_dir))
            
# Also collect the entire arviz package
arviz_pkg = collect_data_files('arviz', include_py_files=True)
print("\nCollecting arviz package files:")
for src, dst in arviz_pkg:
    print(f"Adding arviz package file: {src} -> {dst}")
    
# Initialize data collection with both static files and package files
datas = arviz_static_files + arviz_pkg

# Debug information
print("\nCollected files for bundling:")
for src, dst in datas:
    print(f"  {src} -> {dst}")

# Function to collect all data files from a directory
def collect_all_from_dir(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        return []
    collected = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, os.path.relpath(root, src_dir))
            collected.append((src_file, dst_path))
    return collected

# Create hook-arviz.py for static files
hook_content = '''
from PyInstaller.utils.hooks import collect_data_files

# Collect all static files
datas = collect_data_files('arviz', includes=['**/static/**'])
'''

hook_file = SPEC_DIR / 'hook-arviz.py'
if not hook_file.exists():
    print(f"\nCreating arviz hook file at: {hook_file}")
    hook_file.write_text(hook_content)

# Add NLTK data files
nltk_data_dir = SITE_PACKAGES / 'nltk_data'
if nltk_data_dir.exists():
    print(f"\nFound NLTK data directory at: {nltk_data_dir}")
    nltk_files = []
    for file in nltk_data_dir.rglob('*.*'):
        if file.is_file():
            rel_path = file.relative_to(nltk_data_dir)
            dest_dir = os.path.join('nltk_data', str(rel_path.parent))
            print(f"Adding NLTK file: {file} -> {dest_dir}")
            nltk_files.append((str(file), dest_dir))
    datas.extend(nltk_files)

# Core package data files with explicit includes
packages_to_collect = [
    ('pymc', ['**/*.dat', '**/*.json', '**/*.txt']),  # PyMC data files
    ('nltk', ['**/*']),  # All NLTK files
    'vaderSentiment',
    'transformers',
    'torch',
    'numpy',
    'scipy',
    'pandas',
    'xarray',
    'tokenizers',
    'filelock',
    'regex',
    'sacremoses',
    'sentencepiece',
    'torchvision'
]

# Add Codette's unique components
codette_files = [
    'cognitive_processor.py',
    'codette_quantum_core.py',
    'models/cognitive_engine.py',
    'agireasoning.py',
    'ai_core_system.py',
    'ai_core_identityscan.py',
    'analyze_cocoons*.py',  # Include all cocoon analysis files
]

# Add Codette files to datas
for pattern in codette_files:
    for file in SPEC_DIR.glob(pattern):
        if file.is_file():
            print(f"\nAdding Codette component: {file}")
            rel_path = file.relative_to(SPEC_DIR)
            dest_dir = str(rel_path.parent)
            datas.append((str(file), dest_dir))

# Collect package data files with specific patterns where needed
for pkg in packages_to_collect:
    if isinstance(pkg, tuple):
        pkg_name, patterns = pkg
        for pattern in patterns:
            pkg_files = collect_data_files(pkg_name, includes=[pattern])
            print(f"\nCollecting {pkg_name} files with pattern {pattern}:")
            for src, dst in pkg_files:
                print(f"Adding {pkg_name} file: {src} -> {dst}")
            datas.extend(pkg_files)
    else:
        pkg_files = collect_data_files(pkg, include_py_files=True)
        print(f"\nCollecting {pkg} files:")
        for src, dst in pkg_files:
            print(f"Adding {pkg} file: {src} -> {dst}")
        datas.extend(pkg_files)
    
# Add fallback models
models_dir = os.path.join(SPECPATH, 'models', 'fallback')
if os.path.exists(models_dir):
    datas.extend([(models_dir, 'models/fallback')])

print("\nCollected Data Files:")
for src, dst in datas:
    print(f"  {src} -> {dst}")

a = Analysis(
    ['test_codette_exe.py'],
    pathex=[SPECPATH],  # Add spec file directory to path
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Core scientific packages
        'numpy', 'scipy', 'pandas',
        'nltk', 'nltk.data',
        'arviz', 'arviz.data', 'arviz.utils', 'arviz.data.base', 'arviz.utils',
        'xarray', 'netCDF4', 'cftime',  # Required by arviz
        'pkg_resources.py2_warn',  # For compatibility
        'pathlib',  # For path handling
        
        # PyMC and its dependencies
        'pymc', 'pymc.gp', 'pymc.gp.cov', 'pymc.pytensorf', 'pymc.util',
        
        # Codette's unique components
        'models.cognitive_engine',
        'cognitive_processor',
        'codette_quantum_core',
        'agireasoning',
        'ai_core_system',
        'ai_core_identityscan',
        'analyze_cocoons',
        'BroaderPerspectiveEngine'  # Codette's advanced reasoning engine
    ],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=['hook-runtime-arviz.py'],  # Add runtime hook for arviz
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Clean up any existing files
import shutil
if os.path.exists('dist'):
    print("Cleaning up existing dist directory...")
    shutil.rmtree('dist', ignore_errors=True)
if os.path.exists('build'):
    print("Cleaning up existing build directory...")
    shutil.rmtree('build', ignore_errors=True)

print("\nCreating executable...")
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='test_codette_exe',
    debug=True,  # Enable debug mode
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip debug symbols
    upx=False,  # Don't compress with UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version='1.0.0')
