# ==============================================================================
# Portable OnCue Forced-Alignment Environment Builder (Windows + RTX 40/50 GPU)
# Includes: Micromamba, Python 3.11, PyTorch CUDA 12.8, FFmpeg, 
#           CTC-Forced-Aligner, num2words, Whisper-Base.en, and MMS-300M.
# ==============================================================================

$ErrorActionPreference = "Stop"
$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = Get-Location }

$BinDir    = Join-Path $RootDir "bin"
$EnvDir    = Join-Path $RootDir "env"
$CacheDir  = Join-Path $RootDir "models"
$TempDir   = Join-Path $RootDir "temp_setup"

Write-Host "[1/6] Creating portable directory structure in $RootDir..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BinDir, $EnvDir, $CacheDir, $TempDir | Out-Null

# 1. Download Portable Micromamba
$MambaExe = Join-Path $BinDir "micromamba.exe"
if (-not (Test-Path $MambaExe)) {
    Write-Host "[2/6] Downloading standalone Micromamba..." -ForegroundColor Cyan
    $MambaUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"
    $MambaTar = Join-Path $TempDir "micromamba.tar.bz2"
    Invoke-WebRequest -Uri $MambaUrl -OutFile $MambaTar
    tar -xvf $MambaTar -C $TempDir "Library/bin/micromamba.exe"
    Move-Item (Join-Path $TempDir "Library\bin\micromamba.exe") $MambaExe -Force
}

# 2. Download Portable Static FFmpeg
$FfmpegExe = Join-Path $BinDir "ffmpeg.exe"
if (-not (Test-Path $FfmpegExe)) {
    Write-Host "[3/6] Downloading portable FFmpeg static binary..." -ForegroundColor Cyan
    $FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $FfmpegZip = Join-Path $TempDir "ffmpeg.zip"
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $FfmpegZip
    Expand-Archive -Path $FfmpegZip -DestinationPath $TempDir -Force
    $ExtractedFfmpeg = Get-ChildItem -Path $TempDir -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
    Copy-Item $ExtractedFfmpeg.FullName -Destination $FfmpegExe -Force
}

# 3. Create Portable Python 3.11 + Git Environment
Write-Host "[4/6] Creating portable Python 3.11 environment..." -ForegroundColor Cyan
& $MambaExe create -p $EnvDir -c conda-forge python=3.11 git -y

$PythonExe = Join-Path $EnvDir "python.exe"
$PipExe    = Join-Path $EnvDir "Scripts\pip.exe"

# 4. Install PyTorch (CUDA 12.8 for RTX 40 & 50 series) + Forced Aligner + Normalizers
Write-Host "[5/6] Installing PyTorch (CUDA 12.8), CTC-Forced-Aligner, num2words, and Transformers..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade pip
& $PipExe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $PipExe install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git --extra-index-url https://download.pytorch.org/whl/cu128
& $PipExe install num2words nltk transformers accelerate

# Verify CUDA Architecture Support (sm_89 for 40-series, sm_120 for 50-series)
& $PythonExe -c "import torch; print('Verified Torch:', torch.__version__, '| Archs:', torch.cuda.get_arch_list())"

# 5. Pre-Cache Whisper-Base.en (~290 MB) & MMS-300M Aligner (~1.2 GB) for Offline Use
Write-Host "[6/6] Pre-caching Whisper-Base.en & MMS-300M models for offline execution..." -ForegroundColor Cyan
$env:HF_HOME = $CacheDir
$env:TORCH_HOME = $CacheDir

$PreCacheScript = @"
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from ctc_forced_aligner import load_alignment_model

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Downloading models to {os.environ.get('HF_HOME')} (Target device: {device})...")

print("1/2 Caching openai/whisper-base.en for Acoustic Preamble Anchoring...")
AutoProcessor.from_pretrained("openai/whisper-base.en")
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en")

print("2/2 Caching MahmoudAshraf/mms-300m-1130-forced-aligner...")
model, tokenizer = load_alignment_model(
    device,
    model_path="MahmoudAshraf/mms-300m-1130-forced-aligner",
    dtype=dtype
)
print("All models successfully cached for offline use!")
"@

$PreCachePy = Join-Path $TempDir "precache.py"
Set-Content -Path $PreCachePy -Value $PreCacheScript -Encoding UTF8
& $PythonExe $PreCachePy

Remove-Item -Recurse -Force $TempDir

# 6. Create Offline Launcher Batch File
$LauncherBat = Join-Path $RootDir "run_aligner.bat"
$LauncherContent = @"
@echo off
SETLOCAL
SET "ROOT_DIR=%~dp0"
SET "PATH=%ROOT_DIR%bin;%ROOT_DIR%env;%ROOT_DIR%env\Scripts;%ROOT_DIR%env\Library\bin;%PATH%"
SET "HF_HOME=%ROOT_DIR%models"
SET "TORCH_HOME=%ROOT_DIR%models"
SET "HF_HUB_OFFLINE=1"
SET "TRANSFORMERS_OFFLINE=1"

"%ROOT_DIR%env\python.exe" "%ROOT_DIR%align_deposition.py" %*
"@
Set-Content -Path $LauncherBat -Value $LauncherContent -Encoding ASCII

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " BUILD COMPLETE! Portable environment ready at: $RootDir" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
