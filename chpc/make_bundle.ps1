# Builds the offline install bundle for the CHPC on your LAPTOP (Lengau has no internet).
# Run from the repo folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File chpc\make_bundle.ps1
# Produces .\chpc_bundle\ (~140 MB): a standalone Linux CPython 3.12 + Linux wheels for
# every package in requirements.txt. Upload it with:
#     scp -r chpc_bundle spatel1@lengau.chpc.ac.za:~/lustre/cos700/
$ErrorActionPreference = "Stop"
$bundle = Join-Path (Get-Location) "chpc_bundle"
New-Item -ItemType Directory -Force -Path (Join-Path $bundle "wheels") | Out-Null

# Standalone CPython 3.12 for Linux (python-build-standalone; needs glibc >= 2.17, Lengau has 2.17).
$pyUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
$pyOut = Join-Path $bundle "python312.tar.gz"
if (-not (Test-Path $pyOut)) {
    Write-Host ">> downloading standalone Python 3.12 for Linux"
    Invoke-WebRequest -Uri $pyUrl -OutFile $pyOut
}

# Linux (manylinux2014 = glibc 2.17) wheels for Python 3.12 -- pip can download these from Windows.
Write-Host ">> downloading Linux wheels for Python 3.12"
python -m pip download -r requirements.txt `
    --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 `
    --only-binary=:all: -d (Join-Path $bundle "wheels")

Write-Host ""
Write-Host "Done. Files:" (Get-ChildItem (Join-Path $bundle "wheels")).Count "wheels +" (Get-Item $pyOut).Length "bytes of Python"
Write-Host "Upload with:  scp -r chpc_bundle spatel1@lengau.chpc.ac.za:~/lustre/cos700/"
