# PowerShell Script for One-Click Windows Environment Setup for RAG Pipeline V3
# Usage: Right-click this file and choose "Run with PowerShell"

$ErrorActionPreference = "Stop"

# Clear host and print a beautiful header
Clear-Host
Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host " 🛠️  RAG PIPELINE V3 - ONE-CLICK WINDOWS SETUP WORKER" -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta

# 1. Check if Python is installed
Write-Host "[1/5] Checking for Python installation..." -ForegroundColor Cyan
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python is not installed or not in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from python.org and check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Exit
}
$pyVersion = python --version
Write-Host "✓ Python found: $pyVersion" -ForegroundColor Green

# 2. Set up virtual environment
Write-Host "`n[2/5] Setting up virtual environment (venv)..." -ForegroundColor Cyan
if (!(Test-Path venv)) {
    Write-Host "   Creating new venv..." -ForegroundColor Gray
    python -m venv venv
    Write-Host "✓ Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment (venv/) already exists. Skipping creation." -ForegroundColor Green
}

# 3. Install Python Dependencies
Write-Host "`n[3/5] Installing core dependencies from requirements.txt..." -ForegroundColor Cyan
& .\venv\Scripts\pip install --upgrade pip
& .\venv\Scripts\pip install -r requirements.txt

Write-Host "   Installing Windows-specific 'python-magic-bin' for file type partition..." -ForegroundColor Cyan
& .\venv\Scripts\pip install python-magic-bin
Write-Host "✓ Python packages installed successfully." -ForegroundColor Green

# 4. Download and setup localized Poppler (Required for PDF rendering)
Write-Host "`n[4/5] Checking for Poppler dependency (required by pdf2image)..." -ForegroundColor Cyan
$popplerBaseDir = Join-Path (Get-Location) "Utilities\poppler"
if (!(Test-Path $popplerBaseDir)) {
    New-Item -ItemType Directory -Force -Path $popplerBaseDir | Out-Null
}

$popplerBin = Get-ChildItem -Path $popplerBaseDir -Filter "bin" -Recurse | Select-Object -First 1

if ($null -eq $popplerBin) {
    Write-Host "   Downloading Poppler for Windows (Pre-compiled binaries)..." -ForegroundColor Cyan
    $popplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v26.02.0-0/Release-26.02.0-0.zip"
    $zipPath = Join-Path $popplerBaseDir "poppler.zip"
    
    # Download zip file
    Invoke-WebRequest -Uri $popplerUrl -OutFile $zipPath
    
    Write-Host "   Extracting Poppler binaries locally..." -ForegroundColor Cyan
    Expand-Archive -Path $zipPath -DestinationPath $popplerBaseDir -Force
    Remove-Item $zipPath
    
    $popplerBin = Get-ChildItem -Path $popplerBaseDir -Filter "bin" -Recurse | Select-Object -First 1
}

if ($null -ne $popplerBin) {
    $binPath = $popplerBin.FullName
    Write-Host "✓ Local Poppler setup complete! Bin directory: $binPath" -ForegroundColor Green
    
    # Update local .env file
    $envFile = Join-Path (Get-Location) ".env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile
        # Remove any existing POPPLER_PATH lines
        $envContent = $envContent | Where-Object { $_ -notmatch "^POPPLER_PATH=" }
        # Escape backslashes for .env compatibility
        $escapedPath = $binPath.Replace("\", "\\")
        $envContent += "POPPLER_PATH=""$escapedPath"""
        Set-Content -Path $envFile -Value $envContent
        Write-Host "✓ Local .env updated with POPPLER_PATH." -ForegroundColor Green
    } else {
        $escapedPath = $binPath.Replace("\", "\\")
        New-Item -ItemType File -Path $envFile -Value "POPPLER_PATH=""$escapedPath""" | Out-Null
        Write-Host "✓ Created new .env with POPPLER_PATH." -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ Warning: Poppler bin directory could not be located." -ForegroundColor Yellow
}

# 5. Check/Install Tesseract OCR (Optional for scanned PDF support)
Write-Host "`n[5/5] Checking for Tesseract OCR..." -ForegroundColor Cyan
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Write-Host "✓ Tesseract OCR is already installed and in your system PATH." -ForegroundColor Green
} else {
    Write-Host "   Tesseract OCR not detected in PATH. Attempting automated install via winget..." -ForegroundColor Cyan
    try {
        & winget install --id UB.TesseractOCR --silent --accept-source-agreements --accept-package-agreements
        Write-Host "✓ Tesseract OCR installed successfully via winget!" -ForegroundColor Green
        Write-Host "   Please restart your terminal to apply the Tesseract PATH update." -ForegroundColor Yellow
    } catch {
        Write-Host "⚠️ Automated installation failed. If you need scanned PDF OCR support, please manually install Tesseract OCR:" -ForegroundColor Yellow
        Write-Host "   Download link: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    }
}

# Final Summary Page
Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " 🎉 SUCCESS: WINDOWS ENVIRONMENT SETUP IS COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host " To run the V3 test suite in Windows PowerShell, execute:" -ForegroundColor Cyan
Write-Host "   .\venv\Scripts\python v3_hierarchical_summary_rag\test\test_v3.py" -ForegroundColor Yellow
Write-Host "======================================================================`n" -ForegroundColor Green
