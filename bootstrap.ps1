# Oil Blotter Analyzer — автоустановка Python и зависимостей
# Запуск: двойной щелчок по run.bat или Запуск.bat

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonVersion = "3.12.10"
$PythonMinor = "312"

function Write-Info([string]$Text) {
    Write-Host ""
    Write-Host "[Анализатор] $Text" -ForegroundColor Cyan
}

function Write-Download([string]$Text) {
    Write-Host "[Загрузка]   $Text" -ForegroundColor Yellow
}

function Write-Ok([string]$Text) {
    Write-Host "[Готово]     $Text" -ForegroundColor Green
}

function Test-PythonExe([string]$Exe) {
    if ([string]::IsNullOrWhiteSpace($Exe)) { return $false }
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    try {
        $ver = & $Exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $parts = $ver.Trim().Split(".")
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        return ($major -ge 3 -and $minor -ge 10)
    } catch {
        return $false
    }
}

function Find-SystemPython {
    $tryCommands = @(
        @{ Exe = "py"; Args = "-3" }
        @{ Exe = "python"; Args = $null }
        @{ Exe = "python3"; Args = $null }
    )

    foreach ($entry in $tryCommands) {
        $cmd = $entry.Exe
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { continue }
        try {
            if ($entry.Args) {
                $exe = & $cmd $entry.Args -c "import sys; print(sys.executable)" 2>$null
            } else {
                $exe = & $cmd -c "import sys; print(sys.executable)" 2>$null
            }
            if ($LASTEXITCODE -eq 0 -and $exe) {
                $exe = $exe.Trim()
                if (Test-PythonExe $exe) { return $exe }
            }
        } catch { }
    }

    $localPatterns = @(
        "$env:LOCALAPPDATA\Programs\Python\Python$PythonMinor\python.exe"
        "$env:LOCALAPPDATA\Programs\Python\Python$PythonMinor-32\python.exe"
        "$env:ProgramFiles\Python$PythonMinor\python.exe"
    )
    foreach ($path in $localPatterns) {
        if (Test-PythonExe $path) { return $path }
    }

    return $null
}

function Install-PythonWinget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        return $null
    }

    Write-Info "Python не найден."
    Write-Download "Устанавливаю Python $PythonVersion через winget."
    Write-Download "Пакет: Python.Python.$PythonMinor (Python Software Foundation, python.org)."
    Write-Host "           Может появиться запрос UAC — разрешите установку." -ForegroundColor DarkYellow

    $wingetArgs = @(
        "install", "--id", "Python.Python.$PythonMinor",
        "-e", "--accept-package-agreements", "--accept-source-agreements"
    )

    & winget @wingetArgs
    if ($LASTEXITCODE -ne 0) { return $null }

    Start-Sleep -Seconds 3
    return Find-SystemPython
}

function Install-PythonInstaller {
    $installerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $installerPath = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"

    Write-Info "Python не найден, winget недоступен или установка не удалась."
    Write-Download "Скачиваю официальный установщик Python $PythonVersion (~25 МБ)."
    Write-Download "Источник: $installerUrl"

    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

    Write-Info "Запускаю установку Python для текущего пользователя (без прав администратора, если возможно)."
    Write-Host "           Добавлю Python в PATH. Может появиться окно UAC." -ForegroundColor DarkYellow

    $installArgs = @(
        "/quiet"
        "InstallAllUsers=0"
        "PrependPath=1"
        "Include_test=0"
        "Include_launcher=1"
        "Include_pip=1"
    )

    $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Установщик Python завершился с кодом $($proc.ExitCode)."
    }

    Start-Sleep -Seconds 2

    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "Machine")

    return Find-SystemPython
}

function Ensure-Python {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-PythonExe $venvPython) {
        Write-Ok "Используется виртуальное окружение: $venvPython"
        return $venvPython
    }

    $systemPython = Find-SystemPython
    if (-not $systemPython) {
        $systemPython = Install-PythonWinget
    }
    if (-not $systemPython) {
        $systemPython = Install-PythonInstaller
    }
    if (-not $systemPython) {
        throw @"
Не удалось установить Python автоматически.
Установите Python 3.10+ вручную с https://www.python.org/downloads/
При установке отметьте «Add python.exe to PATH», затем снова запустите run.bat
"@
    }

    Write-Ok "Python найден: $systemPython"

    Write-Info "Создаю изолированное окружение .venv (один раз, ~1–2 мин)."
    & $systemPython -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось создать виртуальное окружение (.venv)."
    }

    if (-not (Test-PythonExe $venvPython)) {
        throw "Виртуальное окружение создано, но python.exe не найден: $venvPython"
    }

    Write-Ok "Виртуальное окружение готово."
    return $venvPython
}

function Ensure-Tkinter([string]$PythonExe) {
    Write-Info "Проверяю модуль tkinter (нужен для окна программы)..."
    & $PythonExe -c "import tkinter" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw @"
В установленном Python нет tkinter (графический интерфейс).
Переустановите полную версию Python с python.org (не embeddable/portable).
При установке выберите «Customize» и включите tcl/tk и tkinter.
"@
    }
    Write-Ok "tkinter доступен."
}

function Ensure-Packages([string]$PythonExe) {
    $requirements = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $requirements)) {
        throw "Файл requirements.txt не найден: $requirements"
    }

    $packageDescriptions = [ordered]@{
        "opencv-python" = "OpenCV — анализ изображения капли, зоны C/A/D/T, метрики"
        "numpy"         = "NumPy — математические расчёты (DS, MD, CI и др.)"
        "Pillow"        = "Pillow — загрузка снимков и отображение в окне программы"
    }

    Write-Info "Проверяю и устанавливаю необходимые пакеты Python..."
    foreach ($name in $packageDescriptions.Keys) {
        Write-Download "$name — $($packageDescriptions[$name])"
    }

    Write-Host ""
    Write-Host "           pip скачает пакеты с PyPI (pypi.org) — это может занять несколько минут." -ForegroundColor DarkYellow
    Write-Host "           opencv-python — самый крупный (~40–90 МБ в зависимости от версии)." -ForegroundColor DarkYellow
    Write-Host ""

    & $PythonExe -m pip install --upgrade pip --quiet
    & $PythonExe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка установки пакетов. Проверьте интернет и запустите run.bat снова."
    }

    Write-Ok "Все пакеты установлены."
}

function Start-Application([string]$PythonExe) {
    Write-Info "Запуск программы..."
    & $PythonExe (Join-Path $ProjectRoot "main.py")
    exit $LASTEXITCODE
}

try {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor White
    Write-Host "  Oil Blotter Analyzer" -ForegroundColor White
    Write-Host "  Подготовка окружения..." -ForegroundColor White
    Write-Host "========================================" -ForegroundColor White

    $python = Ensure-Python
    Ensure-Tkinter $python
    Ensure-Packages $python
    Start-Application $python
}
catch {
    Write-Host ""
    Write-Host "[Ошибка] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Нажмите Enter, чтобы закрыть окно"
    exit 1
}
