# This script automates:
#   1) Launching the AI service in a new PowerShell window
#   2) Waiting briefly for the service to start
#   3) Activating the virtualenv in this window and running the simulation

# 1. Determine the script’s directory
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 2. Define paths relative to the project root
$VenvActivate = Join-Path $Root '.venv\Scripts\Activate.ps1'
$AiServiceDir  = Join-Path $Root 'ai_service'
$SimScript     = Join-Path $Root 'demo_simulation.py'

# 3. Start the AI service in a new PowerShell window (keeps it open for logs)
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-ExecutionPolicy','Bypass',
    '-NoProfile',
    '-Command',
    "& {
        . '$VenvActivate'               # activate virtualenv
        Set-Location '$AiServiceDir'    # change to ai_service directory
        uvicorn ai_service:app --host 127.0.0.1 --port 8001 --reload
    }"
)

# 4. Pause to allow the AI service to initialize
Start-Sleep -Seconds 5

# 5. In this window: activate virtualenv and run the simulation
. $VenvActivate
Set-Location $Root
python $SimScript
