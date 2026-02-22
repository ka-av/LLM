$Proj="C:\LLM\reward_misspec_grid"

New-Item -ItemType Directory -Force -Path `
  $Proj, "$Proj\llm_rewards", "$Proj\scripts", "$Proj\outputs" | Out-Null

Set-Location $Proj

if (-not (Test-Path ".\requirements.txt")) {
@"
google-genai>=0.7.0
pydantic>=2.6.0
python-dotenv>=1.0.1
pandas>=2.2.0
numpy>=1.26.0
tqdm>=4.66.0
"@ | Set-Content -Encoding UTF8 .\requirements.txt
}

if (-not (Test-Path ".\.venv")) {
  py -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -U pip
& .\.venv\Scripts\pip.exe install -r .\requirements.txt

if (-not (Test-Path ".\.env.example")) {
@"
# Copy to .env and put your real key here
GEMINI_API_KEY=YOUR_KEY_HERE
"@ | Set-Content -Encoding UTF8 .\.env.example
}

if (-not (Test-Path ".\.env")) {
  Copy-Item -Force .\.env.example .\.env
}

Write-Host "✅ Setup complete."
Write-Host "Edit .env to set GEMINI_API_KEY, then run scripts/run.ps1"