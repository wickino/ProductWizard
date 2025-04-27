@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Setup complete! Activate the venv with:
echo venv\Scripts\activate
echo.
pause
