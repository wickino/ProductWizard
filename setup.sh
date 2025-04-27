#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "✅ Setup complete! Run the app with:"
echo "streamlit run app/main.py"
