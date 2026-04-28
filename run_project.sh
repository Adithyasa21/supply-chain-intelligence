#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
echo "Run dashboard with: streamlit run app.py"
