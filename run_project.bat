python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
echo Run dashboard with: streamlit run app.py
