web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 4
release: python load_data.py && python run_dbt.py test
