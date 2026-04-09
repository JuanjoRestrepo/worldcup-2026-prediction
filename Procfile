web: gunicorn -w 4 -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker src.api.main:app
release: python load_data.py && python run_dbt.py test
