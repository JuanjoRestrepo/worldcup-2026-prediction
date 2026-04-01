import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def getEngine():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)