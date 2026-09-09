import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_FILE = PROJECT_ROOT / ".env"


def get_database_settings(require_env_file: bool = True) -> dict:
    """Load PostgreSQL connection settings from the project environment."""

    if require_env_file and not ENVIRONMENT_FILE.exists():
        raise FileNotFoundError(f"Environment file was not found:\n{ENVIRONMENT_FILE}")

    load_dotenv(ENVIRONMENT_FILE, override=False)

    settings = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "dbname": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "connect_timeout": 10,
    }

    missing = [
        key
        for key, value in settings.items()
        if key != "connect_timeout" and not value
    ]
    if missing:
        raise ValueError(f"Missing database settings: {missing}")

    return settings
