# CAM Classificator - Backend Application

This is the backend application of the CAM classificator. It handles authentication,
database interaction, and here resides our classificator

## Start the server locally
- Setup environment variables
  - Copy the base file `cp .env-local-base .env` (On Windows, you can use `copy .env-local-base .env`)
  - Fill the .env file with variables needed by you
- Run in backend folder `python -m venv .venv`
- Activate the virtual environment:
  - **On Linux/macOS:** `source .venv/bin/activate`
  - **On Windows:** `.\.venv\Scripts\activate`
- Next run `pip install -r requirements.txt`
- After successfully installing, run `python run.py`