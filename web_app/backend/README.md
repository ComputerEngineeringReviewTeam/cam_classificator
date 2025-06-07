# CAM Classificator - Backend Application

This is the backend application of the CAM classificator. It handles authentication,
database interaction, and here resides our classificator

## Start the server locally
- Setup environment variables
  - Copy the base file `cp .env-local-base .env`
  - Fill the .env file with variables needed by you
- Run in backend folder `python -m venv .venv`
- Then if on Linux run `source .venv/bin/activate`
- Next run `pip install -r requirements.txt`
- After successfull instalation run `python run.py`