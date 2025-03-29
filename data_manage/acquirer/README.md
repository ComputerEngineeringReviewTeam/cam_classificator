# CAM Classificator project

TODO: Description

# Running the application

### When running locally as a standalone application

- Copy the .env-local-base as .env
- Uncomment two lines in app.config.config.py to enable reading .env files
- Make sure the mongodb is running


### When running with docker
- Make sure, the environment variables inside docker-compose.yml are correct
- Set up photo and data volume localisation in docker-compose.yml
- Run `docker compose up -d`
