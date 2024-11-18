FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r /app/app/requirements.txt

EXPOSE ${SERVER_PORT}

ENTRYPOINT ["python"]
CMD ["/app/run.py"]
