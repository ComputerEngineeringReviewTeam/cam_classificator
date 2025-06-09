FROM node:18-alpine AS builder

WORKDIR /app/frontend

COPY ./frontend/package.json ./
COPY ./frontend/package-lock.json ./

ENV REACT_APP_SERVER_ADDR=http://localhost:5000

RUN npm ci

COPY ./frontend/ ./

RUN npm run build



FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./backend .
COPY --from=builder /app/frontend/build /app/frontend/build

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "600", "run:app"]
