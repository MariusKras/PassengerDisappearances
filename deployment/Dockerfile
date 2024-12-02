FROM python:3.11
WORKDIR /app
COPY . /app
COPY trained_model.pkl /app/trained_model.pkl
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "fastapi_model_server:app", "--host", "0.0.0.0", "--port", "8000"]


