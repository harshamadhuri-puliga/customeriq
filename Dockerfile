FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train on the data in ./data (put the real Telco CSV there before building)
RUN python -m customeriq.train
EXPOSE 8000 8501
# API:        docker run -p 8000:8000 customeriq
# Dashboard:  docker run -p 8501:8501 customeriq streamlit run app.py --server.address 0.0.0.0
CMD ["uvicorn", "customeriq.api:app", "--host", "0.0.0.0", "--port", "8000"]
