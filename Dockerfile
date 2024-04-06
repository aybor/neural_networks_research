FROM rocm/pytorch:latest
LABEL authors="aybor"

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]