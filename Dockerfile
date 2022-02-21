FROM python:3.7

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r ./requirements.txt

COPY ./app.py /app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]