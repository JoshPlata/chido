FROM python:3.7
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 80
COPY . .
CMD ["uvicorn", "deploy2:app","--host","0.0.0.0","--port","80"]

