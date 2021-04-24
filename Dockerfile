FROM python:3.7-slim-buster
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt -r requirements_text.txt
RUN python -m spacy download en_core_web_sm

# for flask web server
EXPOSE 8080

# Default command to run the Gunicorn 
WORKDIR /app/app
CMD gunicorn --bind 0.0.0.0:8080 wsgi:app