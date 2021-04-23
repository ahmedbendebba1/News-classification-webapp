FROM python:3.7
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

# for flask web server
EXPOSE 8080

# Default command to run the Gunicorn 
WORKDIR /app/app
CMD gunicorn --bind 0.0.0.0:8080 wsgi:app