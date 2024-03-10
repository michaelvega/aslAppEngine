FROM python:3.9

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Installing all python modules specified
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#Copy App Contents
ADD . /app
WORKDIR /app

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

#Start Flask Server
CMD ["python", "app.py"]
#Expose server port
EXPOSE 8080