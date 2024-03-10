FROM python:3.9

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Installing all python modules specified
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

#Copy App Contents
ADD . /app
WORKDIR /app

#Start Flask Server
CMD ["python", "app.py"]
#Expose server port
EXPOSE 8080