# Use an official Python runtime as a parent image
FROM python:3.9.6

# Set environment variables
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV PORT 8080

# Run app.py when the container launches
CMD gunicorn --bind :$PORT --workers 4 --worker-class eventlet --threads 8 --timeout 120 app:app
