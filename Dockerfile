# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=chat99.py

# Print debug information and run the Flask app
CMD echo "Current directory:" && \
    pwd && \
    echo "Directory contents:" && \
    ls -la && \
    echo "Starting Flask app..." && \
    flask run --host=0.0.0.0