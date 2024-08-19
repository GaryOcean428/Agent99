# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files, including hidden ones, into the container at /app
COPY . /app

# Create a default .env file if it doesn't exist
RUN touch /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=chat99.py

# Print debug information, environment variables, and run the Flask app
CMD echo "Current directory:" && \
    pwd && \
    echo "Directory contents:" && \
    ls -la && \
    echo "Environment variables:" && \
    env | grep GROQ_API_KEY && \
    echo "Contents of .env file:" && \
    cat .env && \
    echo "Starting Flask app..." && \
    flask run --host=0.0.0.0