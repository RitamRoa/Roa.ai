# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy backend dependencies and install them
# These paths are now relative to the root 'ROA' directory where the Dockerfile is
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask app
COPY backend/roa.py .

# Add execute permission to roa.py
RUN chmod +x roa.py

# Copy frontend index.html (it's at the root of the build context)
COPY index.html .

# Expose the port that the Flask app will run on
ENV PORT 8080
EXPOSE $PORT

# Run the Flask app when the container launches
# Assumes roa.py is copied to the root of /app
CMD ["python", "roa.py"]