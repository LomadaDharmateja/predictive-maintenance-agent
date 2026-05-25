# 1. Update the base image to your explicit environment version
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# 2. FIX FOR SLIM IMAGE: Install essential build tools needed to compile heavy AI libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# 3. Upgrade pip *inside* the container to handle modern format packages natively
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]