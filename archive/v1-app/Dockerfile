# Use official stable Python 3.11 image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install critical system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the updated requirements file into the container
COPY requirements.txt .

# Upgrade installation toolwheels and install requirements without caching dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]