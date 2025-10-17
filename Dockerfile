# Use a lightweight, stable Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Streamlit to use port 8080 (Cloud Run default)
ENV PORT=8080

# Expose port for local debugging (Cloud Run auto-detects 8080)
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
