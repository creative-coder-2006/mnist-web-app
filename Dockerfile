FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# Install dependencies, bypassing cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Hugging Face Spaces default port is 7860
EXPOSE 7860

# Run the app with gunicorn binding to Hugging Face's required port
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
