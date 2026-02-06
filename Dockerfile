# 1. Use a lightweight Python base image
FROM python:3.10

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Create a non-root user (Security requirement for Hugging Face)
# They generally run containers as user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 4. Copy requirements and install dependencies
# We copy this first to cache dependencies (makes future builds faster)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of your application code (including your models)
COPY --chown=user . .

# 6. Expose the port Hugging Face expects (7860)
# Note: This is just for documentation; the CMD command below actually opens it.
EXPOSE 7860

# 7. Run the application
# IMPORTANT: You must bind to 0.0.0.0 and port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]