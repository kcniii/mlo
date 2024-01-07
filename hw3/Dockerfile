FROM python:3.8

WORKDIR /home/app
RUN mkdir -p /home/app
# Install required dependencies
RUN pip install --no-cache-dir numpy matplotlib

# Copy your application files
COPY ./app /home/app

# Generate and save the requirements file
RUN pip freeze > requirements.txt

CMD ["python", "main.py"]
