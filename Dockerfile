FROM python:3.9.13

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

# Upgrade pip
RUN pip install --upgrade pip

# Install dependancies
RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000

# Run application
CMD [ "flask", "--app", "gateway_service", "run", "--host", "0.0.0.0", "--port", "5000"]