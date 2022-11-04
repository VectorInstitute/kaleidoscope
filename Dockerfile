# start by pulling the python image
FROM --platform=linux/amd64 python:3.8

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

ENTRYPOINT [ "python" ]
CMD ["gateway_service.py" ]

