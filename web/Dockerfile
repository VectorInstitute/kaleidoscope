# start by pulling the python image
FROM --platform=linux/amd64 python:3.8

# Create app directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Debug
RUN ls -alp
RUN pwd

# Install dependancies
RUN pip install -r web/requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["gateway_service.py" ]

