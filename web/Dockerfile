# start by pulling the python image
FROM --platform=linux/amd64 python:3.8

# Create app directory
WORKDIR /app

# Install apt packages
RUN apt update && apt install -y ldap-utils

# Copy gateway/web dependencies
COPY requirements.txt ./

# Upgrade pip
RUN pip install --upgrade pip

# Install dependancies
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["gateway_service.py" ]

# Gunicorn config for production environment
# CMD ["gunicorn", "--config", "gunicorn.conf.py"]