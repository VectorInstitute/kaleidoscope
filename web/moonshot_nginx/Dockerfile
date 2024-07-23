FROM ubuntu:22.04

# Install nginx and necessary packages
RUN apt update
RUN apt install -y vim nginx
RUN apt install -y certbot python3-certbot-nginx

# Add nginx configuration + SSL certs
ADD ./default /etc/nginx/sites-available
ADD ./letsencrypt /etc
