# lingua
A user toolkit for analyzing and interfacing with Large Language Models (LLMs)

## Docker Image

**Build Docker image**
```bash
docker build -t gateway-service .
```

**Run Interactive Docker image**
```bash
sudo docker run --network=host -it -p 5000:5000 -d gateway-service
```

**Verify running container**
```bash
sudo docker ps
```

**Retrieve webserver URL**
```bash
sudo docker logs [CONTAINER]
```