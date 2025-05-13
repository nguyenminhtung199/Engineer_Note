## 🐳 Docker Commands Cheat Sheet

### Build & Run
```bash
# Build image from Dockerfile
docker build -t your-image-name .

# Run a container from an image
docker run -d -p 3000:3000 --name your-container-name your-image-name

# Build and run using docker-compose
docker-compose up --build
````

### Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop container_name_or_id

# Start a container
docker start container_name_or_id

# Remove a container
docker rm container_name_or_id

# View container logs
docker logs container_name_or_id
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi image_name_or_id
```

### Exec & Debug

```bash
# Open shell inside a running container
docker exec -it container_name_or_id /bin/bash

# Run a one-off command inside container
docker exec -it container_name_or_id command
```

### Clean Up

```bash
# Remove all stopped containers
docker container prune

# Remove unused images, networks, etc.
docker system prune
```

