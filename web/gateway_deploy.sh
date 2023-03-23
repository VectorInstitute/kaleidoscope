#!/bin/bash

# Parse argument for deployment version. Accepts either "production" or "staging", defaults to production.
deploy_version=$1
branch=$2

# Validate arguments
if [ -z "$deploy_version" ]; then
	echo "Must specify a deployment version (staging or production)."
	echo "Usage: ./gateway_deploy.sh <staging|production> [branch-name]"
	exit 1
fi

if [ "$deploy_version" == "production" ]; then
	echo "Deploying production gateway..."

	# Update git repository
	cd ..
	git checkout main
	git pull

	# If user specified a branch, switch to that
	if [ ! -z "$branch" ]; then
		git checkout --track $branch
	fi

	# Rebuild/restart docker service
	cd web
	docker compose --file docker-compose.yaml down
	docker image rm gateway:latest
	docker compose --file docker-compose.yaml up

elif [ "$deploy_version" == "staging" ]; then
	echo "Deploying staging gateway..."

	# Update git repository
	cd ..
	git checkout develop
	git pull

	# If user specified a branch, switch to that
	if [ ! -z "$branch" ]; then
		git checkout --track $branch
	fi

	# Rebuild/restart docker service
	cd web
	docker compose --file docker-compose-staging.yaml down
	docker image rm kaleidoscope-staging-web-worker kaleidoscope-staging-web
	docker compose --file docker-compose-staging.yaml up
else
	echo "Deployment version $deploy_version is not valid. Please specify either staging or production."
	exit 1
fi
