# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

SHELL := /bin/bash

include config.env
export $(shell cut -d= f1 config.env)
export PYTHONPATH := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))):$(PYTHONPATH)

PYTHON_VERSION = $(shell cat .python-version)
DOCKERFILE = Dockerfile
TAG = $(shell date +"%Y%m%d_%H%M")



# Default target
.PHONY: all
all: install

.PHONY: build
docker:
	docker compose up --build


####### RUN TOOLCHAIN #######
.PHONY: run
run:
	source /opt/ros/humble/setup.bash &&  set -a && source config.env && uv run src/infqm/main.py

yolo:
	source /opt/ros/humble/setup.bash && set -a && source config.env && uv run src/object_detector/yolo.py

fcos:
	source /opt/ros/humble/setup.bash && $ set -a && source config.env && uv run src/infqm/object_detector/fcos.py


.PHONY: fox
fox:
	source /opt/ros/humble/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml



####### Get Camera Input #######

.PHONY: test
test:
	@echo "Publishing webcam at $(INPUT_TOPIC)"
	@source /opt/ros/humble/setup.bash && ros2 run v4l2_camera v4l2_camera_node --ros-args -r /image_raw:=$(INPUT_TOPIC)

api:
	uv run src/infqm/api/app.py

unreal:
	source /opt/ros/humble/setup.bash && set -a && source config.env && uv run src/unreal/ros_unreal_bridge.py



####### Set-Up #######

install:
# Create conda environment and install requirements
	uv sync
# sudo apt install fonts-cmu

extract_features:
	uv run src/normalizing_flow/generate_training_data.py

train_flow:
	uv run src/normalizing_flow/train.py


####### Helper #######
push:
	git init
	git add .
	git commit -m "$(TAG)"
	git push origin github

# Lint Python Files
.PHONY: lint
lint:
	uvx ruff check src/


.PHONY: format
format:
	uvx black src
	uvx black Tests
	uvx ruff check src/ --fix --ignore E501
	uvx isort src/
	uvx docformatter --in-place --recursive src/


# Update requirements.txt using pipreqs
.PHONY: update
update:
	@echo "Updating requirements using pipreqs and conda export";
	pipreqs --force --encoding=iso-8859-1 --ignore ".venv"

stats:
	@echo "-- Conda Channels --"
	@conda config --show channels
	@echo "-- Current Branch --"
	@git branch --show-current
	@echo "-- Git Origin --"
	@git remote -v
	echo $(ENV_NAME)

name:
	@echo $(ENV_NAME)


prek:
	prek run -a && git add -update && git commit -m "chore: fixed prek"



# Clean the virtual environment
.PHONY: clean
clean:
	rm -rf .venv
