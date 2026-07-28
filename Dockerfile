# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
FROM ros:humble

# Switch to bash — ROS setup scripts require it
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN source /opt/ros/humble/setup.bash && uv sync --frozen --no-install-project

COPY . .

RUN source /opt/ros/humble/setup.bash && uv sync --frozen
