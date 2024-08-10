# Use an official Ubuntu image as a parent image
FROM ubuntu:20.04

# Install any dependencies you need here
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (optional, for security)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN sudo apt-get update && sudo apt-get install -y libnotify4 libxtst6 xdg-utils libxcb-dri3-0 gnome-keyring && \
    wget https://downloads.mongodb.com/compass/mongodb-compass_1.43.5_amd64.deb -O mongodb-compass.deb && \
    sudo dpkg -i mongodb-compass.deb && \
    sudo apt-get install -f
