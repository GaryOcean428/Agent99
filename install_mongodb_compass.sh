#!/bin/bash

# Install dependencies
sudo apt-get update
sudo apt-get install -y libnotify4 libxtst6 xdg-utils libxcb-dri3-0 gnome-keyring

# Download MongoDB Compass
wget https://downloads.mongodb.com/compass/mongodb-compass_1.43.5_amd64.deb -O mongodb-compass.deb

# Install MongoDB Compass
sudo dpkg -i mongodb-compass.deb

# Fix any remaining dependencies
sudo apt-get install -f
