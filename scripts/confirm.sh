#!/bin/bash

read -p "Are you sure you want to delete all untracked files? [y/N] " -n 1 -r answer
if [ "$answer" != "y" ]; then printf "\nAborting"; exit 1; fi
