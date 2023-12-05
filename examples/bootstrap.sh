#! /bin/bash

set -e

sudo apt install -y python3.10-venv emacs

python3 -m venv .venv

. .venv/bin/activate

