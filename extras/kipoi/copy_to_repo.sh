#!/usr/bin/env bash

rsync -van --exclude="*/downloaded" --exclude="*/__pycache__" TREDNet models_repo

