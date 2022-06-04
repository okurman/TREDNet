#!/usr/bin/env bash

trg=/panfs/pan1/devdcode/sanjar/TREDNet/kipoi/models_repo/

rsync -va --exclude="*/downloaded" --exclude="*/__pycache__" TREDNet $trg

