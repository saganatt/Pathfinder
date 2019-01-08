#!/bin/bash

git fetch origin
git rebase origin/master &&
make clean
make
