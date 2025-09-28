#!/bin/bash

pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

# NOTE(hspadim): exit on error
# Get base folders
# - `root` folder
# - `scripts` folder
# - `logs` folder
# - `src` folder
SCRIPTS_PATH="$(dirname "$(realpath "$BASH_SOURCE")")"
pushd $SCRIPTS_PATH
pushd ..
ROOT_PATH="${PWD}"
popd
popd

DATABASE_FILENAME=database.db
DATABASE_FILEPATH=$ROOT_PATH/data/$DATABASE_FILENAME

RAM_DISK_FOLDER=/mnt/ramdisk
sudo mkdir -p $RAM_DISK_FOLDER
sudo mount -t tmpfs -o size=32G tmpfs $RAM_DISK_FOLDER
cp $DATABASE_FILEPATH $RAM_DISK_FOLDER
# sqlite3 $RAM_DISK_FOLDER/$DATABASE_FILENAME
