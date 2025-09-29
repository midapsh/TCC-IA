#!/bin/bash
# unmount_ramdisk.sh

RAM_DISK_FOLDER=/mnt/ramdisk

if mountpoint -q "$RAM_DISK_FOLDER"; then
    echo "Unmounting $RAM_DISK_FOLDER ..."
    sudo umount "$RAM_DISK_FOLDER"
    sudo rmdir "$RAM_DISK_FOLDER"
    echo "RAM disk unmounted and folder removed."
else
    echo "$RAM_DISK_FOLDER is not mounted."
fi