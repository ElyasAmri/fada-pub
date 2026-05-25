#!/bin/bash
DIR="/home/ubuntu/fada-v3/data/Fetal Ultrasound/Abdomen"
cd "$DIR"
count=0
for f in Abodomen_*.png; do
    newname="${f/Abodomen_/Abdomen_}"
    mv "$f" "$newname"
    count=$((count + 1))
done
echo "Renamed $count files"
ls | head -5
