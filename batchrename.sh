#!/bin/bash
FILES=~/Desktop/DigitisedPhotos/Album1/*.JPG
ROOT=~/Desktop/DigitisedPhotos/Album1
a=1
for i in $FILES; do
  new=$(printf "$ROOT/NDF_%04d.JPG" "$a") #04 pad to length of 4
  mv "$i" "$new"
  let a=a+1
done