#!/bin/bash

date

echo "Started creating HaloFiles Template"

mkdir HaloFiles
cd HaloFiles
mkdir h148
mkdir h229
mkdir h242
mkdir h329

dirs=`ls`
for i in $dirs
do 
 `mkdir Host`
 `mkdir Zombie`
 `mkdir Survivor`
done

echo "Finished creating HaloFiles Template"