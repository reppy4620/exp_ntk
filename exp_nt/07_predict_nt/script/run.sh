#!/bin/bash

output_dir=../out

datasets=(fashion_mnist mnist:3.*.* cifar10)

for d in ${datasets[@]};
do
    od=$output_dir/$d
    python ../tool/predict_color.py $d $od
done