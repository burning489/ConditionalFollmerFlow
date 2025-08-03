#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generate data.

python data.py

# Abaltion study on different methods

for p1 in gan vae; do
	python train.py --mode $p1
	for p2 in 0.1 0.05 0.01; do
		python test.py --mode $p1 --alpha $p2 --ckpt "logs/wine-n5847-$p1/network-snapshot-000040K.pt"
	done
done

for p1 in vesde trig follmer; do
	python train.py --mode $p1
	for p2 in 0.1 0.05 0.01; do
		python test.py --mode $p1 --alpha $p2 --ckpt "logs/wine-n5847-$p1/network-snapshot-000006K.pt"
	done
done

# Ablation study on stopping time

# copy pre-trained checkpoints

cd logs
for p3 in 0.999 0.9995 0.9999; do
    cp -r wine-n5847-follmer wine-n5847-follmer-T$p3
done; 

# compute metrics

cd ..
for p3 in 0.999 0.9995 0.9999; do
    python test.py --mode follmer --T $p3 --ckpt "logs/wine-n5847-follmer-T$p3/network-snapshot-000006K.pt" 
done; 

# reproduce Table 3, Table T5

python post.py




