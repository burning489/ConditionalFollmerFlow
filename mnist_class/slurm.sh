#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Abaltion study on different methods

# train

for p1 in gan vae vesde trig follmer; do
	python train.py --mode $p1
done

# generate samples

for p1 in gan vae vesde trig; do
	python test.py --mode $p1 --T 0.95 --ckpt "logs/$p1-n60000/network-snapshot-000010K.pt"
done
python test.py --mode follmer --ckpt "logs/follmer-n60000/network-snapshot-000010K.pt"

# generate reference stats

python fid.py

# compute FIDs

for p1 in gan vae vesde trig follmer; do
	python plot.py --rundir "logs/$p1-n60000"
	python fid.py --mode gen --ckpt "logs/$p1-n60000/samples.npy"
	python fid.py --mode fid --ckpt "logs/$p1-n60000/samples.npy"
done

# Ablation study on stopping time

# copy pre-trained checkpoints

cd logs
for p2 in 0.999 0.9995 0.9999; do
    cp -r follmer-n60000 follmer-n60000-T$p2
done;

# compute FIDs

cd ..
for p3 in 0.999 0.9995 0.9999; do
	python test.py --mode follmer --T $p3 --ckpt "logs/follmer-n60000-T$p3/network-snapshot-000010K.pt"
	python fid.py --mode gen --ckpt "logs/follmer-n60000-T$p3/samples.npy"
	python fid.py --mode fid --ckpt "logs/follmer-n60000-T$p3/samples.npy"
done; 



