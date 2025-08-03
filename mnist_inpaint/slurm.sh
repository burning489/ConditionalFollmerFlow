#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# train

for p1 in 1 2 3; do
	for p2 in gan vae vesde trig follmer; do
		python train.py --mask_mode $p1 --mode $p2
	done
done

# generate samples

for p1 in 1 2 3; do
	for p2 in gan vae vesde trig; do
		python test.py --mask_mode $p1 --mode $p2 --T 0.95 --ckpt "logs/$p2-mask$p1-n60000/network-snapshot-000010K.pt"
	done
	python test.py --mask_mode $p1 --mode follmer --ckpt "logs/follmer-mask$p1-n60000/network-snapshot-000010K.pt"
done

# generate reference stats

python fid.py

# compute FIDs

for p1 in 1 2 3; do
	for p2 in gan vae vesde trig follmer; do
	python fid.py --mode gen --ckpt "logs/$p2-mask$p1-n60000/samples.npy"
	python fid.py --mode fid --ckpt "logs/$p2-mask$p1-n60000/samples.npy"
	python plot.py --mode $p2 --mask_mode $p2 --ckpt "logs/$p2-mask$p1-n60000/network-snapshot-000010K.pt"
done


# Ablation study on stopping time

# copy pre-trained checkpoints

cd logs
for p1 in 1 2 3; do
	for p2 in 0.999 0.9995 0.9999; do
	    cp -r follmer-maske$p1-n60000 follmer-mask$p1-n60000-T$p2
	done
done

# compute FIDs

cd ..
for p1 in 1 2 3; do
	for p3 in 0.999 0.9995 0.9999; do
		python test.py --mode follmer --T $p3 --ckpt "logs/follmer-mask$p1-n60000-T$p3/network-snapshot-000010K.pt"
		python fid.py --mode gen --ckpt "logs/follmer-mask$p1-n60000-T$p3/samples.npy"
		python fid.py --mode fid --ckpt "logs/follmer-mask$p1-n60000-T$p3/samples.npy"
	done
done