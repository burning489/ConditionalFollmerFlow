#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generate data.

for p1 in 1 2 3; do
	python data.py --data $p1 --output dataset/$p1.pkl
done

# Abaltion study on different methods

for p1 in 1 2 3; do
	for p2 in gan vae vesde trig; do
		python train.py --mode $p2 --data dataset/$p1.pkl
		python test.py --mode $p2 --T 0.95 --data dataset/$p1.pkl --ckpt "logs/$p1-n10000-$p2/network-snapshot-000003K.pt"
	done
	python train.py --mode follmer --data dataset/$p1.pkl
	python test.py --mode follmer --T 0.999 --data dataset/$p1.pkl --ckpt "logs/$p1-n10000-follmer/network-snapshot-000003K.pt"
	python thirdparty.py --mode nnkcde --data dataset/$p1.pkl  
	python thirdparty.py --mode flexcode --data dataset/$p1.pkl  
done

# Ablation study on stopping time

# copy pre-trained checkpoints

cd logs
for p1 in 1 2 3; do 
    for p3 in 0.999 0.9995 0.9999; do
        cp -r $p1-n10000-follmer $p1-n10000-follmer-T$p3; 
    done; 
done

# compute metrics

cd ..
for p1 in 1 2 3; do 
    for p3 in 0.999 0.9995 0.9999; do
        python test.py --mode follmer --T $p3 --data dataset/$p1.pkl --ckpt "logs/$p1-n10000-follmer-T$p3/network-snapshot-000003K.pt" 
    done; 
done


# Ablation study on the number of training samples

for p1 in 1 2 3; do
    for p4 in 1250 2500 5000; do
        python train.py --mode follmer --data dataset/$p1.pkl --n_train $p4
        python test.py --mode follmer --T 0.999 --data dataset/$p1.pkl --ckpt "logs/$p1-n$p4-follmer/network-snapshot-000010K.pt"
    done
done

# reproduce Table 2, Table T2, Table T4

python post.py
 