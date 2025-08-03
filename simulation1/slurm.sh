#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generate data

for p1 in 4squares checkerboard pinwheel swissroll; do
    python data.py --data $p1 --output dataset/$p1.pkl
done


# Abaltion study on different methods

for p1 in 4squares checkerboard pinwheel swissroll; do
    for p2 in gan vae; do
        python train.py --mode $p2 --data dataset/$p1.pkl --n_train 50000
        python test.py --mode $p2 --data dataset/$p1.pkl --ckpt "logs/$p1-n50000-$p2/network-snapshot-000020K.pt" 
    done
    for p2 in vesde trig; do
        python train.py --mode $p2 --data dataset/$p1.pkl --n_train 50000
        python test.py --mode $p2 --T 0.95 --data dataset/$p1.pkl --ckpt "logs/$p1-n50000-$p2/network-snapshot-000016K.pt" 
    done
    python train.py --mode follmer --data dataset/$p1.pkl --n_train 50000
    python test.py --mode follmer --T 0.999 --data dataset/$p1.pkl --ckpt "logs/$p1-n50000-follmer/network-snapshot-000016K.pt" 
    python thirdparty.py --mode nnkcde --data dataset/$p1.pkl   
    python thirdparty.py --mode flexcode --data dataset/$p1.pkl  
done

# reproduce Figure 1

python plot.py

# Ablation study on stopping time

# copy pre-trained checkpoints

cd logs
for p1 in 4squares checkerboard pinwheel swissroll; do 
    for p3 in 0.999 0.9995 0.9999; do
        cp -r $p1-n50000-follmer $p1-n50000-follmer-T$p3; 
    done; 
done

# compute metrics

cd ..
for p1 in 4squares checkerboard pinwheel swissroll; do 
    for p3 in 0.999 0.9995 0.9999; do
        python test.py --mode follmer --T $p3 --data dataset/$p1.pkl --ckpt "logs/$p1-n50000-follmer-T$p3/network-snapshot-000016K.pt" 
    done; 
done

# reproduce Figure F1

for p1 in 4squares checkerboard pinwheel swissroll; do 
    for p3 in 0.999 0.9995 0.9999; do
        python testT.py --data $p1 --ckpt logs/$p1-n50000-follmer-T$p3/network-snapshot-000016K.pt --T $p3
    done
done
python plotT.py

# Ablation study on the number of training samples

for p1 in 4squares checkerboard pinwheel swissroll; do
    for p4 in 1000 2000 10000 40000; do
        python train.py --mode follmer --data dataset/$p1.pkl --n_train $p4 --bsz 500
        python test.py --mode follmer --T 0.999 --data dataset/$p1.pkl --ckpt "logs/$p1-n$p4-follmer/network-snapshot-000010K.pt"
    done
done

# reproduce Table 1, Table T1, Table T3

python post.py
 
