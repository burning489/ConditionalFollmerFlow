from easydict import EasyDict


scheme = "ode"
sde_name = "SchrodingerFollmer" if scheme == "sde" else "Follmer"

cfg = EasyDict({
    "meta": {
        "desc": f"{scheme}-mnist",
        "seed": 42,
    },
    "data": {
        "shape": [1, 28, 28],
        "num_workers": 1,
    },
    "diffusion": {
        "name": sde_name,
    },
    "model": {
        "img_resolution": 28,
        "in_channels": 1,
        "out_channels": 1,
        "model_channels": 64,
        "channel_mult": [1, 2],
        "channel_mult_emb": 2,
        "num_blocks": 2,
        "attn_resolutions": [16, ],
        "dropout": 0.1,
        "embedding_type": "positional",
        "channel_mult_noise": 1,
        "encoder_type": "standard",
        "decoder_type": "standard",
        "resample_filter": [1, 1],
    },
    "train": {
        "max_epochs": 5000,
        "batch_size": 256,
        "ema_decay": 0.999,
        "eps0": 1e-3,
        "eps1": 1e-3,
        "save_freq": 100,
        "eval_freq": 10,
        "sampling_freq": 100,
    },
    "optim": {
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "warmup": 100,
        "grad_clip": 1.0
    },
    "eval": {
        "batch_size": 256,
    },
    "sampling": {
        "batch_size": 64,
        "eps0": 1e-3,
        "eps1": 1e-3,
        "T": 1000,
    }
})


if __name__ == "__main__":
    print(cfg)