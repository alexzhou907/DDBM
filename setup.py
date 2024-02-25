from setuptools import setup

setup(
    name="ddbm",
    py_modules=["ddbm", "evaluations", "datasets", "scripts",],
    install_requires=[
        "blobfile>=1.0.5",
        "packaging",
        "tqdm",
        "numpy",
        "scipy",
        "pandas",
        "Cython",
        "piq==0.7.0",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "lmdb",
        "clip @ git+https://github.com/openai/CLIP.git",
        "mpi4py",
        "flash-attn==2.0.4",
        "pillow",
        "wandb",
        'omegaconf',
        'torchmetrics[image]',
        'prdc',
        'clean-fid==0.1.35'
    ],
)
