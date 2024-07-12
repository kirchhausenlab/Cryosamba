from setuptools import setup

setup(
    name="cryosamba",
    version="0.1",
    description="",
    author="Jose Inacio da Costa Filho",
    author_email="",
    license="MIT",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "tensorboard",
        "cupy-cuda11x",
        "easydict",
        "loguru",
        "mrcfile",
        "numpy",
        "tifffile",
    ],
    python_requires=">=3.8, <3.10",
)
