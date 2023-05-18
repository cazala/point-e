# Install

```
python3 -m pip install -e
```

In case it doesn't work, on Windows you can try:
```
python setup.py
```

# Run

```
python3 text2mesh.py
```

# Troubleshoot

```
ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with LibreSSL 2.8.3
```

Install an older version of urllib3: `pip install urllib3==1.26.6`


# Using the Dedicated Graphic Card
1. You need to install the CUDA toolkit and drivers: https://developer.nvidia.com/cuda-downloads
2. Uninstall the `torch` version installed in `Install` section, and go here https://pytorch.org/get-started/locally/, select the proper OS and requirement, and run the command resulted.