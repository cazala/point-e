# Install

```
python3 -m pip install -e
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
