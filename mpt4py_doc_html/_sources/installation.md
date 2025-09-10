# Installation

1. Install [gmp](https://gmplib.org) and [cddlib](https://github.com/cddlib/cddlib):
    - For **Ubuntu**: 

        ```
        sudo apt-get install libgmp-dev libcdd-dev
        ```
    - For **MacOS**: 
    
        ```
        brew install gmp cddlib
        ```
2. Install mpt4py in a virtual environment
    ```bash
    # Create a virtual environment
    python -m venv venv  # to speficy a python version, e.g. 3.12, use python3.12 -m venv venv
    source venv/bin/activate  # activate the virtual environment
    pip install -e .
    ```