# Downloads the dataset used by this experiment into src/datasets/, where train.py reads it.
# The Google Drive file ID lives in src/datasets/download_fashion_mnist.py.

import os
import runpy

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets'))
    runpy.run_path('download_fashion_mnist.py', run_name='__main__')
