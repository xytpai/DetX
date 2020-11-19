pip install pyyaml
pip install numpy
pip install scipy
pip install matplotlib
pip install cython
pip install opencv-python
pip install pytest
pip install pybind11

git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI
python setup.py install
cd ../../
rm -rf cocoapi

cd layers/detx_ext_cuda
python setup.py install
