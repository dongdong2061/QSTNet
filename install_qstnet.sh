echo "****************** Installing pytorch ******************"
#conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
apt-get install libturbojpeg
pip install jpeg4py -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm==0.5.4 -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom -i https://mirrors.aliyun.com/pypi/simple

echo ""
echo ""
echo "****************** Installing rgbt ******************"
pip install rgbt -i https://mirrors.aliyun.com/pypi/simple


echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
# Hi~ We employ the vot-toolkit==0.5.3 with vot-trax==3.0.3
pip install git+https://github.com/votchallenge/vot-toolkit-python

echo "****************** Installation complete! ******************"