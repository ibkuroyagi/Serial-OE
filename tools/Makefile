DOT:= .
.PHONY: all clean
all: virtualenv
virtualenv:
	./setup_anaconda.sh venv asd_env 3.9
	. ./activate_python.sh; conda update -n base -y -c defaults conda
	. ./activate_python.sh; conda update -n base -y -c conda-forge conda
	git clone https://github.com/rwightman/pytorch-image-models.git
	. ./activate_python.sh; cd pytorch-image-models; pip install .
	. ./activate_python.sh; cd ../; pip install -e .
	. ./activate_python.sh; conda install -y ipython ipykernel numpy pandas scikit-learn matplotlib librosa h5py pyyaml tensorboard umap-learn black flake8 -c conda-forge
	. ./activate_python.sh; conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
clean:
	rm -fr venv miniconda.sh activate_python.sh pytorch-image-models
	find -iname "*.pyc" -delete