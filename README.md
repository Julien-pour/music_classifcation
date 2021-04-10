# music genre classifcation

## Intro
Use fma dataset for genre classification with different deep learning architecture like Vision Transformer (ViT) [`arXiv:2010.11929`][paper] and other CNN like resnet.
[paper]: https://arxiv.org/abs/2010.11929



## Data
see <https://github.com/mdeff/fma>

## Code 
1. [`torch_transformer_classification.ipynb`]: preprocess dataset downsample sample to 22050 Hz and save them in .wav create and train transformeur
[`torch_transformer_classification.ipynb`]:     https://nbviewer.jupyter.org/github/Julien-pour/music_classifcation/blob/main/torch_transformer_classification.ipynb
2. need to clean others notebooks and merge them

## Usage
look at https://medium.com/france-school-of-ai/installer-sa-premi%C3%A8re-machine-avec-gpu-dans-le-cloud-98798fdc4406 (FR) for google cloud computing to train transformer I used Tesla v100 (GCP)
test with python 3.7 and linux
1. Create a Python 3.7 environment
    ```
    conda create -n nom_env pyton=3.7
    conda activate nom_env
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install -U pip
    conda install matplotlib scikit-learn numpy pandas scipy -y
    conda install ipykernel
    python -m ipykernel install --user --name=fastai
    ```
1. Clone the repository
    ```
    git clone https://github.com/Julien-pour/music_classifcation.git
    ```
1. Install dependencies and dl dataset
    ```
    sudo apt-get update
    sudo apt-get install -y p7zip-full
    cd /data
    sudo wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O fma_small.zip   
    #use python for unzip  (problem when using 7z: sudo 7z x fma_small.zip && rm fma_small.zip)
    sudo wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip wget -O fma_metadata.zip
    sudo 7z x fma_metadata.zip && rm fma_metadata.zip
    ```
1. To launch jupyter notebook for GCP
    ```
    jupyter notebook --no-browser --port=6969 --ip='0.0.0.0'
    ```
1. look for vram (in terminal)
    ```
    watch -n 0.2 nvidia-smi
    ```
    
## To do
