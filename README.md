# music_classifcation
use fma dataset for genre classification with != deepnet

look at https://medium.com/france-school-of-ai/installer-sa-premi%C3%A8re-machine-avec-gpu-dans-le-cloud-98798fdc4406 
(FR) for google cloud computing

conda create -n nom_env pyton=3.7
conda activate nom_env
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -U pip
conda install matplotlib scikit-learn numpy pandas scipy -y
conda install ipykernel
python -m ipykernel install --user --name=fastai

git clone https://github.com/Julien-pour/music_classifcation.git
sudo apt-get update
sudo apt-get install -y p7zip-full
cd /data
sudo wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O fma_small.zip
use python for unzip  (or sudo 7z x fma_small.zip && rm fma_small.zip)
sudo wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip wget -O fma_metadata.zip
sudo 7z x fma_metadata.zip && rm fma_metadata.zip

to launch jupyter notebook
jupyter notebook --no-browser --port=6969 --ip='0.0.0.0'

look for vram (terminal)
watch -n 0.2 nvidia-smi
