# CP-AF

```bash
conda create -n myenv python=3.10
conda activate myenv
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# other packages
pip install tqdm
pip install pandas
pip install pytorch_metric_learning

# Pre-training
python Pre-training.py

# Fine-tuning
python Fine-tuning


```
