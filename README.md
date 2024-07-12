# Aligning Neuronal Coding of Dynamic Visual Scenes with Foundation Vision Models

> Due to the dataset size, we had to only include the code part and the checkpoint. The dataset is not included in this repository.

```bash
# Install the required packages
pip3 install torch torchvision torchaudio lightning torchmetrics zarr
python -m pip install tslearn

# run train loop
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
PYTHONPATH=. python src/trainer/Trainer.py -m \
task_name=eccv model.featrue_key=dinov2_feats_0 \
model=eccv dataset.cross_val_movie=True dataset.movie_name=movie01,movie03 tags='["eccv"]'

# the predictions will be saved in the following directory
# Mov1 -> Mov2
checkpoints/0/checkpoints
# Mov2 -> Mov1
checkpoints/1/checkpoints
```
