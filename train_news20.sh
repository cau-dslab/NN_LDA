export PYTHONPATH=.:$PYTHONPATH

mkdir -p models/news20

python new_train.py \
  --dataset "data/news20/news20_train/" \
  --model_dir "models/news20/" \
  --initial_learning_rate 0.001 \
  --batch_size 256 \
  --embedding "one_hot" \
  --cache \
  --max_steps 100000
  --overwrite
