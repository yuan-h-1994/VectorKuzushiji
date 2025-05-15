python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        --output_dir exps/ \
        --epochs 300 \
        --batch_size 64