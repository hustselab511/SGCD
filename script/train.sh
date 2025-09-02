# zh
export CUDA_VISIBLE_DEVICES=1

python main.py \
    --seed 45 \
    --result_file_name result \
    --lang zh \
    --dscgnn_layer_num 2 \
    --gnn_layer_num 3 \
    --loss_w 296 \
    --topk 0.7 \
    --warmup_steps 350 \
    --adam_epsilon 1e-7 \
    --input_files "train_dependent_guide_trf valid_dependent_guide_trf test_dependent_guide_trf"


# en
python main.py \
    --seed 42 \
    --result_file_name result \
    --lang en \
    --dscgnn_layer_num 2 \
    --gnn_layer_num 3 \
    --loss_w 296 \
    --topk 0.5 \
    --warmup_steps 400 \
    --adam_epsilon 1e-8 \
    --input_files "train_dependent_guide_trf valid_dependent_guide_trf test_dependent_guide_trf"