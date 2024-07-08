export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/llama2-7b \
--wandb Tinyllama_longcontext \
--max-train-steps 500 \
--learning-rate 2e-5 \
--dataset /home/yuhao/From-Fragment-to-Fabric-Long-Context-Scaling-with-Short-Instruction-Tuning-Data/F2F_dataset_building/F2F_data.parquet \
--model meta-llama/Llama-2-7b-hf \
--save-interval 50 \
--seq-length 32768 \
--rope-theta 1000000 \
--parallel_mode data_parallel