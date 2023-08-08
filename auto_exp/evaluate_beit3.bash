# ##  beit3_base_itc_patch16_224.pth

# # 128 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output' --log_dir './log/base_128_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 128 --eval_batch_size 1024 
 

# # # 256 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output' --log_dir './log/base_256_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 256 --eval_batch_size 512 

# # # 512 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/' --log_dir './log/base_512_itc' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 512 --eval_batch_size 512

# # 768 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base_768_itc/' --log_dir './log/base_768_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 768 --eval_batch_size 256

# # 1024 tokens

# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base_1024_itc/' --log_dir './log/base_1024_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 1024 --eval_batch_size 128 


# ## beit3_base_patch16_224.pth

# # 128 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base_128_itc/' --log_dir './log/base_128_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --num_max_bpe_tokens 128 --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 128 --eval_batch_size 1024 
 

# # 256 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base_256_itc/' --log_dir './log/base_256_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --num_max_bpe_tokens 128 --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 256 --eval_batch_size 512 

# # 512 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base__64_itc/' --log_dir './log/base__64_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --num_max_bpe_tokens 128 --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 512 --eval_batch_size 512 

# # 768 tokens
# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/768_itc/' --log_dir './log/768_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --num_max_bpe_tokens 128 --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 768 --eval_batch_size 256

# # 1024 tokens

# python retrieval.py --model 'beit3_base_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_base_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/base_1024_itc/' --log_dir './log/base_1024_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --num_max_bpe_tokens 128 --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 1024 --eval_batch_size 128 





## beit3_large_patch16_224.pth

# 128 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_128_itc/' --log_dir './log/large_128_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 128 --eval_batch_size 256 
 

# # 256 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_256_itc/' --log_dir './log/large_256_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 256 --eval_batch_size 128

# 512 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_64_itc/' --log_dir './log/large_64_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 512 --eval_batch_size 64

# 768 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_768_itc/' --log_dir './log/large_768_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 768 --eval_batch_size 32

# # 1024 tokens

# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_1024_itc/' --log_dir './log/large_1024_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 1024 --eval_batch_size 16 




## beit3_large_itc_patch16_224.pth

# 128 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output' --log_dir './log/large_128_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 128 --eval_batch_size 256 
 

# 256 tokens
python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
--epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_itc_patch16_224.pth' \
--data_path '../datasets/ATOMIC/' --output_dir './output' --log_dir './log/large_256_itc/' --weight_decay 0.05 --seed 42 \
--save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 256 --eval_batch_size 64

# 512 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output' --log_dir './log' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 512 --eval_batch_size 64

# # 768 tokens
# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_768_itc/' --log_dir './log/large_768_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0  --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 768 --eval_batch_size 32

# # 1024 tokens

# python retrieval.py --model 'beit3_large_patch16_224' --input_size 224 --task 'atomic' --batch_size 128 --layer_decay 0.65 --lr 2e-4  \
# --epochs 5 --warmup_epochs 2 --drop_path 0.2 --sentencepiece_model 'beit3.spm' --finetune '../ckpt/beit3/beit3_large_itc_patch16_224.pth' \
# --data_path '../datasets/ATOMIC/' --output_dir './output/large_1024_itc/' --log_dir './log/large_1024_itc/' --weight_decay 0.05 --seed 42 \
# --save_ckpt_freq 1 --num_workers 0   --load_image_from_huggingface_hub --eval --num_max_bpe_tokens 1024 --eval_batch_size 16 
