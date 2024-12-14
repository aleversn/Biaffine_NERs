python s1_predict_data.py \
    --n_gpu=1 \
    --skip=-1 \
    --file_dir="/home/lpc/repos/CNNNER/datasets/few_shot" \
    --file_name=msr \
    --save_type_name=GLM4 \
    --model_from_pretrained="/home/lpc/models/glm-4-9b-chat/" \
    --batch_size=20 \
    --eval_mode=test.jsonl
