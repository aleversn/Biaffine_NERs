python s1_predict_data.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="/home/lpc/repos/CNNNER/datasets/few_shot" \
    --file_name=pku \
    --save_type_name=GLM4 \
    --model_from_pretrained="/home/lpc/models/glm-4-9b-chat/" \
    --batch_size=20 \
    --eval_mode=test.jsonl


python s4_predict_data_from_lora.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="/home/lpc/repos/CNNNER/datasets/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM3 \
    --model_from_pretrained="/home/lpc/models/chatglm3-6b/" \
    --peft_pretrained="/home/lpc/repos/ChatGLM_PEFT/save_model/fewshot_ner_1000/ChatGLM_20000" \
    --batch_size=20 \
    --eval_mode=test.jsonl
