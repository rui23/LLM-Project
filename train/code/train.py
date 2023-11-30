import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from zmq import device
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# Step2 加载数据集
# ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
ds = Dataset.load_from_disk("/home/wangrui/LLM/huggingface/transformers-code/04-Kbit Training/data/alpaca_data_zh")


# Step3 数据集预处理
model_name = 'THUDM/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("silver/chatglm-6b-int4-qe", trust_remote_code=True)
tokenizer(tokenizer.eos_token), tokenizer.eos_token_id

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join([example["instruction"], example["input"]]).strip()     # query
    instruction = tokenizer.build_chat_input(instruction, history=[], role="user")  # [gMASK]sop<|user|> \n query<|assistant|>
    response = tokenizer("\n" + example["output"], add_special_tokens=False)        # \n response, 缺少eos token
    input_ids = instruction["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0].numpy().tolist()) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

# tokenizer.decode(tokenized_ds[1]["input_ids"])
# tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))


# Step4 创建模型
import torch
# model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, low_cpu_mem_usage=True, 
#                                              torch_dtype=torch.half, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.half,
#                                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.half, device_map="auto")

print("step4\n", model.chat(tokenizer, "数学考试怎么考高分？", history=[])[0])

for name, param in model.named_parameters():
    print(name, param.dtype)


# Lora
# PEFT Step1 配置文件
config = LoraConfig(
    target_modules=["query_key_value"],
    task_type=TaskType.CAUSAL_LM, 
    # inference_mode=True,
    # r=8,
    # lora_alpha=32, 
    # lora_dropout=0.1
    )

# #PEFT Step2 创建模型
model = get_peft_model(model, config)

model.enable_input_require_grads()
model.print_trainable_parameters()


# Step5 配置训练参数
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=1e-4,
    remove_unused_columns=False,
    # gradient_checkpointing=True,
    # optim="paged_adamw_32bit",
    # use_cpu=False,
    use_mps_device=False,
    # fp16=True,
)
args.n_gpu
args.device


# Step6 创建训练器
# tokenized_ds.select(range(6858))
trainer = Trainer(
    model=model,
    # model=model.to(args.device),
    args=args,
    # train_dataset=tokenized_ds.select(range(6858)),
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# Step7 模型训练
trainer.train()

# Step8 保存微调的模型
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')

# Step9 模型推理
model.eval()
print("step9\n", model.chat(tokenizer, "数学考试怎么考高分？", history=[])[0])

# Step10 加载微调后的模型和tokenizer
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model_saved = PeftModel.from_pretrained(model, '/home/wangrui/LLM/huggingface/transformers-code/fine_tuned_model')
tokenizer_saved = AutoTokenizer.from_pretrained('/home/wangrui/LLM/huggingface/transformers-code/fine_tuned_model',trust_remote_code=True)

# step11 使用重新加载的模型进行推导
print("step11\n", model_saved.chat(tokenizer_saved, "数学考试怎么考高分？", history=[])[0])

