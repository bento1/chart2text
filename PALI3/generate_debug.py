from CustomPALI3.SummaryChartDataset import SummaryChartDataset
from CustomPALI3.CustomPALI3 import CustomPALI3Config,CustomPALI3
from transformers import T5Tokenizer
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
from pytorch_model_summary import summary
import numpy as np
import gc
dataset_repo=[{'dataset':'timm/imagenet-12k-wds','config':'default','type':'vision-text'},
                {'dataset':'wikimedia/wikipedia','config':'20231101.en','type':'text'},
                {'dataset':'conceptual_captions','config':'labeled','type':'vision-text'},
                {'dataset':'poloclub/diffusiondb','config':'2m_random_1m','type':'vision-text'},
                ]
def my_collate_fn(samples):
    image_batch = []
    input_batch = []
    attn_mask_batch = []
    
    batch_size=len(samples)
    
    image_batch_ = []
    input_batch_ = []
    attn_mask_batch_ = []
    for sample in samples:
        image_batch_.extend(sample['image'])
        input_batch_.extend(sample['input_ids'])
        attn_mask_batch_.extend(sample['attn_mask'])
    
    total_b=len(image_batch_)//batch_size # 14 //4   3.xx 3  
    total_b=total_b+1 if len(image_batch_)%batch_size!=0  else total_b
    for i in range(total_b):
        if (i+1)*batch_size<len(image_batch_):
            image_batch.append(torch.stack(image_batch_[i*batch_size:(i+1)*batch_size]))
            input_batch.append(torch.stack(input_batch_[i*batch_size:(i+1)*batch_size]))
            attn_mask_batch.append(torch.stack(attn_mask_batch_[i*batch_size:(i+1)*batch_size]))
        else:
            image_batch.append(torch.stack(image_batch_[i*batch_size:]))
            input_batch.append(torch.stack(input_batch_[i*batch_size:]))
            attn_mask_batch.append(torch.stack(attn_mask_batch_[i*batch_size:]))

    return {'image': image_batch, 'input_ids': input_batch,'attn_mask':attn_mask_batch}
args={
    'output_dir':'/Users/dongunyun/study/datascience/chart2text/PALI3/output',
    'lr':1e-4,
    'max_steps':1e4,
    'valid_steps':1e2,
    'num_epochs':100,
    'batch_size':4,
    'num_training_samples_per_epoch':10,
    'max_epochs':100,
    "warmup_steps":100,
    'num_workers':1,
    'num_nodes':1,
    }

tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-base", bos_token = '<s>',add_bos_token = True)
train_loader=DataLoader(SummaryChartDataset(dataset_repo,1024,tokenizer,'</s>','train'), batch_size=1, shuffle=True, num_workers=1,collate_fn=my_collate_fn)
val_loader=DataLoader(SummaryChartDataset(dataset_repo,1024,tokenizer,'</s>','validation'), batch_size=1, shuffle=True, num_workers=1,collate_fn=my_collate_fn)
config=CustomPALI3Config(version=1,model_name='test',
                    dim=1024,enc_num_tokens=32100,enc_max_seq_len=1024,
                    dec_num_tokens=32100,dec_max_seq_len=1024,enc_depth=12,enc_heads=16,dec_depth=12,dec_heads=16,seq_len=1024
                    ,device='mps',vit_fix=False)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device =torch.device("cpu")


model=CustomPALI3(config)
model=model.from_pretrained("/Users/dongunyun/study/datascience/chart2text/PALI3/output_temp")
model.model.seq_len=128

from torchvision import transforms
from PIL import Image
img=Image.open('getImageNet.png').convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
label='Explain this picture. '

input_image_tensor = transforms.transforms.ToTensor()(img).squeeze(0)
input_image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image_tensor).unsqueeze(0).to(device)
outputs = tokenizer(label, 
                    max_length=1024, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt",
                    return_length=True,
                    )['input_ids'].to(device)
# model.to(device)
gen_=model.generate(input_image_tensor,outputs)

for gen in gen_:
    result_text = tokenizer.decode(gen, skip_special_tokens=True)
    print(result_text)