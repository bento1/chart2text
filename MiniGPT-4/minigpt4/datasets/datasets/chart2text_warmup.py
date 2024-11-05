import os
import json
import torch
from PIL import Image
import numpy as np

def load_dataset(root_path):
    #pew_dataset 1000ea
    #statista_dataset 1000ea
    #scicap_data nosubfig 1000ea
    #simulated_scatter 2000ea
    #test 100
    # print(root_path)
    pew_dataset_root_path=root_path+'/pew_dataset_reduced'
    statista_dataset_root_path=root_path+'/statista_dataset_reduced'
    scicap_data_root_path=root_path+'/scicap_data_reduced'
    simulated_scatter_root_path=root_path+'/simulated_scatter_dataset'
    simulated_trace_root_path=root_path+'/simulated_trace_dataset'
    simulated_bar_root_path=root_path+'/simulated_bar_dataset'
    simulated_comb_root_path=root_path+'/simulated_comb_dataset'
    train_dataset=[]
    valid_dataset=[]
    test_dataset=[]
    #####################################################################################
    #####################################################################################

    # for r_path in [pew_dataset_root_path,statista_dataset_root_path]:
    #     imagpath=os.path.join(r_path,'dataset','imgs')
    #     capspath=os.path.join(r_path,'dataset','captions')

    #     fileEx = r'.png'
    #     file_list = [file.split('.')[0] for file in os.listdir(imagpath) if file.endswith(fileEx)]
    #     train_file_list=file_list[:1000]
    #     valid_file_list=file_list[1000:1100]
    #     test_file_list=file_list[1100:1200]

    #     def readTxt(path):
    #         # readline_all.py
    #         f = open(path, 'r')
    #         result=""
    #         while True:
    #             line = f.readline()
    #             if not line: break
    #             result+=line+' '
    #         f.close()
    #         return result

    #     for filename in train_file_list:
    #         image_path=os.path.join(imagpath,f'{filename}.png')
    #         cap_path=os.path.join(capspath,f'{filename}.txt')
    #         train_dataset.append({'image':image_path,'text':readTxt(cap_path)})

    #     for filename in valid_file_list:
    #         image_path=os.path.join(imagpath,f'{filename}.png')
    #         cap_path=os.path.join(capspath,f'{filename}.txt')
    #         valid_dataset.append({'image':image_path,'text':readTxt(cap_path)})

    #     for filename in test_file_list:
    #         image_path=os.path.join(imagpath,f'{filename}.png')
    #         cap_path=os.path.join(capspath,f'{filename}.txt')
    #         test_dataset.append({'image':image_path,'text':readTxt(cap_path)})
    # #####################################################################################
    # #####################################################################################

    # capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','train')
    # fileEx = r'.json'
    # file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    # train_file_list=[]

    # for filename in file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object:
    #         if json_object["contains-subfigure"]==False:
    #             train_file_list.append(filename)
    #             if len(train_file_list)==1000:
    #                 break

    # imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','train')
    # for filename in train_file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
    #         image_file_name=json_object['figure-ID']
    #         image_path=os.path.join(imgpath,image_file_name)
    #         train_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})

    # capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','val')
    # fileEx = r'.json'
    # file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    # valid_file_list=[]

    # for filename in file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object:
    #         if json_object["contains-subfigure"]==False:
    #             valid_file_list.append(filename)
    #             if len(valid_file_list)==100:
    #                 break

    # imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','val')
    # for filename in valid_file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
    #         image_file_name=json_object['figure-ID']
    #         image_path=os.path.join(imgpath,image_file_name)
    #         valid_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})


    # capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','test')
    # fileEx = r'.json'
    # file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    # test_file_list=[]

    # for filename in file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object:
    #         if json_object["contains-subfigure"]==False:
    #             test_file_list.append(filename)
    #             if len(test_file_list)==100:
    #                 break

    # imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','test')
    # for filename in test_file_list:
    #     cap_path=os.path.join(capspath,filename)
    #     with open(cap_path) as f:
    #         json_object = json.load(f)
    #     if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
    #         image_file_name=json_object['figure-ID']
    #         image_path=os.path.join(imgpath,image_file_name)
    #         test_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_scatter_root_path,'data','train')
    imagepath=os.path.join(simulated_scatter_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_scatter_root_path,'data','valid')
    imagepath=os.path.join(simulated_scatter_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_scatter_root_path,'data','test')
    imagepath=os.path.join(simulated_scatter_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_trace_root_path,'data','train')
    imagepath=os.path.join(simulated_trace_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_trace_root_path,'data','valid')
    imagepath=os.path.join(simulated_trace_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_trace_root_path,'data','test')
    imagepath=os.path.join(simulated_trace_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_bar_root_path,'data','train')
    imagepath=os.path.join(simulated_bar_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_bar_root_path,'data','valid')
    imagepath=os.path.join(simulated_bar_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_bar_root_path,'data','test')
    imagepath=os.path.join(simulated_bar_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_comb_root_path,'data','train')
    imagepath=os.path.join(simulated_comb_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_comb_root_path,'data','valid')
    imagepath=os.path.join(simulated_comb_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_comb_root_path,'data','test')
    imagepath=os.path.join(simulated_comb_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})


    dataset=dict()
    dataset['train']=train_dataset
    dataset['valid']=valid_dataset
    dataset['test']=test_dataset
    return dataset

class SummaryChartDataset(torch.utils.data.Dataset):
    def __init__(self, vis_processor,text_processor, root_path, split='train'):
        self.dataset = load_dataset(root_path)
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        print(self.vis_processor)
        print(self.text_processor)
        self.split=split


    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx: int):
        sample = self.dataset[self.split][idx]
        img_path = os.path.join(sample['image'])
        while not os.path.isfile(img_path):#없으면 True 있으면 False 가 되서 빠져나옴
            idx+=1
            if (idx-1)>=len(self.dataset[self.split]):
                idx=0
            sample = self.dataset[self.split][idx]
            img_path = os.path.join(sample['image'])

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        w_scale= np.random.randint(-20,21)
        h_scale= np.random.randint(-20,21)
        new_width = int(width * (1+w_scale/100))
        new_height= int(height * (1+h_scale/100))
        img = img.resize((new_width, new_height), Image.LANCZOS)

        input_tensor = self.vis_processor(img)

        if 'origin_text' in sample:
            cap_path=sample['origin_text']
            with open(cap_path) as f:
                json_object = json.load(f)
            if 'description' in json_object:
                if "This scatterplot is a chart showing the distribution of values over time" in json_object["description"]:
                    query="describe this chart about trend and its abnormalities"
                elif "This scatterplot shows the clustering results of groups between" in json_object["description"]:
                    query="describe this chart about clustering and its abnormalities"
                elif "This scatterplot is a graph that represents the relationship between" in json_object["description"]:
                    query="describe this chart about correlation and its abnormalities"
                else:
                    query="describe this chart"
            else:
                query="describe this chart"
        else:
            query="describe this chart"
        

        if "You are a helpful" in  sample['text'] or "please don't share false information" in sample['text'] or "following sentence is" in sample['text']:
            if 'origin_text' in sample:
                if 'description' in json_object:
                    ans=json_object['description']
                else:
                    ans="I don't have any idea in this chart"
            else:
                ans="I don't have any idea in this chart"
        else:
            ans=sample['text']


        instruction=self.text_processor('[summary] '+query)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        # input_ids=self.text_processor(instruction)
        
        if self.split=="train":
            ans_ids = self.text_processor(ans)
            return {
                "image": input_tensor,
                "instruction_input": instruction,
                "answer": ans_ids,
            }
        else:
            return {
                "image": input_tensor,
                "instruction_input": instruction,
                "answer": ans,
            }

