from datasets import load_dataset
import json
import os
import random
from hashlib import sha256
from datasets import load_dataset
from transformers import LlamaTokenizer

import json
import os
import random
from multiprocessing import Pool
import argparse
import pandas as pd
# 设置文件路径
file_path_1 = "tokenized_data_Orca_all.parquet"
file_path_2 = "tokenized_data_SlimPajama.parquet"
# 加载数据集
dataset_Orca = load_dataset('parquet', data_files=file_path_1)
dataset_Slim = load_dataset('parquet', data_files=file_path_2)
# 查看数据集
dataset_Orca = dataset_Orca['train']
dataset_Slim = dataset_Slim['train']
dataset_Slim = dataset_Slim.shuffle()
dataset_Orca = dataset_Orca.shuffle()



def generate_unique_string():
    """ Generate a unique string based on hashing """
    unique_id = sha256(os.urandom(48)).hexdigest()[:24]
    return unique_id

import random

def word_len(example):
    # 计算单词数量
    lens = len(example.split())
    return lens
  
def token(text,tokenizer):
    """Tokenize text data and generate ID."""
   
    tokenized_text = tokenizer(text, truncation=True, max_length=10000)
    
    return tokenized_text['input_ids']

def input_start(input,label,length,tokenizer):
    context = f"\n### Instruction: Please carefully read the text and answer the questions contained within.\n"
    context += f"### Context:\n"
    add_list = token(context,tokenizer)
    # print(type(add_list))
    # print(type(input))
    input = input + add_list
    label = label +  [-100] * len(add_list)
    length = length + len(add_list)
    return  input, label, length 

def add_ctxs_to_item(items,data_index_slimpajama,data_index_Orca,length,orca_length):
    """ Adds 'ctxs' key to a group of items and updates GPT_answer """
    ctxs = []
    questions_ctxs =[]
    total_length = 0
    Orca_Q_length = []
    Orca_A_length = []
    Squad_Q_length = []
    Squad_A_length = []
    input_ids =[]
    labels = []
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    input_ids,labels,total_length = input_start(input_ids,labels,total_length,tokenizer)

    # Process each item to create questions, answers, and modify GPT_answer
    for idx, item in enumerate(items):
        # mid
        if idx < 5:
            context_id = generate_unique_string()
            question_id = generate_unique_string()
            c_id = token(f"\nID: {context_id}, ",tokenizer)
            q_id = token(f"\nID: {question_id}, Answer question {context_id}: ",tokenizer)
            text = token(item['context']+f" ### Question: "+item['question'],tokenizer)
            GPT_answer = token(item['GPT_answer'],tokenizer)
            ctxs.append({
                    "ID": c_id+text,
                     #"text": text,
                    "type": 'question',
                    "Q_ID": q_id,
                    "GPT_answer":GPT_answer,
                })
            total_length += len(q_id+c_id+text+GPT_answer)
            Squad_Q_length += [len(text)]
            Squad_A_length += [len(GPT_answer)]
        #end 
        elif idx < 10:
            context_id = generate_unique_string()
            question_id = generate_unique_string()
            c_id = token(f"\nID: {context_id}, ",tokenizer)
            q_id = token(f"\nQuestion {idx-5}: {item['question']}\n Retrieval: The answer to question {idx-5} can be found in ID {context_id}.", tokenizer)
            text = token(item['context']+f" ### Question: "+item['question'],tokenizer)
            GPT_answer = token(item['GPT_answer'],tokenizer)
            ctxs.append({
                    "ID": c_id+text,
                    # "text": text,
                    "type": 'end',
                    "Q_ID": q_id+GPT_answer,
                    #"GPT_answer":GPT_answer,
                })
            total_length += len(q_id+c_id+text+GPT_answer)
            Squad_Q_length += [len(text)]
            Squad_A_length += [len(GPT_answer)]
        else:
            context_id = generate_unique_string()
            c_id = token(f"\nID: {context_id}, ",tokenizer)
            text = token(item['context']+f" ### Question: "+item['question'],tokenizer)
            ctxs.append({
                    "ID": c_id+text,
                    "type": False,
                })
            total_length += len(c_id+text)
            Squad_Q_length += [len(text)]

    data_index_Orca,total_length= add_contexts_Orca(dataset_Orca,ctxs,data_index_Orca,total_length,tokenizer,Orca_Q_length,Orca_A_length,orca_length)


    ## context end:
    Instruction = token(f"\n Provide a detailed analysis based on the Context and offer your answers. Question: ",tokenizer)
    total_length += len(Instruction)    

    data_index_slimpajama,total_length= add_contexts_slim(dataset_Slim, ctxs, data_index_slimpajama,total_length,length,tokenizer)


    random.shuffle(ctxs)
    # Reconstruct the mapping of question IDs to answer IDs after shuffling
 
    for i, ctx in enumerate(ctxs):
        if ctx.get('type') == 'question':
            questions_ctxs.append({
                "ID": ctx['Q_ID'],
                "text": ctx['GPT_answer'],
                "type": 'Answer',
                "location" : i+1
            })

    offset = 0
    random.shuffle(questions_ctxs)
    for new_ctx in questions_ctxs:
        start_position = new_ctx['location'] + offset
        # 检查起始位置是否超出列表长度，如果超出，则设置插入位置为列表末尾
        if start_position >= len(ctxs):
            insert_position = len(ctxs)
        else:
            insert_position = random.randint(start_position, len(ctxs))
        ctxs.insert(insert_position, new_ctx)
        offset += 1  

    # get end_data
    for item in ctxs:
        if item.get('type') == 'Answer':
            input_ids += item['ID']
            labels += [-100] * len(item['ID'])
            input_ids += item['text']
            labels += item['text']
        if item.get('type') == False or item.get('type') == 'question' or item.get('type') == 'end':
            input_ids += item['ID']
            labels += [-100] * len(item['ID'])
    
    # print("=="*100)
    # print('input len:',len(input_ids))
    # print('labels len:',len(labels))
    # print('length: ',total_length)

    input_ids += Instruction 
    labels += [-100] * len(Instruction)
  
    
    for item in ctxs:
        if item.get('type') == 'end':
            input_ids += item['Q_ID']
            labels += item['Q_ID']

    # print("=="*100)
    # print('input len:',len(input_ids))
    # print('labels len:',len(labels))
    # print('length: ',total_length)
    return {'input_ids':input_ids,'labels':labels}, {'Orca_Q':Orca_Q_length,'Orca_A':Orca_A_length,'Squad_Q': Squad_Q_length,'Squad_A':Squad_A_length},data_index_slimpajama,data_index_Orca


def add_contexts_slim(dataset, ctxs, data_index,total_length,length,tokenizer):
    """ Adds contexts from the dataset while checking the token conditions """
    total_tokens = total_length
    max_tokens= length
    while total_tokens < max_tokens:
        text = dataset[data_index]['input_ids']
        data_index = data_index+1
        context_id = generate_unique_string()
        c_id = token(f"\nID: {context_id}, ",tokenizer)
        total_tokens += len(c_id+text)
        if (total_tokens <= max_tokens):
            ctxs.append({"ID": c_id+text,"type": False})
        else:
            all = c_id + text
            ctxs.append({"ID": all[:max_tokens-total_tokens], "type": False})
            total_tokens = length
          # Update the token count assuming 1 token per word
    return data_index,total_tokens


def add_contexts_Orca(dataset, ctxs, data_index,total_length,tokenizer,Orca_Q_length,Orca_A_length,orca_length
    ):
    """ Adds contexts from the dataset while checking the token conditions """

    # features: ['question_input_ids', 'response_input_ids'],
    total_tokens = total_length
    num_QA = 0
    max_QA = 80
    while num_QA  < max_QA:
        num_QA += 1
        context_id = generate_unique_string()
        question_id = generate_unique_string()
        text = dataset[data_index]['question_input_ids']
        response = dataset[data_index]['response_input_ids']
        data_index += 1
        c_id = token(f"\nID: {context_id}, ",tokenizer)
        q_id = token(f"\nID: {question_id}, ",tokenizer)  
        q_start = token(f" Answer question {context_id}: ",tokenizer)  
        ctxs.append({
                "ID":  c_id+text,
                # "text": text,
                "type": 'question',
                "Q_ID": q_id+q_start,
                "GPT_answer":response,
            })
        total_tokens += len(c_id+text+q_id+q_start+response)
        Orca_Q_length += [len(text)]
        Orca_A_length += [len(response)]

        if total_tokens > orca_length:
            # print(num_QA)
            break
    return data_index,total_tokens



def load_json_data(filepath):
    """ Loads JSON data from a file """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error loading JSON data from {filepath}: {e}")
        return []
    

def expand_and_enhance_data(data,data_index_slimpajama,data_index_Orca,length,orca_length,data_number):
    """ Expands each item by 32 times and enhances with ctxs """
    enhanced_data = []
    length_distri = []

    while len(enhanced_data) < data_number:
        for item in data:
            if len(enhanced_data) >= data_number:
                break
            other_items = random.sample(data, 11)
            other_items.insert(0, item)  # Include the current item as the first item
            end_data, length_distribution, data_index_slimpajama, data_index_Orca = add_ctxs_to_item(
                other_items,data_index_slimpajama,data_index_Orca,length,orca_length)
            enhanced_data.append(end_data)
            length_distri.append(length_distribution)
            if data_index_Orca + 100 > len(dataset_Orca):
                data_index_Orca = 0
        # print('data_index_Orca: ',data_index_Orca)
    random.shuffle(enhanced_data)  # Shuffle all items to mix them up
    return enhanced_data,length_distri,data_index_slimpajama,data_index_Orca



def save_data_1(data, output_file):
    """ Save the data into a parquet file """
    df = pd.DataFrame(data, columns=['input_ids', 'labels'])
    df.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}")

def save_data_2(data, output_file):
    """ Save the data into a parquet file """
    df = pd.DataFrame(data, columns=['Orca_Q','Orca_A','Squad_Q','Squad_A'])
    df.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}")
    


def process_and_save_data(length,orca_length,data_number):
    merged_file_path = 'Squad_COT_data.json'
    output_file = 'F2F_data.parquet'
    output_file_2 = 'distribution.parquet'
    merged_data = load_json_data(merged_file_path)
    merged_data = merged_data[20:4020]
    if merged_data:
        data_index_slimpajama = 0
        data_index_Orca =0 
        enhanced_data,length_distri,data_index_slimpajama,data_index_Orca= expand_and_enhance_data(
         merged_data,data_index_slimpajama,data_index_Orca,length,orca_length,data_number)
        print(data_index_slimpajama,data_index_Orca)
        save_data_1(enhanced_data, output_file)
        save_data_2(length_distri, output_file_2)
    else:
        print("Failed to load data or context data.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save data.")
    parser.add_argument('--length', type=int, default=32768, help='Total length for data expansion and enhancement.')
    parser.add_argument('--orca_length', type=int, default=16400, help='Length parameter specific to Orca data.')
    parser.add_argument('--data_number', type=int, default=16000, help='The number of data entries to process. ')

    args = parser.parse_args()

    process_and_save_data(
        args.length,
        args.orca_length,
        args.data_number
    )
    # process_and_save_data(32768,16400,20)