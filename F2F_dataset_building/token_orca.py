import os
import pandas as pd
from hashlib import sha256
from datasets import load_dataset
from transformers import LlamaTokenizer
from multiprocessing import Pool

def filter_criteria(example):
    # 计算单词数量
    question_words = len(example['question'].split())
    response_words = len(example['response'].split())
    # 检查 id 是否以 'cot' 开头
    question_lower = example['question'].lower()
    contains_translate = "translate" in question_lower or "translation" in question_lower
    
    # 判断单词数量是否都小于1000且question中不含"translate"和"translation"
    return 80 < question_words < 600 and 80 < response_words < 600 and not contains_translate


def tokenize_data(example, tokenizer):
    """Tokenize question and response data."""
    tokenized_question = tokenizer(example['question'])
    tokenized_response = tokenizer(example['response'])
    return {
        'question_input_ids': tokenized_question['input_ids'],
        'response_input_ids': tokenized_response['input_ids']
    }

def main():
    # Load dataset
    dataset = load_dataset("Open-Orca/OpenOrca", split='train')
    # Filter dataset based on criteria
    filtered_dataset = dataset.filter(filter_criteria)
    print(filtered_dataset)

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Prepare data for multiprocessing
    pool = Pool(os.cpu_count())
    results = [pool.apply_async(tokenize_data, (example, tokenizer)) for example in filtered_dataset]
    results = [result.get() for result in results]
    pool.close()
    pool.join()

    # Convert results to DataFrame and save as Parquet file
    df = pd.DataFrame(results)
    df.to_parquet('tokenized_data_Orca_all.parquet')

    print("Data tokenized and saved successfully as Parquet.")

if __name__ == "__main__":
    main()
