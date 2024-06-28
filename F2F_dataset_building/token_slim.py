import os
import pandas as pd
from hashlib import sha256
from datasets import load_dataset
from transformers import LlamaTokenizer
from multiprocessing import Pool


def tokenize_data(text_id):
    """Tokenize text data and generate ID."""
    text, tokenizer = text_id
    tokenized_text = tokenizer(text, truncation=True, max_length=1000)
    # Convert BatchEncoding to dictionary and extract necessary fields
    tokenized_text_data = {
        'input_ids': tokenized_text['input_ids'],

    }
    return tokenized_text_data

def filter_by_word_count(example):
    """Calculate the number of words in the text."""
    word_count = len(example['text'].split())
    return 50 < word_count < 800

def main(data_row):
    # Load dataset
    dataset = load_dataset("DKYoon/SlimPajama-6B", split='train')
    # Take the first 1 million rows that meet the word count criteria
    # num = 5000000
    num = data_row
    dataset = dataset.select(range(num)).filter(filter_by_word_count)
    print(len(dataset))
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Prepare data for multiprocessing
    texts = [(text, tokenizer) for text in dataset['text']]  # Assuming 'text' field exists

    # Use multiprocessing to tokenize data
    with Pool(os.cpu_count()) as pool:
        results = pool.map(tokenize_data, texts)

    # Convert results to DataFrame and save as Parquet file
    df = pd.DataFrame(results)
    df.to_parquet('tokenized_data_SlimPajama.parquet')

    print("Data tokenized and saved successfully as Parquet.")

if __name__ == "__main__":
    # data_number = 5000
    main(5000)
