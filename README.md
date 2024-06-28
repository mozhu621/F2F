# From-Fragment-to-Fabric-Long-Context-Scaling-with-Short-Instruction-Tuning-Data



# Project Setup and Data Processing Instructions

## Environment Setup

Ensure that you are using Python 3.10.14 with CUDA 12.4 for this project. Follow these steps to prepare your environment and install the necessary libraries:

1. **Download Required Libraries:**
   Use `pip` to install all the required libraries listed in the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

## Data Processing with F2F

Navigate to the F2F directory and perform data processing using the following scripts:

1. **Token Generation:**
   Run the token generation scripts for both Orca and Slim datasets:
  ```
   python token_orca.py
   python token_slim.py
  ```

2. **Dataset Building:**
   Generate the dataset with specified parameters for length, Orca subset length, and the number of data entries:
  ```
   python dataset_building.py --length 32768 --orca_length 16400 --data_number 16000
  ```

Follow these steps in sequence to ensure that the data is processed correctly for your project.



## Evaluation

After completing the data processing, the model can be evaluated using two distinct methods: NIAH and LongBench. Follow these instructions to carry out the evaluations:

### NIAH Evaluation

1. **Run the NIAH Script:**
   Navigate to the NIAH directory and run the evaluation script. Replace `XX` with the appropriate CUDA device ID and `XXX` with your model's path:
   ```
   CUDA_VISIBLE_DEVICES=XX python needle_in_haystack.py --s 1000 --e 32000 --n 32 --model_path XXX
  ```
2. **Visualization:**
   Visualize the results using the Jupyter notebook provided:
     ```
   CreateVizFromLLMTesting.ipynb
  ```

### LongBench Evaluation

1. **Run the Prediction Script:**
   Use the following command to run the prediction script with your model. Replace `xx` with your CUDA device ID and `xxxx` with your model's path:
  ```
   CUDA_VISIBLE_DEVICES=xx python pred.py --model xxxx --max_length 31500
  ```

2. **Evaluation and Record:**
   Evaluate the results and save them as a CSV file using the provided Jupyter notebook:
  ```
   eval_record.ipynb
  ```

Follow these steps to ensure comprehensive evaluation of your model based on both performance and accuracy.
