# From-Fragment-to-Fabric-Long-Context-Scaling-with-Short-Instruction-Tuning-Data



# Project Setup and Data Processing Instructions

## Environment Setup

Ensure that you are using Python 3.10.14 with CUDA 12.4 for this project. Follow these steps to prepare your environment and install the necessary libraries:

1. **Download Required Libraries:**
   Use `pip` to install all the required libraries listed in the `requirements.txt` file:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

## Data Processing with F2F

Navigate to the F2F directory and perform data processing using the following scripts:

1. **Token Generation:**
   Run the token generation scripts for both Orca and Slim datasets:
   \`\`\`
   python token_orca.py
   python token_slim.py
   \`\`\`

2. **Dataset Building:**
   Generate the dataset with specified parameters for length, Orca subset length, and the number of data entries:
   \`\`\`
   python dataset_building.py --length 32768 --orca_length 16400 --data_number 16000
   \`\`\`

Follow these steps in sequence to ensure that the data is processed correctly for your project.



# Evalution
#
#