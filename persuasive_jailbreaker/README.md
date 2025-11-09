# Johnny Persuasive Jailbreaker Demo

## Environment Setup

Navigate to the project directory:
```bash
cd johnny_persuasive_jailbreaker
```

Create the Conda environment:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate pj-demo
```

Install additional Python dependencies (if not already included):
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY="YOUR-KEY-GOES-HERE"
```

Open and run the Jupyter Notebook located at:
```bash
PAP_Better_Incontext_Sample/test.ipynb
```

Also, you can change the model in 
```bash
mutation_utils.py
```