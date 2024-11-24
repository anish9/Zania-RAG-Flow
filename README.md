# Zania-RAG-Flow
a simple solution 

## Summary:
* Parse PDF using a suitable method.
* Build knowledge chunks.
* Create embeddings for the chunks.
* In test time use suitable distance metric to retrieve chunks.
* Prepare the context and question and pass to LLM.

## Setup:
 * Python3.11<=
 * pipenv for virtualenv
```
pipenv shell --python 3.11
pip install -r requirements.txt
```
## Usage
```
python cli_script.py

or

python cli_script.py --pdf_path ./sources/handbook.pdf --questions "what is the org name? what is the termination policy here?"
```
