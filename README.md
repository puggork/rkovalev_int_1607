# int_proj_2025_3.0_1607_writing_assistance
Test task for the internship project "Improving Writing Assistance at JetBrains AI" (2025, 3.0).

## Reproducing the results locally

1. Clone the repository.

2. Change current directory to the repository.

3. Install the requirements:

`pip install -r requirements.txt`

4. Run the script to download the [dataset](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores):

`python data/load_and_prepare_ds.py`

5. Run the main script, choosing one of the options:

* -a / --approach - Choose the formality detection approach ("flesch" for Flesch Reading Ease score or "xlmr" for the XLM-Roberta-based classifier);

* -nr / --nrows - Number of rows to evaluate from the test set.

For example:

`python main.py -a xlmr -nr 25`

6. To evaluate formality with a Llama model, run all the cells from the approaches/llama_formality.ipynb in Google Colab using the available GPU. Set the number of rows to an integer or None to evaluate the whole dataset.