# rkovalev_int_1607

Test assignment for the internship project "Improving Writing Assistance at JetBrains AI" (2025, 3.0).

The report on the assignment is [in this file](1607_writing_assistance_report.pdf).

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

Check the prediction in the .csv reports in the _results_ folder.

6. To evaluate formality with a Llama model, run all the cells from the [llama_formality notebook](approaches/get_formality_llama.ipynb) in Google Colab using the available GPU. Set the number of rows to an integer or None to evaluate the whole dataset.

* Make sure to use your HuggingFace access token when loading the dataset;

* You need to be granted access to the model (fill out the form [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)).
