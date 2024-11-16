# SARS-CoV-2 <img src="media/Logo_small.png" alt="Logo_small.png" width="24"/>
## Machine Learning Drug Hunter
A straightforward App that combines experimental activity data, molecular descriptors and machine 
learning for classifying potential drug candidates against the SARS-CoV-2 Main Protease (MPro).

## Running online
Feel free to run the online app in the Streamlit servers by clicking the link on the repo description.

## Running locally
### Installation
Firstly, please make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, you can follow the steps bellow:    
#### Create a virtual environment
`conda create -n drughunter python=3.7.9`     
`conda activate drughunter`    
#### Install the required packages
`pip install -r requirements.txt`    
`conda install -c rdkit rdkit`    
### Instructions
#### Open the Streamlit App in your local browser, by running
`streamlit run streamlit_covid_view.py`
