# SARS-CoV-2 <img src="logo/Logo_small.png" alt="Logo_small.png" width="24"/>
## Machine Learning Drug Hunter
A straightforward App that combines experimental activity data, molecular descriptors and machine 
learning for classifying potential drug candidates against the SARS-CoV-2 Main Protease (MPro). 
### Requirements
`Linux` or `Mac` operating system    
`Anaconda` or `Miniconda3`    
More info on **requirements.txt**    
### Installation
Firstly, please make sure you have `Anaconda` or `Miniconda3` installed. Then, you can follow the steps bellow:    
#### Create a virtual environment
`conda create -n drughunter python=3.7.9`     
`conda activate drughunter`    
#### Install the required packages
`conda install -c anaconda scikit-learn`    
`conda install -c anaconda seaborn`    
`conda install -c conda-forge imbalanced-learn`  
`conda install -c conda-forge xgboost`     
`conda install -c conda-forge streamlit`    
`conda install -c rdkit rdkit`   
`conda install -c mordred-descriptor mordred`   
#### If needed, update `protobuf` in your virtual environment
`pip install --upgrade protobuf`   
### Instructions
#### Open the Streamlit App in your local browser, by running
`streamlit run streamlit_covid_view.py`
