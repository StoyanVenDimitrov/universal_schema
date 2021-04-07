# universal_schema for relations extraction from manuals 

Create a virtual environment and install the requirements with
* pip install -r requirements.pip


For data preparation, use preprocessing/prepare_train_input.py

For training and evaluation, use scr/sem_rel.py

To trigger parts of the dataset creation process, delete the respective .json files from the data repository.
Please note that the process can last for hours due to the many API calls.