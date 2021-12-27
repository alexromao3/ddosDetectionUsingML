# DDoS Detection Using Machine Learning Techniques
DDoS Detection using Machine Learning techniques and it's classification

This project is based on the use of machine learning algorithms to train a model using data related to network traffic to determine if we are facing a ddos attack or not.
Then, we can input data and get a prediction about the network status based on the trained model.

This project is composed by two parts:

- train.py that loads the dataset, treats the data within the dataset, trains the model and evaluate the model with the choosed classification algorithm.
- test.py that imports the model, tests the dataset and makes a prediction based on the model trained with the input of 7 parameters. This parameters are: "Fwd Seg Size Avg", "Flow IAT Min", "Flow Duration", "Tot Fwd Pkts", "Pkt Size Avg", "Src Port", "Init Bwd Win Byts", by this order.



Project Authored by: Alexandre Romão

## Content:
- Requirements
- How to run?
- Running
- Dataset

## Requirements:
- To run the program you need to have any python version installed 3.6 or above.
- You need to have these python packages: sklearn, pandas, numpy and pickle installed to run the python files listed above.
- You can install them like this:

  - ``` pip install -U scikit-learn ```
  - ``` pip install pandas ```
  - ``` pip install numpy ```


## How to run?

- Go to your OS command prompt or anaconda prompt and change directory to the directory where the .py files are stored.
- Then, type ``` python train.py ``` and the program will start the execution.

Note: Please create a folder named 'saved_model' in the directory of the python scripts.
Note: Please create a folder named 'ddos_balanced' in the directory of the python scripts, download the dataset mentioned in the 'Dataset' part and store it in the folder.

## Running

- The program will start reading the dataset. ``` Reading dataset.. ``` This may take a while!!
- Then, you get the information that it finished. ``` Dataset was successfully read. ```
- After this step, the dataset will be treated because some data can be damaged or some rows/columns empty.
  - ``` Started the process of treating data of the dataset... ```
  - ``` Process concluded with success. ```
- Finally, the train will start. A model should be picked between these:

    ``` 
        0 - DecisionTreeClassifier 
        1 - GaussianNB 
        2 - RandomForest 
        3 - MLPClassifier
    ```
- You will pick the size of the dataset you want to train: ``` What's the size from the dataset you want to train? (between 0 and 100) ```
- The model will be fitted then. ``` Model fitted. ```
- The program shows you some classification metrics in order to check the quality of the train / test model.
    ```
      - Accuracy of the model
      - Precision of the model
      - Recall of the model
      - F1 of the model
    ```
    
- Then, you can save the fitted model or not in the directory: ``` ./saved_model/model_data.sav ```
- When finished, you can get a prediction with test.py. To do this, type ``` python test.py ``` plus 7 parameters. These parameters are mandatory for the program to work. 
  - e.g. ``` python test.py 232209 0 19500 2500 256 4500 900 ```
  - The output: ``` ['Benign'] ```

## Dataset:
- The dataset used can be found here: https://www.kaggle.com/devendra416/ddos-datasets
- This dataset is a combination of multiple datasets, that were combined and balanced. This dataset contains 84 features.
- The explanation for each parameter of the dataset can be found here: https://www.unb.ca/cic/datasets/ids-2018.html
