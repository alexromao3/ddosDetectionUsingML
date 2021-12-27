from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dtypes = {
    'Src IP': 'category',
    'Src Port': 'uint16',
    'Dst IP': 'category',
    'Dst Port': 'uint16',
    'Protocol': 'category',
    'Flow Duration': 'uint32',
    'Tot Fwd Pkts': 'uint32',
    'Tot Bwd Pkts': 'uint32',
    'TotLen Fwd Pkts': 'float32',
    'TotLen Bwd Pkts': 'float32',
    'Fwd Pkt Len Max': 'float32',
    'Fwd Pkt Len Min': 'float32',
    'Fwd Pkt Len Mean': 'float32',
    'Fwd Pkt Len Std': 'float32',
    'Bwd Pkt Len Max': 'float32',
    'Bwd Pkt Len Min': 'float32',
    'Bwd Pkt Len Mean': 'float32',
    'Bwd Pkt Len Std': 'float32',
    'Flow Byts/s': 'float32',
    'Flow Pkts/s': 'float32',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'float32',
    'Flow IAT Min': 'float32',
    'Fwd IAT Tot': 'float32',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'float32',
    'Fwd IAT Min': 'float32',
    'Bwd IAT Tot': 'float32',
    'Bwd IAT Mean': 'float32',
    'Bwd IAT Std': 'float32',
    'Bwd IAT Max': 'float32',
    'Bwd IAT Min': 'float32',
    'Fwd PSH Flags': 'category',
    'Bwd PSH Flags': 'category',
    'Fwd URG Flags': 'category',
    'Bwd URG Flags': 'category',
    'Fwd Header Len': 'uint32',
    'Bwd Header Len': 'uint32',
    'Fwd Pkts/s': 'float32',
    'Bwd Pkts/s': 'float32',
    'Pkt Len Min': 'float32',
    'Pkt Len Max': 'float32',
    'Pkt Len Mean': 'float32',
    'Pkt Len Std': 'float32',
    'Pkt Len Var': 'float32',
    'FIN Flag Cnt': 'category',
    'SYN Flag Cnt': 'category',
    'RST Flag Cnt': 'category',
    'PSH Flag Cnt': 'category',
    'ACK Flag Cnt': 'category',
    'URG Flag Cnt': 'category',
    'CWE Flag Count': 'category',
    'ECE Flag Cnt': 'category',
    'Down/Up Ratio': 'float32',
    'Pkt Size Avg': 'float32',
    'Fwd Seg Size Avg': 'float32',
    'Bwd Seg Size Avg': 'float32',
    'Fwd Byts/b Avg': 'uint32',
    'Fwd Pkts/b Avg': 'uint32',
    'Fwd Blk Rate Avg': 'uint32',
    'Bwd Byts/b Avg': 'uint32',
    'Bwd Pkts/b Avg': 'uint32',
    'Bwd Blk Rate Avg': 'uint32',
    'Subflow Fwd Pkts': 'uint32',
    'Subflow Fwd Byts': 'uint32',
    'Subflow Bwd Pkts': 'uint32',
    'Subflow Bwd Byts': 'uint32',
    'Init Fwd Win Byts': 'uint32',
    'Init Bwd Win Byts': 'uint32',
    'Fwd Act Data Pkts': 'uint32',
    'Fwd Seg Size Min': 'uint32',
    'Active Mean': 'float32',
    'Active Std': 'float32',
    'Active Max': 'float32',
    'Active Min': 'float32',
    'Idle Mean': 'float32',
    'Idle Std': 'float32',
    'Idle Max': 'float32',
    'Idle Min': 'float32',
    'Label': 'category'
}

print("Reading dataset..")

df = pd.read_csv('./ddos_balanced/final_dataset.csv',
     dtype=dtypes,
     parse_dates=['Timestamp'],
     usecols=[*dtypes.keys(), 'Timestamp'],
     engine='c',
     low_memory=True
     )

print("Dataset was successfully read.")

def treatDataset(df):

    print("Started the process of treating data of the dataset...")
    #Drop columns that have only 1 value viewed
    colsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])

    #Drop columns where missing values are more than 40% and Drop rows where a column missing values are no more than 5%
    missing = df.isna().sum()
    missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
    colsToDrop = np.union1d(colsToDrop, missing[missing['% of total'] >= 40].index.values)
    dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values

    #Handling faulty data
    df['Flow Byts/s'].replace(np.inf, np.nan, inplace=True)
    df['Flow Pkts/s'].replace(np.inf, np.nan, inplace=True)
    dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s', 'Flow Pkts/s'])

    #Drop the columns
    df.drop(columns=colsToDrop, inplace=True)
    df.dropna(subset=dropnaCols, inplace=True)
    print("Process concluded with success. \n")
    return df


def train_model(df):
    print("Model Training:")
    features = ["Fwd Seg Size Avg", "Flow IAT Min", "Flow Duration", "Tot Fwd Pkts", "Pkt Size Avg", "Src Port", "Init Bwd Win Byts"]
    target = "Label"

    #Fwd Seg Size Avg - Average size observed in the forward direction
    #Flow IAT Min - Minimum time between two flows
    #Flow Duration - Duration of flow
    #Tot Fwd Pkts - Total packets in the forward direction
    #Pkt Size Avg - Average size of packet
    #Src Port - Port of the source
    #Label - ddos or benign (goal of prediction)

    x = df.loc[:,features]
    y = df.loc[:,target]

    print("Wich model you want to use: ")
    print("0 - DecisionTreeClassifier")
    print("1 - GaussianNB")
    print("2 - RandomForest")
    print("3 - MLPClassifier")
    choice = input("Model: ")
    choice.lower()

    if (str(choice) == "0"):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        modelname = "DecisionTree"
    elif (str(choice) == "1"):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        modelname = "GaussianNB"
    elif (str(choice) == "2"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        modelname = "RandomForest" 
    elif (str(choice) == "3"):
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        modelname = "MLP"
    else:
        print("You picked an invalid option. Using default model 0 to training.")
        model = DecisionTreeClassifier()

    print("Option valid. \n")
    sizeTrain = input("What's the size from the dataset you want to train? (between 0 and 100): ")
    
    print("Fitting model..")

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=1.0-float(sizeTrain)/100)   

    if modelname == "MLP":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    #Fitting the model
    model.fit(x_train, y_train)
    print("Model fitted. \n")

    #Accuracy - number of correct predictions divided by the number of total predictions
    #Precision - number of true positives divided by the total number of positive predictions
    #Recall - number of true positives that were recalled (found)
    #F1 - weighted average of the precision and recall values 
    
    #Getting results
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)*100
    print("Accuracy of the model "+modelname+" is: ", score) 
    precision = precision_score(y_test, y_pred, pos_label='Benign')*100
    print("Precision of the model "+modelname+" is: ", precision) 
    recall = recall_score(y_test, y_pred, pos_label='Benign')*100
    print("Recall of the model "+modelname+ " is: ", recall)
    f1 = f1_score(y_test, y_pred, pos_label='Benign')*100
    print("F1 score of the model "+modelname+ " is: ", f1)
    matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print( matrix)

    #Structure of the matrix
    #[  True Negative   False Positive ]
    #[ False Negative   True Positive  ]

    print("Save the fitted model?(y/n):")
    choice = input().lower()
    if choice == "y":
        pickle.dump(model, open("./saved_model/model_data.sav", 'wb'))

if __name__ == "__main__":
    dfTreated = treatDataset(df)
    train_model(dfTreated)