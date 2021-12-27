import pickle
import sys

def model_test(attributes):
    model = pickle.load(open('./saved_model/model_data.sav', 'rb'))
    result = model.predict([attributes])
    print(result)


if __name__ == "__main__":
    model_test(sys.argv[1:])