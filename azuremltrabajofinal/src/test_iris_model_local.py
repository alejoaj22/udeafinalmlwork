import pickle
import numpy as np
import json



# Load the trained model from current directory
with open('output/iris_model.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)

if __name__ == "__main__":

 new_record = np.array([[6.7, 3.3, 5.7, 1.2]])

 predict_result = knn.predict(new_record)

 setosa_clases = ['I. Setosa', 'I. Versicolor', 'I. Virginica']
 # return the result back
 result = json.dumps({"predicted_class": setosa_clases[int(predict_result)]})

 print(result)

