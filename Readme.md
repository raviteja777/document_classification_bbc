# Document Classification ML
- BBC news dataset has been used to classify models
  - Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
  - Natural Classes: 5 (business, entertainment, politics, sport, tech)
  - http://mlg.ucd.ie/datasets/bbc.html
  - D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
- Models trained:  Naive Bayes, KNN, SVM and Random Forest 
- GridSearch with cross validation = 10

## Training the model 
- place the data files in the data folder
- Run main.py to train the model 
- The models will save to save_model folder
- results and metrics will be displayed on console 

## Predicting using a saved model
- run test_predict.py by proving the full path to the saved model file 
- results will be displayed on console 
