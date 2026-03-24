import pickle

pickle.dump(xgb_model,  open("models/XGBoost.pkl", "wb"))
pickle.dump(rf_model,   open("models/RandomForest.pkl", "wb"))
pickle.dump(knn_model,  open("models/KNN.pkl", "wb"))
pickle.dump(scaler,     open("models/scaler.pkl", "wb"))
pickle.dump(le,         open("models/label_encoder.pkl (1)", "wb"))
