import pickle as pkl 
import numpy as np 
import pandas as pd

with open('models/test_model.pkl', 'rb') as f:
    model = pkl.load(f)

cols = ['default_profile','favourites_count','followers_count','friends_count','listed_count','protected','statuses_count','verified','age_at_collection','generic']

# feature_dict = {
#     'default_profile': 0,
#     'favourites_count': 0,
#     'followers_count': 0,
#     'friends_count': 0,
#     'listed_count': 0,
#     'protected': 0,
#     'statuses_count': 0,
#     'verified': 0,
#     'age_at_collection': 0,
#     'generic': 0}

# def make_prediction(feature_dict): 
#     # convert dict to numerics
#     for key in feature_dict.keys(): 
#         # if feature_dict[key] == 'Yes': 
#         #     feature_dict[key] = 1
#         # if feature_dict[key] == 'No':
#         #     feature_dict[key] = 0
#         feature_dict[key] = float(feature_dict[key])


#     x_input = []
#     for col in cols:
#         x_input_ = float(feature_dict.get(col,0))
#         x_input.append(x_input_)
    

#     pred_df = pd.DataFrame(feature_dict, index=[0])
#     # pred_df = pred_df[cols]
#     prediction = float(model.predict(pred_df))
#     # prediction = 5
#     return (x_input, [prediction])

def make_prediction(feature_dict):
        # convert dict to numerics
    for key in feature_dict.keys(): 
        feature_dict[key] = float(feature_dict[key])
    df = pd.DataFrame(feature_dict, index=[0])
    prediction = model.predict(df)[0]
    return (df, prediction)
    

## Test functionality    
if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what setting all params to 0 predicts")
    features = {f: '0' for f in cols}
    print('Features are')
    pprint(features)

    x_input, pred = make_prediction(features)
    print(f'Input values: {x_input}')
    print(f'Output prediction: {pred}')
    