# 1. Library imports
import uvicorn
from fastapi import FastAPI
from CreditBM import CreditBM
import numpy as np
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE


# 2. Create the app object
app = FastAPI()
pickle_in = open("mypicklefile", "rb")
classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hi, APP of Credit scoring'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome . We try to predict if you will be in default or not': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Credit with the confidence
@app.post('/predict')
def predict_creditstatus(data: CreditBM):
    data = data.dict()
    print(data)
    CREDIT_ANNUITY_RATIO= data['CREDIT_ANNUITY_RATIO']
    REGION_POPULATION_RELATIVE= data['REGION_POPULATION_RELATIVE']
    AMT_REQ_CREDIT_BUREAU_YEAR= data['AMT_REQ_CREDIT_BUREAU_YEAR']
    PREVIOUS_LOANS_COUNT= data['PREVIOUS_LOANS_COUNT']
    CREDIT_GOODS_RATIO= data['CREDIT_GOODS_RATIO']
    EXT_SOURCE_SUM= data['EXT_SOURCE_SUM']
    DAYS_BIRTH= data['DAYS_BIRTH']
    PREV_APPL_MEAN_CNT_PAYMENT= data['PREV_APPL_MEAN_CNT_PAYMENT']
    PREV_BUR_MEAN_AMT_CREDIT_SUM_DEBT= data['PREV_BUR_MEAN_AMT_CREDIT_SUM_DEBT']
    DAYS_REGISTRATION= data['DAYS_REGISTRATION']
    CAR_EMPLOYED_RATIO= data['CAR_EMPLOYED_RATIO']
    DAYS_ID_PUBLISH= data['DAYS_ID_PUBLISH']
    PREV_APPL_MEAN_SELLERPLACE_AREA= data['PREV_APPL_MEAN_SELLERPLACE_AREA']
    EXT_SOURCE_1= data['EXT_SOURCE_1']
    EXT_SOURCE_2= data['EXT_SOURCE_2']
    OBS_30_CREDIT_RATIO= data['OBS_30_CREDIT_RATIO']
    PREVIOUS_APPLICATION_COUNT= data['PREVIOUS_APPLICATION_COUNT']
    PREV_APPL_MEAN_INSTALL_MEAN_AMT_INSTALMENT= data['PREV_APPL_MEAN_INSTALL_MEAN_AMT_INSTALMENT']
    PREV_BUR_MEAN_AMT_CREDIT_SUM= data['PREV_BUR_MEAN_AMT_CREDIT_SUM']
    DAYS_LAST_PHONE_CHANGE= data['DAYS_LAST_PHONE_CHANGE']
    ANNUITY_INCOME_RATIO= data['ANNUITY_INCOME_RATIO']
    AMT_CREDIT= data['AMT_CREDIT']

    prediction = classifier.predict_proba([[
        CREDIT_ANNUITY_RATIO,
    REGION_POPULATION_RELATIVE,
    AMT_REQ_CREDIT_BUREAU_YEAR,
    PREVIOUS_LOANS_COUNT,
    CREDIT_GOODS_RATIO,
    EXT_SOURCE_SUM,
    DAYS_BIRTH,
    PREV_APPL_MEAN_CNT_PAYMENT,
    PREV_BUR_MEAN_AMT_CREDIT_SUM_DEBT,
    DAYS_REGISTRATION,
    CAR_EMPLOYED_RATIO,
    DAYS_ID_PUBLISH,
    PREV_APPL_MEAN_SELLERPLACE_AREA,
    EXT_SOURCE_1,
    EXT_SOURCE_2,
    OBS_30_CREDIT_RATIO,
    PREVIOUS_APPLICATION_COUNT,
    PREV_APPL_MEAN_INSTALL_MEAN_AMT_INSTALMENT,
    PREV_BUR_MEAN_AMT_CREDIT_SUM,
    DAYS_LAST_PHONE_CHANGE,
    ANNUITY_INCOME_RATIO,
    AMT_CREDIT,
                     ]])
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload
