
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from CreditData import CreditData
import numpy as np
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE


# 2. Create the app object
app = FastAPI()
pickle_in = open("finalized_model_fastapi.pkl", "rb")
classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, API of Prêt à dépenser Credit scoring'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to the Credit Scoring API. We try to predict if you will be in default or not': f'{name}'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Credit with the confidence
@app.post('/predict')
def predict_creditstatus(data: CreditData):
    data = data.dict()
    print(data)
    EXT_SOURCE_3 = data['EXT_SOURCE_3']
    DAYS_LAST_PHONE_CHANGE= data['DAYS_LAST_PHONE_CHANGE']
    REGION_POPULATION_RELATIVE= data['REGION_POPULATION_RELATIVE']
    TOTALAREA_MODE= data['TOTALAREA_MODE']
    HOUR_APPR_PROCESS_START= data['HOUR_APPR_PROCESS_START']
    DAYS_BIRTH= data['DAYS_BIRTH']
    DAYS_EMPLOYED= data['DAYS_EMPLOYED']
    AMT_INCOME_TOTAL= data['AMT_INCOME_TOTAL']
    OBS_60_CNT_SOCIAL_CIRCLE= data['OBS_60_CNT_SOCIAL_CIRCLE']
    FLAG_DOCUMENT_3= data['FLAG_DOCUMENT_3']
    OBS_30_CNT_SOCIAL_CIRCLE= data['OBS_30_CNT_SOCIAL_CIRCLE']
    EXT_SOURCE_2= data['EXT_SOURCE_2']
    DEF_60_CNT_SOCIAL_CIRCLE= data['DEF_60_CNT_SOCIAL_CIRCLE']
    AMT_ANNUITY= data['AMT_ANNUITY']
    DAYS_ID_PUBLISH= data['DAYS_ID_PUBLISH']
    DEF_30_CNT_SOCIAL_CIRCLE= data['DEF_30_CNT_SOCIAL_CIRCLE']
    AMT_GOODS_PRICE= data['AMT_GOODS_PRICE']

    prediction = classifier.predict_proba([[
                    EXT_SOURCE_3,
                    DAYS_LAST_PHONE_CHANGE,
                    REGION_POPULATION_RELATIVE,
                    TOTALAREA_MODE,
                    HOUR_APPR_PROCESS_START,
                    DAYS_BIRTH,
                    DAYS_EMPLOYED,
                    AMT_INCOME_TOTAL,
                    OBS_60_CNT_SOCIAL_CIRCLE,
                    FLAG_DOCUMENT_3,
                    OBS_30_CNT_SOCIAL_CIRCLE,
                    EXT_SOURCE_2,
                    DEF_60_CNT_SOCIAL_CIRCLE,
                    AMT_ANNUITY,
                    DAYS_ID_PUBLISH,
                    DEF_30_CNT_SOCIAL_CIRCLE,
                    AMT_GOODS_PRICE
                     ]])
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload
