# -*- coding: utf-8 -*-
"""
@author: MEHAN
"""
from pydantic import BaseModel

# 2. Class which describes Bank Notes measurements
class CreditBM(BaseModel):
    CREDIT_ANNUITY_RATIO: float
    REGION_POPULATION_RELATIVE: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    PREVIOUS_LOANS_COUNT: float
    CREDIT_GOODS_RATIO: float
    EXT_SOURCE_SUM: float
    DAYS_BIRTH: float
    PREV_APPL_MEAN_CNT_PAYMENT : float
    PREV_BUR_MEAN_AMT_CREDIT_SUM_DEBT: float
    DAYS_REGISTRATION: float
    CAR_EMPLOYED_RATIO: float
    DAYS_ID_PUBLISH : float
    PREV_APPL_MEAN_SELLERPLACE_AREA : float
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    OBS_30_CREDIT_RATIO: float
    PREVIOUS_APPLICATION_COUNT: float
    PREV_APPL_MEAN_INSTALL_MEAN_AMT_INSTALMENT: float
    PREV_BUR_MEAN_AMT_CREDIT_SUM: float
    DAYS_LAST_PHONE_CHANGE: float
    ANNUITY_INCOME_RATIO: float
    AMT_CREDIT: float
