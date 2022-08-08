# -*- coding: utf-8 -*-
"""
@author: Fiacre
"""
from pydantic import BaseModel

# 2. Class which describes Bank Notes measurements
class CreditData(BaseModel):
    EXT_SOURCE_3: float
    DAYS_LAST_PHONE_CHANGE: float
    REGION_POPULATION_RELATIVE: float
    TOTALAREA_MODE: float
    HOUR_APPR_PROCESS_START: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    AMT_INCOME_TOTAL: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    FLAG_DOCUMENT_3: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    EXT_SOURCE_2: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    AMT_ANNUITY: float
    DAYS_ID_PUBLISH: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    AMT_GOODS_PRICE: float
