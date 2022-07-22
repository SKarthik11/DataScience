from multiprocessing.sharedctypes import Array
from typing import List
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI
from LinearRegressionModel import pred_df,figjson,linreg,Accuracy
from pydantic import BaseModel

from fastapi import FastAPI

app=FastAPI()

result=pred_df.to_json()

@app.get("/showPlot")
def GraphData():
    return figjson

@app.get("/showValues")
def Values():
    return "(Accuracy : ",Accuracy,")",result