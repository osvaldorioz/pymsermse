from fastapi import FastAPI
import mse_rmse
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

class Vector(BaseModel):
    vector: List[float]

@app.post("/mse")
async def mse(y_true: Vector, 
              y_pred: Vector):
    
    start = time.time()
    
    # Calcular MSE y RMSE
    mse = mse_rmse.mean_squared_error(y_true.vector, y_pred.vector)
    rmse = mse_rmse.root_mean_squared_error(y_true.vector, y_pred.vector)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse
    }
    jj = json.dumps(j1)

    return jj
