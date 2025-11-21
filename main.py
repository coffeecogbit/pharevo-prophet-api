from fastapi import FastAPI
from pydantic import BaseModel
from prophet import Prophet
import pandas as pd

app = FastAPI()

class ForecastRequest(BaseModel):
    data: list
    periods: int

@app.post("/forecast")
def forecast(req: ForecastRequest):
    df = pd.DataFrame(req.data)

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=req.periods)
    fcst = m.predict(future)

    return {
        "forecast": fcst[["ds","yhat","yhat_lower","yhat_upper"]].to_dict("records")
    }
