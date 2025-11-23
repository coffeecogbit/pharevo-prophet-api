from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prophet import Prophet

app = FastAPI()


class DataPoint(BaseModel):
    ds: datetime = Field(..., description="Timestamp of the observation.")
    y: float = Field(..., description="Observed value at the timestamp.")


class ForecastRequest(BaseModel):
    data: List[DataPoint]
    periods: int = Field(..., gt=0, description="Number of future periods to forecast.")
    freq: str = Field("D", description="Pandas frequency string for the forecast horizon.")


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if not req.data:
        raise HTTPException(status_code=400, detail="Data must contain at least one observation.")

    df = pd.DataFrame([point.dict() for point in req.data])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if df["ds"].isna().any():
        raise HTTPException(status_code=400, detail="All 'ds' values must be valid datetimes.")

    if df["y"].isna().any():
        raise HTTPException(status_code=400, detail="All 'y' values must be present and numeric.")

    df = df.sort_values("ds").reset_index(drop=True)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=req.periods, freq=req.freq)
    forecast_df = model.predict(future)
    forecast_df["ds"] = forecast_df["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    return {
        "forecast": forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict("records")
    }
