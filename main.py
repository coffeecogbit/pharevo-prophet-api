from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator, validator
from prophet import Prophet

app = FastAPI()


# -----------------------------
# Pydantic models
# -----------------------------
class DataPoint(BaseModel):
    ds: datetime = Field(..., description="Timestamp of the observation.")
    y: float = Field(
        ...,
        description="Observed value at the timestamp.",
        allow_inf_nan=False,
    )


class ForecastRequest(BaseModel):
    data: List[DataPoint]

    # พื้นฐานเดิม
    periods: int = Field(..., gt=0, description="Number of future periods to forecast.")
    freq: str = Field("D", description="Pandas frequency string for the forecast horizon.")

    # Seasonality options
    yearly_seasonality: bool = Field(
        True, description="Enable yearly seasonality."
    )
    weekly_seasonality: bool = Field(
        True, description="Enable weekly seasonality."
    )
    daily_seasonality: bool = Field(
        False, description="Enable daily seasonality."
    )

    # Growth model
    growth: str = Field(
        "linear",
        description="Growth model: 'linear' or 'logistic'.",
    )
    cap: Optional[float] = Field(
        None, description="Capacity for logistic growth (required if growth='logistic')."
    )
    floor: Optional[float] = Field(
        None, description="Floor for logistic growth (optional)."
    )

    # Additional regressors
    # ตัวอย่างรูปแบบ:
    # "regressors": {
    #   "promo": [0, 1, 0, ...],
    #   "rain": [10.0, 5.0, 0.0, ...]
    # }
    regressors: Optional[Dict[str, List[float]]] = Field(
        None, description="Optional additional regressors aligned with data points."
    )

    # ขอ components เพิ่มใน response หรือไม่
    include_components: bool = Field(
        False,
        description="If true, include Prophet components (trend, weekly, yearly, etc.) in the response.",
    )

    @validator("freq")
    def validate_freq(cls, value: str) -> str:
        try:
            pd.tseries.frequencies.to_offset(value)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid frequency string.") from exc
        return value

    @validator("growth")
    def validate_growth(cls, value: str) -> str:
        value = value.lower()
        if value not in {"linear", "logistic"}:
            raise ValueError("growth must be 'linear' or 'logistic'.")
        return value

    @validator("cap", "floor")
    def validate_bounds(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not np.isfinite(value):
            raise ValueError("cap and floor must be finite numbers when provided.")
        return value
        
    @root_validator(skip_on_failure=True)
    def validate_logistic_and_regressors(cls, values):
        growth = values.get("growth")
        cap = values.get("cap")
        floor = values.get("floor")
        data = values.get("data") or []
        regressors = values.get("regressors") or {}

        # ถ้าใช้ logistic ต้องมี cap
        if growth == "logistic" and cap is None:
            raise ValueError("cap is required when growth='logistic'.")

        # floor ถ้ามี ต้องน้อยกว่า cap
        if growth == "logistic" and floor is not None and cap is not None:
            if floor >= cap:
                raise ValueError("floor must be less than cap when using logistic growth.")

        # เช็กความยาว regressors ให้เท่ากับ data
        if regressors:
            n = len(data)
            reserved = {"ds", "y", "cap", "floor"}
            for name, series in regressors.items():
                if name in reserved:
                    raise ValueError(
                        f"Regressor '{name}' is reserved and cannot be used as an additional regressor."
                    )

                # ความยาวต้องตรงกับ data
                if len(series) != n:
                    raise ValueError(
                        f"Regressor '{name}' must have the same length as data ({n})."
                    )

                # ต้องเป็นตัวเลข finite เท่านั้น
                series_arr = np.array(series, dtype=float)
                if not np.all(np.isfinite(series_arr)):
                    raise ValueError(
                        f"Regressor '{name}' contains non-finite values (NaN or inf)."
                    )

        return values


# -----------------------------
# Simple in-memory cache
# -----------------------------
_FORECAST_CACHE: Dict[str, Dict] = {}


def _make_cache_key(req: ForecastRequest) -> str:
    """สร้าง key สำหรับ cache จากเนื้อหา request ทั้งหมด"""
    payload = {
        "data": [
            {"ds": point.ds.isoformat(), "y": point.y}
            for point in req.data
        ],
        "periods": req.periods,
        "freq": req.freq,
        "yearly_seasonality": req.yearly_seasonality,
        "weekly_seasonality": req.weekly_seasonality,
        "daily_seasonality": req.daily_seasonality,
        "growth": req.growth,
        "cap": req.cap,
        "floor": req.floor,
        "regressors": req.regressors,
        "include_components": req.include_components,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------
# Main forecasting endpoint
# -----------------------------
@app.post("/forecast")
def forecast(req: ForecastRequest):
    # ใช้ cache ถ้ามี request เดิมเป๊ะ ๆ
    cache_key = _make_cache_key(req)
    if cache_key in _FORECAST_CACHE:
        return _FORECAST_CACHE[cache_key]

    if not req.data:
        raise HTTPException(
            status_code=400,
            detail="Data must contain at least one observation.",
        )

    # สร้าง DataFrame จาก data
    df = pd.DataFrame([{"ds": point.ds, "y": point.y} for point in req.data])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # ตรวจสอบคุณภาพข้อมูล
    if df["ds"].isna().any():
        raise HTTPException(
            status_code=400,
            detail="All 'ds' values must be valid datetimes.",
        )

  if not df["ds"].is_unique:
        raise HTTPException(
            status_code=400,
            detail="All 'ds' values must be unique timestamps.",
        )

    if df["y"].isna().any():
        raise HTTPException(
            status_code=400,
            detail="All 'y' values must be present and numeric.",
        )

    if (~np.isfinite(df["y"])).any():
        raise HTTPException(
            status_code=400,
            detail="All 'y' values must be finite numbers.",
        )

    # Logistic growth ต้องการให้ค่า y อยู่ระหว่าง floor และ cap
    if req.growth == "logistic":
        if (df["y"] >= req.cap).any():
            raise HTTPException(
                status_code=400,
                detail="All 'y' values must be less than cap when using logistic growth.",
            )

        if req.floor is not None and (df["y"] <= req.floor).any():
            raise HTTPException(
                status_code=400,
                detail="All 'y' values must be greater than floor when using logistic growth.",
            )
    if df.shape[0] < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two observations are required to build a forecast.",
        )

    if df["ds"].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Observations must span at least two distinct timestamps.",
        )

    df = df.sort_values("ds").reset_index(drop=True)

    # ใส่ regressors ลงใน df ถ้ามี
    regressors = req.regressors or {}
    for name, series in regressors.items():
        df[name] = series

    # logistic growth: ต้องมี cap/floor ใน df และ future
    if req.growth == "logistic":
        df["cap"] = req.cap
        if req.floor is not None:
            df["floor"] = req.floor

    # สร้างและตั้งค่าโมเดล Prophet
    model = Prophet(
        growth=req.growth,
        yearly_seasonality=req.yearly_seasonality,
        weekly_seasonality=req.weekly_seasonality,
        daily_seasonality=req.daily_seasonality,
    )

    # ลงทะเบียน regressors
    for name in regressors.keys():
        model.add_regressor(name)

    # เทรนโมเดล
    model.fit(df)

    # สร้าง future dataframe
    future = model.make_future_dataframe(
        periods=req.periods,
        freq=req.freq,
    )

    # รวม regressors ในอดีต + เติมค่าในอนาคตด้วยค่าล่าสุด
    if regressors:
        # merge ค่าในอดีตตาม ds
        cols = ["ds"] + list(regressors.keys())
        future = future.merge(df[cols], on="ds", how="left")

        # สำหรับอนาคตที่ไม่มีค่า regressors → เติมด้วยค่าล่าสุดของแต่ละตัว
        for name in regressors.keys():
            last_val = df[name].iloc[-1]
            future[name].fillna(last_val, inplace=True)

    # logistic: ใส่ cap/floor ใน future ด้วย
    if req.growth == "logistic":
        future["cap"] = req.cap
        if req.floor is not None:
            future["floor"] = req.floor

    # พยากรณ์
    forecast_df = model.predict(future)

    # แปลง ds ให้เป็น string ISO (ไม่เอา timezone แปลก ๆ)
    forecast_df["ds"] = forecast_df["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    # เตรียม output หลัก
    forecast_records = forecast_df[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].to_dict("records")

    result: Dict[str, object] = {
        "forecast": forecast_records,
        "used_params": {
            "periods": req.periods,
            "freq": req.freq,
            "yearly_seasonality": req.yearly_seasonality,
            "weekly_seasonality": req.weekly_seasonality,
            "daily_seasonality": req.daily_seasonality,
            "growth": req.growth,
            "cap": req.cap,
            "floor": req.floor,
            "regressors": list(regressors.keys()),
        },
    }

    # ถ้าขอ components เพิ่ม
    if req.include_components:
        possible_components = [
            "trend",
            "weekly",
            "yearly",
            "daily",
            "holidays",
            "weekly_lower",
            "weekly_upper",
            "yearly_lower",
            "yearly_upper",
        ]
        component_cols = [
            c for c in possible_components if c in forecast_df.columns
        ]
        if component_cols:
            result["components"] = forecast_df[["ds"] + component_cols].to_dict(
                "records"
            )

    # เก็บ cache
    _FORECAST_CACHE[cache_key] = result

    return result
