#!/usr/bin/env bash
# ถ้า PORT ไม่ได้ตั้ง ให้ใช้ 8000 เป็น default
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"