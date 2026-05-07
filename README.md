

## Deploy on Render

This repository is a monorepo, and the Flask app lives in `FlaskApi/`.

Use the included `render.yaml` blueprint, or set these Render service settings manually:

- Root Directory: `FlaskApi`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn time_series_forecasting_using_fbprophet:app`

Render will read `runtime.txt` from `FlaskApi/`, which keeps the app on Python 3.11.

