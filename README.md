# MP171 — Used Two-Wheeler Price Prediction (Jaipur)

- **Group ID:** G257
- **Problem ID:** MP171

## Group Members

| Role | Name | Enrolment No. |
|---|---|---|
| Data Lead | Tanmay Kaul | S24CSEU0357 |
| Model Lead | Member 2 | S24CSEU0389 |
| Service Lead | Sarthak Jain | S24CSEU0403 |
| Ops Lead | Member 4 | S24CSEU |

## Usage

```bash
python -m pip install -r requirements.txt
python -m src.mp171.run_all
pytest -v
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
