# MP171 — Used Two-Wheeler Price Prediction (Jaipur)

- **Group ID:** G257
- **Problem ID:** MP171

## Group Members

| Role | Name | Enrolment No. |
|---|---|---|
| Data Lead | Tanmay Kaul | S24CSEU0357 |
| Model Lead | Sameer Anand | S24CSEU0389 |
| Service Lead | Sarthak Jain | S24CSEU0403 |
| Ops Lead | saransh.  | S24CSEU1768 |

## Usage

```bash
python -m pip install -r requirements.txt
python -m src.mp171.run_all
pytest -v
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
