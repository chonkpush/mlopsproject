# MP171 — Used Two-Wheeler Price Prediction (Jaipur)

- **Group ID:** G257
- **Problem ID:** MP171

## Group Members

| Role | Name | Enrolment No. |
|---|---|---|
| Data Lead | Tanmay Kaul | S24CSEU0357 |
| Model Lead | Member 2 | S24CSEU |
| Service Lead | Member 3 | S24CSEU |
| Ops Lead | Member 4 | S24CSEU |

## Structure & Usage

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m mp171.run_all
pytest -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
