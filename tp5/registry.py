import json
from pathlib import Path
from datetime import date
ROOT = Path(__file__).resolve().parent

def warranty(serial, today=None):
    records = json.loads((ROOT/'data/garanties.json').read_text())
    serial = serial.strip().upper()
    if serial not in records:
        return {'serial': serial, 'status': 'inconnu'}
    end = records[serial]
    return {'serial': serial, 'expiration': end, 'status': 'active' if (today or date.today()) <= date.fromisoformat(end) else 'expiree'}
