from registry import warranty
from datetime import date
def test_active(): assert warranty('PX200-0042',date(2026,11,24))['status']=='active'
def test_expired(): assert warranty('PX200-0001',date(2026,11,24))['status']=='expiree'
def test_unknown(): assert warranty('XXX')['status']=='inconnu'
