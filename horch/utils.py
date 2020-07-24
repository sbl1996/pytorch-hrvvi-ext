from datetime import datetime, timedelta

__all__ = ['time_now']

def time_now():
    now = datetime.utcnow() + timedelta(hours=8)
    now = now.strftime("%H:%M:%S")
    return now
