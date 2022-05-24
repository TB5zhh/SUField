from datetime import datetime

def current_timestr(format='%Y-%m-%d.%H:%M:%S'):
    return datetime.now().strftime(format)