
from logger_setup import *

__all__ = [
    'format_time',
]

def format_time(timesec):
    m, s = divmod(timesec, 60)
    h, m = divmod(m, 60)
    str_format_time= ''
    if h == 0:
        if m == 0:
            str_format_time= "%02ds" % (s)
            #return "%02ds" % (s)
        else:
            str_format_time= "%02dm%02ds" % (m, s)
            #return "%02dm%02ds" % (m, s)
    else:
        str_format_time= "%dh%02dm%02ds" % (h, m, s)
        #return "%dh%02dm%02ds" % (h, m, s)

    return str_format_time, h, m, s