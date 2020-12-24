import datetime
import logging

import pytz

def add_datetime_to_pars(pars):
    pars["time_interval_start"] = parse_time_tuple(convert_to_time_tuple(pars["time_interval_start"]))
    pars["time_interval_end"] = parse_time_tuple(convert_to_time_tuple(pars["time_interval_end"]))
    return pars


def convert_to_time_tuple(time):
    time_tuple = (time['time'], time['zone'])
    return time_tuple


def parse_time_tuple(time_tuple):
    if len(time_tuple) == 2:
        if time_tuple[1] == 'utc':
            return parse_utc(time_tuple[0])
        else:
            return parse_oklahoma_time(time_tuple[0])
    else:
        return parse_utc(time_tuple)


def parse_utc(utc_str):
    utc = pytz.utc
    if len(str(utc_str)) == 12:
        d = datetime.datetime.strptime(str(utc_str), '%Y%m%d%H%M')
    d = utc.localize(d)
    return d


def parse_oklahoma_time(oklahoma_str):
    utc = pytz.utc
    okla = pytz.timezone('America/Chicago')
    if len(str(oklahoma_str)) == 12:
        d = datetime.datetime.strptime(str(oklahoma_str), '%Y%m%d%H%M')
    d = okla.localize(d)
    return d.astimezone(utc)


def curate_by_time(sbws, lsrs, pars):
    print('Ensuring event times match')
    logging.info('Ensuring event times match')
    logging.debug(f'num sbws {len(sbws)}')
    logging.debug(f'num lsrs {len(lsrs)}')
    lsrs = [lsr for lsr in lsrs if ensure_in_interval(pars, parse_time_tuple((lsr.get('valid'), 'utc')))]
    sbws = [sbw for sbw in sbws if
            ensure_in_interval(pars, parse_time_tuple((sbw.get('expired'), 'utc'))) or
            ensure_in_interval(pars, parse_time_tuple((sbw.get('issued'), 'utc'))) or
            ensure_in_interval(pars, parse_time_tuple((sbw.get('init_iss'), 'utc'))) or
            ensure_in_interval(pars, parse_time_tuple((sbw.get('init_exp'), 'utc')))
            ]
    logging.debug(f'num sbws {len(sbws)}')
    logging.debug(f'num lsrs {len(lsrs)}')
    return sbws, lsrs


def ensure_in_interval(pars, time):
    return pars.get('time_interval_start', datetime.date.min) <= time <= pars.get('time_interval_end',
                                                                                  datetime.date.min)
