import logging


def find_applicable_sbws(sbws, phenom=None, sig=None, gtype=None):
    print('finding relevant sbws')
    logging.info('finding relevant sbws')
    logging.debug(f'phenom {phenom}')
    logging.debug(f'sig {sig}')
    logging.debug(f'gtype {gtype}')
    logging.debug(f'num sbws {len(sbws)}')
    if phenom is not None:
        sbws = [sbw for sbw in sbws if sbw['phenom'] in phenom]
    if sig is not None:
        sbws = [sbw for sbw in sbws if sbw['sig'] in sig]
    if gtype is not None:
        sbws = [sbw for sbw in sbws if sbw['gtype'] in gtype]
    logging.debug(f'num sbws {len(sbws)}')
    return sbws


def find_applicable_lsrs(lsrs, typecode=None, typetext=None):
    print('finding relevant lsrs')
    logging.info('finding relevant lsrs')
    logging.debug(f'typecode {typecode}')
    logging.debug(f'typetext {typetext}')
    logging.debug(f'num lsrs {len(lsrs)}')
    typecode = (None,) if typecode is None else typecode
    typetext = (None,) if typetext is None else typetext
    lsrs = [lsr for lsr in lsrs if lsr['typecode'] in typecode or lsr['typetext'] in typetext]
    logging.debug(f'num lsrs {len(lsrs)}')
    return lsrs
