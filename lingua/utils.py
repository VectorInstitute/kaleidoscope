"""
TODOS:
    1. eventually we need to seperate this out to client and server utils

"""
import logging

import requests

from protocols import pack, unpack

logger = logging.getLogger(__name__)


def check_response(resp):
    if not resp.ok:
        raise ValueError(
            "request to {} not sucessful, error code: {} msg: {}".format(
                resp.url,
                resp.status_code,
                resp.json()["detail"],
            )
        )
    logger.debug("addr %s response code %s", resp.url, resp.status_code)


def get(addr):
    resp = requests.get(addr)
    check_response(resp)
    return unpack(resp.content)


def post(addr, obj, field_name="json"):
    # NOTE: the field_name must match the server's field name
    resp = requests.post(addr, json=obj)  # allign with metaseq json request
    check_response(resp)
    return resp


def server_send(obj):
    return pack(obj)


def server_parse(bytestream):
    return unpack(bytestream)
