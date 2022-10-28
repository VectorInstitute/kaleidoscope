"""messaging protocols, eventually the user can specify
    which to use
"""
import pickle

import cloudpickle


class NaiveProtocol:
    """Protocol to pack arbitary objects for sending
        and unpack arbitary bytestreams for recieving

    NOTE: HUGE security risk since we are pickling.. but
        just going with the flow for now

    TODO: isn't pickle a bit slow?
    one optimization we can make
    is to not pickle the stuff
    that can be sent over
    """

    # TODO: isn't pickle a bit slow?
    # one optimization we can make
    # is to not pickle the stuff
    # that can be sent over
    @staticmethod
    def pack(obj):
        return cloudpickle.dumps(obj)

    @staticmethod
    def unpack(obj_bytes):
        return pickle.loads(obj_bytes)


def pack(obj):
    # can change based on user config
    return NaiveProtocol.pack(obj)


def unpack(obj_bytes):
    return NaiveProtocol.unpack(obj_bytes)
