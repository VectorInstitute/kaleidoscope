import datetime
import os
from pathlib import Path


class Config(object):

    GATEWAY_BIND_HOST = os.environ["GATEWAY_BIND_HOST"]
    GATEWAY_ADVERTISED_HOST = os.environ["GATEWAY_ADVERTISED_HOST"]
    GATEWAY_PORT = os.environ["GATEWAY_PORT"]

    JOB_SCHEDULER = os.environ["JOB_SCHEDULER"]
    JOB_SCHEDULER_HOST = os.environ["JOB_SCHEDULER_HOST"]
    JOB_SCHEDULER_USER = os.environ["JOB_SCHEDULER_USER"]
    JOB_SCHEDULER_BIN = os.environ["JOB_SCHEDULER_BIN"]

    JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(
        days=int(os.environ["JWT_ACCESS_TOKEN_EXPIRES_DAYS"])
    )
    JWT_COOKIE_SECURE = os.getenv("JWT_COOKIE_SECURE", "False") == "True"
    JWT_REFRESH_COOKIE_PATH = os.environ["JWT_REFRESH_COOKIE_PATH"]

    LDAP_HOST = os.environ["LDAP_HOST"]
    LDAP_BIND_DIRECT_PREFIX = os.environ["LDAP_BIND_DIRECT_PREFIX"]
    LDAP_BIND_DIRECT_SUFFIX = os.environ["LDAP_BIND_DIRECT_SUFFIX"]
    LDAP_BIND_DIRECT_GET_USER_INFO = (
        os.getenv("LDAP_BIND_DIRECT_GET_USER_INFO", "False") == "True"
    )

    SQLALCHEMY_DATABASE_URI = os.environ["SQLALCHEMY_DATABASE_URI"]

    CELERY_BROKER_URL = os.environ["CELERY_BROKER_URL"]
    CELERY_BACKEND_URL = os.environ["CELERY_BACKEND_URL"]
