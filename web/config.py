import datetime
import os
from pathlib import Path


class Config(object):

    GATEWAY_HOST = os.environ["GATEWAY_HOST"]
    GATEWAY_PORT = os.environ["GATEWAY_PORT"]

    JOB_SCHEDULER_HOST = os.environ["JOB_SCHEDULER_HOST"]
    JOB_SCHEDULER_USER = os.environ["JOB_SCHEDULER_USER"]
    JOB_SCHEDULER_REMOTE_BIN = os.environ["JOB_SCHEDULER_REMOTE_BIN"]

    JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(
        days=int(os.environ["JWT_ACCESS_TOKEN_EXPIRES_DAYS"])
    )
    JWT_COOKIE_SECURE = os.getenv("JWT_COOKIE_SECURE", "False") == "True"
    JWT_REFRESH_COOKIE_PATH = os.environ["JWT_REFRESH_COOKIE_PATH"]

    LDAP_HOST = os.environ["LDAP_HOST"]
    LDAP_BASE_DN = os.environ["LDAP_BASE_DN"]
    LDAP_USER_ACCESS_GROUP = os.environ["LDAP_USER_ACCESS_GROUP"]
    LDAP_USER_DN = os.environ["LDAP_USER_DN"]
    LDAP_USER_RDN_ATTR = os.environ["LDAP_USER_RDN_ATTR"]
    LDAP_USER_LOGIN_ATTR = os.environ["LDAP_USER_LOGIN_ATTR"]
    LDAP_USER_OBJECT_FILTER = os.environ["LDAP_USER_OBJECT_FILTER"]
    LDAP_USER_SEARCH_SCOPE = os.environ["LDAP_USER_SEARCH_SCOPE"]
    LDAP_GROUP_DN = os.environ["LDAP_GROUP_DN"]
    LDAP_GROUP_MEMBERS_ATTR = os.environ["LDAP_GROUP_MEMBERS_ATTR"]
    LDAP_GROUP_OBJECT_FILTER = os.environ["LDAP_GROUP_OBJECT_FILTER"]

    SQLALCHEMY_DATABASE_URI = os.environ["SQLALCHEMY_DATABASE_URI"]

    CELERY_BROKER_URL = os.environ["CELERY_BROKER_URL"]
    CELERY_BACKEND_URL = os.environ["CELERY_BACKEND_URL"]
