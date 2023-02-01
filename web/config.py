import datetime
import os
from pathlib import Path


class Config(object):
    DEBUG = True

    GATEWAY_HOST = "0.0.0.0"
    GATEWAY_PORT = 3001

    JOB_SCHEDULER_HOST = "vremote"
    JOB_SCHEDULER_USER = "llm"

    JWT_SECRET_KEY = "abc123"
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(days=30)
    JWT_COOKIE_SECURE = False
    # JWT_ACCESS_COOKIE_PATH = '/api/'
    JWT_REFRESH_COOKIE_PATH = "/token/refresh"

    LDAP_HOST = "172.17.15.251"
    LDAP_BIND_DIRECT_PREFIX = "uid="
    LDAP_BIND_DIRECT_SUFFIX = ",ou=People,dc=vector,dc=local"
    LDAP_BIND_DIRECT_GET_USER_INFO = False

    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:vector@lingua-db-1/test"

    CELERY_BROKER_URL = "sqla+postgresql+psycopg2://postgres:vector@lingua-db-1/test"
    CELERY_BACKEND_URL = "db+postgresql://postgres:vector@lingua-db-1/test"
