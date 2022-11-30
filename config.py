import os

class Config(object):
    DEBUG = True

    GATEWAY_HOST = "0.0.0.0"
    GATEWAY_PORT = 3001

    JWT_SECRET_KEY = 'abc123'

    LDAP_HOST = "172.17.15.251"
    LDAP_BIND_DIRECT_PREFIX = 'uid='
    LDAP_BIND_DIRECT_SUFFIX = ',ou=People,dc=vector,dc=local'
    LDAP_BIND_DIRECT_GET_USER_INFO = False

    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:vector@lingua-db-1/test'