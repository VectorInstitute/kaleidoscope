import os

class Config(object):
    DEBUG = True

    JWT_SECRET_KEY = 'abc123'

    LDAP_HOST = "172.17.15.252"
    LDAP_BIND_DIRECT_PREFIX = 'uid='
    LDAP_BIND_DIRECT_SUFFIX = ',ou=People,dc=vector,dc=local'
    LDAP_BIND_DIRECT_GET_USER_INFO = False