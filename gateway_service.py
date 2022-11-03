#!/usr/bin/env python3

from flask import Flask
from flask_ldap3_login import LDAP3LoginManager
from flask_jwt_extended import JWTManager

from config import Config
from auth import auth
from routes import gateway

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    ldap_manager = LDAP3LoginManager(app)  
    jwt = JWTManager(app)


    app.register_blueprint(auth)
    app.register_blueprint(gateway)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
