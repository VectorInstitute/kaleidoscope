#!/usr/bin/env python3

from flask import Flask, redirect, url_for
from flask_ldap3_login import LDAP3LoginManager
from flask_jwt_extended import JWTManager


from config import Config
from auth import auth
from db import db
from resources.home import home
from resources.models import models

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    ldap_manager = LDAP3LoginManager(app)  
    jwt = JWTManager(app)

    app.register_blueprint(auth)
    app.register_blueprint(home.home_bp)
    app.register_blueprint(models.models_bp, url_prefix='/models')

    db.init_app(app)
    with app.app_context():
        db.create_all()

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.GATEWAY_HOST, port=Config.GATEWAY_PORT)
