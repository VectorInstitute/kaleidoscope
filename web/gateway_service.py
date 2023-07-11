#!/usr/bin/env python3
"""Module for building and configuring the gateway service application"""
import logging

from flask import Flask
from flask_ldap3_login import LDAP3LoginManager
from flask_jwt_extended import JWTManager
from celery import Celery
from flask_migrate import Migrate

from config import Config
from auth import auth
from db import db
from home.routes import home_bp
from model_instances.routes import model_instances_bp

def create_app():
    """Create Flask application with authentication and DB configs"""
    app = Flask(__name__)
    app.config.from_object(Config)
    app.logger.setLevel(logging.INFO)  # ToDo: move this to config

    ldap_manager = LDAP3LoginManager(app)
    jwt = JWTManager(app)

    app.register_blueprint(auth)
    app.register_blueprint(home_bp)
    app.register_blueprint(model_instances_bp, url_prefix="/models")

    db.init_app(app)
    migrate = Migrate(app, db)

    return app


def make_celery(app):
    """Create asynchronous workers using local environment variables"""
    celery = Celery(
        app.import_name,
        backend=app.config["CELERY_BACKEND_URL"],
        broker=app.config["CELERY_BROKER_URL"],
        include=["tasks"],
    )
    celery.conf.update(app.config)

    celery.conf.beat_schedule = {
        "verify_health": {
            "task": "tasks.verify_model_instance_health",
            "schedule": 30.0,
        },
        "verify_active": {
            "task": "tasks.verify_model_instance_active",
            "schedule": 30.0,
        }
    }

    class ContextTask(celery.Task):
        """Class to define the contexts for celery tasks"""

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    app.celery = celery
    return celery

app = create_app()
celery = make_celery(app)

if __name__ == "__main__":
    app.run(host=Config.GATEWAY_BIND_HOST, port=Config.GATEWAY_PORT)
