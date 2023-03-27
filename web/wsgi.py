import gateway_service

if __name__ == "__main__":
    app = create_app()
    celery = make_celery(app)
    app.run()