from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class BaseMixin(object):
    @classmethod
    def create(cls, **kw):
        obj = cls(**kw)
        db.session.add(obj)
        db.session.commit()

    @classmethod
    def get_by_id(cls, id):
        return db.session.query(cls).filter_by(id=id).first()

    def destroy(self):
        db.session.delete(self)
        db.session.commit()
