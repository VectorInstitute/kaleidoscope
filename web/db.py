"""Module to represent the database"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class BaseMixin:
    """Class to represent the database object"""

    @classmethod
    def create(cls, **kw):
        """Create session object"""
        obj = cls(**kw)
        db.session.add(obj)
        db.session.commit()
        return obj

    @classmethod
    def find_by_id(cls, id):
        """Performs an SQL query to retrieve session IDs"""
        return db.session.query(cls).filter_by(id=id).first()

    def save(self):
        """Performs the ending of a transaction"""
        # db.session.add(self)
        db.session.commit()

    def destroy(self):
        """Removes the current transaction session"""
        db.session.delete(self)
        db.session.commit()
