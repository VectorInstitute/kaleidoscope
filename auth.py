from flask import Blueprint, request, current_app, make_response
from flask_ldap3_login import AuthenticationResponseStatus
from flask_jwt_extended import create_access_token

auth = Blueprint('auth', __name__)

@auth.route("/authenticate", methods=["POST"])
def authenticate():

    auth_params = request.authorization
    response = current_app.ldap3_login_manager.authenticate_direct_credentials(auth_params['username'], auth_params['password'])

    if (response.status == AuthenticationResponseStatus.success):
        return make_response({
            'token': create_access_token(identity=auth_params['username'])
        }, 200)
    else:
        return make_response({"msg": "Bad username or password"}, 401)
        