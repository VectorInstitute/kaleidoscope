from flask import Blueprint, redirect, request, current_app, jsonify, make_response
from flask_ldap3_login import AuthenticationResponseStatus
from flask_jwt_extended import create_access_token, create_refresh_token, set_access_cookies, set_refresh_cookies

auth = Blueprint('auth', __name__)

@auth.route("/authenticate", methods=["POST"])
def authenticate():

    auth_params = request.authorization
    ldapAuthResponse = current_app.ldap3_login_manager.authenticate_direct_credentials(auth_params['username'], auth_params['password'])

    if (ldapAuthResponse.status == AuthenticationResponseStatus.success):
        access_token = create_access_token(identity=auth_params['username'])
        response = make_response({"token": access_token}, 200)
        # set_access_cookies(response, access_token)
        return response 
    else:
        return make_response({"msg": "Bad username or password"}, 401)
