"""Module for Vector AI Institute HPC authentication"""
from flask import Blueprint, request, current_app, make_response
from flask_ldap3_login import AuthenticationResponseStatus
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
)

from config import Config

auth = Blueprint("auth", __name__)


@auth.route("/authenticate", methods=["POST"])
def authenticate():
    """Authenticate a user on predefined credentials"""
    auth_params = request.authorization

    # Verify that we can connect to the LDAP server
    try:
        connection = current_app.ldap3_login_manager.make_connection()
        connection.bind()
    except Exception as err:
        return make_response(
            {"msg": f"Could not connect to LDAP server at {Config.LDAP_HOST} ({err})"},
            500,
        )

    # Before authenticating, verify that user is a member of the user access group
    groups = current_app.ldap3_login_manager.get_user_groups(auth_params["username"])
    if not any(str(group["cn"]) == f"['{Config.LDAP_USER_ACCESS_GROUP}']" for group in groups):
        return make_response(
            {
                "msg": f"User {auth_params['username']} not a member of \
                the {Config.LDAP_USER_ACCESS_GROUP} group "
            },
            403,
        )

    ldapAuthResponse = current_app.ldap3_login_manager.authenticate(
        auth_params["username"], auth_params["password"]
    )

    if ldapAuthResponse.status == AuthenticationResponseStatus.success:
        access_token = create_access_token(identity=auth_params["username"])
        response = make_response({"token": access_token}, 200)
        return response
    return make_response({"msg": "Bad username or password"}, 401)


@auth.route("/verify_token", methods=["POST"])
@jwt_required()
def verify_token():
    """Verify if the user authentication token is valid"""
    # If we get this far, the token is valid, so we can just return success
    return make_response({"msg": "Token is valid"}, 200)
