from flask import Blueprint, request, current_app, make_response
from flask_ldap3_login import AuthenticationResponseStatus
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
)
import subprocess

from config import Config

auth = Blueprint("auth", __name__)


@auth.route("/authenticate", methods=["POST"])
def authenticate():

    auth_params = request.authorization

    # Before authenticating, verify that user is a member of the llm_user group
    try:
        ldapsearch_cmd = f"""ldapsearch -x -b "cn=llm_user,ou=Group,dc=vector,dc=local" -H ldap://{Config.LDAP_HOST} | grep {auth_params['username']}"""
        subprocess.check_output(ldapsearch_cmd, shell=True)
    except subprocess.CalledProcessError:
        return make_response(
            {
                "msg": f"User {auth_params['username']} not a member of the llm_user group "
            },
            403,
        )

    ldapAuthResponse = current_app.ldap3_login_manager.authenticate_direct_credentials(
        auth_params["username"], auth_params["password"]
    )

    if ldapAuthResponse.status == AuthenticationResponseStatus.success:
        access_token = create_access_token(identity=auth_params["username"])
        response = make_response({"token": access_token}, 200)
        return response
    else:
        return make_response({"msg": "Bad username or password"}, 401)


@auth.route("/verify_token", methods=["POST"])
@jwt_required()
def verify_token():
    # If we get this far, the token is valid, so we can just return success
    return make_response({"msg": "Token is valid"}, 200)
