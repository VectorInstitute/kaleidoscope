#!/bin/bash
export FLASK_APP=gateway_service.py

flask db upgrade
python gateway_service.py
