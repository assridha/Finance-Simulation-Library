"""
API blueprint for the web interface.
"""
from flask import Blueprint

bp = Blueprint('api', __name__)

from . import routes  # noqa 