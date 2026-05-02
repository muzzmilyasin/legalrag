import sys
import os

# Ensure the root project directory is on the path so web_app can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from web_app import app  # noqa: F401 – Vercel requires a symbol named `app`
