"""The six tools, as plain functions.

Each takes an open `Database` plus its arguments and returns a result model. They know nothing
about MCP; `server.py` wraps them. That keeps them easy to read and to test directly.
"""
