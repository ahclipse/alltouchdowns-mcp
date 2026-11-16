"""Builds the slim public database (`mcp.db`) from a copy of the site's database.

Run it with `python -m alltouchdowns_mcp.export --help`. The steps, in order:

1. `source`   copy the site database into a scratch directory and verify the copy
2. `build`    select the rows and columns that may be published
3. `validate` check the result, then swap it into place atomically
"""
