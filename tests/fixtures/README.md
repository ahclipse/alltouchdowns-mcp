# Test fixture

`seasons-2007-2019.db` is a slim database holding two complete seasons, 2007 and 2019. Tests and CI evals use it instead of the full dataset.

It is built by the same export code that builds the production database, so it has the same tables and the same rules applied. It contains no play descriptions.

The two seasons were chosen because together they cover both eras of the relocated teams: St. Louis, San Diego and Oakland in 2007, and both Los Angeles teams (plus Oakland's last season) in 2019.

## Regenerating

Regenerate only when the export schema changes. It needs the private site database:

```bash
uv run python -m alltouchdowns_mcp.export --source-file /path/to/site.db --seasons 2007,2019 --out tests/fixtures/seasons-2007-2019.db
```

```bash
cd tests/fixtures && shasum -a 256 seasons-2007-2019.db > fixture.sha256
```

A regenerated file has a new hash, which invalidates any stored eval baseline. Baselines must be re-measured afterwards.

## Attribution

Play-by-play data from nflverse (https://github.com/nflverse), licensed CC BY 4.0. Processed and reformatted. Not affiliated with, endorsed by, or sponsored by the NFL or any NFL team. See [NOTICE](../../NOTICE).
