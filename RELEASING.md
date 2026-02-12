# Releasing `evenet-lite`

This repository publishes only `evenet-lite`.

`evenet-core` is maintained in a separate git repository (checked out here as the `evenet/` submodule) and must be released from that repository.

For `evenet-core` release steps, see:

- `/Users/avencastmini/PycharmProjects/EveNet-Lite/evenet/RELEASING.md`

## Auto publish flow

- Tag push matching `v*` triggers `.github/workflows/build-packages.yml`.
- `build-lite` builds and validates artifacts.
- `publish-lite` runs only if `build-lite` succeeds and uploads to PyPI.

No separate manual publish workflow is required.

## What you must set on PyPI

Set up **Trusted Publishing** for package `evenet-lite`:

1. Package name: `evenet-lite` (create it if it does not exist yet).
2. Publisher type: GitHub Actions.
3. GitHub owner: your org/user.
4. GitHub repository: your `EveNet-Lite` repository.
5. Workflow filename: `build-packages.yml`.
6. Environment name: `pypi`.

Notes:
- First release may require creating a **pending publisher** in PyPI before the workflow can publish.
- You do not need a `PYPI_API_TOKEN` secret when using trusted publishing.

## One-time GitHub requirement

Create GitHub Actions environment `pypi` in this repository (Settings -> Environments).

## Release steps

1. Ensure `evenet-core` release exists on PyPI.
2. Bump version in `pyproject.toml`.
3. Commit and push.
4. Tag and push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow will build, then auto-publish on success.
