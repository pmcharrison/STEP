# Change Log

All important changes to the psynet-step package will be documented here.

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.1]

### Changed

- Replaced ``custom_network_filter`` with PsyNet 14's ``custom_chain_filter`` so STEP still respects the participant time budget under the new trial-selection API.

## [v0.1.0]

### Added

- Added a modern PEP 621 `pyproject.toml` (Hatchling backend) so the package can be built and published to PyPI as `psynet-step` (the Python import name remains `step`).
- Added Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13 classifiers and `requires-python = ">=3.9"`.
- Added `[project.urls]` for Homepage / Repository / Issues.
- Added `dev` (pytest), `psynet`, and `publish` optional-dependency extras. The `psynet` extra is only required at runtime when using STEP in a PsyNet experiment; declaring PsyNet as a hard dependency would create a circular dependency if PsyNet ever adds `psynet-step` to its own extras.
- Added `markupsafe` and `sqlalchemy` as explicit runtime dependencies (previously only available transitively through psynet/dallinger).
- Added `step/VERSION` as the single source of truth for the package version, read at build time by Hatchling and at import time by `step/__init__.py`.
- Added `CHANGELOG.md`.

### Changed

- Bumped version from 0.0.1 to 0.1.0 for the first PyPI release.
- Filled in the LICENSE copyright holder (was a placeholder `[year] [fullname]`).
- Updated README with PyPI installation instructions and corrected repository URLs.
- Updated `.gitignore` to exclude `dist/` build artifacts.

### Removed

- Removed `setup.py` (replaced by `pyproject.toml`).
- Removed `MANIFEST.in` (not used by Hatchling; replaced by explicit wheel/sdist include/exclude rules in `pyproject.toml`).
- Removed the empty root `__init__.py` (it was not part of the `step` package and served no purpose).
- Removed `psynet` and `dallinger` from hard runtime dependencies (moved to optional `[psynet]` extra to avoid potential circular dependencies).

## [v0.0.1]

Initial development release (not published to PyPI).
