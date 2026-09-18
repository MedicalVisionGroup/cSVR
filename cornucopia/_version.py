# Static replacement for the versioneer-generated file of the upstream package. The
# vendored copy is not a git checkout of cornucopia, so the git-based lookup would report
# the surrounding repository's revision instead.
_VERSION = "0.4.0+23.g58369ea.csvr"


def get_versions():
    return {
        "version": _VERSION,
        "full-revisionid": "58369ea",
        "dirty": True,
        "error": None,
        "date": "2025-12-08",
    }
