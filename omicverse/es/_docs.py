# Stubbed for omicverse — was decoupler's DocstringProcessor that
# templated %(net)s / %(tmin)s placeholders into kernel docstrings.
# omicverse's public API documents these on the `__init__.py` wrappers,
# so the heavy machinery isn't needed in-tree. `@docs.dedent` reduces
# to identity; original kernel docstrings remain with their %()s
# placeholders unsubstituted (harmless — they're internal).


class _DocsNoop:
    def dedent(self, fn):
        return fn

    # Keep ``docs(...)`` style template calls (used by a few kernels) as
    # an identity formatter so the import surface stays intact.
    def __call__(self, fn):
        return fn

    def __getattr__(self, _name):
        return self.dedent


docs = _DocsNoop()
