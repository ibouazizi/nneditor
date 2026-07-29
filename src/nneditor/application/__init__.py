"""Application services: sessions, jobs, and renderer-facing queries.

This layer owns everything between the immutable source/IR and the UI shell:
sessions, background jobs, navigation, hierarchy overrides, statistics,
transactional revisions, and export. The renderer remains replaceable and
source artifacts are never mutated.
"""
