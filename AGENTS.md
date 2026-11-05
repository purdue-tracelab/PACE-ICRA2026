# Repository maintenance

## External data and parameter provenance

For every code update, review the external data, published constants, standards,
and empirical assumptions used by the affected code. Update
[the external data register](docs/external-data-sources.md) in the same change.
Add a dated audit-log entry even when the review finds no provenance changes.

For each new or changed input, record the code location, value and unit, source
link, source version or immutable commit, and any dataset split or sample
filter. Document how the source was transformed or fitted and how the result
was checked. Distinguish source values, derived values, inherited settings,
and modeling assumptions. Identify validation gaps and untraceable values
explicitly. For unchanged inputs, record the scope of the check and that the
source entries required no changes.

Include the register update in the PR diff and summarize the provenance check
in the PR description. Keep upstream data files out of the repository unless
their inclusion is part of the requested change.
