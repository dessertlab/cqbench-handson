# Semgrep rule snapshot

`semgrep.json` is the resolved, immutable Semgrep configuration used by full
CQBench evaluation. The benchmark records its SHA-256 digest. Structural-only
evaluation does not require this file.

The study used the following registry configurations:

```text
p/trailofbits
p/default
p/comment
p/python
p/java
p/c
p/cwe-top-25
p/owasp-top-ten
p/r2c-security-audit
p/insecure-transport
p/secrets
p/findsecbugs
p/gitlab
p/mobsfscan
p/command-injection
p/sql-injection
```

Refresh it only through:

```bash
python -m cqbench vendor-rules --output cqbench/rules/semgrep.json --overwrite
```

The command uses Semgrep 1.120.0, deduplicates by rule ID, fails on conflicting
definitions, and records the digest in `manifest.json`. Do not run a release
evaluation against mutable registry aliases.
