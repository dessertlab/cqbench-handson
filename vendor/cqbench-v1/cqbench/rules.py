from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import write_json_atomic


REGISTRY_PACKS = (
    "p/trailofbits",
    "p/default",
    "p/comment",
    "p/python",
    "p/java",
    "p/c",
    "p/cwe-top-25",
    "p/owasp-top-ten",
    "p/r2c-security-audit",
    "p/insecure-transport",
    "p/secrets",
    "p/findsecbugs",
    "p/gitlab",
    "p/mobsfscan",
    "p/command-injection",
    "p/sql-injection",
)


def vendor_rules(output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    try:
        import semgrep
        from semgrep.config_resolver import ConfigLoader
    except ImportError as exc:
        raise RuntimeError("Semgrep 1.120.0 is required to vendor rules") from exc

    rules: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = {}
    for pack in REGISTRY_PACKS:
        files = ConfigLoader(pack, None).load_config()
        assert len(files) == 1, f"Unexpected Semgrep response count for {pack}"
        data = json.loads(files[0].contents)
        pack_rules = data.get("rules", [])
        assert isinstance(pack_rules, list) and pack_rules, f"No rules in {pack}"
        for rule in pack_rules:
            rule_id = rule.get("id")
            assert isinstance(rule_id, str) and rule_id
            sources.setdefault(rule_id, []).append(pack)
            if rule_id in rules:
                assert rules[rule_id] == rule, (
                    f"Conflicting definitions for Semgrep rule {rule_id}: "
                    f"{sources[rule_id]}"
                )
            else:
                rules[rule_id] = rule

    payload = {"rules": [rules[rule_id] for rule_id in sorted(rules)]}
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    manifest = {
        "semgrep_version": getattr(semgrep, "__VERSION__", "1.120.0"),
        "packs": list(REGISTRY_PACKS),
        "rule_count": len(rules),
        "sha256": digest,
        "duplicate_rule_ids": sum(len(value) > 1 for value in sources.values()),
    }
    write_json_atomic(
        output.with_name("manifest.json"),
        manifest,
        overwrite=overwrite,
    )
    return manifest
