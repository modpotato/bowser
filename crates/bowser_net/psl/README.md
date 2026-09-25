# Vendored Public Suffix List

`../src/psl_data.bin` is a compact binary trie built from the canonical
Public Suffix List at <https://publicsuffix.org/list/public_suffix_list.dat>.

- Source: Mozilla Public License 2.0 — the list data is © Mozilla
  contributors; see the terms in the
  [list header](https://publicsuffix.org/list/public_suffix_list.dat).
- Vendored version: `2026-09-24_13-26-36_UTC`
  (commit `a179a48c465e818cfd8d626691cb317985da87fb`).
- 10,334 rules (289 wildcard, 8 exception) compressed into a 157,204-byte
  reversed-label trie (magic `BPSLv001`, vs ~335 KB raw). Unicode rules are
  IDNA-encoded to ASCII at build time, so the trie is pure ASCII. The
  committed file must be the raw binary trie, not base64 or any other
  text encoding.

## Updating

The upstream list changes several times a week. Per publicsuffix.org
guidance, fetch a fresh copy at most once per day, then regenerate:

    python3 crates/bowser_net/psl/gen_psl.py /path/to/public_suffix_list.dat

Verify the generator round-trips its own output (re-reads the generated
bin and asserts the exception rules, wildcards, IDN rules, and a sample
of exact rules) without touching the committed bin:

    python3 crates/bowser_net/psl/gen_psl.py --check /path/to/public_suffix_list.dat

`src/psl.rs` (`list_version()`) reports the embedded version; its unit
tests assert the expected version string, so a stale regeneration is
caught by `cargo test -p bowser_net`.
