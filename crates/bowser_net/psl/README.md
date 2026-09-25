# Vendored Public Suffix List

`../src/psl_data.bin` is a compact binary trie built from the canonical
Public Suffix List at <https://publicsuffix.org/list/public_suffix_list.dat>.

- Source: Mozilla Public License 2.0 (see the list header for terms).
- Vendored version: `2026-09-24_13-26-36_UTC`
  (commit `a179a48c465e818cfd8d626691cb317985da87fb`).
- 10,334 rules (289 wildcard, 8 exception) compressed into a 157,204-byte
  reversed-label trie (vs ~335 KB raw). Unicode rules are IDNA-encoded to
  ASCII at build time, so the trie is pure ASCII.

Regenerate after pulling a fresh list:

    python3 crates/bowser_net/psl/gen_psl.py /path/to/public_suffix_list.dat

`src/psl.rs` (`list_version()`) reports the embedded version; its unit
tests assert the expected version string, so a stale regeneration is
caught by `cargo test -p bowser_net`.
