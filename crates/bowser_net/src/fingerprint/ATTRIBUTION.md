# Attribution

The fingerprint profile data in this module was transcribed from the profile
*constants* (cipher lists, curve lists, signature-scheme lists, extension
flags, HTTP/2 settings, header templates, User-Agent strings) in
[wreq-util](https://github.com/0x676e67/wreq-util) (`src/emulate/profile/`),
copyright its contributors, licensed under the Apache License, Version 2.0.

No wreq-util *code* is included: only the data values were taken, and they
were re-encoded into bowser_net's own table format with a different
representation (interned/compressed string pools, per-profile structs).

A copy of the Apache-2.0 license under which wreq-util is distributed:

https://www.apache.org/licenses/LICENSE-2.0
