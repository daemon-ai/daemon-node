//! CanonicalStore — port of `canonical.py`.
//!
//! Owner-scoped identity cards with monotonic version chains and a partial unique index on live rows
//! (`WHERE valid_until IS NULL`). `remember()` returns `created | unchanged | updated`
//! (`canonical.py` L196-L287). Scaffold.
