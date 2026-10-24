# Transform support — status ledger

Last updated: 2026-10-04 (tree at `bc65fcb`). Status of CSS transforms in
Bowser: what is wired, what is parsed-but-not-wired, and what does not
exist. A status ledger, not a design doc.

Everything below was verified against the code at this commit. All
transform work to date landed in **wave 34** (Oct 2026).

## Works today

- **CSS parsing of `transform`** (`crates/bowser_css/src/value.rs:137`,
  `parse_transform_function` at `:985`; registered in `COMPUTED_PROPS`,
  `crates/bowser_css/src/cascade.rs:401`, initial `none` at `:2640`,
  grammar validation at `:2985`).
  `transform: none` parses to `Keyword("none")`; anything else becomes
  `CssValue::Transform(Vec<TransformFunction>)` (`value.rs:121-135`) —
  one vec no matter how the list was written. `translate()` / `scale()` /
  `rotate()` (and single-axis forms) parse structurally; every other
  function (`matrix()` / `matrix3d()`, `skew*()`, `perspective()`,
  `*3d()` / `*z()` forms, nested `calc()` args) is retained opaque as
  `TransformFunction::Other`. A valid-but-unrenderable function still
  counts as non-`none` (it makes the box the CB for fixed descendants);
  only truly invalid syntax fails the declaration.
  Commits: `01d17d6` (w34 css: transform property registration),
  `d7b3c11` (w34 review: opaque transform functions, single
  `Transform(Vec)` shape).
- **`position: fixed` containing-block establishment.** A non-`none`
  transform makes the box the CB for fixed descendants (CSS Transforms 1
  §3): `resolve_containing_block`
  (`crates/bowser_layout/src/layout_position.rs:139`) stops at the
  *nearest* transformed ancestor for fixed boxes and uses its padding
  box; otherwise fixed is against the viewport. `has_transform` (`:125`)
  is the predicate; `crate::has_non_none_transform`
  (`crates/bowser_layout/src/lib.rs:680`) is the public style-level check.
  NOTE: this covers **fixed only** — see "Known gaps" below for absolute.
- **Shift carry for transformed ancestors.** `apply_pending_shifts`
  (`crates/bowser_layout/src/layout.rs:2738`) restarts shift accumulation
  at each fixed box (viewport-anchored fixed boxes must not inherit
  in-flow ancestor shifts), but a fixed box under a transformed CB rides
  the transformed ancestor's *carried* applied shift (`layout.rs:2758-2765`),
  since it was positioned against the ancestor's provisional rect.
  Commit `5ece536`.
- **Auto-offset fixed boxes in transformed frames.**
  `fix_fixed_static_frames` (`layout_position.rs:589`): for fixed boxes
  with `auto` offsets, ancestor pending shifts are added to the static
  position's frame, walking up to (but not through) the nearest
  transformed ancestor, so no double-counting with the carried shift.
  Commit `5ece536`.
- **Scrollable-extent fixed-in-transformed handling.**
  `scrollable_extent` (`crates/bowser_layout/src/lib.rs:751`):
  viewport-anchored fixed boxes never extend a scroller's scrollable
  overflow range; a fixed box whose CB is a transformed ancestor *inside*
  the scroller scrolls with it and contributes (`fixed_cb_inside`
  flag).
- **Scroll-container viewport-anchoredness.**
  `scroll_containers_with_offset` (`crates/bowser_layout/src/lib.rs:844`):
  tracks per-box viewport-anchoredness and the nearest transformed
  ancestor's frame; a fixed box takes its containing block's frame
  (viewport's when no transformed ancestor intervenes, else the
  transformed ancestor's).
- **Tests.** `crates/bowser_layout/src/layout_position.rs:1264-1400`:
  `fixed_insets_resolve_against_transformed_ancestor`,
  `fixed_without_transformed_ancestor_uses_viewport`,
  `transform_none_does_not_create_containing_block`,
  `fixed_uses_nearest_transformed_ancestor`,
  `fixed_prefers_transformed_ancestor_over_positioned`,
  `fixed_explicit_offsets_follow_shifted_transformed_cb`,
  `fixed_auto_offsets_with_fixed_ancestor_before_transformed_cb`.

## Parsed but not wired (parser-only)

- **Unrenderable transform functions.** `TransformFunction::Other`
  retains `matrix()`/`matrix3d()`, `skew*()`, `perspective()`, 3D forms
  only so they count as non-`none` for CB purposes. Nothing evaluates
  them into a matrix (`value.rs:110-130`: "layout only needs `none` vs
  non-`none` today").

## Not started / known gaps

- **`will-change`, `filter`, `contain`, `perspective` (property),
  `transform-origin`, `backface-visibility`, `transform-style`, and the
  individual `translate` / `rotate` / `scale` properties are not
  registered in `COMPUTED_PROPS` at all** — no parsing, no wiring.
  (Earlier exploration notes described `will-change`/`filter`/
  `perspective`/`contain` CB establishment as partially done; that was
  wrong. As of this commit there is nothing for these properties.)
- **Absolute descendants of transformed ancestors are NOT diverted.**
  Per CSS Transforms 1 §3 a non-`none` transform also establishes the CB
  for `position: absolute` descendants, but `resolve_containing_block`
  (`layout_position.rs:139-160`) only stops at transformed ancestors
  for fixed; absolute resolves against the nearest *positioned*
  ancestor only.
- **Visual rendering of transforms: none.** No matrix evaluation in CSS,
  no transform-bearing `DisplayItem` (`crates/bowser_paint/src/display_list.rs:306`),
  no support in either backend. `crates/bowser_paint/src/render.rs:125-126`
  lists transforms as a layout/paint TODO that renders as ignored
  (un-transformed).
- **Transform stacking contexts: none.**
  `creates_stacking_context` (`display_list.rs:1831`) explicitly excludes
  `transform` (and `filter`, `isolation`, `perspective`, `mask`): only
  `opacity < 1` and non-`normal` `mix-blend-mode` create stacking contexts
  outside positioning/`z-index` (`:1826-1842`).
- **SVG `transform` attribute:** N/A — there is no SVG crate and no SVG
  box tree (`crates/` has `bowser_url` … `bowser_app`; no `bowser_svg`).

## File pointers (jump targets for future workers)

| What | Where |
|---|---|
| `CssValue::Transform` shape + none-vs-non-none policy | `crates/bowser_css/src/value.rs:110-135` |
| `TransformFunction` enum | `crates/bowser_css/src/value.rs:137-` |
| Transform function parsing | `crates/bowser_css/src/value.rs:985` (`parse_transform_function`) |
| `transform` property registration / initial / grammar | `crates/bowser_css/src/cascade.rs:401`, `:2640`, `:2985` |
| Non-`none` check used across layout | `crates/bowser_layout/src/lib.rs:680` (`has_non_none_transform`) |
| Fixed CB resolution (transform walk) | `crates/bowser_layout/src/layout_position.rs:139` (`resolve_containing_block`), predicate at `:125` |
| Transformed-CB shift carry | `crates/bowser_layout/src/layout.rs:2738` (`apply_pending_shifts`), carried frame at `:2758-2765` |
| Auto-offset static-frame fixup | `crates/bowser_layout/src/layout_position.rs:589` (`fix_fixed_static_frames`) |
| Scrollable-extent fixed-in-transformed | `crates/bowser_layout/src/lib.rs:751` (`scrollable_extent`) |
| Scroll-container viewport-anchoredness | `crates/bowser_layout/src/lib.rs:844` (`scroll_containers_with_offset`) |
| Transform CB tests | `crates/bowser_layout/src/layout_position.rs:1264-1400` |
| Stacking-context exclusion note | `crates/bowser_paint/src/display_list.rs:1826-1844` |
| Render TODO (transforms ignored) | `crates/bowser_paint/src/render.rs:125-126` |
| Wave-34 attribution | `git log --grep=transform -i`: `01d17d6`, `60d07b9`, `d7b3c11`, `5ece536` |
