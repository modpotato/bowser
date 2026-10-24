# Sync reflow on geometry getters — design probe (wave 72)

## Problem

Two WPT groups need it (16 + 10 subtests):

- `css-position/sticky/position-sticky-nested-*.html` (16): script writes
  `scroller.scrollTop = N`, then reads `getBoundingClientRect()` / `offsetTop`
  of the sticky box. Two gaps: (a) the write is dropped when the scroller has
  no published `ScrollState` yet (`scroll_state_for` → `None` until
  `rebuild_after_layout`); (b) the read sees the pre-script layout tree —
  nothing re-runs layout between the write and the read.
- dynamic `static` → `fixed` (10): script flips `el.style.position`, then
  reads geometry. Same (b): the style dirt sits in the document's dirty
  report until the next `refresh`; the getter measures the stale tree.

## Where the pieces live

- Geometry getters: `bowser_js/src/bindings.rs` (`node_geom_long` →
  `measure_node_geometry`, `n_get_bounding_client_rect`, `n_scroll_top_get`,
  …). They read `WebApis::layout_tree` (the last published `Rc<LayoutBox>`)
  plus the `geom_cache`. They have **no path to the `Tab`**.
- Mutation dirt: `Document::take_dirty` → `Tab::sync_dirty_flags` →
  `Tab::dirty.{style,layout,paint}` → serviced by `refresh(tab, fonts)` in
  `bowser_app/src/window.rs` on the window worker's next tick.
- `refresh` runs exactly the dirty stages (style → layout → scroll-tree
  rebuild → `publish_layout_tree` → paint prep). The layout stage bakes
  sticky constraints at the live scroll offsets.
- Threading: script natives run on the window worker thread (the same thread
  that owns the `Tab`), via `page_vm.pump()` inside
  `tab.installed_mut()` → `page_vm_mut()`.

## The borrow wall (the crux)

A geometry getter cannot call `refresh` today because the worker holds
`&mut Tab` (through `installed_mut()`) across `page_vm.pump()`. Handing the
getter a `&mut Tab` would be a second live mutable borrow — instant UB, even
single-threaded. `JsInvoker` (the only embedder hook `WebApis` carries) is
JS-callback-only.

So a sound sync-reflow hook needs, in order:

1. **End the `&mut Tab` borrow across the pump.** Change the pump call sites
   (`window.rs`, `script.rs` ×11, `tab.rs` ×5) from `tab.installed_mut()` to
   cloning an `Rc<RefCell<InstalledDocument>>` (or `Rc<RefCell<Tab>>`) and
   pumping through the `Rc`. During the pump the worker holds no `Tab`
   borrow, so a native *may* borrow it.
2. **A re-entrant layout entry point.** New trait next to `JsInvoker`, e.g.
   `LayoutEnsurer::ensure_layout(&self)`, implemented by the app: it upgrades
   a `Weak<RefCell<Tab>>`, runs `sync_dirty_flags()`, and — only if
   `dirty.style || dirty.layout` — runs the style+layout stages and
   `publish_layout_tree` (paint stays dirty for the next frame, like Blink).
   Must be audited for re-entrancy: no script may run during these stages
   (layout is already script-free), and the `fonts: &Mutex<FontCache>` must
   not be held across the call.
3. **Dirty-checked call sites.** Every geometry getter calls
   `ensure_layout()` first. Cost model = Blink's forced sync layout: O(1)
   flag check when clean; one style+layout pass per mutation→read pair when
   dirty. Mitigations already in the tree: the wave-51 partial-layout path
   services single-subtree dirt incrementally, and `geom_cache` dedups
   getter clusters within an epoch.

## Invalidation-cost estimate

- Clean read: one flag test — negligible.
- Dirty read: full `refresh` style+layout when the partial path declines
  (today it declines most multi-subtree dirt). That is the same price as the
  next frame would pay anyway; the getter just pays it early. Worst case is
  a script that interleaves writes and reads (layout thrash) — exactly the
  pattern Blink penalizes too, and our partial path + cache make the common
  single-mutation case cheaper than Blink's full reflow.
- No new allocations on the read path: the hook itself is a trait-object
  call; the layout pass reuses the existing arenas and pooled scratch.

## Narrower half shipped in wave 72 (no `Tab` borrow needed)

Gap (a) — dropped `scrollTop`/`scrollLeft`/`scrollTo`/`scrollBy` writes to
not-yet-published scrollers — is fixed without touching the borrow
discipline: `ElementScrollStates` gains a `pending_script` stash (mirroring
`pending_shim`), written by the setters when `scroll_state_for` is `None`,
applied (clamped) at the next `publish_with_origins`, pruned for nodes that
never become containers, and consulted by the getters so write→read
round-trips. This is a prerequisite for the full fix and fixes a real
dropped-write bug on its own, but it does **not** by itself make the
nested-sticky WPT tests pass — the read half still needs the reflow above.

## Recommendation

Land the `pending_script` stash now (wave 72). Schedule the borrow-discipline
restructure + `LayoutEnsurer` as its own wave: it touches the window worker's
core borrow pattern (22 `installed_mut()` sites) and needs a harsh-review
pass over re-entrancy before it can be trusted.
