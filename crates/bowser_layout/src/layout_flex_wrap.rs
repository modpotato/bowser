//! Multi-line flexbox (`flex-wrap: wrap | wrap-reverse`).
//!
//! The single-line driver lives in the parent module (`layout_flex` and
//! friends); this module runs that same per-line machinery once per flex
//! line and then stacks the lines across the container's cross size per
//! `align-content` (CSS flexbox sections 9.3-9.5).
//!
//! Entry is [`place`], chosen by `layout_flex` when `flex-wrap` is not
//! `nowrap`. Row containers collect lines immediately (their main size is
//! always definite); column containers with content-sized items measure
//! first, mirroring the single-line `ColumnMeasured` stage, and collect
//! lines on that resume. Each line runs the existing [`flex_lengths`]
//! (section 9.7) and [`flex_justify`] (section 9.6); the cross axis is
//! finished on the `WrapRowFinish` / `WrapColumnDone` resumes.
//!
//! Documented simplifications (on top of the single-line ones):
//! - Forced `break-before` / `break-after` values break flex lines
//!   (flexbox §9.3); `avoid` values are ignored.
//! - `inline-flex` shrink-to-fit collects lines at the definite
//!   available width first and fits to the longest line.
//! - Column wrap needs a definite container height to break; with an auto
//!   height everything stays on one line (flexbox section 9.3).
//! - Items measured for a column wrap (`needs_measure`) measure at
//!   their keyword width when the basis is a content keyword, and at
//!   the cross size otherwise, mirroring the single-line path.

use super::{
    AlignContent, BlockInput, FlexAlign, FlexCtx, FlexItem, FlexStage, FontMetrics, Forest,
    LayoutScratch, PrefWidthCache, SpecLen, Task, clear_pending_shifts, compute_mt_place,
    flex_finish_container, flex_justify, flex_lengths, fresh_floats, preferred_width_mcw,
    record_shift, resolve_metrics,
};

/// Collect flex lines per CSS flexbox section 9.3: items pack in
/// `order`-sorted sequence; before an item whose hypothetical outer main
/// size would overflow the container's definite main size, the line breaks
/// (a non-empty line never breaks before its first item, so an oversize
/// item still gets its own line rather than stranding an empty line).
/// Each item after the first on a line consumes the main-axis gap before
/// it, so gaps count toward the break decision.
/// A forced `break-before` starts a new line at the item and a forced
/// `break-after` ends the line after the item (flexbox §9.3); forced
/// breaks apply in both row and column collection. Forced breaks apply
/// even when the main size is indefinite (`None`); only size-driven
/// breaks need a definite main size.
pub(super) fn collect_lines(
    items: &[FlexItem],
    main_size: Option<f32>,
    main_gap: f32,
) -> Vec<(usize, usize)> {
    let mut lines = Vec::new();
    let (mut start, mut used) = (0usize, 0.0f32);
    for (i, it) in items.iter().enumerate() {
        // Forced breaks apply in both row and column collection, with or
        // without a definite main size (flexbox §9.3).
        if i > start && it.force_break_before {
            lines.push((start, i));
            start = i;
            used = 0.0;
        }
        let outer = it.base + it.main_bp + it.m_ms + it.m_me;
        // The item carries the gap stride before it unless it starts the
        // line.
        if let Some(main) = main_size
            && i > start
            && used + main_gap + outer > main
        {
            lines.push((start, i));
            start = i;
            used = 0.0;
        }
        used += outer + if i > start { main_gap } else { 0.0 };
        if it.force_break_after {
            lines.push((start, i + 1));
            start = i + 1;
            used = 0.0;
        }
    }
    if start < items.len() || lines.is_empty() {
        lines.push((start, items.len()));
    }
    lines
}

/// Wrap-path entry, called from `layout_flex` when `flex-wrap` is not
/// `nowrap`. Row containers break lines immediately; column containers
/// with content-sized items measure first (mirroring the single-line
/// `ColumnMeasured` stage) and break lines on that resume.
pub(super) fn place(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    ctx: &mut FlexCtx,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    horizontal: bool,
    scratch: &mut LayoutScratch,
) {
    if horizontal {
        ctx.lines = collect_lines(&ctx.items, Some(ctx.content_w), ctx.main_gap);
        wrap_place_row(forest, tasks, ctx, scratch);
    } else if ctx.items.iter().any(|it| it.needs_measure) {
        // Column items with auto flex-basis and auto height need their
        // natural main size measured before lines can be collected.
        ctx.stage = FlexStage::WrapColumnMeasured;
        // Staged in scratch and restored after: capacity survives
        // across containers.
        let mut measure_tasks = std::mem::take(&mut scratch.measure_tasks);
        for it in ctx.items.iter().rev() {
            if !it.needs_measure {
                continue;
            }
            measure_tasks.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: it.idx,
                cb_x: 0.0,
                cb_w: ctx.content_w,
                cb_h: ctx.hspec,
                shrink: false,
                // w76fix2 M4: mirror the single-line column measuring
                // task — keyword-basis items measure at the keyword
                // width, not the final cross size, and a zero
                // minimum-height override replaces a specified height
                // for generic keyword items, so the basis keyword —
                // not the height property — determines the measured
                // height.
                width_override: Some(it.measure_cross.unwrap_or(it.cross)),
                width_override_final: false,
                height_override: if it.measure_height_min {
                    Some(0.0)
                } else {
                    None
                },
                height_is_min: it.measure_height_min,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: false,
            }));
        }
        tasks.push(Task::Flex(ctx.clone()));
        tasks.append(&mut measure_tasks);
        scratch.measure_tasks = measure_tasks;
    } else {
        wrap_place_column(forest, tasks, ctx, fonts, pw_cache, scratch);
    }
}

/// Distribute flex lines across the container cross size per
/// `align-content` (flexbox section 9.5). Updates `line_cross` in place
/// (`stretch` grows the lines) and returns each line's offset from the
/// container content-box cross-start edge. `mirror` flips the offsets so
/// lines stack from the cross-end edge: the row path passes
/// `wrap_reverse` (the cross axis is vertical, unaffected by direction),
/// while the column path passes `cross_rtl ^ wrap_reverse` — under
/// `direction: rtl` the cross-start edge is the physical right, so lines
/// stack from the right unless `wrap-reverse` flips the axis again.
fn distribute_lines(
    line_cross: &mut [f32],
    cross_size: f32,
    align_content: AlignContent,
    mirror: bool,
    cross_gap: f32,
) -> Vec<f32> {
    let n = line_cross.len();
    let total: f32 = line_cross.iter().sum();
    // One cross-axis gap stride between adjacent lines: it is fixed, so
    // `align-content` distributes only the remaining free space.
    let gaps = cross_gap * n.saturating_sub(1) as f32;
    let free = cross_size - total - gaps;
    let mut offset = vec![0.0f32; n];
    // Stride between two adjacent lines (never after the last).
    let stride = |li: usize| if li + 1 < n { cross_gap } else { 0.0 };
    if align_content == AlignContent::Stretch && free > 0.0 {
        // Free space is split equally among the lines (flexbox 9.5).
        let each = free / n as f32;
        let mut cursor = 0.0;
        for (li, (lc, off)) in line_cross.iter_mut().zip(offset.iter_mut()).enumerate() {
            *off = cursor;
            *lc += each;
            cursor += *lc + stride(li);
        }
    } else {
        let (cursor, gap) = match align_content {
            AlignContent::End if free > 0.0 => (free, 0.0),
            AlignContent::Center if free > 0.0 => (free / 2.0, 0.0),
            AlignContent::Between if n > 1 && free > 0.0 => (0.0, free / (n - 1) as f32),
            AlignContent::Around if free > 0.0 => {
                let g = free / n as f32;
                (g / 2.0, g)
            }
            AlignContent::Evenly if free > 0.0 => {
                let g = free / (n + 1) as f32;
                (g, g)
            }
            // `Start`, `Stretch` with no free space, or overflow: lines
            // pack from the cross-start edge.
            _ => (0.0, 0.0),
        };
        let mut cursor = cursor;
        for (li, (lc, off)) in line_cross.iter().zip(offset.iter_mut()).enumerate() {
            *off = cursor;
            cursor += *lc + gap + stride(li);
        }
    }
    if mirror {
        for (lc, off) in line_cross.iter().zip(offset.iter_mut()) {
            *off = cross_size - *off - *lc;
        }
    }
    offset
}

/// Place a wrapping row container's items: resolve flexible lengths and
/// justification per line, lay out children at provisional (flex-start)
/// cross positions, and resume in [`wrap_row_finish`] to stack the lines.
fn wrap_place_row(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    ctx: &mut FlexCtx,
    scratch: &mut LayoutScratch,
) {
    for &(s, e) in ctx.lines.iter() {
        flex_lengths(
            &mut ctx.items[s..e],
            Some(ctx.content_w),
            ctx.main_gap,
            scratch,
        );
    }
    // Final horizontal metrics and collapsed top margins before
    // positioning.
    for it in ctx.items.iter_mut() {
        resolve_metrics(
            forest,
            it.idx,
            0.0,
            ctx.content_w,
            false,
            Some(it.target),
            true,
        );
        compute_mt_place(forest, it.idx, ctx.hspec);
        it.m_cs = forest.nodes[it.idx].mt_place;
    }
    for &(s, e) in ctx.lines.iter() {
        flex_justify(
            &mut ctx.items[s..e],
            Some(ctx.content_w),
            ctx.main_gap,
            ctx.justify,
            ctx.reverse,
        );
    }
    ctx.stage = FlexStage::WrapRowFinish;
    // Staged in scratch and restored after: capacity survives across
    // containers.
    let mut child_tasks = std::mem::take(&mut scratch.child_tasks);
    for it in ctx.items.iter_mut().rev() {
        let bx = ctx.content_x + it.main_pos;
        // Provisional cross position; `wrap_row_finish` shifts to the
        // final one once line cross sizes are known.
        let by = ctx.content_top + it.m_cs;
        it.cross_pos = it.m_cs;
        let n = &mut forest.nodes[it.idx];
        n.rect.x = bx;
        n.rect.y = by;
        child_tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: it.idx,
            cb_x: bx - it.m_ms,
            cb_w: ctx.content_w,
            cb_h: ctx.hspec,
            shrink: false,
            width_override: Some(it.target),
            width_override_final: true,
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
    tasks.push(Task::Flex(ctx.clone()));
    tasks.append(&mut child_tasks);
    scratch.child_tasks = child_tasks;
}

/// Row wrap cross axis (`WrapRowFinish` resume): with natural item heights
/// known, size each line, stack the lines per `align-content`, align items
/// within their line (stretching where applicable, re-laying those out),
/// and shift the rest into place.
pub(super) fn wrap_row_finish(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    ctx: &mut FlexCtx,
    scratch: &mut LayoutScratch,
) {
    for it in ctx.items.iter_mut() {
        it.natural_cross = forest.nodes[it.idx].content_h;
        it.baseline = forest.nodes[it.idx].baseline;
    }
    let nlines = ctx.lines.len();
    let mut line_cross = vec![0.0f32; nlines];
    let mut line_bl = vec![0.0f32; nlines];
    for (li, &(s, e)) in ctx.lines.iter().enumerate() {
        let mut lc = 0.0f32;
        let mut bl = 0.0f32;
        for it in &ctx.items[s..e] {
            lc = lc.max(it.m_cs + it.natural_cross + it.cross_bp + it.m_ce);
            if it.align == FlexAlign::Baseline {
                bl = bl.max(it.m_cs + it.baseline);
            }
        }
        line_cross[li] = lc;
        line_bl[li] = bl;
    }
    // The lines fill the definite container cross size, or exactly their
    // own extent plus the inter-line gaps when the container height is
    // auto.
    let cross_size = ctx.hspec.unwrap_or_else(|| {
        line_cross.iter().sum::<f32>() + ctx.cross_gap * nlines.saturating_sub(1) as f32
    });
    let line_off = distribute_lines(
        &mut line_cross,
        cross_size,
        ctx.align_content,
        ctx.wrap_reverse,
        ctx.cross_gap,
    );
    for (li, &(s, e)) in ctx.lines.iter().enumerate() {
        let lc = line_cross[li];
        let top = line_off[li];
        let max_bl = line_bl[li];
        for it in ctx.items[s..e].iter_mut() {
            let outer = it.natural_cross + it.cross_bp;
            // Cross-axis auto margins absorb positive free space before
            // `align-self` (flexbox 8.3); with any auto cross margin
            // present alignment is skipped, and `stretch` does not apply
            // either (flexbox 9.8: stretch needs both cross margins
            // non-auto).
            // `wrap-reverse` flips the cross axis: cross-start is the
            // line's physical end, so Start/End/Center and auto margins
            // resolve against the swapped margins, and the border-box
            // offset is measured from the cross-start edge, then
            // mirrored back to physical (top/left-relative) coordinates.
            let rev = ctx.wrap_reverse;
            let (m_s, m_e) = if rev {
                (it.m_ce, it.m_cs)
            } else {
                (it.m_cs, it.m_ce)
            };
            let (auto_s, auto_e) = if rev {
                (it.auto_ce, it.auto_cs)
            } else {
                (it.auto_cs, it.auto_ce)
            };
            let free = lc - m_s - outer - m_e;
            let auto_n = usize::from(auto_s) + usize::from(auto_e);
            let to_phys = |d: f32| if rev { lc - d - outer } else { d };
            let (cross, pos) = if free > 0.0 && auto_n > 0 {
                let each = free / auto_n as f32;
                (
                    it.natural_cross,
                    to_phys(m_s + if auto_s { each } else { 0.0 }),
                )
            } else {
                match it.align {
                    FlexAlign::Stretch => {
                        // Stretch only applies to auto cross sizes.
                        let auto_cross = !it.auto_cs
                            && !it.auto_ce
                            && matches!(forest.nodes[it.idx].data.height, SpecLen::Auto);
                        if auto_cross {
                            // Aspect-ratio arm (flexbox §9.4 step 5): a
                            // replaced item with an intrinsic ratio keeps
                            // its hypothetical cross size — transferred
                            // from the §9.7 main size by `layout_block`'s
                            // sequential min/max pass, already in
                            // `natural_cross` — instead of stretching.
                            if super::intrinsic_ratio(forest, it.idx).is_some() {
                                (it.natural_cross, to_phys(m_s))
                            } else {
                                // A stretched item fills the line exactly, so
                                // its physical offset is the same either way.
                                let c = (lc - m_s - m_e - it.cross_bp).max(0.0);
                                it.stretched = true;
                                (c, it.m_cs)
                            }
                        } else {
                            (it.natural_cross, to_phys(m_s))
                        }
                    }
                    FlexAlign::Start => (it.natural_cross, to_phys(m_s)),
                    FlexAlign::End => (it.natural_cross, to_phys(lc - m_e - outer)),
                    FlexAlign::Center => (
                        it.natural_cross,
                        to_phys(m_s + (lc - m_s - m_e - outer) / 2.0),
                    ),
                    FlexAlign::Baseline => (it.natural_cross, max_bl - it.baseline),
                }
            };
            it.cross = cross;
            // Position relative to the container content-box cross-start
            // edge; the provisional rect sits at `cross_pos`.
            let final_pos = top + pos;
            if it.stretched {
                // Final position is known; the re-layout below fills the
                // stretched cross size (children need the rect first).
                let n = &mut forest.nodes[it.idx];
                n.rect.y = ctx.content_top + final_pos;
                it.cross_pos = final_pos;
            } else {
                let dy = final_pos - it.cross_pos;
                if dy != 0.0 {
                    // Deferred cross-axis shift, applied once by
                    // `apply_pending_shifts` after `run_tasks`.
                    record_shift(forest, it.idx, 0.0, dy);
                }
                it.cross_pos = final_pos;
            }
        }
    }
    // `cross_size` is the content-box cross extent the container needs:
    // with an auto height it is exactly the stacked lines.
    ctx.container_cross = cross_size;
    if ctx.items.iter().any(|it| it.stretched) {
        ctx.stage = FlexStage::WrapRowDone;
        // Staged in scratch and restored after: capacity survives
        // across containers.
        let mut relayout = std::mem::take(&mut scratch.relayout);
        for it in ctx.items.iter().rev() {
            if !it.stretched {
                continue;
            }
            // The subtree is re-laid out from provisional rects below;
            // drop its deferred inline-block shifts first so the fresh
            // pass re-records them instead of applying stale ones twice.
            clear_pending_shifts(forest, it.idx);
            relayout.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: it.idx,
                cb_x: ctx.content_x + it.main_pos - it.m_ms,
                cb_w: ctx.content_w,
                cb_h: ctx.hspec,
                shrink: false,
                width_override: Some(it.target),
                width_override_final: true,
                height_override: Some(it.cross),
                height_is_min: false,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: false,
            }));
        }
        tasks.push(Task::Flex(ctx.clone()));
        tasks.append(&mut relayout);
        scratch.relayout = relayout;
    } else {
        flex_finish_container(forest, ctx, cross_size);
    }
}

/// Row wrap finalize (`WrapRowDone` resume): stretched items are re-laid
/// out; the container height was fixed in [`wrap_row_finish`].
pub(super) fn wrap_row_done(forest: &mut Forest, ctx: &FlexCtx) {
    flex_finish_container(forest, ctx, ctx.container_cross);
}

/// Place a wrapping column container's items: collect flex lines (the
/// main size must be definite to break, flexbox 9.3), resolve flexible
/// lengths and justification per line, size each line (column) across the
/// container width per `align-content`, and lay children out exactly once
/// at final positions.
pub(super) fn wrap_place_column(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    ctx: &mut FlexCtx,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    if ctx.lines.is_empty() {
        ctx.lines = collect_lines(&ctx.items, ctx.hspec, ctx.main_gap);
    }
    for &(s, e) in ctx.lines.iter() {
        flex_lengths(&mut ctx.items[s..e], ctx.hspec, ctx.main_gap, scratch);
    }
    // Collapse each item's top margin with its first child before
    // distributing main-axis positions.
    for it in ctx.items.iter_mut() {
        compute_mt_place(forest, it.idx, Some(it.target));
        it.m_ms = forest.nodes[it.idx].mt_place;
    }
    for &(s, e) in ctx.lines.iter() {
        flex_justify(
            &mut ctx.items[s..e],
            ctx.hspec,
            ctx.main_gap,
            ctx.justify,
            ctx.reverse,
        );
    }
    // Hypothetical cross size per item for line sizing (flexbox 9.2,
    // simplified): fixed-width items keep their used width; auto-width
    // blocks contribute their max-content width so columns shrink to fit.
    let nlines = ctx.lines.len();
    let mut line_cross = vec![0.0f32; nlines];
    for (li, &(s, e)) in ctx.lines.iter().enumerate() {
        let mut lc = 0.0f32;
        for i in s..e {
            let it = &ctx.items[i];
            let hypo = if it.cross_auto {
                match forest.nodes[it.idx].intrinsic {
                    // Replaced items contribute the intrinsic width
                    // clamped by min/max-width, mirroring the
                    // engine's `resolve_metrics` auto-intrinsic arm
                    // (percentage clamps resolve against the definite
                    // container width, as there).
                    Some(dims) => {
                        let d = &forest.nodes[it.idx].data;
                        let cw = ctx.content_w;
                        let r = |sp: SpecLen| {
                            sp.resolve(
                                d.font_size,
                                forest.root_font_size,
                                forest.vp_w,
                                forest.vp_h,
                                cw,
                            )
                        };
                        let min_w = r(d.min_width).unwrap_or(0.0).max(0.0);
                        let max_w = r(d.max_width).unwrap_or(f32::INFINITY).max(0.0);
                        dims.width_or_default().max(0.0).min(max_w).max(min_w)
                    }
                    // One shared `mcw` and one pooled walk pair serve every
                    // item (see `LayoutScratch`).
                    None => preferred_width_mcw(
                        forest,
                        it.idx,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                    ),
                }
            } else {
                it.cross
            };
            lc = lc.max(it.m_cs + hypo + it.cross_bp + it.m_ce);
        }
        line_cross[li] = lc;
    }
    let line_left = distribute_lines(
        &mut line_cross,
        ctx.content_w,
        ctx.align_content,
        // The cross axis is horizontal here: under `direction: rtl` the
        // cross-start edge is the physical right, so lines stack from
        // the right; `wrap-reverse` flips the axis once more.
        ctx.cross_rtl ^ ctx.wrap_reverse,
        ctx.cross_gap,
    );
    // Each item's line, for the placement pass below.
    let mut item_line = vec![0usize; ctx.items.len()];
    for (li, &(s, e)) in ctx.lines.iter().enumerate() {
        item_line[s..e].fill(li);
    }
    // Per-item cross size and position within its line, mirroring the
    // single-line column path with the line's cross size in place of the
    // container's.
    for (li, &(s, e)) in ctx.lines.iter().enumerate() {
        let lc = line_cross[li];
        for it in ctx.items[s..e].iter_mut() {
            // Cross-axis auto margins absorb positive free space before
            // `align-self` (flexbox 8.3) and suppress the auto-width fill
            // (flexbox 9.8: stretch needs both cross margins non-auto).
            // `wrap-reverse` swaps cross-start/cross-end (flexbox §9.3);
            // the margins below are already in cross-start/cross-end
            // order for `direction: rtl` (swapped at item build), so only
            // the wrap flip applies here. Positions are measured from
            // the cross-start edge and mirrored to physical coordinates
            // when that edge is the physical right (`direction: rtl`
            // XOR `wrap-reverse`).
            let rev = ctx.wrap_reverse;
            let (m_s, m_e) = if rev {
                (it.m_ce, it.m_cs)
            } else {
                (it.m_cs, it.m_ce)
            };
            let (auto_s, auto_e) = if rev {
                (it.auto_ce, it.auto_cs)
            } else {
                (it.auto_cs, it.auto_ce)
            };
            let auto_n = usize::from(auto_s) + usize::from(auto_e);
            // Auto cross sizes fill the line (like stretch); specified
            // widths keep their size and are aligned within the line.
            if it.cross_auto && auto_n == 0 {
                // Aspect-ratio arm (flexbox §9.4 step 5): a replaced item
                // with an intrinsic ratio sizes its cross axis through
                // the ratio from the §9.7 main size instead of filling
                // the line — the fill must not distort it. The main size
                // is final, so the transfer never grows it back.
                it.cross = match super::intrinsic_ratio(forest, it.idx) {
                    Some(ratio) => {
                        let (w, _) = super::replaced_minmax_transfer(
                            forest,
                            it.idx,
                            ctx.content_w,
                            Some(it.target),
                            (it.target * ratio).max(0.0),
                            it.target,
                            ratio,
                            true,
                            false,
                            false,
                        );
                        w
                    }
                    None => (lc - m_s - m_e - it.cross_bp).max(0.0),
                };
            }
            let outer = it.cross + it.cross_bp;
            let free = lc - m_s - outer - m_e;
            let to_phys = |d: f32| {
                if ctx.cross_rtl ^ rev {
                    lc - d - outer
                } else {
                    d
                }
            };
            it.cross_pos = if free > 0.0 && auto_n > 0 {
                let each = free / auto_n as f32;
                to_phys(m_s + if auto_s { each } else { 0.0 })
            } else {
                match it.align {
                    FlexAlign::Start | FlexAlign::Stretch => to_phys(m_s),
                    FlexAlign::Baseline => to_phys(m_s),
                    FlexAlign::End => to_phys(lc - m_e - outer),
                    FlexAlign::Center => to_phys(m_s + (lc - m_s - m_e - outer) / 2.0),
                }
            };
        }
    }
    ctx.stage = FlexStage::WrapColumnDone;
    // Staged in scratch and restored after: capacity survives across
    // containers.
    let mut child_tasks = std::mem::take(&mut scratch.child_tasks);
    for i in (0..ctx.items.len()).rev() {
        let it = ctx.items[i].clone();
        let bx = ctx.content_x + line_left[item_line[i]] + it.cross_pos;
        let by = ctx.content_top + it.main_pos;
        resolve_metrics(
            forest,
            it.idx,
            0.0,
            ctx.content_w,
            false,
            Some(it.cross),
            false,
        );
        let n = &mut forest.nodes[it.idx];
        n.rect.x = bx;
        n.rect.y = by;
        child_tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: it.idx,
            // `cb_x` is the physical margin-box left edge: with
            // `direction: rtl` the physical left margin is the
            // cross-end one (margins were swapped at item build).
            cb_x: bx - if ctx.cross_rtl { it.m_ce } else { it.m_cs },
            cb_w: ctx.content_w,
            cb_h: Some(it.target),
            shrink: false,
            width_override: Some(it.cross),
            width_override_final: false,
            height_override: Some(it.target),
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
    tasks.push(Task::Flex(ctx.clone()));
    tasks.append(&mut child_tasks);
    scratch.child_tasks = child_tasks;
}

/// Column wrap finalize (`WrapColumnDone` resume): the container's main
/// extent is the largest line extent (a definite height wins inside
/// `flex_finish_container`).
pub(super) fn wrap_column_done(forest: &mut Forest, ctx: &FlexCtx) {
    let mut natural = 0.0f32;
    for &(s, e) in &ctx.lines {
        let items_sum: f32 = ctx.items[s..e]
            .iter()
            .map(|it| it.m_ms + it.target + it.main_bp + it.m_me)
            .sum();
        // One main-axis gap stride between adjacent items on the line.
        let line_main = items_sum + ctx.main_gap * (e.saturating_sub(s).saturating_sub(1)) as f32;
        natural = natural.max(line_main);
    }
    flex_finish_container(forest, ctx, natural);
}

#[cfg(test)]
mod flex_wrap_tests {
    use crate::{FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn body(l: &LayoutBox) -> &LayoutBox {
        &l.children[0]
    }

    /// The first child of `<body>`: the flex container under test.
    fn first(l: &LayoutBox) -> &LayoutBox {
        &body(l).children[0]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    fn three_items() -> &'static str {
        r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#
    }

    #[test]
    fn row_wrap_breaks_items_into_lines() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 120px; height: 20px; } \
             .b { flex-basis: 120px; height: 20px; } \
             .d { flex-basis: 120px; height: 20px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // 120 + 120 fits; the third item (360 > 300) starts a new line.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 120.0));
        assert!(approx(d.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(d.rect.y, c.rect.y + 20.0));
        assert!(approx(c.rect.h, 40.0));
    }

    #[test]
    fn row_wrap_resolves_flex_per_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 120px; flex-grow: 1; height: 10px; } \
             .b { flex-basis: 120px; flex-grow: 1; height: 10px; } \
             .d { flex-basis: 120px; flex-grow: 1; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // Free space is distributed within each line, not across lines:
        // line 1 splits 60px over a and b; d alone takes the whole line.
        assert!(approx(a.rect.w, 150.0));
        assert!(approx(b.rect.w, 150.0));
        assert!(approx(b.rect.x, c.rect.x + 150.0));
        assert!(approx(d.rect.w, 300.0));
        assert!(approx(d.rect.x, c.rect.x));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
    }

    #[test]
    fn row_wrap_justify_center_applies_per_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; justify-content: center; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // 120px items: two fit per 300px line, so each line centers its
        // own free space (30px on line 1, 90px on line 2).
        assert!(approx(a.rect.x, c.rect.x + 30.0));
        assert!(approx(b.rect.x, c.rect.x + 150.0));
        assert!(approx(d.rect.x, c.rect.x + 90.0));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
    }

    #[test]
    fn row_wrap_reverse_stacks_lines_bottom_up() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap-reverse; width: 300px; } \
             .a { flex-basis: 120px; height: 20px; } \
             .b { flex-basis: 120px; height: 20px; } \
             .d { flex-basis: 120px; height: 20px; }",
        );
        let c = first(&l);
        let (a, d) = (&c.children[0], &c.children[2]);
        // Two 20px lines in a 40px container: the first line sits at the
        // bottom, the second at the top.
        assert!(approx(a.rect.y, c.rect.y + 20.0));
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 40.0));
    }

    #[test]
    fn row_wrap_reverse_align_items_start_aligns_to_cross_start() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-wrap: wrap-reverse; width: 300px; height: 60px; \
               align-items: flex-start; align-content: flex-start; } \
             .a { flex-basis: 150px; height: 20px; } \
             .b { flex-basis: 150px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // One 20px line stacked from the bottom (y=40..60). With
        // wrap-reverse the cross-start edge is the physical bottom, so
        // `flex-start` puts the 20px item at y=40 and the 10px item at
        // y=50.
        assert!(approx(a.rect.y, c.rect.y + 40.0));
        assert!(approx(b.rect.y, c.rect.y + 50.0));
    }

    #[test]
    fn row_wrap_reverse_align_items_end_aligns_to_cross_end() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-wrap: wrap-reverse; width: 300px; height: 60px; \
               align-items: flex-end; align-content: flex-start; } \
             .a { flex-basis: 150px; height: 20px; } \
             .b { flex-basis: 150px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // `flex-end` is the physical top of the line: both items start
        // at y=40.
        assert!(approx(a.rect.y, c.rect.y + 40.0));
        assert!(approx(b.rect.y, c.rect.y + 40.0));
    }

    #[test]
    fn row_wrap_align_content_center() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; height: 100px; \
               align-content: center; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, d) = (&c.children[0], &c.children[2]);
        // 80px of free space centers the two 10px lines at 40 and 50.
        assert!(approx(a.rect.y, c.rect.y + 40.0));
        assert!(approx(d.rect.y, c.rect.y + 50.0));
        assert!(approx(c.rect.h, 100.0));
    }

    #[test]
    fn row_wrap_align_content_space_between() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; height: 100px; \
               align-content: space-between; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, d) = (&c.children[0], &c.children[2]);
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(d.rect.y, c.rect.y + 90.0));
    }

    #[test]
    fn row_wrap_oversize_item_gets_own_line() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 500px; height: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // No empty line is stranded before it; it shrinks onto its own line.
        assert!(approx(a.rect.w, 300.0));
        assert!(approx(c.rect.h, 10.0));
    }

    #[test]
    fn row_wrap_break_before_column_forces_new_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; break-before: column; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // `break-before: column` is a forced break (flexbox §9.3): b
        // starts a new line even though it would fit beside a (240 < 300).
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x));
        assert!(approx(b.rect.y, c.rect.y + 10.0));
        assert!(approx(d.rect.x, c.rect.x + 120.0));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
        assert!(approx(c.rect.h, 20.0));
    }

    #[test]
    fn row_wrap_break_after_page_forces_new_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 120px; height: 10px; break-after: page; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // `break-after: page` is a forced break value too: the first line
        // ends after a even though b would fit beside it.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x));
        assert!(approx(b.rect.y, c.rect.y + 10.0));
        assert!(approx(d.rect.x, c.rect.x + 120.0));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
        assert!(approx(c.rect.h, 20.0));
    }

    #[test]
    fn row_wrap_break_before_avoid_breaks_nothing() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; break-before: avoid; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // `avoid` is not a forced break: packing is purely size-driven.
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.y, c.rect.y));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
    }

    #[test]
    fn nowrap_stays_single_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: nowrap; width: 300px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // One line, 60px of overflow shrunk evenly: 100px each.
        assert!(approx(a.rect.w, 100.0));
        assert!(approx(b.rect.x, c.rect.x + 100.0));
        assert!(approx(d.rect.x, c.rect.x + 200.0));
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 10.0));
    }

    #[test]
    fn column_wrap_breaks_into_columns() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; \
               width: 400px; height: 100px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // a and b fill the first 100px column; d (120 > 100) wraps to a
        // second column. `align-content: stretch` widens each 50px column
        // to 200px.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x));
        assert!(approx(b.rect.y, c.rect.y + 40.0));
        assert!(approx(d.rect.x, c.rect.x + 200.0));
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(d.rect.w, 50.0));
        assert!(approx(c.rect.h, 100.0));
    }

    // w76fix2 M4: wrapped column keyword bases measure at the keyword
    // width, mirroring the nowrap test: container 130px, min-content
    // width 60 wraps the text to 3 lines, fit-content width 130 to 2
    // lines, max-content width 192 to 1 line, so the heights stand in
    // a 3:2:1 ratio. Before the fix the wrap measuring task used the
    // final cross size for all three, collapsing the distinction.
    #[test]
    fn column_wrap_basis_content_keywords_measure_at_keyword_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div><div class="b">aaaaaaaaaa bbbbbbbbbb cccccccccc</div><div class="d">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; flex-wrap: wrap; width: 130px; } \
             .a { font-size: 10px; flex-basis: min-content; flex-shrink: 0; } \
             .b { font-size: 10px; flex-basis: fit-content; flex-shrink: 0; } \
             .d { font-size: 10px; flex-basis: max-content; flex-shrink: 0; }",
        );
        let c = first(&l);
        let (ha, hb, hd) = (
            c.children[0].rect.h,
            c.children[1].rect.h,
            c.children[2].rect.h,
        );
        assert!(ha.is_finite() && hb.is_finite() && hd.is_finite());
        assert!(
            ha > hb && hb > hd,
            "wrap min {ha} / fit {hb} / max {hd} not 3:2:1 lines"
        );
        assert!(
            approx(ha, hb + hd),
            "wrap min {ha} != fit {hb} + max {hd} (3 lines != 2 + 1)"
        );
    }

    #[test]
    fn column_wrap_reverse_align_items_start_aligns_to_cross_start() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap-reverse; \
               width: 60px; height: 300px; align-items: flex-start; } \
             .a { height: 60px; width: 20px; } \
             .b { height: 60px; width: 10px; } \
             .d { height: 60px; width: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // One line, stretched to the 60px container width. With
        // wrap-reverse the cross-start edge is the physical right edge,
        // so `flex-start` puts the 20px item at x=40 and the 10px items
        // at x=50.
        assert!(approx(a.rect.x, c.rect.x + 40.0));
        assert!(approx(b.rect.x, c.rect.x + 50.0));
        assert!(approx(d.rect.x, c.rect.x + 50.0));
    }

    #[test]
    fn column_wrap_rtl_stacks_columns_from_the_right() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; direction: rtl; \
               width: 400px; height: 100px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // a and b fill the first 100px column; d wraps to a second column.
        // `align-content: stretch` widens each 50px column to 200px. Under
        // `direction: rtl` the cross-start edge is the physical right, so
        // the first column stacks at x=200..400 and the second at x=0..200;
        // items align to the physical right within their 200px column.
        assert!(approx(a.rect.x, c.rect.x + 350.0), "a.x = {}", a.rect.x);
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x + 350.0), "b.x = {}", b.rect.x);
        assert!(approx(b.rect.y, c.rect.y + 40.0));
        assert!(approx(d.rect.x, c.rect.x + 150.0), "d.x = {}", d.rect.x);
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 100.0));
    }

    #[test]
    fn column_wrap_rtl_wrap_reverse_stacks_columns_from_the_left() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap-reverse; direction: rtl; \
               width: 400px; height: 100px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // `wrap-reverse` flips the cross axis a second time, so columns
        // stack from the physical left again and items align to the
        // physical left within their 200px stretched column.
        assert!(approx(a.rect.x, c.rect.x), "a.x = {}", a.rect.x);
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x), "b.x = {}", b.rect.x);
        assert!(approx(b.rect.y, c.rect.y + 40.0));
        assert!(approx(d.rect.x, c.rect.x + 200.0), "d.x = {}", d.rect.x);
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 100.0));
    }

    #[test]
    fn column_wrap_rtl_baseline_aligns_like_flex_start() {
        // Locks in `to_phys(m_s)` for baseline items under column+RTL:
        // the cross-start edge is the physical right.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; \
               width: 100px; align-items: baseline; } \
             .a { height: 20px; width: 20px; } \
             .b { height: 20px; width: 30px; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // Single line stretched to the 100px container width; baseline
        // items sit against the physical right edge.
        assert!(approx(a.rect.x, c.rect.x + 80.0), "a.x = {}", a.rect.x);
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x + 70.0), "b.x = {}", b.rect.x);
        assert!(approx(b.rect.y, c.rect.y + 20.0));
    }

    #[test]
    fn column_wrap_break_after_column_forces_new_column() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; \
               width: 400px; height: 100px; } \
             .a { height: 30px; width: 50px; } \
             .b { height: 30px; width: 50px; break-after: column; } \
             .d { height: 30px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // a and b (60px) would leave room for d in the first 100px
        // column, but `break-after: column` forces d into a second
        // column. `align-content: stretch` widens each 50px column to
        // 200px.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x));
        assert!(approx(b.rect.y, c.rect.y + 30.0));
        assert!(approx(d.rect.x, c.rect.x + 200.0));
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 100.0));
    }

    #[test]
    fn column_wrap_auto_height_stays_single_line() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; width: 400px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (b, d) = (&c.children[1], &c.children[2]);
        // Indefinite main size: no size-driven breaking, one 120px column.
        assert!(approx(b.rect.y, c.rect.y + 40.0));
        assert!(approx(d.rect.y, c.rect.y + 80.0));
        assert!(approx(c.rect.h, 120.0));
    }

    #[test]
    fn column_wrap_auto_height_break_before_column_forces_new_column() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; width: 400px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; break-before: column; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // Forced breaks apply even with an indefinite main size (flexbox
        // §9.3): b starts a second column although everything would fit
        // in one 120px column. `align-content: stretch` widens each 50px
        // column to 200px.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x + 200.0));
        assert!(approx(b.rect.y, c.rect.y));
        assert!(approx(d.rect.x, c.rect.x + 200.0));
        assert!(approx(d.rect.y, c.rect.y + 40.0));
        assert!(approx(c.rect.h, 80.0));
    }

    #[test]
    fn row_wrap_gap_counts_toward_breaks() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 380px; column-gap: 20px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // 120 + 20 + 120 = 260 fits, but 260 + 20 + 120 = 400 > 380, so
        // the third item wraps. Without the gap 240 + 120 = 360 <= 380
        // would keep it on the first line, so the break decision is
        // genuinely gap-driven.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 140.0));
        assert!(approx(d.rect.x, c.rect.x));
        assert!(approx(d.rect.y, c.rect.y + 10.0));
    }

    #[test]
    fn inline_flex_percentage_gap_resolves_against_final_content_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: inline-flex; column-gap: 10%; } \
             .a { width: 100px; height: 10px; } \
             .b { width: 100px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // Available width is 800, but the percentage gap must resolve
        // against the final shrink-to-fit content width (css-align-3
        // §2.2), not the available width: during the estimate pass the
        // gap is 0 (css-sizing-3), so the container fits the 200px of
        // content exactly instead of inflating to 280px, and the used
        // gap is 10% of 200 = 20px.
        //
        // The items keep their 100px widths: their automatic minimum
        // (flexbox §4.5) is their min-content size, which is their
        // definite specified width (css-sizing-3 §5.2), so the 20px of
        // overflow cannot shrink them to 90px — it overflows the
        // container instead.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(c.rect.w, 200.0));
        assert!(approx(a.rect.w, 100.0));
        assert!(approx(b.rect.w, 100.0));
        assert!(approx(b.rect.x, c.rect.x + 120.0));
    }

    #[test]
    fn row_wrap_row_gap_separates_lines() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; row-gap: 10px; } \
             .a { flex-basis: 120px; height: 20px; } \
             .b { flex-basis: 120px; height: 20px; } \
             .d { flex-basis: 120px; height: 20px; }",
        );
        let c = first(&l);
        let (a, d) = (&c.children[0], &c.children[2]);
        // Two 20px lines plus one 10px inter-line gap: 50px tall.
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(d.rect.y, c.rect.y + 30.0));
        assert!(approx(c.rect.h, 50.0));
    }

    #[test]
    fn row_wrap_gap_justify_center_subtracts_gap() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; column-gap: 20px; \
               justify-content: center; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // Line 1 is 260px wide (120 + 20 + 120): 20px of free space on
        // each side. Line 2 centers its lone 120px item.
        assert!(approx(a.rect.x, c.rect.x + 20.0));
        assert!(approx(b.rect.x, c.rect.x + 160.0));
        assert!(approx(d.rect.x, c.rect.x + 90.0));
    }

    #[test]
    fn row_wrap_gap_justify_space_between() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; column-gap: 20px; \
               justify-content: space-between; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // Free space (40px) goes between the items on top of the 20px
        // gap: b ends flush at the line end.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 180.0));
    }

    #[test]
    fn row_wrap_gap_flex_grow_splits_remainder() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; column-gap: 20px; } \
             .a { flex-basis: 120px; flex-grow: 1; height: 10px; } \
             .b { flex-basis: 120px; flex-grow: 1; height: 10px; } \
             .d { flex-basis: 120px; flex-grow: 1; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // Line 1: 300 - 240 - 20 = 40px of free space, 20px per item.
        assert!(approx(a.rect.w, 140.0));
        assert!(approx(b.rect.x, c.rect.x + 160.0));
        assert!(approx(d.rect.w, 300.0));
    }

    #[test]
    fn row_wrap_percentage_row_gap_zero_with_auto_height() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; row-gap: 10%; } \
             .a { flex-basis: 120px; height: 20px; } \
             .b { flex-basis: 120px; height: 20px; } \
             .d { flex-basis: 120px; height: 20px; }",
        );
        let c = first(&l);
        // The height is indefinite, so the percentage row gap is zero:
        // the lines stack with no stride.
        assert!(approx(c.children[2].rect.y, c.rect.y + 20.0));
        assert!(approx(c.rect.h, 40.0));
    }

    #[test]
    fn row_wrap_align_content_center_counts_row_gap() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-wrap: wrap; width: 300px; height: 100px; \
               align-content: center; row-gap: 10px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, d) = (&c.children[0], &c.children[2]);
        // Two 10px lines plus a 10px gap = 30px, centered in 100px: the
        // first line starts at 35, the second at 55.
        assert!(approx(a.rect.y, c.rect.y + 35.0));
        assert!(approx(d.rect.y, c.rect.y + 55.0));
    }

    #[test]
    fn column_wrap_gaps_break_and_stack_columns() {
        let l = layout_html(
            three_items(),
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; \
               width: 400px; height: 130px; align-content: flex-start; gap: 30px 10px; } \
             .a { height: 40px; width: 50px; } \
             .b { height: 40px; width: 50px; } \
             .d { height: 40px; width: 50px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // Main axis: 40 + 30 + 40 = 110 fits, 110 + 30 + 40 = 180 > 130
        // wraps d into a second column. Cross axis: the 10px column gap
        // separates the 50px columns.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.x, c.rect.x));
        assert!(approx(b.rect.y, c.rect.y + 70.0));
        assert!(approx(d.rect.x, c.rect.x + 60.0));
        assert!(approx(d.rect.y, c.rect.y));
        assert!(approx(c.rect.h, 130.0));
    }

    #[test]
    fn wrap_row_cross_auto_margins_center_item() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-wrap: wrap; width: 600px; height: 200px; } \
             .a { flex-basis: 100px; height: 40px; margin: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // `margin: auto` absorbs free space on both axes of the line.
        assert!(approx(a.rect.x, c.rect.x + 250.0));
        assert!(approx(a.rect.y, c.rect.y + 80.0));
    }

    #[test]
    fn wrap_column_cross_auto_margins_center_item() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; \
               width: 600px; height: 200px; } \
             .a { height: 40px; width: 100px; margin: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.y, c.rect.y + 80.0));
        assert!(approx(a.rect.x, c.rect.x + 250.0));
    }

    #[test]
    fn wrap_reverse_row_cross_auto_margin_top_pushes_to_cross_start() {
        // wrap-reverse flips the cross axis: cross-start is the bottom,
        // so `margin-top: auto` (cross-end) absorbs the free space and
        // pushes the item to the bottom.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-wrap: wrap-reverse; width: 600px; height: 200px; } \
             .a { flex-basis: 100px; height: 40px; margin-top: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.y, c.rect.y + 160.0));
    }
}
