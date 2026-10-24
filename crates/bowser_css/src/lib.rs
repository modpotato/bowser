//! CSS parsing: tokenizer, values, selectors, and stylesheet rules.
//!
//! [`parse_stylesheet`] turns stylesheet text into a [`Stylesheet`] (a list
//! of [`Rule`]s in source order) and never fails: per CSS Syntax Level 3,
//! parse errors drop just the offending rule or declaration.
//!
//! Shorthands (`margin`, `font`, `border`, `background`, `list-style`, …)
//! are expanded to longhands at parse time, with omitted longhands reset
//! to their initial values. A shorthand whose value contains a `var()`
//! reference is instead stored as-is under the shorthand property name,
//! so the cascade step can substitute the variables first and expand it
//! afterwards.

#![forbid(unsafe_code)]
// `computed_props!` in `cascade.rs` recurses ~158 deep while assigning
// property indices, past the default 128 macro-recursion limit.
#![recursion_limit = "256"]

mod cascade;
mod cbprops;
mod kwatom;
mod matching;
mod parse;
mod propatom;
mod selector;
mod token;
mod value;

pub use cascade::{
    BEFORE_AFTER_ATOMS, ComputedStyle, MAX_GRID_LINE, SnapAlign, SnapStop, SnapType, Styler,
    Viewport, compute_style, content_text, counter_adjustments, default_ua_stylesheet,
    is_valid_for_property, props, selectors_are_subtree_local,
};
pub use cbprops::{Filter, FilterFunction, establishes_fixed_cb};
pub use kwatom::{Keyword, KeywordAtom};
pub use matching::{
    ElementState, MatchScratch, matches, matches_with_scratch, matches_with_state,
    matches_with_state_and_scratch,
};
pub use parse::{Declaration, MediaQuery, Rule, Stylesheet, parse_declarations, parse_stylesheet};
pub use propatom::{PropertyAtom, PropertyName};
pub use selector::{
    Selector, SelectorParseError, has_combinator, parse_selector_list, specificity,
};
pub use token::Token;
pub use token::serialize_tokens;
pub use value::{
    AttrData, BasicShape, CalcSizeBasis, ClipPath, Color, CounterData, CssValue, LengthUnit,
    RepeatMode, Shadow, ShapeRadius, TransformContext, TransformFunction, TransformMatrix,
    TransformOrigin, compose_transforms, evaluate_transform, length_to_px, parse_css_angle_deg,
    parse_css_color, resolve_absolute_length,
};
