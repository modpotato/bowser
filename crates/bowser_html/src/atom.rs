//! Interned tag and attribute name atoms.
//!
//! Every element used to heap-allocate its tag name (`String` per
//! `<div>`), and the selector-matching hot paths compared those names
//! with `eq_ignore_ascii_case` per candidate. This module interns the
//! well-known tag and attribute names into a single global, immutable,
//! lock-free table: a well-known name is a `Copy` 4-byte [`NameAtom`]
//! (its index in `NAMES`), so hot-path compares are one `u32`
//! comparison with no allocation, no hashing, and no synchronization.
//!
//! Names outside the table (custom elements, `data-*` attributes, ...)
//! are not lost: [`Name`] is a two-tier type — a well-known atom plus an
//! optional heap sidecar that carries the original string exactly when
//! the atom is [`NameAtom::UNKNOWN`]. Equality stays exact: equal atoms
//! imply equal strings (the table holds unique strings, and interning is
//! deterministic), and two unknown names fall back to comparing their
//! sidecars.
//!
//! Case handling is the caller's job, matching the existing semantics:
//! the HTML tokenizer already ASCII-lowercases tag and attribute names,
//! and `document.createElement` / `setAttribute` lowercase for HTML
//! elements, so element-side atoms are canonical. Selector- and
//! API-side queries use [`NameAtom::lookup_html`] for the
//! ASCII case-insensitive HTML rule.

use std::fmt;
use std::ops::Deref;
use std::sync::OnceLock;

/// Well-known tag and attribute names, sorted byte-wise for binary
/// search. Tag and attribute names share one namespace: the string
/// `"div"` is the same atom whether it names an element or an
/// attribute. SVG/MathML names keep their spec case (`foreignObject`,
/// `viewBox`); everything the HTML tokenizer produces is already
/// ASCII-lowercased.
static NAMES: &[&str] = &[
    "a",
    "abbr",
    "accept",
    "accept-charset",
    "accesskey",
    "action",
    "address",
    "align",
    "allow",
    "allowfullscreen",
    "allowpaymentrequest",
    "alt",
    "animate",
    "animateTransform",
    "annotation",
    "annotation-xml",
    "applet",
    "area",
    "article",
    "aside",
    "async",
    "audio",
    "autocapitalize",
    "autocomplete",
    "autofocus",
    "autoplay",
    "b",
    "base",
    "basefont",
    "bdi",
    "bdo",
    "bgcolor",
    "bgsound",
    "big",
    "blockquote",
    "body",
    "border",
    "br",
    "button",
    "canvas",
    "caption",
    "cellpadding",
    "cellspacing",
    "center",
    "challenge",
    "charset",
    "checked",
    "circle",
    "cite",
    "class",
    "clipPath",
    "code",
    "col",
    "colgroup",
    "color",
    "cols",
    "colspan",
    "content",
    "contenteditable",
    "controls",
    "coords",
    "crossorigin",
    "cx",
    "cy",
    "d",
    "data",
    "datalist",
    "datetime",
    "dd",
    "decoding",
    "default",
    "defer",
    "defs",
    "del",
    "desc",
    "details",
    "dfn",
    "dialog",
    "dir",
    "dirname",
    "disabled",
    "div",
    "dl",
    "download",
    "draggable",
    "dt",
    "ellipse",
    "em",
    "embed",
    "encoding",
    "enctype",
    "enterkeyhint",
    "face",
    "feBlend",
    "feColorMatrix",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feDistantLight",
    "feDropShadow",
    "feFlood",
    "feFuncA",
    "feFuncB",
    "feFuncG",
    "feFuncR",
    "feGaussianBlur",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "fePointLight",
    "feSpecularLighting",
    "feSpotLight",
    "feTile",
    "feTurbulence",
    "fieldset",
    "figcaption",
    "figure",
    "fill",
    "font",
    "footer",
    "for",
    "foreignObject",
    "form",
    "formaction",
    "formenctype",
    "formmethod",
    "formnovalidate",
    "formtarget",
    "frame",
    "frameset",
    "g",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "headers",
    "height",
    "hgroup",
    "hidden",
    "high",
    "hr",
    "href",
    "hreflang",
    "html",
    "http-equiv",
    "i",
    "id",
    "iframe",
    "image",
    "img",
    "input",
    "inputmode",
    "ins",
    "integrity",
    "is",
    "ismap",
    "itemid",
    "itemprop",
    "itemref",
    "itemscope",
    "itemtype",
    "kbd",
    "keygen",
    "keytype",
    "kind",
    "label",
    "lang",
    "language",
    "legend",
    "li",
    "line",
    "linearGradient",
    "link",
    "list",
    "listing",
    "loading",
    "longdesc",
    "loop",
    "low",
    "maction",
    "main",
    "malignmark",
    "manifest",
    "map",
    "mark",
    "marker",
    "marquee",
    "mask",
    "math",
    "max",
    "maxlength",
    "media",
    "menclose",
    "menu",
    "merror",
    "meta",
    "metadata",
    "meter",
    "method",
    "mfenced",
    "mfrac",
    "mglyph",
    "mi",
    "min",
    "minlength",
    "mmultiscripts",
    "mn",
    "mo",
    "mover",
    "mpadded",
    "mpath",
    "mphantom",
    "mrow",
    "ms",
    "mspace",
    "msqrt",
    "mstyle",
    "msub",
    "msubsup",
    "msup",
    "mtable",
    "mtd",
    "mtext",
    "mtr",
    "multiple",
    "munder",
    "munderover",
    "muted",
    "name",
    "nav",
    "nobr",
    "noembed",
    "noframes",
    "nomodule",
    "nonce",
    "noscript",
    "novalidate",
    "object",
    "ol",
    "open",
    "optgroup",
    "optimum",
    "option",
    "output",
    "p",
    "param",
    "path",
    "pattern",
    "picture",
    "ping",
    "placeholder",
    "plaintext",
    "points",
    "polygon",
    "polyline",
    "popover",
    "popovertarget",
    "popovertargetaction",
    "poster",
    "pre",
    "preload",
    "progress",
    "property",
    "q",
    "r",
    "radialGradient",
    "radiogroup",
    "rb",
    "readonly",
    "rect",
    "referrerpolicy",
    "rel",
    "required",
    "reversed",
    "role",
    "rows",
    "rowspan",
    "rp",
    "rt",
    "rtc",
    "ruby",
    "s",
    "samp",
    "sandbox",
    "scope",
    "scoped",
    "script",
    "seamless",
    "search",
    "section",
    "select",
    "selected",
    "selectedcontent",
    "semantics",
    "set",
    "shadowrootclonable",
    "shadowrootdelegatesfocus",
    "shadowrootmode",
    "shape",
    "size",
    "sizes",
    "slot",
    "small",
    "source",
    "span",
    "spellcheck",
    "src",
    "srcdoc",
    "srclang",
    "srcset",
    "start",
    "step",
    "stop",
    "strike",
    "stroke",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "switch",
    "symbol",
    "tabindex",
    "table",
    "target",
    "tbody",
    "td",
    "template",
    "text",
    "textPath",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "transform",
    "translate",
    "tspan",
    "tt",
    "type",
    "u",
    "ul",
    "use",
    "usemap",
    "value",
    "var",
    "video",
    "view",
    "viewBox",
    "volume",
    "wbr",
    "width",
    "wrap",
    "x",
    "x1",
    "x2",
    "xmlns",
    "xmp",
    "y",
    "y1",
    "y2",
];

/// Atom of a well-known name: its index in the `NAMES` table.
///
/// `Copy`, so comparing two atoms is a single `u32` comparison with no
/// indirection. Atoms are only meaningful together with the table in
/// this module (there is exactly one table per process).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NameAtom(u32);

impl NameAtom {
    /// Not a well-known name. The owning [`Name`] carries the original
    /// string on its heap sidecar; never index the table with this.
    pub const UNKNOWN: NameAtom = NameAtom(u32::MAX);

    /// Look up a well-known name: exact, case-sensitive. Pure function —
    /// no allocation, no hashing, no locks.
    ///
    /// Generated from the `NAMES` table (do not hand-edit the arms): the
    /// `every_well_known_name_round_trips` test pins each arm's index to
    /// the table order.
    pub fn lookup(name: &str) -> Option<NameAtom> {
        // A `match` over the table compiles to a decision tree on the
        // name bytes — much cheaper than a binary search per lookup on
        // the selector-matching hot path.
        match name {
            "a" => Some(NameAtom(0)),
            "abbr" => Some(NameAtom(1)),
            "accept" => Some(NameAtom(2)),
            "accept-charset" => Some(NameAtom(3)),
            "accesskey" => Some(NameAtom(4)),
            "action" => Some(NameAtom(5)),
            "address" => Some(NameAtom(6)),
            "align" => Some(NameAtom(7)),
            "allow" => Some(NameAtom(8)),
            "allowfullscreen" => Some(NameAtom(9)),
            "allowpaymentrequest" => Some(NameAtom(10)),
            "alt" => Some(NameAtom(11)),
            "animate" => Some(NameAtom(12)),
            "animateTransform" => Some(NameAtom(13)),
            "annotation" => Some(NameAtom(14)),
            "annotation-xml" => Some(NameAtom(15)),
            "applet" => Some(NameAtom(16)),
            "area" => Some(NameAtom(17)),
            "article" => Some(NameAtom(18)),
            "aside" => Some(NameAtom(19)),
            "async" => Some(NameAtom(20)),
            "audio" => Some(NameAtom(21)),
            "autocapitalize" => Some(NameAtom(22)),
            "autocomplete" => Some(NameAtom(23)),
            "autofocus" => Some(NameAtom(24)),
            "autoplay" => Some(NameAtom(25)),
            "b" => Some(NameAtom(26)),
            "base" => Some(NameAtom(27)),
            "basefont" => Some(NameAtom(28)),
            "bdi" => Some(NameAtom(29)),
            "bdo" => Some(NameAtom(30)),
            "bgcolor" => Some(NameAtom(31)),
            "bgsound" => Some(NameAtom(32)),
            "big" => Some(NameAtom(33)),
            "blockquote" => Some(NameAtom(34)),
            "body" => Some(NameAtom(35)),
            "border" => Some(NameAtom(36)),
            "br" => Some(NameAtom(37)),
            "button" => Some(NameAtom(38)),
            "canvas" => Some(NameAtom(39)),
            "caption" => Some(NameAtom(40)),
            "cellpadding" => Some(NameAtom(41)),
            "cellspacing" => Some(NameAtom(42)),
            "center" => Some(NameAtom(43)),
            "challenge" => Some(NameAtom(44)),
            "charset" => Some(NameAtom(45)),
            "checked" => Some(NameAtom(46)),
            "circle" => Some(NameAtom(47)),
            "cite" => Some(NameAtom(48)),
            "class" => Some(NameAtom(49)),
            "clipPath" => Some(NameAtom(50)),
            "code" => Some(NameAtom(51)),
            "col" => Some(NameAtom(52)),
            "colgroup" => Some(NameAtom(53)),
            "color" => Some(NameAtom(54)),
            "cols" => Some(NameAtom(55)),
            "colspan" => Some(NameAtom(56)),
            "content" => Some(NameAtom(57)),
            "contenteditable" => Some(NameAtom(58)),
            "controls" => Some(NameAtom(59)),
            "coords" => Some(NameAtom(60)),
            "crossorigin" => Some(NameAtom(61)),
            "cx" => Some(NameAtom(62)),
            "cy" => Some(NameAtom(63)),
            "d" => Some(NameAtom(64)),
            "data" => Some(NameAtom(65)),
            "datalist" => Some(NameAtom(66)),
            "datetime" => Some(NameAtom(67)),
            "dd" => Some(NameAtom(68)),
            "decoding" => Some(NameAtom(69)),
            "default" => Some(NameAtom(70)),
            "defer" => Some(NameAtom(71)),
            "defs" => Some(NameAtom(72)),
            "del" => Some(NameAtom(73)),
            "desc" => Some(NameAtom(74)),
            "details" => Some(NameAtom(75)),
            "dfn" => Some(NameAtom(76)),
            "dialog" => Some(NameAtom(77)),
            "dir" => Some(NameAtom(78)),
            "dirname" => Some(NameAtom(79)),
            "disabled" => Some(NameAtom(80)),
            "div" => Some(NameAtom(81)),
            "dl" => Some(NameAtom(82)),
            "download" => Some(NameAtom(83)),
            "draggable" => Some(NameAtom(84)),
            "dt" => Some(NameAtom(85)),
            "ellipse" => Some(NameAtom(86)),
            "em" => Some(NameAtom(87)),
            "embed" => Some(NameAtom(88)),
            "encoding" => Some(NameAtom(89)),
            "enctype" => Some(NameAtom(90)),
            "enterkeyhint" => Some(NameAtom(91)),
            "face" => Some(NameAtom(92)),
            "feBlend" => Some(NameAtom(93)),
            "feColorMatrix" => Some(NameAtom(94)),
            "feComposite" => Some(NameAtom(95)),
            "feConvolveMatrix" => Some(NameAtom(96)),
            "feDiffuseLighting" => Some(NameAtom(97)),
            "feDisplacementMap" => Some(NameAtom(98)),
            "feDistantLight" => Some(NameAtom(99)),
            "feDropShadow" => Some(NameAtom(100)),
            "feFlood" => Some(NameAtom(101)),
            "feFuncA" => Some(NameAtom(102)),
            "feFuncB" => Some(NameAtom(103)),
            "feFuncG" => Some(NameAtom(104)),
            "feFuncR" => Some(NameAtom(105)),
            "feGaussianBlur" => Some(NameAtom(106)),
            "feMerge" => Some(NameAtom(107)),
            "feMergeNode" => Some(NameAtom(108)),
            "feMorphology" => Some(NameAtom(109)),
            "feOffset" => Some(NameAtom(110)),
            "fePointLight" => Some(NameAtom(111)),
            "feSpecularLighting" => Some(NameAtom(112)),
            "feSpotLight" => Some(NameAtom(113)),
            "feTile" => Some(NameAtom(114)),
            "feTurbulence" => Some(NameAtom(115)),
            "fieldset" => Some(NameAtom(116)),
            "figcaption" => Some(NameAtom(117)),
            "figure" => Some(NameAtom(118)),
            "fill" => Some(NameAtom(119)),
            "font" => Some(NameAtom(120)),
            "footer" => Some(NameAtom(121)),
            "for" => Some(NameAtom(122)),
            "foreignObject" => Some(NameAtom(123)),
            "form" => Some(NameAtom(124)),
            "formaction" => Some(NameAtom(125)),
            "formenctype" => Some(NameAtom(126)),
            "formmethod" => Some(NameAtom(127)),
            "formnovalidate" => Some(NameAtom(128)),
            "formtarget" => Some(NameAtom(129)),
            "frame" => Some(NameAtom(130)),
            "frameset" => Some(NameAtom(131)),
            "g" => Some(NameAtom(132)),
            "h1" => Some(NameAtom(133)),
            "h2" => Some(NameAtom(134)),
            "h3" => Some(NameAtom(135)),
            "h4" => Some(NameAtom(136)),
            "h5" => Some(NameAtom(137)),
            "h6" => Some(NameAtom(138)),
            "head" => Some(NameAtom(139)),
            "header" => Some(NameAtom(140)),
            "headers" => Some(NameAtom(141)),
            "height" => Some(NameAtom(142)),
            "hgroup" => Some(NameAtom(143)),
            "hidden" => Some(NameAtom(144)),
            "high" => Some(NameAtom(145)),
            "hr" => Some(NameAtom(146)),
            "href" => Some(NameAtom(147)),
            "hreflang" => Some(NameAtom(148)),
            "html" => Some(NameAtom(149)),
            "http-equiv" => Some(NameAtom(150)),
            "i" => Some(NameAtom(151)),
            "id" => Some(NameAtom(152)),
            "iframe" => Some(NameAtom(153)),
            "image" => Some(NameAtom(154)),
            "img" => Some(NameAtom(155)),
            "input" => Some(NameAtom(156)),
            "inputmode" => Some(NameAtom(157)),
            "ins" => Some(NameAtom(158)),
            "integrity" => Some(NameAtom(159)),
            "is" => Some(NameAtom(160)),
            "ismap" => Some(NameAtom(161)),
            "itemid" => Some(NameAtom(162)),
            "itemprop" => Some(NameAtom(163)),
            "itemref" => Some(NameAtom(164)),
            "itemscope" => Some(NameAtom(165)),
            "itemtype" => Some(NameAtom(166)),
            "kbd" => Some(NameAtom(167)),
            "keygen" => Some(NameAtom(168)),
            "keytype" => Some(NameAtom(169)),
            "kind" => Some(NameAtom(170)),
            "label" => Some(NameAtom(171)),
            "lang" => Some(NameAtom(172)),
            "language" => Some(NameAtom(173)),
            "legend" => Some(NameAtom(174)),
            "li" => Some(NameAtom(175)),
            "line" => Some(NameAtom(176)),
            "linearGradient" => Some(NameAtom(177)),
            "link" => Some(NameAtom(178)),
            "list" => Some(NameAtom(179)),
            "listing" => Some(NameAtom(180)),
            "loading" => Some(NameAtom(181)),
            "longdesc" => Some(NameAtom(182)),
            "loop" => Some(NameAtom(183)),
            "low" => Some(NameAtom(184)),
            "maction" => Some(NameAtom(185)),
            "main" => Some(NameAtom(186)),
            "malignmark" => Some(NameAtom(187)),
            "manifest" => Some(NameAtom(188)),
            "map" => Some(NameAtom(189)),
            "mark" => Some(NameAtom(190)),
            "marker" => Some(NameAtom(191)),
            "marquee" => Some(NameAtom(192)),
            "mask" => Some(NameAtom(193)),
            "math" => Some(NameAtom(194)),
            "max" => Some(NameAtom(195)),
            "maxlength" => Some(NameAtom(196)),
            "media" => Some(NameAtom(197)),
            "menclose" => Some(NameAtom(198)),
            "menu" => Some(NameAtom(199)),
            "merror" => Some(NameAtom(200)),
            "meta" => Some(NameAtom(201)),
            "metadata" => Some(NameAtom(202)),
            "meter" => Some(NameAtom(203)),
            "method" => Some(NameAtom(204)),
            "mfenced" => Some(NameAtom(205)),
            "mfrac" => Some(NameAtom(206)),
            "mglyph" => Some(NameAtom(207)),
            "mi" => Some(NameAtom(208)),
            "min" => Some(NameAtom(209)),
            "minlength" => Some(NameAtom(210)),
            "mmultiscripts" => Some(NameAtom(211)),
            "mn" => Some(NameAtom(212)),
            "mo" => Some(NameAtom(213)),
            "mover" => Some(NameAtom(214)),
            "mpadded" => Some(NameAtom(215)),
            "mpath" => Some(NameAtom(216)),
            "mphantom" => Some(NameAtom(217)),
            "mrow" => Some(NameAtom(218)),
            "ms" => Some(NameAtom(219)),
            "mspace" => Some(NameAtom(220)),
            "msqrt" => Some(NameAtom(221)),
            "mstyle" => Some(NameAtom(222)),
            "msub" => Some(NameAtom(223)),
            "msubsup" => Some(NameAtom(224)),
            "msup" => Some(NameAtom(225)),
            "mtable" => Some(NameAtom(226)),
            "mtd" => Some(NameAtom(227)),
            "mtext" => Some(NameAtom(228)),
            "mtr" => Some(NameAtom(229)),
            "multiple" => Some(NameAtom(230)),
            "munder" => Some(NameAtom(231)),
            "munderover" => Some(NameAtom(232)),
            "muted" => Some(NameAtom(233)),
            "name" => Some(NameAtom(234)),
            "nav" => Some(NameAtom(235)),
            "nobr" => Some(NameAtom(236)),
            "noembed" => Some(NameAtom(237)),
            "noframes" => Some(NameAtom(238)),
            "nomodule" => Some(NameAtom(239)),
            "nonce" => Some(NameAtom(240)),
            "noscript" => Some(NameAtom(241)),
            "novalidate" => Some(NameAtom(242)),
            "object" => Some(NameAtom(243)),
            "ol" => Some(NameAtom(244)),
            "open" => Some(NameAtom(245)),
            "optgroup" => Some(NameAtom(246)),
            "optimum" => Some(NameAtom(247)),
            "option" => Some(NameAtom(248)),
            "output" => Some(NameAtom(249)),
            "p" => Some(NameAtom(250)),
            "param" => Some(NameAtom(251)),
            "path" => Some(NameAtom(252)),
            "pattern" => Some(NameAtom(253)),
            "picture" => Some(NameAtom(254)),
            "ping" => Some(NameAtom(255)),
            "placeholder" => Some(NameAtom(256)),
            "plaintext" => Some(NameAtom(257)),
            "points" => Some(NameAtom(258)),
            "polygon" => Some(NameAtom(259)),
            "polyline" => Some(NameAtom(260)),
            "popover" => Some(NameAtom(261)),
            "popovertarget" => Some(NameAtom(262)),
            "popovertargetaction" => Some(NameAtom(263)),
            "poster" => Some(NameAtom(264)),
            "pre" => Some(NameAtom(265)),
            "preload" => Some(NameAtom(266)),
            "progress" => Some(NameAtom(267)),
            "property" => Some(NameAtom(268)),
            "q" => Some(NameAtom(269)),
            "r" => Some(NameAtom(270)),
            "radialGradient" => Some(NameAtom(271)),
            "radiogroup" => Some(NameAtom(272)),
            "rb" => Some(NameAtom(273)),
            "readonly" => Some(NameAtom(274)),
            "rect" => Some(NameAtom(275)),
            "referrerpolicy" => Some(NameAtom(276)),
            "rel" => Some(NameAtom(277)),
            "required" => Some(NameAtom(278)),
            "reversed" => Some(NameAtom(279)),
            "role" => Some(NameAtom(280)),
            "rows" => Some(NameAtom(281)),
            "rowspan" => Some(NameAtom(282)),
            "rp" => Some(NameAtom(283)),
            "rt" => Some(NameAtom(284)),
            "rtc" => Some(NameAtom(285)),
            "ruby" => Some(NameAtom(286)),
            "s" => Some(NameAtom(287)),
            "samp" => Some(NameAtom(288)),
            "sandbox" => Some(NameAtom(289)),
            "scope" => Some(NameAtom(290)),
            "scoped" => Some(NameAtom(291)),
            "script" => Some(NameAtom(292)),
            "seamless" => Some(NameAtom(293)),
            "search" => Some(NameAtom(294)),
            "section" => Some(NameAtom(295)),
            "select" => Some(NameAtom(296)),
            "selected" => Some(NameAtom(297)),
            "selectedcontent" => Some(NameAtom(298)),
            "semantics" => Some(NameAtom(299)),
            "set" => Some(NameAtom(300)),
            "shadowrootclonable" => Some(NameAtom(301)),
            "shadowrootdelegatesfocus" => Some(NameAtom(302)),
            "shadowrootmode" => Some(NameAtom(303)),
            "shape" => Some(NameAtom(304)),
            "size" => Some(NameAtom(305)),
            "sizes" => Some(NameAtom(306)),
            "slot" => Some(NameAtom(307)),
            "small" => Some(NameAtom(308)),
            "source" => Some(NameAtom(309)),
            "span" => Some(NameAtom(310)),
            "spellcheck" => Some(NameAtom(311)),
            "src" => Some(NameAtom(312)),
            "srcdoc" => Some(NameAtom(313)),
            "srclang" => Some(NameAtom(314)),
            "srcset" => Some(NameAtom(315)),
            "start" => Some(NameAtom(316)),
            "step" => Some(NameAtom(317)),
            "stop" => Some(NameAtom(318)),
            "strike" => Some(NameAtom(319)),
            "stroke" => Some(NameAtom(320)),
            "strong" => Some(NameAtom(321)),
            "style" => Some(NameAtom(322)),
            "sub" => Some(NameAtom(323)),
            "summary" => Some(NameAtom(324)),
            "sup" => Some(NameAtom(325)),
            "svg" => Some(NameAtom(326)),
            "switch" => Some(NameAtom(327)),
            "symbol" => Some(NameAtom(328)),
            "tabindex" => Some(NameAtom(329)),
            "table" => Some(NameAtom(330)),
            "target" => Some(NameAtom(331)),
            "tbody" => Some(NameAtom(332)),
            "td" => Some(NameAtom(333)),
            "template" => Some(NameAtom(334)),
            "text" => Some(NameAtom(335)),
            "textPath" => Some(NameAtom(336)),
            "textarea" => Some(NameAtom(337)),
            "tfoot" => Some(NameAtom(338)),
            "th" => Some(NameAtom(339)),
            "thead" => Some(NameAtom(340)),
            "time" => Some(NameAtom(341)),
            "title" => Some(NameAtom(342)),
            "tr" => Some(NameAtom(343)),
            "track" => Some(NameAtom(344)),
            "transform" => Some(NameAtom(345)),
            "translate" => Some(NameAtom(346)),
            "tspan" => Some(NameAtom(347)),
            "tt" => Some(NameAtom(348)),
            "type" => Some(NameAtom(349)),
            "u" => Some(NameAtom(350)),
            "ul" => Some(NameAtom(351)),
            "use" => Some(NameAtom(352)),
            "usemap" => Some(NameAtom(353)),
            "value" => Some(NameAtom(354)),
            "var" => Some(NameAtom(355)),
            "video" => Some(NameAtom(356)),
            "view" => Some(NameAtom(357)),
            "viewBox" => Some(NameAtom(358)),
            "volume" => Some(NameAtom(359)),
            "wbr" => Some(NameAtom(360)),
            "width" => Some(NameAtom(361)),
            "wrap" => Some(NameAtom(362)),
            "x" => Some(NameAtom(363)),
            "x1" => Some(NameAtom(364)),
            "x2" => Some(NameAtom(365)),
            "xmlns" => Some(NameAtom(366)),
            "xmp" => Some(NameAtom(367)),
            "y" => Some(NameAtom(368)),
            "y1" => Some(NameAtom(369)),
            "y2" => Some(NameAtom(370)),
            _ => None,
        }
    }

    /// Look up a name under the HTML ASCII case-insensitivity rule:
    /// names that are already lowercase go straight to the table;
    /// anything with ASCII uppercase is lowercased into a stack buffer
    /// first (no allocation for names up to 64 bytes).
    pub fn lookup_html(name: &str) -> Option<NameAtom> {
        if name.as_bytes().iter().all(|b| !b.is_ascii_uppercase()) {
            return Self::lookup(name);
        }
        let mut buf = [0u8; 64];
        if name.len() <= buf.len() {
            for (i, b) in name.as_bytes().iter().enumerate() {
                buf[i] = b.to_ascii_lowercase();
            }
            // ASCII-lowercasing only rewrites A-Z bytes, so the buffer is
            // still valid UTF-8.
            let lowered = core::str::from_utf8(&buf[..name.len()])
                .expect("ASCII-lowercased bytes are valid UTF-8");
            Self::lookup(lowered)
        } else {
            Self::lookup(&name.to_ascii_lowercase())
        }
    }

    /// The well-known name behind this atom. Panics on
    /// [`NameAtom::UNKNOWN`] — go through [`Name::as_str`] instead.
    pub fn as_str(self) -> &'static str {
        NAMES[self.0 as usize]
    }

    /// Atom of the `id` attribute name: the hot path for id-selector
    /// matching and id-index maintenance. Resolved once per process.
    pub fn id() -> NameAtom {
        static ID: OnceLock<NameAtom> = OnceLock::new();
        *ID.get_or_init(|| NameAtom::lookup("id").expect("`id` is a well-known name"))
    }

    /// Atom of the `class` attribute name: the hot path for class
    /// matching. Resolved once per process.
    pub fn class() -> NameAtom {
        static CLASS: OnceLock<NameAtom> = OnceLock::new();
        *CLASS.get_or_init(|| NameAtom::lookup("class").expect("`class` is a well-known name"))
    }
}

/// An interned tag or attribute name: two tiers.
///
/// Well-known names (the `NAMES` table) are just their [`NameAtom`] —
/// 4 bytes, `Copy`-cheap to compare, zero heap use. Anything else keeps
/// its atom as [`NameAtom::UNKNOWN`] and carries the original string on
/// a heap sidecar, so unknown names round-trip exactly (custom elements,
/// `data-*` attributes, foreign names).
///
/// [`TagName`] and [`AttrName`] are aliases documenting intent at the
/// use sites; they are the same type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Name {
    atom: NameAtom,
    /// The original string iff `atom == NameAtom::UNKNOWN`.
    spill: Option<Box<str>>,
}

/// An interned element tag name.
pub type TagName = Name;
/// An interned attribute name.
pub type AttrName = Name;

impl Default for Name {
    /// Placeholder with no heap use: the [`NameAtom::UNKNOWN`] atom and no
    /// sidecar. Only meaningful as a [`std::mem::take`]/`replace` target —
    /// [`as_str`](Name::as_str) on a default name panics, exactly like
    /// [`NameAtom::UNKNOWN`]'s [`as_str`](NameAtom::as_str).
    fn default() -> Self {
        Name {
            atom: NameAtom::UNKNOWN,
            spill: None,
        }
    }
}

impl Name {
    /// Intern a borrowed name: well-known names become pure atoms (no
    /// allocation); anything else is copied onto the heap sidecar.
    pub fn intern(name: &str) -> Self {
        match NameAtom::lookup(name) {
            Some(atom) => Name { atom, spill: None },
            None => Name {
                atom: NameAtom::UNKNOWN,
                spill: Some(name.into()),
            },
        }
    }

    /// Intern, ASCII-lowercasing first when the name needs it (HTML tag
    /// and attribute names). Skips the intermediate `String` when the
    /// name is already lowercase.
    pub fn intern_lowered(name: &str) -> Self {
        if name.as_bytes().iter().all(|b| !b.is_ascii_uppercase()) {
            return Self::intern(name);
        }
        Self::from(name.to_ascii_lowercase())
    }

    /// The atom for this name, or [`NameAtom::UNKNOWN`]. Comparing atoms
    /// is exactly string equality: interning is deterministic and the
    /// table holds unique strings.
    pub fn atom(&self) -> NameAtom {
        self.atom
    }

    /// The original string.
    pub fn as_str(&self) -> &str {
        match &self.spill {
            Some(s) => s,
            None => self.atom.as_str(),
        }
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Self::intern(s)
    }
}

impl From<&String> for Name {
    fn from(s: &String) -> Self {
        Self::intern(s.as_str())
    }
}

impl From<String> for Name {
    /// Zero-copy for unknown names (the `String` becomes the sidecar);
    /// well-known names drop the `String` and keep just the atom.
    fn from(s: String) -> Self {
        match NameAtom::lookup(&s) {
            Some(atom) => Name { atom, spill: None },
            None => Name {
                atom: NameAtom::UNKNOWN,
                spill: Some(s.into_boxed_str()),
            },
        }
    }
}

impl Deref for Name {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for Name {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<Name> for str {
    fn eq(&self, other: &Name) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<Name> for &str {
    fn eq(&self, other: &Name) -> bool {
        *self == other.as_str()
    }
}

impl NameAtom {
    /// Atom for the well-known name `"a"`.
    pub const A: NameAtom = NameAtom(0);
    /// Atom for the well-known name `"address"`.
    pub const ADDRESS: NameAtom = NameAtom(6);
    /// Atom for the well-known name `"annotation-xml"`.
    pub const ANNOTATION_XML: NameAtom = NameAtom(15);
    /// Atom for the well-known name `"applet"`.
    pub const APPLET: NameAtom = NameAtom(16);
    /// Atom for the well-known name `"area"`.
    pub const AREA: NameAtom = NameAtom(17);
    /// Atom for the well-known name `"article"`.
    pub const ARTICLE: NameAtom = NameAtom(18);
    /// Atom for the well-known name `"aside"`.
    pub const ASIDE: NameAtom = NameAtom(19);
    /// Atom for the well-known name `"b"`.
    pub const B: NameAtom = NameAtom(26);
    /// Atom for the well-known name `"base"`.
    pub const BASE: NameAtom = NameAtom(27);
    /// Atom for the well-known name `"basefont"`.
    pub const BASEFONT: NameAtom = NameAtom(28);
    /// Atom for the well-known name `"bgsound"`.
    pub const BGSOUND: NameAtom = NameAtom(32);
    /// Atom for the well-known name `"big"`.
    pub const BIG: NameAtom = NameAtom(33);
    /// Atom for the well-known name `"blockquote"`.
    pub const BLOCKQUOTE: NameAtom = NameAtom(34);
    /// Atom for the well-known name `"body"`.
    pub const BODY: NameAtom = NameAtom(35);
    /// Atom for the well-known name `"br"`.
    pub const BR: NameAtom = NameAtom(37);
    /// Atom for the well-known name `"button"`.
    pub const BUTTON: NameAtom = NameAtom(38);
    /// Atom for the well-known name `"caption"`.
    pub const CAPTION: NameAtom = NameAtom(40);
    /// Atom for the well-known name `"center"`.
    pub const CENTER: NameAtom = NameAtom(43);
    /// Atom for the well-known name `"code"`.
    pub const CODE: NameAtom = NameAtom(51);
    /// Atom for the well-known name `"col"`.
    pub const COL: NameAtom = NameAtom(52);
    /// Atom for the well-known name `"colgroup"`.
    pub const COLGROUP: NameAtom = NameAtom(53);
    /// Atom for the well-known name `"color"`.
    pub const COLOR: NameAtom = NameAtom(54);
    /// Atom for the well-known name `"datalist"`.
    pub const DATALIST: NameAtom = NameAtom(66);
    /// Atom for the well-known name `"dd"`.
    pub const DD: NameAtom = NameAtom(68);
    /// Atom for the well-known name `"desc"`.
    pub const DESC: NameAtom = NameAtom(74);
    /// Atom for the well-known name `"details"`.
    pub const DETAILS: NameAtom = NameAtom(75);
    /// Atom for the well-known name `"dialog"`.
    pub const DIALOG: NameAtom = NameAtom(77);
    /// Atom for the well-known name `"dir"`.
    pub const DIR: NameAtom = NameAtom(78);
    /// Atom for the well-known name `"div"`.
    pub const DIV: NameAtom = NameAtom(81);
    /// Atom for the well-known name `"dl"`.
    pub const DL: NameAtom = NameAtom(82);
    /// Atom for the well-known name `"dt"`.
    pub const DT: NameAtom = NameAtom(85);
    /// Atom for the well-known name `"em"`.
    pub const EM: NameAtom = NameAtom(87);
    /// Atom for the well-known name `"embed"`.
    pub const EMBED: NameAtom = NameAtom(88);
    /// Atom for the well-known name `"encoding"`.
    pub const ENCODING: NameAtom = NameAtom(89);
    /// Atom for the well-known name `"face"`.
    pub const FACE: NameAtom = NameAtom(92);
    /// Atom for the well-known name `"fieldset"`.
    pub const FIELDSET: NameAtom = NameAtom(116);
    /// Atom for the well-known name `"figcaption"`.
    pub const FIGCAPTION: NameAtom = NameAtom(117);
    /// Atom for the well-known name `"figure"`.
    pub const FIGURE: NameAtom = NameAtom(118);
    /// Atom for the well-known name `"font"`.
    pub const FONT: NameAtom = NameAtom(120);
    /// Atom for the well-known name `"footer"`.
    pub const FOOTER: NameAtom = NameAtom(121);
    /// Atom for the well-known name `"foreignObject"`.
    pub const FOREIGNOBJECT: NameAtom = NameAtom(123);
    /// Atom for the well-known name `"form"`.
    pub const FORM: NameAtom = NameAtom(124);
    /// Atom for the well-known name `"frame"`.
    pub const FRAME: NameAtom = NameAtom(130);
    /// Atom for the well-known name `"frameset"`.
    pub const FRAMESET: NameAtom = NameAtom(131);
    /// Atom for the well-known name `"h1"`.
    pub const H1: NameAtom = NameAtom(133);
    /// Atom for the well-known name `"h2"`.
    pub const H2: NameAtom = NameAtom(134);
    /// Atom for the well-known name `"h3"`.
    pub const H3: NameAtom = NameAtom(135);
    /// Atom for the well-known name `"h4"`.
    pub const H4: NameAtom = NameAtom(136);
    /// Atom for the well-known name `"h5"`.
    pub const H5: NameAtom = NameAtom(137);
    /// Atom for the well-known name `"h6"`.
    pub const H6: NameAtom = NameAtom(138);
    /// Atom for the well-known name `"head"`.
    pub const HEAD: NameAtom = NameAtom(139);
    /// Atom for the well-known name `"header"`.
    pub const HEADER: NameAtom = NameAtom(140);
    /// Atom for the well-known name `"hgroup"`.
    pub const HGROUP: NameAtom = NameAtom(143);
    /// Atom for the well-known name `"hr"`.
    pub const HR: NameAtom = NameAtom(146);
    /// Atom for the well-known name `"html"`.
    pub const HTML: NameAtom = NameAtom(149);
    /// Atom for the well-known name `"i"`.
    pub const I: NameAtom = NameAtom(151);
    /// Atom for the well-known name `"iframe"`.
    pub const IFRAME: NameAtom = NameAtom(153);
    /// Atom for the well-known name `"image"`.
    pub const IMAGE: NameAtom = NameAtom(154);
    /// Atom for the well-known name `"img"`.
    pub const IMG: NameAtom = NameAtom(155);
    /// Atom for the well-known name `"input"`.
    pub const INPUT: NameAtom = NameAtom(156);
    /// Atom for the well-known name `"keygen"`.
    pub const KEYGEN: NameAtom = NameAtom(168);
    /// Atom for the well-known name `"language"`.
    pub const LANGUAGE: NameAtom = NameAtom(173);
    /// Atom for the well-known name `"li"`.
    pub const LI: NameAtom = NameAtom(175);
    /// Atom for the well-known name `"link"`.
    pub const LINK: NameAtom = NameAtom(178);
    /// Atom for the well-known name `"listing"`.
    pub const LISTING: NameAtom = NameAtom(180);
    /// Atom for the well-known name `"main"`.
    pub const MAIN: NameAtom = NameAtom(186);
    /// Atom for the well-known name `"malignmark"`.
    pub const MALIGNMARK: NameAtom = NameAtom(187);
    /// Atom for the well-known name `"marquee"`.
    pub const MARQUEE: NameAtom = NameAtom(192);
    /// Atom for the well-known name `"math"`.
    pub const MATH: NameAtom = NameAtom(194);
    /// Atom for the well-known name `"menu"`.
    pub const MENU: NameAtom = NameAtom(199);
    /// Atom for the well-known name `"meta"`.
    pub const META: NameAtom = NameAtom(201);
    /// Atom for the well-known name `"mglyph"`.
    pub const MGLYPH: NameAtom = NameAtom(207);
    /// Atom for the well-known name `"mi"`.
    pub const MI: NameAtom = NameAtom(208);
    /// Atom for the well-known name `"mn"`.
    pub const MN: NameAtom = NameAtom(212);
    /// Atom for the well-known name `"mo"`.
    pub const MO: NameAtom = NameAtom(213);
    /// Atom for the well-known name `"ms"`.
    pub const MS: NameAtom = NameAtom(219);
    /// Atom for the well-known name `"mtext"`.
    pub const MTEXT: NameAtom = NameAtom(228);
    /// Atom for the well-known name `"multiple"`.
    pub const MULTIPLE: NameAtom = NameAtom(230);
    /// Atom for the well-known name `"nav"`.
    pub const NAV: NameAtom = NameAtom(235);
    /// Atom for the well-known name `"nobr"`.
    pub const NOBR: NameAtom = NameAtom(236);
    /// Atom for the well-known name `"noembed"`.
    pub const NOEMBED: NameAtom = NameAtom(237);
    /// Atom for the well-known name `"noframes"`.
    pub const NOFRAMES: NameAtom = NameAtom(238);
    /// Atom for the well-known name `"noscript"`.
    pub const NOSCRIPT: NameAtom = NameAtom(241);
    /// Atom for the well-known name `"object"`.
    pub const OBJECT: NameAtom = NameAtom(243);
    /// Atom for the well-known name `"ol"`.
    pub const OL: NameAtom = NameAtom(244);
    /// Atom for the well-known name `"optgroup"`.
    pub const OPTGROUP: NameAtom = NameAtom(246);
    /// Atom for the well-known name `"option"`.
    pub const OPTION: NameAtom = NameAtom(248);
    /// Atom for the well-known name `"p"`.
    pub const P: NameAtom = NameAtom(250);
    /// Atom for the well-known name `"param"`.
    pub const PARAM: NameAtom = NameAtom(251);
    /// Atom for the well-known name `"plaintext"`.
    pub const PLAINTEXT: NameAtom = NameAtom(257);
    /// Atom for the well-known name `"pre"`.
    pub const PRE: NameAtom = NameAtom(265);
    /// Atom for the well-known name `"rb"`.
    pub const RB: NameAtom = NameAtom(273);
    /// Atom for the well-known name `"rp"`.
    pub const RP: NameAtom = NameAtom(283);
    /// Atom for the well-known name `"rt"`.
    pub const RT: NameAtom = NameAtom(284);
    /// Atom for the well-known name `"rtc"`.
    pub const RTC: NameAtom = NameAtom(285);
    /// Atom for the well-known name `"ruby"`.
    pub const RUBY: NameAtom = NameAtom(286);
    /// Atom for the well-known name `"s"`.
    pub const S: NameAtom = NameAtom(287);
    /// Atom for the well-known name `"script"`.
    pub const SCRIPT: NameAtom = NameAtom(292);
    /// Atom for the well-known name `"search"`.
    pub const SEARCH: NameAtom = NameAtom(294);
    /// Atom for the well-known name `"section"`.
    pub const SECTION: NameAtom = NameAtom(295);
    /// Atom for the well-known name `"select"`.
    pub const SELECT: NameAtom = NameAtom(296);
    /// Atom for the well-known name `"selected"`.
    pub const SELECTED: NameAtom = NameAtom(297);
    /// Atom for the well-known name `"selectedcontent"`.
    pub const SELECTEDCONTENT: NameAtom = NameAtom(298);
    /// Atom for the well-known name `"size"`.
    pub const SIZE: NameAtom = NameAtom(305);
    /// Atom for the well-known name `"small"`.
    pub const SMALL: NameAtom = NameAtom(308);
    /// Atom for the well-known name `"source"`.
    pub const SOURCE: NameAtom = NameAtom(309);
    /// Atom for the well-known name `"span"`.
    pub const SPAN: NameAtom = NameAtom(310);
    /// Atom for the well-known name `"strike"`.
    pub const STRIKE: NameAtom = NameAtom(319);
    /// Atom for the well-known name `"strong"`.
    pub const STRONG: NameAtom = NameAtom(321);
    /// Atom for the well-known name `"style"`.
    pub const STYLE: NameAtom = NameAtom(322);
    /// Atom for the well-known name `"sub"`.
    pub const SUB: NameAtom = NameAtom(323);
    /// Atom for the well-known name `"summary"`.
    pub const SUMMARY: NameAtom = NameAtom(324);
    /// Atom for the well-known name `"sup"`.
    pub const SUP: NameAtom = NameAtom(325);
    /// Atom for the well-known name `"svg"`.
    pub const SVG: NameAtom = NameAtom(326);
    /// Atom for the well-known name `"table"`.
    pub const TABLE: NameAtom = NameAtom(330);
    /// Atom for the well-known name `"tbody"`.
    pub const TBODY: NameAtom = NameAtom(332);
    /// Atom for the well-known name `"td"`.
    pub const TD: NameAtom = NameAtom(333);
    /// Atom for the well-known name `"template"`.
    pub const TEMPLATE: NameAtom = NameAtom(334);
    /// Atom for the well-known name `"textarea"`.
    pub const TEXTAREA: NameAtom = NameAtom(337);
    /// Atom for the well-known name `"tfoot"`.
    pub const TFOOT: NameAtom = NameAtom(338);
    /// Atom for the well-known name `"th"`.
    pub const TH: NameAtom = NameAtom(339);
    /// Atom for the well-known name `"thead"`.
    pub const THEAD: NameAtom = NameAtom(340);
    /// Atom for the well-known name `"title"`.
    pub const TITLE: NameAtom = NameAtom(342);
    /// Atom for the well-known name `"tr"`.
    pub const TR: NameAtom = NameAtom(343);
    /// Atom for the well-known name `"track"`.
    pub const TRACK: NameAtom = NameAtom(344);
    /// Atom for the well-known name `"tt"`.
    pub const TT: NameAtom = NameAtom(348);
    /// Atom for the well-known name `"type"`.
    pub const TYPE: NameAtom = NameAtom(349);
    /// Atom for the well-known name `"u"`.
    pub const U: NameAtom = NameAtom(350);
    /// Atom for the well-known name `"ul"`.
    pub const UL: NameAtom = NameAtom(351);
    /// Atom for the well-known name `"var"`.
    pub const VAR: NameAtom = NameAtom(355);
    /// Atom for the well-known name `"wbr"`.
    pub const WBR: NameAtom = NameAtom(360);
    /// Atom for the well-known name `"xmp"`.
    pub const XMP: NameAtom = NameAtom(367);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_unique() {
        for w in NAMES.windows(2) {
            assert!(w[0] < w[1], "atom table not sorted/unique at {:?}", w);
        }
    }

    #[test]
    fn every_well_known_name_round_trips() {
        for name in NAMES {
            let atom = NameAtom::lookup(name).expect("well-known name must resolve");
            assert_eq!(atom.as_str(), *name);
            let n = Name::intern(name);
            assert_eq!(n.atom(), atom);
            assert!(n.spill.is_none(), "well-known name must not spill: {name}");
            assert_eq!(n.as_str(), *name);
        }
    }

    #[test]
    fn unknown_names_spill_and_round_trip() {
        for unknown in [
            "my-widget",
            "data-custom-thing",
            "aria-label-x",
            "x:still-unknown",
        ] {
            assert_eq!(NameAtom::lookup(unknown), None);
            let n = Name::intern(unknown);
            assert_eq!(n.atom(), NameAtom::UNKNOWN);
            assert_eq!(n.as_str(), unknown);
            assert_eq!(n, Name::from(unknown.to_string()));
        }
        // Unknown names compare by string: distinct unknowns differ, and
        // an unknown never equals a well-known name.
        assert_ne!(Name::intern("my-a"), Name::intern("my-b"));
        assert_ne!(Name::intern("my-div"), Name::intern("div"));
        assert_ne!(Name::intern("div"), Name::intern("my-div"));
        assert_eq!(Name::intern("my-div"), Name::intern("my-div"));
    }

    #[test]
    fn lookup_html_is_ascii_case_insensitive() {
        assert_eq!(NameAtom::lookup_html("DIV"), NameAtom::lookup("div"));
        assert_eq!(NameAtom::lookup_html("Div"), NameAtom::lookup("div"));
        assert_eq!(NameAtom::lookup_html("div"), NameAtom::lookup("div"));
        // Case-preserved SVG names are found by the exact lookup...
        assert_eq!(
            NameAtom::lookup("foreignObject").map(NameAtom::as_str),
            Some("foreignObject")
        );
        // ...but the HTML case-insensitive lookup lowercases first, so a
        // mixed-case query misses the table. Callers fall back to the
        // string compare there, which stays correct.
        assert_eq!(NameAtom::lookup_html("foreignObject"), None);
        // ...a wrong-case query misses the table (callers fall back to
        // the string compare, which stays correct).
        assert_eq!(NameAtom::lookup_html("FOREIGNOBJECT"), None);
        assert_eq!(NameAtom::lookup_html("my-widget"), None);
        // Long names with uppercase fall back to the allocating path.
        let long = "A".repeat(70);
        assert_eq!(NameAtom::lookup_html(&long), None);
    }

    #[test]
    fn tag_name_casing_semantics() {
        // `tagName` uppercases for HTML elements: the interned name keeps
        // its canonical (lowercase) form, so uppercasing still works.
        assert_eq!(TagName::from("div").as_str().to_ascii_uppercase(), "DIV");
        // Foreign names keep their case exactly.
        assert_eq!(TagName::from("foreignObject").as_str(), "foreignObject");
        assert_eq!(
            TagName::from("foreignObject").as_str().to_ascii_uppercase(),
            "FOREIGNOBJECT"
        );
        // Attribute names behave the same way.
        assert_eq!(AttrName::from("viewBox").as_str(), "viewBox");
    }

    #[test]
    fn atom_consts_match_table() {
        let cases: &[(&str, NameAtom)] = &[
            ("a", NameAtom::A),
            ("address", NameAtom::ADDRESS),
            ("annotation-xml", NameAtom::ANNOTATION_XML),
            ("applet", NameAtom::APPLET),
            ("area", NameAtom::AREA),
            ("article", NameAtom::ARTICLE),
            ("aside", NameAtom::ASIDE),
            ("b", NameAtom::B),
            ("base", NameAtom::BASE),
            ("basefont", NameAtom::BASEFONT),
            ("bgsound", NameAtom::BGSOUND),
            ("big", NameAtom::BIG),
            ("blockquote", NameAtom::BLOCKQUOTE),
            ("body", NameAtom::BODY),
            ("br", NameAtom::BR),
            ("button", NameAtom::BUTTON),
            ("caption", NameAtom::CAPTION),
            ("center", NameAtom::CENTER),
            ("code", NameAtom::CODE),
            ("col", NameAtom::COL),
            ("colgroup", NameAtom::COLGROUP),
            ("color", NameAtom::COLOR),
            ("datalist", NameAtom::DATALIST),
            ("dd", NameAtom::DD),
            ("desc", NameAtom::DESC),
            ("details", NameAtom::DETAILS),
            ("dialog", NameAtom::DIALOG),
            ("dir", NameAtom::DIR),
            ("div", NameAtom::DIV),
            ("dl", NameAtom::DL),
            ("dt", NameAtom::DT),
            ("em", NameAtom::EM),
            ("embed", NameAtom::EMBED),
            ("encoding", NameAtom::ENCODING),
            ("face", NameAtom::FACE),
            ("fieldset", NameAtom::FIELDSET),
            ("figcaption", NameAtom::FIGCAPTION),
            ("figure", NameAtom::FIGURE),
            ("font", NameAtom::FONT),
            ("footer", NameAtom::FOOTER),
            ("foreignObject", NameAtom::FOREIGNOBJECT),
            ("form", NameAtom::FORM),
            ("frame", NameAtom::FRAME),
            ("frameset", NameAtom::FRAMESET),
            ("h1", NameAtom::H1),
            ("h2", NameAtom::H2),
            ("h3", NameAtom::H3),
            ("h4", NameAtom::H4),
            ("h5", NameAtom::H5),
            ("h6", NameAtom::H6),
            ("head", NameAtom::HEAD),
            ("header", NameAtom::HEADER),
            ("hgroup", NameAtom::HGROUP),
            ("hr", NameAtom::HR),
            ("html", NameAtom::HTML),
            ("i", NameAtom::I),
            ("iframe", NameAtom::IFRAME),
            ("image", NameAtom::IMAGE),
            ("img", NameAtom::IMG),
            ("input", NameAtom::INPUT),
            ("keygen", NameAtom::KEYGEN),
            ("language", NameAtom::LANGUAGE),
            ("li", NameAtom::LI),
            ("link", NameAtom::LINK),
            ("listing", NameAtom::LISTING),
            ("main", NameAtom::MAIN),
            ("malignmark", NameAtom::MALIGNMARK),
            ("marquee", NameAtom::MARQUEE),
            ("math", NameAtom::MATH),
            ("menu", NameAtom::MENU),
            ("meta", NameAtom::META),
            ("mglyph", NameAtom::MGLYPH),
            ("mi", NameAtom::MI),
            ("mn", NameAtom::MN),
            ("mo", NameAtom::MO),
            ("ms", NameAtom::MS),
            ("mtext", NameAtom::MTEXT),
            ("multiple", NameAtom::MULTIPLE),
            ("nav", NameAtom::NAV),
            ("nobr", NameAtom::NOBR),
            ("noembed", NameAtom::NOEMBED),
            ("noframes", NameAtom::NOFRAMES),
            ("noscript", NameAtom::NOSCRIPT),
            ("object", NameAtom::OBJECT),
            ("ol", NameAtom::OL),
            ("optgroup", NameAtom::OPTGROUP),
            ("option", NameAtom::OPTION),
            ("p", NameAtom::P),
            ("param", NameAtom::PARAM),
            ("plaintext", NameAtom::PLAINTEXT),
            ("pre", NameAtom::PRE),
            ("rb", NameAtom::RB),
            ("rp", NameAtom::RP),
            ("rt", NameAtom::RT),
            ("rtc", NameAtom::RTC),
            ("ruby", NameAtom::RUBY),
            ("s", NameAtom::S),
            ("script", NameAtom::SCRIPT),
            ("search", NameAtom::SEARCH),
            ("section", NameAtom::SECTION),
            ("select", NameAtom::SELECT),
            ("selected", NameAtom::SELECTED),
            ("selectedcontent", NameAtom::SELECTEDCONTENT),
            ("size", NameAtom::SIZE),
            ("small", NameAtom::SMALL),
            ("source", NameAtom::SOURCE),
            ("span", NameAtom::SPAN),
            ("strike", NameAtom::STRIKE),
            ("strong", NameAtom::STRONG),
            ("style", NameAtom::STYLE),
            ("sub", NameAtom::SUB),
            ("summary", NameAtom::SUMMARY),
            ("sup", NameAtom::SUP),
            ("svg", NameAtom::SVG),
            ("table", NameAtom::TABLE),
            ("tbody", NameAtom::TBODY),
            ("td", NameAtom::TD),
            ("template", NameAtom::TEMPLATE),
            ("textarea", NameAtom::TEXTAREA),
            ("tfoot", NameAtom::TFOOT),
            ("th", NameAtom::TH),
            ("thead", NameAtom::THEAD),
            ("title", NameAtom::TITLE),
            ("tr", NameAtom::TR),
            ("track", NameAtom::TRACK),
            ("tt", NameAtom::TT),
            ("type", NameAtom::TYPE),
            ("u", NameAtom::U),
            ("ul", NameAtom::UL),
            ("var", NameAtom::VAR),
            ("wbr", NameAtom::WBR),
            ("xmp", NameAtom::XMP),
        ];
        for (name, atom) in cases {
            assert_eq!(atom.as_str(), *name, "const for {name}");
            assert_eq!(Name::intern(name).atom(), *atom);
        }
    }

    #[test]
    fn hot_consts_resolve() {
        assert_eq!(NameAtom::id().as_str(), "id");
        assert_eq!(NameAtom::class().as_str(), "class");
        assert_eq!(Name::intern("id").atom(), NameAtom::id());
        assert_eq!(Name::intern("class").atom(), NameAtom::class());
    }
}
