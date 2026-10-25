use bowser_html::{ScriptElement, ScriptExecutor, ScriptHost, parse_with_scripting_and_executor};
use bowser_js::{Value, Vm, install_parse_document_shim};

struct ShimProbe {
    vm: Vm,
    /// One probe per `<script>` in the document, consumed in order by
    /// `execute_script`. Single-script tests pass exactly one.
    probes: Vec<String>,
    probe_idx: usize,
    result: Option<Result<String, String>>,
    /// Viewport threaded into the shim install; `None` exercises the
    /// 800x600 fallback.
    viewport: Option<(u32, u32)>,
}

impl ScriptExecutor for ShimProbe {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        _script: &str,
    ) {
        #[allow(clippy::unnecessary_cast)]
        let host_ptr =
            host as *mut ScriptHost<'_, '_, '_> as *mut ScriptHost<'static, 'static, 'static>;
        let probe = self.probes.get(self.probe_idx).cloned().unwrap_or_default();
        self.probe_idx += 1;
        let outcome = match install_parse_document_shim(&mut self.vm, host_ptr, self.viewport) {
            Ok(guard) => {
                let r = match self.vm.eval(&probe) {
                    Ok(_) => match self.vm.eval("globalThis.__result") {
                        Ok(Value::Str(s)) => Ok(s.as_wtf8().to_string()),
                        Ok(v) => Ok(format!("{v:?}")),
                        Err(e) => Err(format!("readback: {e:?}")),
                    },
                    Err(e) => Err(format!("eval: {e:?}")),
                };
                guard.uninstall(&mut self.vm);
                r
            }
            Err(e) => Err(format!("install: {e:?}")),
        };
        self.result = Some(outcome);
    }
}

fn run_probe(html: &str, probe_js: &str) -> String {
    run_probes(html, &[probe_js])
}

/// `run_probe` for multi-script documents: `probes[i]` runs in the
/// `i`-th `<script>` element, each with its own shim install. Used to
/// exercise cross-script wrapper re-keying (a wrapper cached by an
/// earlier script's install is repointed at the next script's slot).
fn run_probes(html: &str, probes: &[&str]) -> String {
    run_probes_with_viewport(html, probes, None)
}

/// `run_probes` with an explicit embedder viewport threaded into the
/// shim install.
fn run_probes_with_viewport(html: &str, probes: &[&str], viewport: Option<(u32, u32)>) -> String {
    let mut ex = ShimProbe {
        vm: Vm::new(),
        probes: probes.iter().map(|s| s.to_string()).collect(),
        probe_idx: 0,
        result: None,
        viewport,
    };
    let scripts: String = (0..probes.len())
        .map(|i| format!("<script>{i}</script>"))
        .collect();
    let full = format!("{html}{scripts}");
    parse_with_scripting_and_executor(full.as_bytes(), true, &mut ex);
    match ex.result {
        Some(Ok(r)) => r,
        Some(Err(e)) => panic!("probe failed: {e}"),
        None => panic!("probe produced no result"),
    }
}

/// `run_probe` with an explicit embedder viewport threaded into the
/// shim install.
fn run_probe_with_viewport(html: &str, probe_js: &str, viewport: Option<(u32, u32)>) -> String {
    run_probes_with_viewport(html, &[probe_js], viewport)
}

#[test]
fn shim_style_set_get_round_trip() {
    let r = run_probe(
        "<div id=t style=\"color: red\"></div>",
        "var el = document.getElementById('t');
         el.style.display = 'block';
         el.style['border-collapse'] = 'collapse';
         el.style.borderSpacing = '10px';
         globalThis.__result = [
           el.style.display,
           el.style.borderCollapse,
           el.style['border-spacing'],
           el.style.getPropertyValue('display'),
           el.getAttribute('style')
         ].join('|');",
    );
    assert!(r.starts_with("block|collapse|10px|block|"), "{r}");
    assert!(r.contains("display: block"), "{r}");
    assert!(r.contains("border-collapse: collapse"), "{r}");
    assert!(r.contains("border-spacing: 10px"), "{r}");
}

#[test]
fn shim_style_remove_and_css_text() {
    let r = run_probe(
        "<div id=t style=\"display: block; color: red\"></div>",
        "var el = document.getElementById('t');
         var old = el.style.removeProperty('display');
         el.style.setProperty('margin-top', '5px');
         var mt = el.style.marginTop;
         el.style.cssText = 'width: 10px';
         globalThis.__result = [old, mt, el.style.display, el.style.width, el.getAttribute('style')].join('|');",
    );
    assert_eq!(r, "block|5px||10px|width: 10px");
}

#[test]
fn shim_class_name_and_id_reflection() {
    let r = run_probe(
        "<div id=t class=\"a b\"></div>",
        "var el = document.getElementById('t');
         var c0 = el.className;
         el.className = 'x y';
         var i0 = el.id;
         el.id = 'u';
         globalThis.__result = [c0, el.className, el.getAttribute('class'), i0, el.id,
           document.getElementById('u').id].join('|');",
    );
    assert_eq!(r, "a b|x y|x y|t|u|u");
}

#[test]
fn shim_child_nodes_and_text_content() {
    let r = run_probe(
        "<div id=t>hello <span id=s>world</span>!</div>",
        "var el = document.getElementById('t');
         var kinds = [];
         for (var i = 0; i < el.childNodes.length; i++) {
           var n = el.childNodes[i];
           kinds.push(n.nodeType + ':' + (n.textContent || n.id));
         }
         var before = el.childNodes.length + '/' + el.children.length;
         var tc = el.textContent;
         el.textContent = 'replaced';
         globalThis.__result = [before, kinds.join(','), tc, el.textContent, el.children.length].join('|');",
    );
    assert_eq!(r, "3/1|3:hello ,1:world,3:!|hello world!|replaced|0");
}

#[test]
fn shim_text_child_nodes_empty_node_list() {
    // A text node's `childNodes` is an empty Array-kind NodeList, so
    // `Array.prototype.forEach.call(text.childNodes, fn)` does not throw
    // (WPT `check-layout-th.js` recurses through every node's `childNodes`).
    let r = run_probe(
        "<div id=t>hello</div>",
        "var t = document.getElementById('t').firstChild;
         var cn = t.childNodes;
         var threw = false;
         var seen = 0;
         try { Array.prototype.forEach.call(cn, function() { seen++; }); }
         catch (e) { threw = true; }
         globalThis.__result = [t.nodeType, cn.length, (cn instanceof NodeList),
           seen, !threw].join('|');",
    );
    assert_eq!(r, "3|0|true|0|true");
}

#[test]
fn shim_first_last_child() {
    let r = run_probe(
        "<div id=t><b id=b></b>mid<i id=i></i></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.firstChild.id, el.lastChild.id].join('|');",
    );
    assert_eq!(r, "b|i");
}

#[test]
fn shim_inner_html() {
    let r = run_probe(
        "<div id=t><p>a</p></div>",
        "var el = document.getElementById('t');
         var before = el.innerHTML;
         el.innerHTML = '<span id=n>hi</span>';
         var after = el.innerHTML;
         var found = document.getElementById('n');
         globalThis.__result = [before, after, found && found.textContent].join('|');",
    );
    assert_eq!(r, "<p>a</p>|<span id=\"n\">hi</span>|hi");
}

#[test]
fn shim_element_query_selector() {
    let r = run_probe(
        "<div id=o><p class=a><span id=s></span></p></div><p class=a id=p2></p>",
        "var o = document.getElementById('o');
         var one = o.querySelector('p.a');
         var all = o.querySelectorAll('p.a');
         var span = o.querySelector('p > span');
         var none = o.querySelector('div > span');
         globalThis.__result = [one && one.querySelector('span') ? 'yes' : 'no', all.length,
           span ? span.id : 'null', none ? 'bad' : 'null'].join('|');",
    );
    assert_eq!(r, "yes|1|s|null");
}

#[test]
fn shim_append_insert_remove_child() {
    let r = run_probe(
        "<div id=p><b id=b></b></div>",
        "var p = document.getElementById('p');
         var d = document.createElement('div');
         d.id = 'd';
         p.appendChild(d);
         var s = document.createElement('span');
         s.id = 's';
         p.insertBefore(s, d);
         var kids = [];
         for (var i = 0; i < p.children.length; i++) kids.push(p.children[i].id);
         var order1 = kids.join(',');
         p.removeChild(document.getElementById('b'));
         globalThis.__result = [order1, p.children.length, p.innerHTML].join('|');",
    );
    assert!(r.starts_with("b,s,d|2|"), "{r}");
    assert!(r.contains("id=\"s\""), "{r}");
    assert!(r.contains("id=\"d\""), "{r}");
    assert!(!r.contains("\"b\""), "{r}");
}

#[test]
fn shim_document_body_and_element() {
    let r = run_probe(
        "<div id=t></div>",
        "globalThis.__result = [document.body ? 'body-yes' : 'body-no',
           document.documentElement ? 'html-yes' : 'html-no',
           document.body.firstChild.id].join('|');",
    );
    assert_eq!(r, "body-yes|html-yes|t");
}

#[test]
fn shim_get_computed_style_inline() {
    let r = run_probe(
        "<div id=t style=\"border-collapse: collapse; width: 10px\"></div>",
        "var cs = getComputedStyle(document.getElementById('t'));
         globalThis.__result = [
           cs['border-collapse'], cs.borderCollapse,
           cs.width, cs.display,
           ('border-collapse' in cs), ('color' in cs),
           cs.getPropertyValue('width')
         ].join('|');",
    );
    assert_eq!(r, "collapse|collapse|10px|block|true|true|10px");
}

#[test]
fn shim_get_computed_style_inheritance() {
    // Mirrors css/css-tables/inheritance.html's assert_inherited flow.
    let r = run_probe(
        "<div id=container><div id=target></div></div>",
        "var container = document.getElementById('container');
         var target = document.getElementById('target');
         var out = [];
         out.push(getComputedStyle(target)['border-collapse']); // separate (initial)
         container.style['border-collapse'] = 'collapse';
         target.style['border-collapse'] = 'unset';
         out.push(getComputedStyle(target)['border-collapse']); // collapse (inherited)
         target.style['border-collapse'] = 'initial';
         out.push(getComputedStyle(target)['border-collapse']); // separate
         target.style['table-layout'] = 'unset';
         container.style['table-layout'] = 'fixed';
         out.push(getComputedStyle(target)['table-layout']); // auto (not inherited)
         target.style['table-layout'] = 'inherit';
         out.push(getComputedStyle(target)['table-layout']); // fixed
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "separate|collapse|separate|auto|fixed");
}

#[test]
fn shim_get_computed_style_ua_display() {
    let r = run_probe(
        "<table id=t><tr id=r><td id=c></td></tr></table><span id=s></span>",
        "globalThis.__result = [
           getComputedStyle(document.getElementById('t')).display,
           getComputedStyle(document.getElementById('r')).display,
           getComputedStyle(document.getElementById('c')).display,
           getComputedStyle(document.getElementById('s')).display
         ].join('|');",
    );
    assert_eq!(r, "table|table-row|table-cell|inline");
}

#[test]
fn shim_get_computed_style_stylesheet_rule() {
    // The sticky-computed shape: a `<style>` rule (`table *`) must
    // reach the shim's computed style, not just inline style.
    let r = run_probe(
        "<style>table * { position: sticky; }</style>\
         <table><thead id=h></thead><tbody><tr><td id=c>Test</td></tr></tbody></table>",
        "globalThis.__result = [
           getComputedStyle(document.getElementById('h')).position,
           getComputedStyle(document.getElementById('c')).position,
           getComputedStyle(document.querySelector('tr')).position,
           getComputedStyle(document.querySelector('table')).position
         ].join('|');",
    );
    // The table itself is not a descendant of a table: static.
    assert_eq!(r, "sticky|sticky|sticky|static");
}

#[test]
fn shim_get_computed_style_table_layout_from_stylesheet() {
    // fixed-layout-2's computed subtests: `table-layout` from a
    // stylesheet rule (and from inline style on a custom element).
    let r = run_probe(
        "<style>.f { table-layout: fixed; }</style>\
         <table class=f id=t></table><x-table id=x style=\"table-layout: fixed\"></x-table>",
        "globalThis.__result = [
           getComputedStyle(document.getElementById('t')).tableLayout,
           getComputedStyle(document.getElementById('x')).tableLayout,
           getComputedStyle(document.getElementById('t'))['table-layout']
         ].join('|');",
    );
    assert_eq!(r, "fixed|fixed|fixed");
}

#[test]
fn shim_get_computed_style_border_mapping() {
    // html-to-css-mapping-2 cases 1-5 (left side, plus the cell and
    // inline-style assertions), through the parse shim.
    let r = run_probe(
        "<style>table { border-collapse: separate !important; }</style>\
         <div><table id=t1 border=\"0\" rules=\"groups\"><tbody><tr><td></td></tr></tbody></table></div>\
         <div><table id=t2 border=\"0\" rules=\"groups\" frame=\"box\"><tbody><tr><td></td></tr></tbody></table></div>\
         <div><table id=t3 border=\"10\" rules=\"groups\" frame=\"box\"><tbody><tr><td id=c3></td></tr></tbody></table></div>\
         <div><table id=t4 border=\"yes\" rules=\"all\"><tr><td id=c4></td></tr></table></div>\
         <div><table id=t5 border=\"10\" frame=\"hsides\" style=\"border-right-style: solid\"><tr><td></td></tr></table></div>",
        "function b(id, side, kind) { return getComputedStyle(document.getElementById(id))['border' + side + kind]; }
         function c(id, side, kind) { return getComputedStyle(document.getElementById(id))['border' + side + kind]; }
         globalThis.__result = [
           b('t1','Left','Width'), b('t1','Left','Style'), b('t1','Left','Color'),
           b('t2','Left','Width'), b('t2','Left','Style'), b('t2','Left','Color'),
           b('t3','Left','Width'), b('t3','Left','Style'), b('t3','Left','Color'),
           c('c3','Top','Width'), c('c3','Top','Style'), c('c3','Top','Color'),
           c('c3','Left','Width'), c('c3','Left','Style'),
           b('t4','Left','Width'), b('t4','Left','Style'), b('t4','Left','Color'),
           c('c4','Left','Width'), c('c4','Left','Style'), c('c4','Left','Color'),
           b('t5','Left','Style'),
           b('t5','Right','Width'), b('t5','Right','Style'), b('t5','Right','Color')
         ].join('|');",
    );
    assert_eq!(
        r,
        "0px|hidden|rgb(128, 128, 128)|\
         1px|outset|rgb(0, 0, 0)|\
         10px|outset|rgb(128, 128, 128)|\
         1px|inset|rgb(128, 128, 128)|\
         1px|inset|\
         1px|outset|rgb(128, 128, 128)|\
         1px|solid|rgb(128, 128, 128)|\
         hidden|\
         10px|solid|rgb(128, 128, 128)",
        "{r}"
    );
}

#[test]
fn shim_get_computed_style_border_spacing_computed() {
    let r = run_probe(
        "<div id=a style=\"font-size: 10px; border-spacing: 1em 2px\"></div>\
         <div id=b style=\"border-spacing: 5px\"></div>\
         <div id=c style=\"border-spacing: -5px\"></div>\
         <div id=d></div>",
        "globalThis.__result = [
           getComputedStyle(document.getElementById('a')).borderSpacing,
           getComputedStyle(document.getElementById('b')).borderSpacing,
           getComputedStyle(document.getElementById('c')).borderSpacing,
           getComputedStyle(document.getElementById('d')).borderSpacing
         ].join('|');",
    );
    // A negative length is invalid: the declaration drops and the
    // initial value reads back. A single value serializes as one
    // length, and the initial value as `0px` (CSSOM shortest
    // serialization: an equal pair collapses).
    assert_eq!(r, "10px 2px|5px|0px|0px");
}

#[test]
fn shim_get_computed_style_detached_fallback() {
    // A detached element is not in the whole-tree rebuild: the
    // inline-only fallback serves it, with the same adjustments.
    let r = run_probe(
        "<div></div>",
        "var el = document.createElement('div');
         el.setAttribute('style', 'border-top-style: solid; border-top-width: thick; color: red');
         var cs = getComputedStyle(el);
         globalThis.__result = [cs.borderTopWidth, cs.borderTopStyle, cs.borderTopColor, cs.borderLeftWidth].join('|');",
    );
    assert_eq!(r, "5px|solid|rgb(255, 0, 0)|0px");
}

#[test]
fn shim_css_supports() {
    let r = run_probe(
        "<div></div>",
        "globalThis.__result = [
           CSS.supports('border-collapse', 'collapse'),
           CSS.supports('display', 'block'),
           CSS.supports('nope', 'x'),
           CSS.supports('display: table')
         ].join('|');",
    );
    assert_eq!(r, "true|true|false|true");
}

#[test]
fn shim_css_supports_validates_values() {
    // w52: the value is validated against the property's real
    // grammar (the same check `el.style` assignment uses), not just
    // non-empty. `calc-size()` parses (w51/w52) and validates as a
    // length; unknown keywords fail; `!important` never validates;
    // `var()` counts as supported. w66: `-moz-available` and
    // `-webkit-fill-available` normalize to `stretch`, so they validate.
    let r = run_probe(
        "<div></div>",
        "globalThis.__result = [
           CSS.supports('width', '10px'),
           CSS.supports('width', 'calc-size(100px, size * 2)'),
           CSS.supports('width', 'calc(10px + 100%)'),
           CSS.supports('width', 'fit-content'),
           CSS.supports('width', 'inherit'),
           CSS.supports('width', 'var(--x)'),
           CSS.supports('width', '-moz-available'),
           CSS.supports('width', '-webkit-fill-available'),
           CSS.supports('width', 'intrinsic'),
           CSS.supports('width', 'stretch'),
           // w63: `stretch` is a valid sizing keyword (CSS Sizing 3 §5)
           // for width/height/min-width/min-height/max-width/max-height/
           // flex-basis, so the probe above now reports true.
           CSS.supports('width', '10px !important'),
           CSS.supports('width', ''),
           CSS.supports('width', '10px garbage'),
           CSS.supports('display', 'bogus'),
           CSS.supports('width: 10px'),
           CSS.supports('width: -moz-available')
         ].join('|');",
    );
    assert_eq!(
        r, "true|true|true|true|true|true|true|true|false|true|false|false|false|false|true|true",
        "{r}"
    );
}

#[test]
fn shim_id_named_globals() {
    let r = run_probe(
        "<div id=container><span id=target></span></div>",
        "globalThis.__result = [typeof container, container.id, typeof target, target.id].join('|');",
    );
    assert_eq!(r, "object|container|object|target");
}

#[test]
fn shim_geometry_measures_mid_parse() {
    // The parse-phase `ParseLayoutHook` runs a partial layout on demand:
    // mid-parse geometry reads return computed values, not zeros. The
    // empty block-level div fills the 800px layout viewport; its rect
    // is relative to the partial-layout root (the containing block).
    let r = run_probe(
        "<div id=t></div>",
        "var el = document.getElementById('t');
         var r = el.getBoundingClientRect();
         globalThis.__result = [el.offsetWidth, el.offsetHeight, el.clientWidth, r.width, r.top, el.offsetParent.tagName].join('|');",
    );
    assert_eq!(r, "800|0|800|800|0|BODY");
}

#[test]
fn shim_geometry_rect_is_dom_rect() {
    // The parse-shim `getBoundingClientRect` rect links to the
    // shim realm's `DOMRect.prototype`: `instanceof DOMRect` holds
    // and the eight fields read back.
    let r = run_probe(
        "<div id=t></div>",
        "var r = document.getElementById('t').getBoundingClientRect();
         var out = [r instanceof DOMRect, r.constructor.name === 'DOMRect'];
         out.push([r.x, r.y, r.width, r.height, r.top, r.right, r.bottom, r.left].join(','));
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "true|true|0,0,800,0,0,800,0,0");
}

#[test]
fn shim_geometry_uses_embedder_viewport() {
    // The embedder's viewport threads through the install into the
    // mid-parse partial layout: a block div fills 1280px and `50vw`
    // resolves against the threaded width, not the 800x600 fallback.
    let r = run_probe_with_viewport(
        "<div id=t></div><div id=u style=\"width:50vw\"></div>",
        "var t = document.getElementById('t');
         var u = document.getElementById('u');
         globalThis.__result = [t.offsetWidth, u.offsetWidth].join('|');",
        Some((1280, 800)),
    );
    assert_eq!(r, "1280|640");
}

#[test]
fn shim_geometry_display_none_still_zero() {
    // No box, no geometry: `display: none` still measures all zeros and
    // reports a null offsetParent mid-parse.
    let r = run_probe(
        "<div id=t style=\"display:none;width:100px\"></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.offsetWidth, el.offsetHeight, el.clientWidth, el.offsetParent].join('|');",
    );
    // JS Array.join stringifies null/undefined as "".
    assert_eq!(r, "0|0|0|");
}

#[test]
fn parse_layout_hook_forced_reflow() {
    // The classic forced-sync-layout pattern mid-parse: a style write
    // bumps the tree's `mutation_count`, so the next geometry read
    // re-runs the partial layout and observes the new size.
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"></div>",
        "var el = document.getElementById('t');
         var before = el.offsetWidth;
         el.style.width = '200px';
         var after = el.offsetWidth;
         var r = el.getBoundingClientRect();
         globalThis.__result = [before, after, el.offsetHeight, r.width, r.height].join('|');",
    );
    assert_eq!(r, "100|200|50|200|50");
}

#[test]
fn parse_layout_hook_applies_style_element_rules() {
    // `<style>` element rules are harvested from the snapshot and apply
    // to the partial layout document-wide.
    let r = run_probe(
        "<style>#t{width:120px;height:33px}</style><div id=t></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.offsetWidth, el.offsetHeight].join('|');",
    );
    assert_eq!(r, "120|33");
}

#[test]
fn parse_layout_hook_offset_parent() {
    // `offsetParent` resolves through the partial layout tree via
    // `table_offset_parent`: the positioned ancestor for a static
    // child, null for `body` itself. `body` is measured (nonzero
    // `offsetWidth`), proving the null comes from the real
    // `table_offset_parent → None` path rather than a missing box.
    let r = run_probe(
        "<div id=outer style=\"position:relative\"><div id=inner style=\"width:100px\"></div></div>",
        "var el = document.getElementById('inner');
         globalThis.__result = [el.offsetParent.id, el.offsetParent.tagName, document.body.offsetParent, document.body.offsetWidth].join('|');",
    );
    // JS Array.join stringifies null/undefined as "".
    assert_eq!(r, "outer|DIV||800");
}

/// A [`ScriptExecutor`] that installs the parse shim, evals a probe,
/// and records the [`ParseLayoutHook`] pass count before uninstalling.
struct LayoutPassProbe {
    vm: Vm,
    probe: String,
    result: Option<String>,
    passes: u64,
}

impl ScriptExecutor for LayoutPassProbe {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        _script: &str,
    ) {
        #[allow(clippy::unnecessary_cast)]
        let host_ptr =
            host as *mut ScriptHost<'_, '_, '_> as *mut ScriptHost<'static, 'static, 'static>;
        let guard = install_parse_document_shim(&mut self.vm, host_ptr, None).expect("install");
        self.vm.eval(&self.probe).expect("probe eval");
        // Read the counter before uninstall pops the slot.
        self.passes = guard.layout_passes();
        let result = self.vm.eval("globalThis.__result").expect("readback");
        self.result = Some(match result {
            Value::Str(s) => s.as_wtf8().to_string(),
            v => format!("{v:?}"),
        });
        guard.uninstall(&mut self.vm);
    }
}

fn run_layout_probe(html: &str, probe_js: &str) -> (String, u64) {
    let mut ex = LayoutPassProbe {
        vm: Vm::new(),
        probe: probe_js.to_string(),
        result: None,
        passes: 0,
    };
    let full = format!("{html}<script>0</script>");
    parse_with_scripting_and_executor(full.as_bytes(), true, &mut ex);
    (ex.result.expect("probe produced no result"), ex.passes)
}

#[test]
fn parse_layout_hook_caches_by_mutation_count() {
    // Repeat reads between DOM mutations are served from the cache:
    // exactly two partial layout passes for five reads.
    let (r, passes) = run_layout_probe(
        "<div id=t style=\"width:100px;height:50px\"></div>",
        "var el = document.getElementById('t');
         var a = el.offsetWidth;   // pass 1
         var b = el.offsetWidth;   // cached
         var c = el.clientWidth;   // cached
         el.style.width = '200px'; // mutation
         var d = el.offsetWidth;   // pass 2
         var e = el.offsetWidth;   // cached
         globalThis.__result = [a, b, c, d, e].join('|');",
    );
    assert_eq!(r, "100|100|100|200|200");
    assert_eq!(
        passes, 2,
        "repeat reads between mutations must not re-layout"
    );
}

#[test]
fn parse_layout_hook_stale_free_reads() {
    // Zero-mutation reads return identical values without re-layout; a
    // `setAttribute` mutation invalidates the cache and re-measures.
    let (r, passes) = run_layout_probe(
        "<div id=t style=\"width:100px;height:50px\"></div>",
        "var el = document.getElementById('t');
         var a = el.offsetWidth;  // pass 1
         var b = el.offsetWidth;  // cached, stale-free
         el.setAttribute('style', 'width:150px;height:50px'); // mutation
         var c = el.offsetWidth;  // pass 2
         var d = el.offsetWidth;  // cached, stale-free
         globalThis.__result = [a, b, c, d].join('|');",
    );
    assert_eq!(r, "100|100|150|150");
    assert_eq!(
        passes, 2,
        "repeat reads between mutations must not re-layout"
    );
}

#[test]
fn shim_style_valid_invalid_values() {
    // Mirrors /css/support/parsing-testcommon.js valid/invalid flows.
    let r = run_probe(
        "<div></div>",
        "var div = document.createElement('div');
         var out = [];
         div.style['border-collapse'] = '';
         div.style['border-collapse'] = 'collapse';
         out.push(div.style.getPropertyValue('border-collapse'));
         div.style['border-collapse'] = 'none';
         out.push('[' + div.style.getPropertyValue('border-collapse') + ']');
         div.style['border-spacing'] = '10px 20px';
         out.push(div.style.getPropertyValue('border-spacing'));
         div.style['border-spacing'] = '10%';
         out.push('[' + div.style.getPropertyValue('border-spacing') + ']');
         div.style['table-layout'] = 'fixed';
         out.push(div.style.getPropertyValue('table-layout'));
         div.style['caption-side'] = 'auto';
         out.push('[' + div.style.getPropertyValue('caption-side') + ']');
         globalThis.__result = out.join('|');",
    );
    // Invalid assignments are no-ops per CSSOM: the old declaration
    // survives (WPT parsing-testcommon `testInvalidValue`).
    assert_eq!(r, "collapse|[collapse]|10px 20px|[10px 20px]|fixed|[]");
}

#[test]
fn shim_siblings_clone_template() {
    let r = run_probe(
        "<div id=p><b id=b></b><i id=i></i></div><template id=tmpl><p id=tp>hi</p></template>",
        "var b = document.getElementById('b');
         var out = [];
         out.push(b.nextElementSibling.id);
         out.push(b.previousElementSibling ? 'bad' : 'null');
         out.push(document.getElementById('i').previousElementSibling.id);
         var c = b.cloneNode(false);
         out.push(c.tagName + ':' + c.children.length);
         var d = document.getElementById('p').cloneNode(true);
         out.push(d.children.length + ':' + d.querySelector('i').id);
         var tmpl = document.getElementById('tmpl');
         out.push(tmpl.content ? tmpl.content.querySelector('p').textContent : 'no-content');
         out.push(document.getElementById('b').content ? 'bad' : 'null');
         var frag = tmpl.content.cloneNode(true);
         document.getElementById('p').appendChild(frag);
         out.push(document.getElementById('p').querySelectorAll('p').length);
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "i|null|b|B:0|2:i|hi|null|1");
}

#[test]
fn shim_tag_name_and_fonts() {
    let r = run_probe(
        "<div id=t><span></span></div>",
        "var el = document.getElementById('t');
         var out = [el.tagName, el.firstChild.tagName];
         var thenable = (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === 'function');
         out.push(thenable ? 'fonts-ok' : 'fonts-missing');
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "DIV|SPAN|fonts-ok");
}

#[test]
fn shim_remove_and_class_list() {
    let r = run_probe(
        "<div id=p><b id=b class=\"x\"></b></div>",
        "var p = document.getElementById('p');
         var b = document.getElementById('b');
         var out = [];
         out.push(b.classList.contains('x'));
         b.classList.toggle('x');
         out.push(b.classList.contains('x'));
         b.classList.toggle('x', true);
         b.classList.add('y', 'z');
         out.push(b.className);
         b.classList.remove('y');
         out.push(b.className);
         b.remove();
         out.push(p.children.length);
         out.push(document.scrollingElement ? 'scroll-ok' : 'scroll-missing');
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "true|false|x y z|x z|0|scroll-ok");
}

#[test]
fn shim_class_list_same_object() {
    // WebIDL `[SameObject]`: repeated `classList` access returns the
    // identical live view, and the cached view tracks attribute writes.
    let r = run_probe(
        "<div id=b class=\"x\"></div>",
        "var b = document.getElementById('b');
         var out = [];
         out.push(b.classList === b.classList);
         b.classList.add('y');
         out.push(b.classList.contains('y') && b.className === 'x y');
         b.classList.remove('x');
         out.push(b.classList.contains('x') === false);
         globalThis.__result = out.join('|');",
    );
    assert_eq!(r, "true|true|true");
}

#[test]
fn shim_computed_testcommon_flow() {
    // Mirrors /css/support/computed-testcommon.js::test_computed_value.
    let r = run_probe(
        "<div id=target></div>",
        "var property = 'border-collapse', specified = 'collapse';
         var target = document.getElementById('target');
         var ok = (property in getComputedStyle(target)) && CSS.supports(property, specified);
         target.style[property] = '';
         target.style[property] = specified;
         var readValue = getComputedStyle(target)[property];
         globalThis.__result = [ok, readValue].join('|');",
    );
    assert_eq!(r, "true|collapse");
}

#[test]
fn parse_layout_hook_scroll_extents_overflow() {
    // The scroll-extent reverse pass in `measure_subtree` (previously
    // untested): an overflowing child pushes `scrollWidth`/`scrollHeight`
    // past the client sizes, which stay at the padding box.
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"><div style=\"width:300px;height:80px\"></div></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.scrollWidth, el.scrollHeight, el.clientWidth, el.clientHeight].join('|');",
    );
    assert_eq!(r, "300|80|100|50");
}

#[test]
fn parse_layout_hook_scroll_extents_transformed_child() {
    // The scroll-extent fold is transform-aware (CSS Overflow 3 §3,
    // mirroring the layout crate's `scrollable_extent`): a translated
    // child contributes its *transformed* border box, so the
    // scroller's `scrollWidth`/`scrollHeight` grow by the translation.
    // Untransformed, the child (100x20 at the origin) would leave both
    // at the client size (100|50).
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"><div style=\"width:100px;height:20px;transform:translate(150px,60px)\"></div></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.scrollWidth, el.scrollHeight, el.clientWidth, el.clientHeight].join('|');",
    );
    assert_eq!(r, "250|80|100|50");
}

#[test]
fn parse_layout_hook_scroll_extents_scaled_child_origin_pivot() {
    // A `scale()` composes through the accumulated matrix and pivots
    // on `transform-origin` (the default 50% 50%): the 100px-wide
    // child spans -50..150 after `scaleX(2)`, so only the far edge
    // (150) extends `scrollWidth` past the 100px client width.
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"><div style=\"width:100px;height:20px;transform:scaleX(2)\"></div></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.scrollWidth, el.scrollHeight, el.clientWidth].join('|');",
    );
    assert_eq!(r, "150|50|100");
}

#[test]
fn parse_layout_hook_scroll_extents_nested_transform_composition() {
    // Transforms compose down the tree: the inner child's own
    // translate(30px) rides on its parent's translate(100px), so the
    // far edge lands at 100 + 30 + 50 (its width) = 180, not 80. (The
    // 40px-wide parent's own transformed far edge is only 140, so the
    // child is what extends the union.)
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"><div style=\"width:40px;transform:translateX(100px)\"><div style=\"width:50px;height:20px;transform:translateX(30px)\"></div></div></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.scrollWidth, el.scrollHeight].join('|');",
    );
    assert_eq!(r, "180|50");
}

/// Authoritative `scrollWidth`/`scrollHeight` for `#id` from a full
/// layout pass, rebuilt from layout's own `scrollable_extent` via the
/// public `scroll_containers`: a container's `max` is its overflow
/// past the border box, so the scroll size is the client size plus
/// that overflow — exactly the quantity the shim's mid-parse walk in
/// `measure_subtree` must produce.
fn full_layout_scroll_size(html: &str, id: &str) -> (f32, f32) {
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let nid = doc
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("no element with id '{id}'"));
    let root = bowser_layout::layout(
        &doc,
        &[],
        bowser_layout::Size { w: 800.0, h: 600.0 },
        &bowser_layout::FixedFontMetrics,
    );
    fn find<'a>(
        b: &'a bowser_layout::LayoutBox,
        nid: bowser_dom::NodeId,
        ancestors: &mut Vec<&'a bowser_layout::LayoutBox>,
    ) -> Option<(
        &'a bowser_layout::LayoutBox,
        Vec<&'a bowser_layout::LayoutBox>,
    )> {
        if b.node == Some(nid) {
            return Some((b, ancestors.clone()));
        }
        ancestors.push(b);
        for c in &b.children {
            if let Some(found) = find(c, nid, ancestors) {
                return Some(found);
            }
        }
        ancestors.pop();
        None
    }
    let (scroller, ancestors) = find(&root, nid, &mut Vec::new()).expect("scroller laid out");
    let borders = bowser_layout::box_used_borders(scroller, &ancestors);
    let client_w = (scroller.rect.w - borders[3] - borders[1]).max(0.0);
    let client_h = (scroller.rect.h - borders[0] - borders[2]).max(0.0);
    let info = bowser_layout::scroll_containers(&root)
        .into_iter()
        .find(|i| i.node == Some(nid))
        .expect("scroller is a scroll container");
    (client_w + info.max.0, client_h + info.max.1)
}

/// The shim's mid-parse `scrollWidth`/`scrollHeight` for `#id` must
/// agree with the full-layout oracle above. `expect_oracle` pins the
/// oracle's own value, proving the case is not vacuous (the oracle
/// really exercises the spill / the skip).
fn assert_shim_matches_full_layout(html: &str, id: &str, expect_oracle: (f32, f32)) {
    let (exp_w, exp_h) = full_layout_scroll_size(html, id);
    assert!(
        (exp_w - expect_oracle.0).abs() < 0.001 && (exp_h - expect_oracle.1).abs() < 0.001,
        "oracle {exp_w}|{exp_h} != pinned {0}|{1} for {html}",
        expect_oracle.0,
        expect_oracle.1
    );
    let r = run_probe(
        html,
        &format!(
            "var el = document.getElementById('{id}'); \
             globalThis.__result = [el.scrollWidth, el.scrollHeight].join('|');"
        ),
    );
    let (w, h) = r.split_once('|').expect("w|h result");
    let got_w: f32 = w.parse().expect("scrollWidth parses");
    let got_h: f32 = h.parse().expect("scrollHeight parses");
    assert!(
        (got_w - exp_w).abs() < 0.001 && (got_h - exp_h).abs() < 0.001,
        "mid-parse scroll size {got_w}|{got_h} disagrees with full layout {exp_w}|{exp_h} for {html}"
    );
}

#[test]
fn parse_layout_hook_scroll_extent_split_inline_spill() {
    // Divergence (a): a `padding: 40px` span around a 10px block
    // paints a 90px border box — the wrapper's own border/padding
    // spill past its content box is real scrollable overflow
    // (layout's `split_inline_spill_insets`), so `scrollHeight`
    // reaches 90 against the 50px scrollport. The shim walk used to
    // skip the wrapper wholesale and report only the child's 10px
    // (`scrollHeight` 50).
    assert_shim_matches_full_layout(
        "<div id=t style=\"width:100px;height:50px;overflow:auto\"><span style=\"padding:40px\"><div style=\"height:10px\"></div></span></div>",
        "t",
        (100.0, 90.0),
    );
}

#[test]
fn parse_layout_hook_scroll_extent_split_inline_spill_transformed() {
    // Transformed branch of the spill arm: the padded wrapper
    // translated 20px down lands its bottom padding edge at 110
    // against the 50px scrollport, while the child alone reaches
    // only 70.
    assert_shim_matches_full_layout(
        "<div id=t style=\"width:100px;height:50px;overflow:auto\"><span style=\"padding:40px;transform:translate(0,20px)\"><div style=\"height:10px\"></div></span></div>",
        "t",
        (100.0, 110.0),
    );
}

#[test]
fn parse_layout_hook_scroll_extent_abspos_outside_cb() {
    // The abspos child's containing block is `#outer`, outside the
    // scroller, so it does not scroll with `#t`: layout's
    // `scrollable_extent` skips it and `scrollWidth` stays at the
    // 100px client width. The shim walk used to fold it in and
    // over-report 250.
    assert_shim_matches_full_layout(
        "<div id=outer style=\"position:relative;width:300px;height:50px\"><div id=t style=\"width:100px;height:50px;overflow:auto\"><div style=\"position:absolute;left:200px;top:10px;width:50px;height:20px\"></div></div></div>",
        "t",
        (100.0, 50.0),
    );
}

#[test]
fn parse_layout_hook_scroll_extent_abspos_inside_cb() {
    // Control: the scroller itself is positioned, so the abspos
    // child's containing block is inside it — it scrolls with `#t`
    // and extends `scrollWidth` to 250 in both passes.
    assert_shim_matches_full_layout(
        "<div id=t style=\"position:relative;width:100px;height:50px;overflow:auto\"><div style=\"position:absolute;left:200px;top:10px;width:50px;height:20px\"></div></div>",
        "t",
        (250.0, 50.0),
    );
}

#[test]
fn parse_layout_hook_scroll_extent_abspos_nested_positioned_cb() {
    // Control: a positioned intermediate box starts a containing
    // block inside the (static) scroller, so the abspos grandchild
    // still scrolls with `#t` and extends `scrollWidth` to 250 in
    // both passes.
    assert_shim_matches_full_layout(
        "<div id=t style=\"width:100px;height:50px;overflow:auto\"><div style=\"position:relative\"><div style=\"position:absolute;left:200px;top:10px;width:50px;height:20px\"></div></div></div>",
        "t",
        (250.0, 50.0),
    );
}

#[test]
fn parse_layout_hook_scroll_extent_nested_scroller_skipped() {
    // A nested scroll container's own box contributes to the outer
    // scroller's extent, but its overflowing content is clipped by its
    // own scrollport: layout's `scrollable_extent` skips the nested
    // subtree (`scroll_containers`' deepest-first `skip` set), so the
    // outer `scrollWidth` stays at the 100px client width. The shim
    // walk had no skip set and folded the 500px content into the
    // outer scroller (scrollWidth 500).
    assert_shim_matches_full_layout(
        "<div id=t style=\"width:100px;height:100px;overflow:auto\"><div style=\"width:100px;height:100px;overflow:auto\"><div style=\"width:500px;height:10px\"></div></div></div>",
        "t",
        (100.0, 100.0),
    );
}

#[test]
fn parse_layout_hook_client_left_top_borders() {
    // `clientLeft`/`clientTop` report the used border widths exactly
    // (previously untested); the client sizes exclude them.
    let r = run_probe(
        "<div id=t style=\"border-left:7px solid black;border-top:3px solid black;width:100px;height:50px\"></div>",
        "var el = document.getElementById('t');
         globalThis.__result = [el.clientLeft, el.clientTop, el.clientWidth, el.clientHeight].join('|');",
    );
    assert_eq!(r, "7|3|100|50");
}

#[test]
fn parse_layout_hook_offset_left_top_positioned_parent() {
    // `offsetLeft`/`offsetTop` are measured from the offset parent's
    // padding-box origin: the positioned ancestor's border widths are
    // subtracted exactly (17px child edge minus 10px border = 7).
    let r = run_probe(
        "<div id=outer style=\"position:relative;border-left:10px solid black;border-top:5px solid black\"><div id=inner style=\"margin-left:7px;margin-top:3px;width:100px;height:20px\"></div></div>",
        "var el = document.getElementById('inner');
         globalThis.__result = [el.offsetParent.id, el.offsetLeft, el.offsetTop].join('|');",
    );
    assert_eq!(r, "outer|7|3");
}

#[test]
fn parse_layout_hook_baseline_td_pct_divs() {
    // w76 bindings task 1: WPT `css-tables/tentative/baseline-td`
    // "%ge children" shape, read mid-parse through the shim: the
    // 50% div's offsetParent is its td and its offsetTop is 30
    // (row baseline 80 minus its own 50 bottom edge), the 80% div
    // sits at 0, and both % heights resolve (50/80). The
    // offsetHeight pair discriminates mechanism (1) (% resolution
    // or baseline shift lost in the shim rebuild) from mechanism
    // (2) (offset-parent / first-box mismatch, via tagName) and
    // mechanism (3) (honest-zero fallback, all zeros).
    let r = run_probe(
        "<style>td{padding:0}</style><div>inline table<table style=\"height:100px;position:relative;display:inline-table;border-spacing:8px 0\">\
         <td style=\"vertical-align:baseline\"><div id=d50 style=\"width:100px;height:50%\"></div></td>\
         <td style=\"vertical-align:baseline\"><div id=d80 style=\"width:100px;height:80%\"></div></td></table></div>",
        "var d50 = document.getElementById('d50');\n         var d80 = document.getElementById('d80');\n         globalThis.__result = [d50.offsetParent.tagName, d50.offsetTop, d50.offsetHeight, d80.offsetTop, d80.offsetHeight].join('|');",
    );
    assert_eq!(r, "TD|30|50|0|80");
}

#[test]
fn parse_layout_hook_baseline_td_pct_divs_widened() {
    // Same shape as `parse_layout_hook_baseline_td_pct_divs`, but a
    // combinator rule forces the finding-2 selector-context widening:
    // the pass roots at the document element (the WPT page's
    // `.ahem td` rule widens the same way), so this pins the
    // whole-document shim path the WPT check actually exercises,
    // not just the table-rooted partial pass.
    let r = run_probe(
        "<style>div td{padding:0}</style><div>inline table<table style=\"height:100px;position:relative;display:inline-table;border-spacing:8px 0\">\
         <td style=\"vertical-align:baseline\"><div id=d50 style=\"width:100px;height:50%\"></div></td>\
         <td style=\"vertical-align:baseline\"><div id=d80 style=\"width:100px;height:80%\"></div></td></table></div>",
        "var d50 = document.getElementById('d50');\n         var d80 = document.getElementById('d80');\n         globalThis.__result = [d50.offsetParent.tagName, d50.offsetTop, d50.offsetHeight, d80.offsetTop, d80.offsetHeight].join('|');",
    );
    assert_eq!(r, "TD|30|50|0|80");
}

#[test]
fn parse_layout_hook_baseline_td_full_page_pct_tail() {
    // w76fix M5-probe: the whole WPT `baseline-td` page (scripts
    // and `<link>` sheets stripped; their rules inlined) read the
    // way `checkLayout` reads it — renamed for what it actually
    // pins: the %ge tail plus shim plumbing. w77fix: the QSA pin
    // below was updated from selector-GROUP order to document
    // order. w77 commit 36744d4 deliberately made shim
    // `querySelectorAll` return the union of comma groups in
    // document order, deduped — the DOM spec order, and the order
    // the production DOM (`bowser_dom`) already returns — so the
    // old group-order expectation (every `table`, then the
    // `.display-table` div) is no longer correct shim behavior,
    // even though it passed at the w76 base: that pass pinned a
    // documented non-spec shim quirk ("shim behavior, not the
    // spec"), not WPT geometry. Document order on this page is
    // TABLE, DIV, TABLE, TABLE, TABLE. The %ge tail pin is
    // unchanged, and no geometry in this probe indexes into the
    // QSA order (the tops come from a separate single-group
    // `.m` query, whose order is unaffected).
    // Subtest 1's table-1 offsets are collected only as the reads
    // that populate the whole-document cache entry before the %ge
    // reads, and are deliberately NOT asserted: they ride on Ahem
    // font metrics, which the shim path (FixedFontMetrics) does not
    // reproduce, so asserting them here would pin fixed-metric
    // accidents rather than WPT behavior. If this passes while WPT
    // fails, the difference is outside the shim geometry path
    // (harness scripts, link-sheet harvest, or the driver
    // viewport), not the table geometry.
    let r = run_probe(
        "<style>table{background:#AAA;border-spacing:8px 0px}td{padding:0;background:#BFB}.ahem{font:25px/2 Ahem}.ahem td,.ahem .display-cell{font:25px/2 Ahem}.m{display:inline-block}.display-table{display:table;background:#AAA;border-spacing:8px 0px}.display-cell{display:table-cell;padding:0}main td{padding:0}p{margin-top:4px;margin-bottom:4px}</style>\
         <main><h2>&lt;TD&gt; baseline tests</h2>\
         <p class=\"testdesc\">All the baseline values.</p>\
         <table style=\"height:100px\" class=\"ahem vgrad\">\
         <td style=\"font: 75px/1 Ahem;vertical-align:baseline;\"><div class=\"m\" data-offset-y=0>Xp</div></td>\
         <td style=\"\" title=\"default\"><div class=\"m\" data-offset-y=25>Xp</div></td>\
         <td style=\"vertical-align: top\"><div class=\"m\" data-offset-y=0>Xp</div></td>\
         <td style=\"vertical-align: middle\"><div class=\"m\" data-offset-y=25>Xp</div></td>\
         <td style=\"vertical-align: bottom\"><div class=\"m\" data-offset-y=50>Xp</div></td>\
         <td style=\"vertical-align: baseline\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: sub\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: super\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: text-top\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: text-bottom\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: 100px\"><div class=\"m\" data-offset-y=28>Xp</div></td>\
         <td style=\"vertical-align: 100%\"><div class=\"m\" data-offset-y=28>Xp</div></td></table>\
         <p class=\"testdesc\">display-table variant.</p>\
         <div style=\"height:100px;position:relative\" class=\"ahem vgrad display-table\">\
         <div class=\"display-cell\" style=\"font: 75px/1 Ahem;vertical-align:baseline;\"><div class=\"m\" data-offset-y=0>Xp</div></div>\
         <div class=\"display-cell\" style=\"\" title=\"default\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: top\"><div class=\"m\" data-offset-y=0>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: middle\"><div class=\"m\" data-offset-y=25>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: bottom\"><div class=\"m\" data-offset-y=50>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: baseline\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: sub\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: super\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: text-top\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: text-bottom\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: 100px\"><div class=\"m\" data-offset-y=28>Xp</div></div>\
         <div class=\"display-cell\" style=\"vertical-align: 100%\"><div class=\"m\" data-offset-y=28>Xp</div></div></div>\
         <p class=\"testdesc\">Baselines with %ge children</p>\
         <div>inline table<table style=\"height:100px;position:relative;display:inline-table\">\
         <td style=\"vertical-align:baseline\"><div id=d50 style=\"width:100px;height:50%\"></div></td>\
         <td style=\"vertical-align:baseline\"><div id=d80 style=\"width:100px;height:80%\"></div></td></table></div>\
         <p class=\"testdesc\">Baseline and td height</p>\
         <table style=\"border: 1px solid black\"><tr><td style=\"vertical-align:baseline;height:200px\">base</td>\
         <td style=\"vertical-align:baseline\"><div style=\"background:yellow;width:50px;height:100px\"></div></td></tr></table>\
         <p class=\"testdesc\">Baseline of rowspanned cell</p>\
         <table class=\"ahem\"><tr><td style=\"vertical-align:baseline;height:100px\">Xp</td>\
         <td style=\"vertical-align:baseline;font-size:40px;line-height:1.5\" rowspan=2><div class=\"m\">Xp</div><br>Xp<br>Xp</td></tr>\
         <tr><td style=\"vertical-align:baseline;height:100px\">Xp</td></tr></table></main>",
        "var nodes = document.querySelectorAll('table, .display-table');\n         var names = [];\n         for (var i = 0; i < nodes.length; i++) names.push(nodes[i].tagName);\n         var ms = document.querySelectorAll('.m');\n         var t1tops = [];\n         for (var i = 0; i < 12 && i < ms.length; i++) t1tops.push(ms[i].offsetTop);\n         var d50 = document.getElementById('d50');\n         var d80 = document.getElementById('d80');\n         globalThis.__result = [names.join(','), t1tops.join(','), d50.offsetParent.tagName, d50.offsetTop, d50.offsetHeight, d80.offsetTop, d80.offsetHeight].join('|');",
    );
    // Only the shim QSA document order (see the test doc) and the
    // %ge tail are asserted here.
    assert!(
        r.starts_with("TABLE,DIV,TABLE,TABLE,TABLE|"),
        "shim QSA must list matches in document order, got: {r}"
    );
    assert!(
        r.ends_with("|TD|30|50|0|80"),
        "%ge divs must read TD|30|50|0|80, got: {r}"
    );
}

#[test]
fn parse_layout_hook_baseline_td_pct_divs_wide_viewport() {
    // Same shape as `parse_layout_hook_baseline_td_pct_divs_widened`,
    // but at the driver's 1280x800 viewport instead of the 800x600
    // fallback: the viewport is the one remaining difference between
    // this probe and the driver's shim pass.
    let r = run_probe_with_viewport(
        "<style>div td{padding:0}</style><div>inline table<table style=\"height:100px;position:relative;display:inline-table;border-spacing:8px 0\">\
         <td style=\"vertical-align:baseline\"><div id=d50 style=\"width:100px;height:50%\"></div></td>\
         <td style=\"vertical-align:baseline\"><div id=d80 style=\"width:100px;height:80%\"></div></td></table></div>",
        "var d50 = document.getElementById('d50');\n         var d80 = document.getElementById('d80');\n         globalThis.__result = [d50.offsetParent.tagName, d50.offsetTop, d50.offsetHeight, d80.offsetTop, d80.offsetHeight].join('|');",
        Some((1280, 800)),
    );
    assert_eq!(r, "TD|30|50|0|80");
}

#[test]
fn parse_layout_hook_offset_left_top_static_body_fallback() {
    // Static element under static blocks: the provisional partial
    // root is the parent block, which contains no positioned ancestor
    // and no `body`, so the pass widens to the document element and
    // CSSOM View's `body` fallback resolves — `#grand`'s 30px margin
    // above the provisional root accumulates into `offsetLeft`, and
    // `offsetParent` is `body`, never null. (Pre-fix this read
    // `0|0|null`: the pass-local approximation.)
    let r = run_probe(
        "<div id=grand style=\"margin-left:30px\"><div id=outer style=\"width:400px\"><div id=inner style=\"width:100px;height:20px\"></div></div></div>",
        "var el = document.getElementById('inner');
         globalThis.__result = [el.offsetLeft, el.offsetTop, el.offsetParent.tagName].join('|');",
    );
    assert_eq!(r, "30|0|BODY");
}

#[test]
fn parse_layout_hook_detached_handle_honest_zero() {
    // A handle detached by `remove()` measures honest zeros: the
    // element has no box in the re-run partial layout (the stale-handle
    // path, distinct from the `display: none` case).
    let r = run_probe(
        "<div id=t style=\"width:100px;height:50px\"></div>",
        "var el = document.getElementById('t');
         var before = el.offsetWidth;
         el.remove();
         var g = el.getBoundingClientRect();
         globalThis.__result = [before, el.offsetWidth, el.clientWidth, g.width, el.offsetParent].join('|');",
    );
    // JS Array.join stringifies null/undefined as "".
    assert_eq!(r, "100|0|0|0|");
}

#[test]
fn parse_layout_hook_two_containing_blocks_b_first() {
    // Regression for the finding-1 cache-root bug: `#a` lives in
    // positioned `#outer`, `#b` outside it. Reading `#b` first roots
    // the pass at `body` (measuring everything); `#a`'s read must still
    // resolve its own offset parent, not the cache root's.
    let (r, passes) = run_layout_probe(
        "<div id=outer style=\"position:relative;width:300px\"><div id=a style=\"width:100px;height:20px\"></div></div><div id=b style=\"width:150px;height:30px\"></div>",
        "var a = document.getElementById('a');
         var b = document.getElementById('b');
         var bw = b.offsetWidth;  // roots at body
         var aw = a.offsetWidth;  // different root: re-runs, not served from #b's entry
         globalThis.__result = [bw, aw, a.offsetParent.id, b.offsetParent.tagName,
           a.offsetLeft, b.offsetTop, b.offsetLeft].join('|');",
    );
    assert_eq!(r, "150|100|outer|BODY|0|20|0");
    assert_eq!(
        passes, 2,
        "the second subtree must re-run its own pass instead of answering from the first subtree's entry"
    );
}

#[test]
fn parse_layout_hook_two_containing_blocks_a_first() {
    // Same geometry, reversed read order: `#a`'s pass roots at `#outer`
    // and cannot serve `#b`, so `#b`'s read re-runs from `body` rather
    // than returning the other subtree's zeros.
    let (r, passes) = run_layout_probe(
        "<div id=outer style=\"position:relative;width:300px\"><div id=a style=\"width:100px;height:20px\"></div></div><div id=b style=\"width:150px;height:30px\"></div>",
        "var a = document.getElementById('a');
         var b = document.getElementById('b');
         var aw = a.offsetWidth;  // roots at #outer
         var bw = b.offsetWidth;  // different root: re-runs, not zeros
         globalThis.__result = [aw, bw, a.offsetParent.id, b.offsetParent.tagName,
           a.offsetLeft, b.offsetTop, b.offsetLeft].join('|');",
    );
    assert_eq!(r, "100|150|outer|BODY|0|20|0");
    assert_eq!(
        passes, 2,
        "the second subtree must re-run its own pass instead of answering zeros"
    );
}

#[test]
fn shim_window_scroll_globals_reachable() {
    // Problem 2: `scrollTo`/`scrollBy`/`scrollX`/`scrollY` must exist on
    // the global object mid-parse — and since `window`/`self` alias
    // the global, `window.scrollTo(…)` must resolve identically to the
    // bare global call (was "callee is not a function" in WPT).
    let r = run_probe(
        "<body style=\"height: 3000px\"></body>",
        "var g = [typeof scrollTo, typeof scroll, typeof scrollBy, typeof scrollX, typeof scrollY,
                  typeof pageXOffset, typeof pageYOffset].join(',');
         var alias = [window.scrollTo === scrollTo, self.scrollTo === scrollTo,
                      window.scrollBy === scrollBy, scrollTo === scroll].join(',');
         scrollTo({top: 120});
         var after_abs = [scrollY, pageYOffset, window.scrollY].join(',');
         scrollBy(0, 30);
         var after_rel = [scrollY, window.pageYOffset].join(',');
         document.documentElement.scrollTop = 200;
         var doc_write = scrollY;
         document.body.scrollTop = 250;
         var body_write = scrollY;
         scrollTo(0, 99999);
         var clamped = scrollY;
         globalThis.__result = [g, alias, after_abs, after_rel, doc_write, body_write, clamped].join('|');",
    );
    // Document is 3000px tall, fallback viewport 600px: max scroll 2400.
    assert_eq!(
        r,
        "function,function,function,number,number,number,number|\
         true,true,true,true|\
         120,120,120|150,150|200|250|2400",
        "{r}"
    );
}

#[test]
fn shim_window_scroll_snaps_on_document_snap_type() {
    // A document with `scroll-snap-type: y mandatory` on `html` snaps
    // script window scrolls mid-parse: scrolling to 800 lands at the
    // 1000px snap area.
    let r = run_probe(
        "<style>html { scroll-snap-type: y mandatory; } \
         .t { height: 1000px; scroll-snap-align: start; }</style>\
         <div class=t></div><div class=t></div><div class=t></div>",
        "scrollTo({top: 800}); globalThis.__result = String(scrollY);",
    );
    assert_eq!(r, "1000", "{r}");
}

#[test]
fn shim_element_scroll_snaps_on_script_scroll_to() {
    // The WPT headline shape (`scrollTo-scrollBy-snaps.html`):
    // `scrollTo({left: 800})` into a `both mandatory` scroller with
    // snap areas at 0/1000 must land at 1000, not 800.
    let r = run_probe(
        "<style>\
           #scroller { overflow: scroll; scroll-snap-type: both mandatory; \
                       width: 400px; height: 400px; position: absolute; } \
           #space { width: 2000px; height: 2000px; } \
           .t { position: absolute; width: 200px; height: 200px; scroll-snap-align: start; } \
           #t1 { left: 0px; top: 0px; } #t2 { left: 1000px; top: 0px; } \
           #t3 { left: 0px; top: 1000px; } #t4 { left: 1000px; top: 1000px; } \
         </style>\
         <div id=scroller><div id=space></div>\
           <div class=t id=t1></div><div class=t id=t2></div>\
           <div class=t id=t3></div><div class=t id=t4></div></div>",
        "var sc = document.getElementById('scroller');
         sc.scrollTo({left: 800, top: 0});
         var l = sc.scrollLeft;
         sc.scrollTo({left: 800, top: 0});
         sc.scrollBy({left: -600, top: 0});
         globalThis.__result = [l, sc.scrollLeft].join(',');",
    );
    // First scrollTo lands at 1000 (not 800); the scrollBy from the
    // re-snapped 1000 by -600 rests at 400, whose nearest snap position
    // is 0.
    assert_eq!(r, "1000,0", "{r}");
}

/// `run_probe` with a seeded document URL, so harvested `<link
/// rel="stylesheet">` hrefs resolve (and fetch) against it like the
/// embedder seeds it in production (`set_parse_shim_document_url`).
fn run_probe_with_document_url(html: &str, probe_js: &str, doc_url: &str) -> String {
    let mut ex = ShimProbe {
        vm: Vm::new(),
        probes: vec![probe_js.to_string()],
        probe_idx: 0,
        result: None,
        viewport: None,
    };
    bowser_js::set_parse_shim_document_url(&mut ex.vm, doc_url);
    let full = format!("{html}<script>0</script>");
    parse_with_scripting_and_executor(full.as_bytes(), true, &mut ex);
    match ex.result {
        Some(Ok(r)) => r,
        Some(Err(e)) => panic!("probe failed: {e}"),
        None => panic!("probe produced no result"),
    }
}

/// A tiny blocking HTTP server for `<link>` sheet tests: serves each
/// `(path, css)` route with `200 text/css`, everything else 404.
/// Returns the `http://127.0.0.1:PORT` base the document URL is built
/// from. `bowser_net` sends absolute-form request targets, so the
/// scheme+authority prefix is stripped before matching.
fn link_css_server(routes: Vec<(&str, &str)>) -> String {
    use std::io::{Read, Write};
    let routes: Vec<(String, String)> = routes
        .into_iter()
        .map(|(p, c)| (p.to_string(), c.to_string()))
        .collect();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind css server");
    let port = listener.local_addr().expect("css server addr").port();
    std::thread::spawn(move || {
        for _ in 0..32 {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let target = req.split_whitespace().nth(1).unwrap_or("/");
            let path = match target.find("://") {
                Some(i) => match target[i + 3..].find('/') {
                    Some(j) => &target[i + 3 + j..],
                    None => "/",
                },
                None => target,
            };
            let path = path.split('?').next().unwrap_or("/");
            let (status, body) = match routes.iter().find(|(p, _)| *p == path) {
                Some((_, css)) => ("200 OK", css.as_str()),
                None => ("404 Not Found", ""),
            };
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/css\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[test]
fn shim_link_sheet_applies_to_template_cloned_table() {
    // The WPT css/css-tables/fixed-layout-2.html shape: the
    // `x-table -> display: table` mapping lives in a LINKED sheet
    // (`support/base.css`), the script clones the table out of
    // `<template>` contents mid-parse, appends the fragment to the
    // document, and reads `offsetWidth`. Before the w52 fix
    // `n_parse_el_clone_node` dropped the fragment marking, so the
    // clone went in as a literal `<template>` element instead of
    // being spliced: the cloned `x-table` sat inside the non-rendered
    // `<template>` and every `offsetWidth` read answered the honest
    // zero. The `<link>` sheet harvest itself was fine (proven by
    // env-gated tracing); the clone now carries the fragment marking,
    // matching DOM cloneNode type preservation.
    let base = link_css_server(vec![(
        "/css/css-tables/support/base.css",
        "x-table{display:table}x-tr{display:table-row}x-td{display:table-cell}",
    )]);
    let r = run_probe_with_document_url(
        "<html><head>\
         <link rel=\"author\" title=\"t\" href=\"mailto:t@example.com\">\
         <link rel=\"stylesheet\" href=\"./support/base.css\">\
         <style>x-table{table-layout:fixed;border-spacing:0}</style>\
         </head><body>\
         <main id=\"main\"><h1>Fixed Layout</h1></main>\
         <template id=\"template\"><div class=\"wrapper\">\
           <x-table><x-tr><x-td><div></div></x-td><x-td></x-td></x-tr></x-table>\
         </div></template>",
        "var fragment = template.content.cloneNode(true);\
         var table = fragment.querySelector('x-table');\
         table.style.width = '10px';\
         main.appendChild(fragment);\
         globalThis.__result = [\
           getComputedStyle(table).tableLayout,\
           getComputedStyle(table).display,\
           table.offsetWidth > 0 ? 'nonzero' : 'zero'\
         ].join('|');",
        &format!("{base}/css/css-tables/fixed-layout-2.html"),
    );
    assert_eq!(r, "fixed|table|nonzero", "{r}");
}

#[test]
fn shim_document_add_event_listener_registers_mid_parse() {
    // A mid-parse `document.addEventListener('x', fn)` must not throw
    // (the parse-shim document wrapper is an EventTarget mid-parse,
    // like the page global), and registers into the same shim listener
    // table as the window's: `adopt_shim_listeners` drains that one
    // table into the production document root, so WPT's
    // `document.addEventListener('TestRendered', …)` (css/css-overflow/
    // scrollbar-gutter-dynamic-004.html) survives the production
    // install.
    let r = run_probe(
        "<div id=t></div>",
        "var log = [];
         document.addEventListener('x', function() { log.push('doc'); });
         window.addEventListener('x', function() { log.push('win'); });
         window.dispatchEvent({type: 'x'});
         globalThis.__result = [typeof document.addEventListener,
           typeof document.removeEventListener, log.join(',')].join('|');",
    );
    assert_eq!(r, "function|function|doc,win", "{r}");
}

#[test]
fn shim_document_remove_event_listener() {
    // `document.removeEventListener` mirrors the window shim: it
    // removes the (type, callback, capture) registration from the
    // shared shim listener table.
    let r = run_probe(
        "<div id=t></div>",
        "var log = [];
         function onx() { log.push('x'); }
         document.addEventListener('x', onx);
         document.removeEventListener('x', onx);
         window.dispatchEvent({type: 'x'});
         document.addEventListener('x', onx);
         window.dispatchEvent({type: 'x'});
         globalThis.__result = log.join(',');",
    );
    assert_eq!(r, "x", "{r}");
}

#[test]
fn shim_rekeyed_wrapper_class_list_not_stale() {
    // Finding 1: script 1 touches `b.classList` on the id-named
    // element (caching the `[SameObject]` view baked with slot 1's
    // `(id, el_key)`); the slot pops at script end; script 2's
    // install repoints the wrapper at slot 2. The cached view must be
    // dropped on re-key, so `classList.contains` mints a fresh view
    // for the live slot instead of throwing
    // `TypeError: document is not available outside the parser`.
    let r = run_probes(
        "<div id=b class=\"a\"></div>",
        &[
            "b.classList; globalThis.__result = 'touched';",
            "globalThis.__result = b.classList.contains('a') ? 'hit' : 'miss';",
        ],
    );
    assert_eq!(r, "hit", "{r}");
}

#[test]
fn shim_rekeyed_wrapper_style_not_stale() {
    // The `style` view has the identical caching hole: script 1
    // caches it under slot 1, script 2's re-key must drop it so the
    // write below hits the live element.
    let r = run_probes(
        "<div id=b></div>",
        &[
            "b.style.color; globalThis.__result = 'touched';",
            "b.style.setProperty('color', 'red'); globalThis.__result = b.getAttribute('style');",
        ],
    );
    assert!(r.contains("color:"), "{r}");
}

#[test]
fn shim_dom_rect_ctor_coerces_args() {
    // Finding 2: `new DOMRect` applies WebIDL `unrestricted double`
    // conversion — strings, booleans, and `null` convert; a missing
    // or `undefined` argument defaults to `0.0`; `NaN` is preserved.
    let r = run_probe(
        "",
        "var d = new DOMRect('1', '2', '3', '4');\n         var e = new DOMRect();\n         var n = new DOMRect(NaN, undefined, null, true);\n         var o = new DOMRect({valueOf: function() { return 5; }});\n         globalThis.__result = [\n           [d.x, d.y, d.width, d.height].join(','),\n           [e.x, e.y, e.width, e.height].join(','),\n           [String(n.x), n.y, n.width, n.height].join(','),\n           o.x\n         ].join('|');",
    );
    assert_eq!(r, "1,2,3,4|0,0,0,0|NaN,0,0,1|5", "{r}");
}

#[test]
fn shim_dom_rect_derived_live_and_writable() {
    // Findings 3/4: `top`/`right`/`bottom`/`left` are not own data
    // properties — the prototype getters derive them live, so `d.x =
    // 10` updates `d.right`. `x`/`y`/`width`/`height` are writable
    // (own data on instances); `top` is read-only, so the sloppy-mode
    // assignment is ignored. On `Object.create(DOMRect.prototype)`
    // the prototype's writable setters fire.
    let r = run_probe(
        "",
        "var d = new DOMRect(1, 2, 3, 4);\n         d.x = 10;\n         var right1 = d.right;\n         d.top = 5;\n         var top1 = d.top;\n         var p = Object.create(DOMRect.prototype);\n         p.x = 7; p.width = 2;\n         globalThis.__result = [right1, top1, d.top, p.right, p.x].join('|');",
    );
    assert_eq!(r, "13|2|2|9|7", "{r}");
}

#[test]
fn production_document_fonts_probe() {
    // The production document carries a real FontFaceSet (not the
    // parse shim's `{ ready }` stub): constructor shape, SameObject
    // `document.fonts`, add/has/iteration, and `check()` false for
    // an unloaded family when no font cache is shared.
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut vm = Vm::new();
    let (doc, _) = bowser_dom::Document::parse_html(b"<html><head></head><body></body></html>");
    let doc = Rc::new(RefCell::new(doc));
    let base = bowser_url::parse("http://example.com/", None).unwrap();
    let ctx = vm
        .install_page_web_apis(doc, base, "BowserTest/1.0")
        .expect("production install");
    let v = vm
        .eval(
            "var out = []; \
             out.push(typeof FontFace); \
             out.push(document.fonts === document.fonts); \
             out.push(document.fonts.size); \
             out.push(document.fonts.status); \
             out.push(typeof document.fonts.ready.then); \
             var f = new FontFace('ProbeFam', 'url(probe.woff)'); \
             document.fonts.add(f); \
             out.push(document.fonts.has(f)); \
             out.push(document.fonts.size); \
             out.push([...document.fonts][0] === f); \
             out.push(document.fonts.check('16px ProbeFam')); \
             out.push(document.fonts.check('16px NoSuchFontXYZ')); \
             document.fonts.delete(f); \
             out.push(document.fonts.size); \
             out.join('|');",
        )
        .expect("fonts probe eval");
    let s = match v {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("unexpected probe value: {v:?}"),
    };
    assert_eq!(
        s, "function|true|0|loaded|function|true|1|true|false|false|0",
        "{s}"
    );
    vm.remove_page_web_apis(ctx);
}

#[test]
fn shim_dom_rect_ctor_props_attributes() {
    // Finding 4: per WebIDL the interface object's `prototype` is
    // non-writable, non-enumerable (non-configurable) and `name` is
    // non-writable, non-enumerable (configurable).
    let r = run_probe(
        "",
        "var dp = Object.getOwnPropertyDescriptor(DOMRect, 'prototype');\n         var dn = Object.getOwnPropertyDescriptor(DOMRect, 'name');\n         globalThis.__result = [dp.writable, dp.enumerable, dp.configurable,\n           dn.writable, dn.enumerable, dn.configurable].join('|');",
    );
    assert_eq!(r, "false|false|false|false|false|true", "{r}");
}

#[test]
fn parse_layout_hook_baseline_td_pct_divs_wpt_whitespace() {
    let r = run_probe(
        "<style>div td{padding:0}</style><div>inline table<table style=\"height:100px;position:relative;display:inline-table;border-spacing:8px 0\">\n         <td style=\"vertical-align:baseline\">\n         <div id=d50 style=\"width:100px;height:50%\"></div>\n         </td>\n         <td style=\"vertical-align:baseline\">\n         <div id=d80 style=\"width:100px;height:80%\"></div>\n         </td></table></div>",
        "var d50 = document.getElementById('d50');\n         var d80 = document.getElementById('d80');\n         globalThis.__result = [d50.offsetParent.tagName, d50.offsetTop, d50.offsetHeight, d80.offsetTop, d80.offsetHeight].join('|');",
    );
    assert_eq!(r, "TD|30|50|0|80");
}
