//! `bowser-wpt`: run WPT tests against the real headless `bowser` binary.
//!
//! Unlike `bowser-wpt-run` (stub driver), this wires
//! [`bowser_app::wpt_driver::BowserDriver`] into the
//! [`bowser_wpt::runner::Runner`], so testharness tests actually execute
//! through the headless pipeline (`bowser --check` + the injected
//! completion-callback reporter), reftests render and compare pixels, and
//! crashtests load pages.
//!
//! Build with the `wpt-driver` feature (off by default so the shipping
//! `bowser` binary carries no test infra):
//!
//! ```sh
//! cargo build --release -p bowser_app --features wpt-driver --bin bowser-wpt
//! ./target/release/bowser-wpt --wpt-root ../wpt --kind testharness \
//!   --include dom/nodes --timeout-secs 30
//! ```
//!
//! `--wpt-root` doubles as the fixture-server document root: the driver
//! serves test pages (and their subresources) from the checkout.
//!
//! Machine-readable output: `--output-json FILE` writes the run as
//! wptreport JSON (see [`bowser_wpt::results::export_wptreport`]).
//! Expected-failure metadata: `--metadata-dir DIR` (default
//! `<wpt-root>/metadata` when it exists) loads every `*.ini` beneath it
//! into [`bowser_wpt::results::ExpectedMetadata`]; tests whose outcome
//! matches their expectation are reported as expected-fail rather than
//! failures.

use std::path::{Path, PathBuf};
use std::time::Duration;

use bowser_app::wpt_driver::BowserDriver;
use bowser_wpt::manifest::{Manifest, TestKind};
use bowser_wpt::results::{ExpectedMetadata, export_wptreport};
use bowser_wpt::runner::{RunConfig, Runner};

fn usage() -> ! {
    eprintln!(
        "usage: bowser-wpt --wpt-root DIR [options]\n\
         options:\n  \
         --manifest FILE      read test kinds from FILE instead of walking\n  \
         --list               print \"<id> <kind>\" for every selected test and exit\n  \
         --include SUB        only run tests whose id contains SUB (repeatable)\n  \
         --exclude SUB        skip tests whose id contains SUB (repeatable)\n  \
         --kind KIND          only run tests of KIND (repeatable;\n  \
         testharness|reftest|print-reftest|manual|crashtest)\n  \
         --timeout-secs N     per-test timeout (default 60)\n  \
         --output-json FILE   write the run as wptreport JSON to FILE\n  \
         --metadata-dir DIR   expected-failure metadata dir\n  \
         (default: <wpt-root>/metadata when it exists)\n  \
         --exe PATH           bowser binary (default: `bowser` next to this binary)"
    );
    std::process::exit(2);
}

/// Locate the `bowser` binary: `--exe` override, else a sibling of this
/// binary (cargo puts both bins in the same target directory).
fn bowser_exe(override_path: Option<PathBuf>) -> PathBuf {
    if let Some(p) = override_path {
        return p;
    }
    let mut me = std::env::current_exe().unwrap_or_else(|_| usage());
    me.pop();
    me.join("bowser")
}

/// Load every `*.ini` under `dir` (recursively, sorted for
/// determinism) into one [`ExpectedMetadata`]. A missing dir is not an
/// error — it just means no expectations. Later files override earlier
/// ones on duplicate test ids.
fn load_metadata(dir: Option<&Path>) -> ExpectedMetadata {
    let Some(dir) = dir else {
        return ExpectedMetadata::default();
    };
    let mut files = Vec::new();
    collect_ini_files(dir, &mut files);
    files.sort();
    let mut merged = ExpectedMetadata::default();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| {
            eprintln!("cannot read {}: {e}", f.display());
            std::process::exit(1);
        });
        let one = ExpectedMetadata::parse(&text).unwrap_or_else(|e| {
            eprintln!("bad expected-failure metadata {}: {e}", f.display());
            std::process::exit(1);
        });
        merged.merge(one);
    }
    if !merged.is_empty() {
        eprintln!(
            "loaded {} expected-failure {} from {}",
            merged.len(),
            if merged.len() == 1 {
                "expectation"
            } else {
                "expectations"
            },
            dir.display()
        );
    }
    merged
}

fn collect_ini_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_ini_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "ini") {
            out.push(path);
        }
    }
}

fn main() {
    let mut wpt_root: Option<PathBuf> = None;
    let mut manifest_file: Option<PathBuf> = None;
    let mut list = false;
    let mut include = Vec::new();
    let mut exclude = Vec::new();
    let mut kinds = Vec::new();
    let mut timeout_secs: u64 = 60;
    let mut output_json: Option<PathBuf> = None;
    let mut metadata_dir: Option<PathBuf> = None;
    let mut exe_override: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wpt-root" => wpt_root = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--manifest" => {
                manifest_file = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--list" => list = true,
            "--include" => include.push(args.next().unwrap_or_else(|| usage())),
            "--exclude" => exclude.push(args.next().unwrap_or_else(|| usage())),
            "--kind" => {
                let k = args.next().unwrap_or_else(|| usage());
                kinds.push(match k.as_str() {
                    "testharness" => TestKind::Testharness,
                    "reftest" => TestKind::Reftest,
                    "print-reftest" => TestKind::PrintReftest,
                    "manual" => TestKind::Manual,
                    "crashtest" => TestKind::Crashtest,
                    _ => {
                        eprintln!("unknown kind: {k}");
                        usage();
                    }
                });
            }
            "--timeout-secs" => {
                timeout_secs = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage());
            }
            "--output-json" => {
                output_json = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--metadata-dir" => {
                metadata_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--exe" => exe_override = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            _ => {
                eprintln!("unknown argument: {arg}");
                usage();
            }
        }
    }

    let wpt_root = wpt_root.unwrap_or_else(|| {
        eprintln!("need --wpt-root DIR");
        usage();
    });

    let manifest = match manifest_file.or_else(|| {
        let candidate = wpt_root.join("MANIFEST.json");
        candidate.is_file().then_some(candidate)
    }) {
        Some(file) => Manifest::from_json(&std::fs::read_to_string(&file).unwrap_or_else(|e| {
            eprintln!("cannot read {}: {e}", file.display());
            std::process::exit(1);
        })),
        None => Manifest::walk(&wpt_root),
    };
    let manifest = manifest.unwrap_or_else(|e| {
        eprintln!("manifest error: {e}");
        std::process::exit(1);
    });

    let metadata_dir = metadata_dir.or_else(|| {
        let candidate = wpt_root.join("metadata");
        candidate.is_dir().then_some(candidate)
    });
    let metadata = load_metadata(metadata_dir.as_deref());

    let mut config = RunConfig {
        include,
        exclude,
        kinds,
        timeout: Duration::from_secs(timeout_secs),
        metadata,
        ..RunConfig::default()
    };

    if list {
        for item in manifest.items() {
            if config.wants(item) {
                println!("{} {}", item.id, item.kind);
            }
        }
        return;
    }

    let exe = bowser_exe(exe_override);
    if !exe.is_file() {
        eprintln!(
            "bowser binary not found at {} (build it with \
             `cargo build -p bowser_app --features wpt-driver`, or pass --exe)",
            exe.display()
        );
        std::process::exit(1);
    }
    let mut driver = BowserDriver::new(exe, &wpt_root).unwrap_or_else(|e| {
        eprintln!("fixture server failed: {e}");
        std::process::exit(1);
    });
    driver.check_timeout = Duration::from_secs(timeout_secs);
    // Join manifest-relative URLs against the fixture server base so the
    // runner hands the driver absolute URLs (see bowser_wpt runner).
    config.server_base_url = driver.run_config().server_base_url;

    let metadata = config.metadata.clone();
    let mut runner = Runner::new(driver, config);
    let report = runner.run(&manifest);
    for r in &report.reports {
        let status = match &r.outcome {
            bowser_wpt::runner::TestOutcome::Harness(h) => {
                format!(
                    "harness {} ({}/{} pass)",
                    h.status.as_str(),
                    h.passed(),
                    h.subtests.len()
                )
            }
            bowser_wpt::runner::TestOutcome::Reftest { passed, .. } => {
                format!("reftest {}", if *passed { "PASS" } else { "FAIL" })
            }
            bowser_wpt::runner::TestOutcome::Crashtest { passed } => {
                format!("crashtest {}", if *passed { "PASS" } else { "FAIL" })
            }
            bowser_wpt::runner::TestOutcome::Skipped { reason } => format!("SKIP ({reason})"),
            bowser_wpt::runner::TestOutcome::DriverFailed(e) => format!("DRIVER-FAIL ({e})"),
            bowser_wpt::runner::TestOutcome::DriverTimeout { after, .. } => {
                format!("WATCHDOG-TIMEOUT (no harness report after {after:?})")
            }
            bowser_wpt::runner::TestOutcome::FilesystemFailure(e) => format!("FS-FAIL ({e})"),
        };
        let expected_fail = if r.expected_fail {
            " (expected-fail)"
        } else {
            ""
        };
        println!("{} [{}] {status}{expected_fail}", r.id, r.kind);
    }
    println!();
    println!("{}", report.summary_text());
    if manifest.skipped > 0 {
        println!(
            "note: {} manifest entries had an unknown shape",
            manifest.skipped
        );
    }
    if let Some(path) = &output_json {
        let json = export_wptreport(&report, &metadata);
        std::fs::write(path, json).unwrap_or_else(|e| {
            eprintln!("cannot write {}: {e}", path.display());
            std::process::exit(1);
        });
        eprintln!("wrote wptreport JSON to {}", path.display());
    }
}
