//! `bowser-wpt-run`: enumerate and (eventually) run WPT tests.
//!
//! Today the only driver is the stub, so `--list` works and a real run
//! reports every test as a driver failure. That is expected: the point right
//! now is manifest handling, filtering, and report plumbing.
//!
//! ```sh
//! bowser-wpt-run --wpt-root ../wpt --list --kind testharness --include dom/
//! bowser-wpt-run --manifest ../wpt/MANIFEST.json --include css/
//! ```

use std::path::PathBuf;
use std::time::Duration;

use bowser_wpt::driver::StubDriver;
use bowser_wpt::manifest::{Manifest, TestKind};
use bowser_wpt::runner::{RunConfig, Runner};

fn usage() -> ! {
    eprintln!(
        "usage: bowser-wpt-run (--wpt-root DIR | --manifest FILE) [options]\n\
         options:\n  \
         --list               print \"<id> <kind>\" for every selected test and exit\n  \
         --include SUB        only run tests whose id contains SUB (repeatable)\n  \
         --exclude SUB        skip tests whose id contains SUB (repeatable)\n  \
         --kind KIND          only run tests of KIND (repeatable;\n  \
         testharness|reftest|print-reftest|manual|crashtest)\n  \
         --timeout-secs N     per-test timeout for testharness tests (default 60)"
    );
    std::process::exit(2);
}

fn main() {
    let mut wpt_root: Option<PathBuf> = None;
    let mut manifest_file: Option<PathBuf> = None;
    let mut list = false;
    let mut include = Vec::new();
    let mut exclude = Vec::new();
    let mut kinds = Vec::new();
    let mut timeout_secs: u64 = 60;

    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wpt-root" => wpt_root = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--manifest" => {
                manifest_file = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())));
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
            _ => {
                eprintln!("unknown argument: {arg}");
                usage();
            }
        }
    }

    let manifest = match (wpt_root, manifest_file) {
        (Some(root), None) => Manifest::walk(&root),
        (None, Some(file)) => {
            Manifest::from_json(&std::fs::read_to_string(&file).unwrap_or_else(|e| {
                eprintln!("cannot read {}: {e}", file.display());
                std::process::exit(1);
            }))
        }
        _ => {
            eprintln!("need exactly one of --wpt-root or --manifest");
            usage();
        }
    };
    let manifest = manifest.unwrap_or_else(|e| {
        eprintln!("manifest error: {e}");
        std::process::exit(1);
    });

    let config = RunConfig {
        include,
        exclude,
        kinds,
        timeout: Duration::from_secs(timeout_secs),
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

    let mut runner = Runner::new(StubDriver, config);
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
            bowser_wpt::runner::TestOutcome::FilesystemFailure(e) => format!("FS-FAIL ({e})"),
        };
        println!("{} [{}] {status}", r.id, r.kind);
    }
    println!();
    println!("{}", report.summary_text());
    if manifest.skipped > 0 {
        println!(
            "note: {} manifest entries had an unknown shape",
            manifest.skipped
        );
    }
}
