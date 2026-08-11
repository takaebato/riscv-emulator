//! Run the official riscv-tests suites through the tohost harness.
//!
//! RV64I (plus Zifencei's FENCE.I) is complete, so the whole rv64ui-p
//! directory is swept and every test in it must pass. Further suites
//! (rv64um-p-*, ...) join as their extensions are implemented.
//! Requires the ELFs to be built in vendor/riscv-tests (see docs/setup.md).

use riscv_emulator::{
    cpu::Cpu,
    harness::{self, Outcome},
    loader,
};

/// Far above any rv64ui test's real length (rv64ui-p-add finishes in ~500).
const STEP_LIMIT: u64 = 100_000;

fn run_isa_test(name: &str) -> Outcome {
    let path = format!("vendor/riscv-tests/isa/{name}");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e} (build riscv-tests first)"));
    let elf = loader::load(&bytes).unwrap();
    let mut cpu = Cpu::new();
    cpu.load_elf(&elf).unwrap();
    let tohost = elf.symbol("tohost").expect("riscv-tests ELFs define tohost");
    harness::run(&mut cpu, tohost, STEP_LIMIT)
}

/// Every ELF in the directory whose name starts with `prefix` (skipping the
/// .dump disassembly listings), sorted for stable output.
fn suite(prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir("vendor/riscv-tests/isa")
        .expect("cannot read vendor/riscv-tests/isa (build riscv-tests first)")
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            (name.starts_with(prefix) && !name.ends_with(".dump")).then_some(name)
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no {prefix}* tests found");
    names
}

#[test]
fn the_whole_rv64ui_p_suite_passes() {
    let failures: Vec<_> = suite("rv64ui-p-")
        .into_iter()
        .map(|name| {
            let outcome = run_isa_test(&name);
            (name, outcome)
        })
        .filter(|(_, outcome)| *outcome != Outcome::Pass)
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}
