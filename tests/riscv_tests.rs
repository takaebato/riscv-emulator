//! Run official riscv-tests ELFs through the tohost harness.
//!
//! Only tests whose instruction mix is fully implemented are listed; the list
//! grows as decode fills out, until it can become a directory sweep.
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

/// rv64ui-p tests expected to pass with the current instruction set.
/// Still missing (and excluded here): slti, sltiu, xori, ori, andi, jalr,
/// and fence_i (Zifencei).
const PASSING: &[&str] = &[
    "rv64ui-p-add",
    "rv64ui-p-addi",
    "rv64ui-p-addiw",
    "rv64ui-p-addw",
    "rv64ui-p-and",
    "rv64ui-p-auipc",
    "rv64ui-p-beq",
    "rv64ui-p-bge",
    "rv64ui-p-bgeu",
    "rv64ui-p-blt",
    "rv64ui-p-bltu",
    "rv64ui-p-bne",
    "rv64ui-p-jal",
    "rv64ui-p-lb",
    "rv64ui-p-lbu",
    "rv64ui-p-ld",
    "rv64ui-p-ld_st",
    "rv64ui-p-lh",
    "rv64ui-p-lhu",
    "rv64ui-p-lui",
    "rv64ui-p-lw",
    "rv64ui-p-lwu",
    "rv64ui-p-ma_data",
    "rv64ui-p-or",
    "rv64ui-p-sb",
    "rv64ui-p-sd",
    "rv64ui-p-sh",
    "rv64ui-p-simple",
    "rv64ui-p-sll",
    "rv64ui-p-slli",
    "rv64ui-p-slliw",
    "rv64ui-p-sllw",
    "rv64ui-p-slt",
    "rv64ui-p-sltu",
    "rv64ui-p-sra",
    "rv64ui-p-srai",
    "rv64ui-p-sraiw",
    "rv64ui-p-sraw",
    "rv64ui-p-srl",
    "rv64ui-p-srli",
    "rv64ui-p-srliw",
    "rv64ui-p-srlw",
    "rv64ui-p-st_ld",
    "rv64ui-p-sub",
    "rv64ui-p-subw",
    "rv64ui-p-sw",
    "rv64ui-p-xor",
];

#[test]
fn implemented_rv64ui_tests_pass() {
    let failures: Vec<_> = PASSING
        .iter()
        .map(|name| (name, run_isa_test(name)))
        .filter(|(_, outcome)| *outcome != Outcome::Pass)
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}
