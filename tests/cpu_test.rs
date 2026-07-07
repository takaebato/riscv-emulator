//! Cpu/Bus integration test: loading a real ELF and the first fetch.

use riscv_emulator::{cpu::Cpu, loader};

fn read_test_elf(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/vendor/riscv-tests/isa/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read {path} ({e}); build riscv-tests first (see docs/setup.md)"))
}

#[test]
fn power_on_and_fetch_first_instruction() {
    let elf = loader::load(&read_test_elf("rv64ui-p-add")).unwrap();
    let mut cpu = Cpu::new();
    cpu.load_elf(&elf).unwrap();

    // Right after power-on: pc = entry point
    assert_eq!(cpu.pc, 0x8000_0000);

    // The fetched instruction word matches the first 4 bytes of the ELF code segment
    // (i.e. the copy into the Bus is faithful).
    let expected = u32::from_le_bytes(elf.segments[0].data[0..4].try_into().unwrap());
    let inst = cpu.fetch().unwrap();
    assert_eq!(inst, expected);
    assert_ne!(inst, 0, "first instruction must not be 0 (invalid)");
}
