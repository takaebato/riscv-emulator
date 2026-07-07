//! Integration tests for the ELF loader, using real prebuilt riscv-tests binaries.
//! Precondition: vendor/riscv-tests/isa has been built (see docs/setup.md).

use riscv_emulator::loader;

fn read_test_elf(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/vendor/riscv-tests/isa/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read {path} ({e}); build riscv-tests first (see docs/setup.md)"))
}

#[test]
fn loads_rv64ui_p_add() {
    let elf = loader::load(&read_test_elf("rv64ui-p-add")).unwrap();

    // The entry point is the start of DRAM (dictated by the env/p linker script).
    assert_eq!(elf.entry, 0x8000_0000);

    // Only PT_LOAD segments are returned (the RISCV_ATTRIBUTES segment is not).
    assert_eq!(elf.segments.len(), 2);

    // First segment = .text.init (code)
    let text = &elf.segments[0];
    assert_eq!(text.addr, 0x8000_0000);
    assert!(!text.data.is_empty());
    assert!(text.mem_size >= text.data.len() as u64);

    // The tohost symbol used for pass/fail reporting (address is stable: linker script).
    assert_eq!(elf.symbol("tohost"), Some(0x8000_1000));
    assert_eq!(elf.symbol("no_such_symbol"), None);
}

#[test]
fn rejects_non_elf_bytes() {
    assert!(matches!(
        loader::load(b"not an elf"),
        Err(loader::LoadError::Parse(_))
    ));
}

#[test]
fn rejects_non_riscv_elf() {
    // Forge a minimal ELF64 header (machine = x86-64) and confirm it is rejected.
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    bytes[4] = 2; // ELFCLASS64
    bytes[5] = 1; // little endian
    bytes[6] = 1; // EV_CURRENT
    bytes[16] = 2; // ET_EXEC
    bytes[18] = 62; // EM_X86_64
    bytes[20] = 1; // e_version
    bytes[52] = 64; // e_ehsize
    match loader::load(&bytes) {
        Err(loader::LoadError::UnsupportedArch(_)) | Err(loader::LoadError::Parse(_)) => {}
        other => panic!("non-RISC-V ELF was not rejected: {other:?}"),
    }
}
