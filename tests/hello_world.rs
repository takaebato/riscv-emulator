//! The milestone 1 boss fight: a statically linked Linux ELF (guest/hello.s,
//! raw write/exit syscalls, RV64I only) runs to completion under the Linux
//! user-mode emulation and its output reaches the host.

use riscv_emulator::{
    cpu::Cpu,
    linux::{self, Outcome},
    loader,
};

#[test]
fn hello_world_runs() {
    let bytes = std::fs::read("guest/hello").unwrap_or_else(|e| {
        panic!("cannot read guest/hello: {e} (build it first; see guest/hello.s)")
    });
    let elf = loader::load(&bytes).unwrap();
    let mut cpu = Cpu::new();
    cpu.load_elf(&elf).unwrap();

    let mut out = Vec::new();
    assert_eq!(linux::run(&mut cpu, &mut out, 10_000), Outcome::Exited(0));
    assert_eq!(String::from_utf8(out).unwrap(), "Hello, world!\n");
}