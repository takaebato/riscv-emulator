//! Guest programs (statically linked Linux ELFs from guest/*.s) run to
//! completion under the Linux user-mode emulation and their output reaches
//! the host. hello is the milestone 1 boss fight; fizzbuzz is the
//! M-extension side quest (REMU branching, DIVU/REMU decimal conversion).

use riscv_emulator::{
    cpu::Cpu,
    linux::{self, Outcome},
    loader,
};

fn run_guest(name: &str) -> String {
    let path = format!("guest/{name}");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("cannot read {path}: {e} (build it first; see {path}.s)")
    });
    let elf = loader::load(&bytes).unwrap();
    let mut cpu = Cpu::new();
    cpu.load_elf(&elf).unwrap();

    let mut out = Vec::new();
    assert_eq!(linux::run(&mut cpu, &mut out, 10_000), Outcome::Exited(0));
    String::from_utf8(out).unwrap()
}

#[test]
fn hello_world_runs() {
    assert_eq!(run_guest("hello"), "Hello, world!\n");
}

#[test]
fn fizzbuzz_runs() {
    // The same rules in Rust serve as the oracle for the guest's output.
    let expected: String = (1..=30)
        .map(|i| match (i % 3, i % 5) {
            (0, 0) => "FizzBuzz\n".to_string(),
            (0, _) => "Fizz\n".to_string(),
            (_, 0) => "Buzz\n".to_string(),
            _ => format!("{i}\n"),
        })
        .collect();
    assert_eq!(run_guest("fizzbuzz"), expected);
}