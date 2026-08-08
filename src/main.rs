//! Interpreter loop demo: load an ELF, then fetch → decode → execute with a trace.
//! Stops on an escaped exception (e.g. an unimplemented instruction), on a
//! riscv-tests result reported through tohost, or at the step limit.

use riscv_emulator::{cpu::Cpu, harness, inst, loader};

/// Safety cap so a legal infinite loop cannot hang the demo.
const STEP_LIMIT: u64 = 10_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: riscv-emulator <ELF file>")?;
    let bytes = std::fs::read(&path)?;
    let elf = loader::load(&bytes)?;

    let mut cpu = Cpu::new();
    cpu.load_elf(&elf)?;
    println!("entry: {:#x}", cpu.pc);

    // riscv-tests ELFs publish the address of their result variable as the
    // `tohost` symbol. A generic ELF does not have one; then we just trace.
    let tohost = elf.symbol("tohost");
    match tohost {
        Some(addr) => println!("tohost: {addr:#x}\n"),
        None => println!("tohost: (no symbol)\n"),
    }

    let mut executed = 0u64;
    while executed < STEP_LIMIT {
        let pc = cpu.pc;
        // fetch/decode by hand (instead of cpu.step()) so the trace can show the
        // instruction before executing it.
        let stop = match cpu.fetch().and_then(|raw| {
            inst::decode(raw).map(|inst| (raw, inst))
        }) {
            Ok((raw, inst)) => {
                println!("{pc:#010x}: {raw:08x}  {inst}");
                cpu.execute(inst).err()
            }
            Err(e) => Some(e),
        };
        if let Some(e) = stop {
            println!("\nstopped at {pc:#010x}: {e}");
            break;
        }
        executed += 1;
        if let Some(outcome) = tohost.and_then(|addr| harness::check_tohost(&cpu, addr)) {
            println!("\n{outcome}");
            break;
        }
    }

    println!("executed {executed} instruction(s)");
    Ok(())
}
