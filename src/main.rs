//! Emulator entry point. The ELF picks its own environment: a `tohost`
//! symbol marks a riscv-tests binary (traced run, pass/fail via tohost);
//! anything else runs under Linux user-mode emulation (untraced, so the
//! guest owns stdout).

use riscv_emulator::{cpu::Cpu, harness, linux, loader};

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

    // riscv-tests ELFs publish the address of their result variable as the
    // `tohost` symbol. An ELF without one is a Linux program: hand it to the
    // syscall environment instead.
    let Some(tohost) = elf.symbol("tohost") else {
        let outcome = linux::run(&mut cpu, &mut std::io::stdout(), STEP_LIMIT);
        println!("{outcome}");
        return Ok(());
    };
    println!("entry: {:#x}", cpu.pc);
    println!("tohost: {tohost:#x}\n");

    let mut executed = 0u64;
    while executed < STEP_LIMIT {
        cpu.tick();
        if cpu.take_pending_interrupt() {
            println!("{:#010x}: --------  [interrupt: mcause {:#x}]", cpu.csrs[0x341], cpu.csrs[0x342]);
        }
        let pc = cpu.pc;
        // fetch/decode by hand (instead of cpu.step()) so the trace can show the
        // instruction before executing it.
        let stop = match cpu.fetch_decode() {
            Ok((inst, len, raw)) => {
                println!("{pc:#010x}: {raw:08x}  {inst}");
                cpu.execute(inst, len).err()
            }
            Err(e) => Some(e),
        };
        if let Some(e) = stop {
            // Architectural behavior: exceptions trap into the guest's
            // handler (mtvec) and the run continues.
            println!("{pc:#010x}: --------  [trap: {e}]");
            cpu.trap(&e);
        }
        executed += 1;
        if let Some(outcome) = harness::check_tohost(&cpu, tohost) {
            println!("\n{outcome}");
            break;
        }
    }

    println!("executed {executed} instruction(s)");
    Ok(())
}
