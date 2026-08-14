//! Diff-test driver: run a riscv-tests ELF on Spike (the reference
//! simulator) and on our emulator in lockstep, and report the first
//! instruction where the architectural state disagrees.
//!
//!     cargo run --bin difftest vendor/riscv-tests/isa/rv64ui-p-add

use riscv_emulator::bus::DRAM_BASE;
use riscv_emulator::difftest::{self, DiffResult, Divergence, Event};
use riscv_emulator::{cpu::Cpu, inst, loader};

/// Events shown before the divergence point for context.
const CONTEXT: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: difftest <riscv-tests ELF>")?;

    // The reference run. Spike logs to stderr; its own boot ROM at 0x1000
    // runs first, so comparison starts at the first commit in DRAM.
    // --misaligned: handling misaligned accesses in hardware is
    // implementation-defined, and our Bus does; this tells Spike to
    // simulate the same choice instead of trapping (rv64ui-p-ma_data
    // exercises exactly this difference).
    let output = std::process::Command::new("spike")
        .args(["--misaligned", "-l", "--log-commits", &path])
        .output()
        .map_err(|e| format!("cannot run spike: {e} (install per docs/setup.md)"))?;
    let log = String::from_utf8_lossy(&output.stderr);
    let events = difftest::parse_log(&log);
    let start = events
        .iter()
        .position(|ev| ev.pc() == DRAM_BASE)
        .ok_or("spike's log never reaches DRAM_BASE; is this a bare-metal test ELF?")?;
    let events = &events[start..];

    // Our run, over the same ELF.
    let elf = loader::load(&std::fs::read(&path)?)?;
    let mut cpu = Cpu::new();
    cpu.load_elf(&elf)?;

    match difftest::run(&mut cpu, events) {
        DiffResult::Lockstep { instructions } => {
            println!("lockstep OK: {instructions} instructions match spike");
        }
        DiffResult::LockstepUntilEcall { instructions } => {
            println!(
                "lockstep OK: {instructions} instructions match spike \
                 (compared up to the final ecall; the trap path waits for \
                 phase 3 privilege modes)"
            );
        }
        DiffResult::Diverged { index, kind } => {
            println!("DIVERGED at event {index} (pc {:#x})\n", events[index].pc());
            println!("last events from spike's log:");
            for event in &events[index.saturating_sub(CONTEXT)..=index] {
                print_event(event);
            }
            println!();
            match kind {
                Divergence::Pc { spike, ours } => {
                    println!("pc mismatch: spike is at {spike:#x}, we are at {ours:#x}");
                }
                Divergence::Reg { rd, spike, ours } => {
                    println!("x{rd} mismatch: spike has {spike:#018x}, we have {ours:#018x}");
                }
                Divergence::OurException(e) => {
                    println!("spike retired this instruction, but we raised: {e}");
                }
            }
            std::process::exit(1);
        }
    }
    Ok(())
}

/// One log event, disassembled with our own Display (which doubles as a
/// cross-check that our decoder can read everything Spike retired).
fn print_event(event: &Event) {
    match event {
        Event::Commit { pc, raw, write } => {
            let disasm = match inst::decode(*raw) {
                Ok(inst) => inst.to_string(),
                Err(_) => "(not decodable by us)".to_string(),
            };
            print!("  {pc:#010x}: {raw:08x}  {disasm:24}");
            if let Some((rd, value)) = write {
                print!("  x{rd} <- {value:#x}");
            }
            println!();
        }
        Event::Trap { epc, cause, tval } => {
            print!("  {epc:#010x}: --------  [{cause}");
            if let Some(tval) = tval {
                print!(", tval {tval:#x}");
            }
            println!("]");
        }
    }
}
