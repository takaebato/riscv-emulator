//! Spike diff testing: replay Spike's commit log on our Cpu in lockstep and
//! find the first instruction where the two implementations disagree.
//!
//! `spike -l --log-commits` writes one commit line per retired instruction
//! (pc, instruction word, and the register it wrote). We parse that into
//! [`Event`]s, then step our Cpu alongside: before each event the pc must
//! match, after it the written register must hold the same value. Where
//! Spike takes a trap we either trap too or the next pc check catches us.
//!
//! Two Spike behaviors shape the loop:
//! - A trapped instruction does not commit: the log shows an `exception`
//!   line instead, and execution continues at mtvec. The riscv-tests
//!   startup relies on this (it probes optional CSRs like mnstatus with
//!   mtvec aimed at the next instruction, so trap and no-trap converge).
//! - Spike drops to U-mode for the rv64ui tests, so the final ECALL raises
//!   cause 8 (user ecall) where our M-mode-only Cpu raises 11. Both land in
//!   the same handler, but the handler's `csrr t5, mcause` would diverge.
//!   Until phase 3 brings privilege modes, comparison therefore stops (and
//!   counts as success) at the first ecall trap: every test case body has
//!   run by then.

use crate::cpu::Cpu;
use crate::exception::Exception;
use crate::inst::{Inst, decode};

/// One line of Spike's log that matters for the comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A retired instruction: its pc, raw word, and x-register write (if
    /// any). CSR/memory effects in the log are ignored: memory follows from
    /// matching stores, and CSRs are compared indirectly when read back.
    Commit {
        pc: u64,
        raw: u32,
        write: Option<(usize, u64)>,
    },
    /// A taken trap: the faulting pc and Spike's cause name (plus mtval).
    Trap {
        epc: u64,
        cause: String,
        tval: Option<u64>,
    },
}

impl Event {
    /// The pc our Cpu must be at when this event begins.
    pub fn pc(&self) -> u64 {
        match self {
            Event::Commit { pc, .. } => *pc,
            Event::Trap { epc, .. } => *epc,
        }
    }
}

/// Parse Spike's stderr. Unknown lines (disassembly from -l, `>>>>` symbol
/// annotations, ...) are skipped, so the parser accepts the combined
/// `-l --log-commits` output.
pub fn parse_log(log: &str) -> Vec<Event> {
    let mut events = Vec::new();
    for line in log.lines() {
        let tok: Vec<&str> = line.split_whitespace().collect();
        match tok.as_slice() {
            // `core   0: exception trap_illegal_instruction, epc 0x...`
            ["core", _, "exception", cause, "epc", epc] => {
                if let Some(epc) = hex(epc) {
                    events.push(Event::Trap {
                        epc,
                        cause: cause.trim_end_matches(',').to_string(),
                        tval: None,
                    });
                }
            }
            // `core   0:           tval 0x...` — belongs to the trap above.
            ["core", _, "tval", val] => {
                if let (Some(v), Some(Event::Trap { tval, .. })) = (hex(val), events.last_mut()) {
                    *tval = Some(v);
                }
            }
            // `core   0: 3 0x<pc> (0x<raw>) x5  0x<val> ...` — the leading
            // single digit is the privilege level, which distinguishes
            // commit lines from -l disassembly lines (pc comes third there).
            ["core", _, prv, pc, raw, rest @ ..] if prv.len() == 1 => {
                let (Some(pc), Some(raw)) = (hex(pc), hex(raw)) else {
                    continue;
                };
                // Scan the effects for an x-register write: `x<n>` followed
                // by the value. `mem`/`c<csr>`/float entries are skipped.
                let mut write = None;
                for pair in rest.windows(2) {
                    if let Some(rd) = pair[0].strip_prefix('x').and_then(|n| n.parse().ok()) {
                        write = hex(pair[1]).map(|v| (rd, v));
                    }
                }
                events.push(Event::Commit { pc, raw: raw as u32, write });
            }
            _ => {}
        }
    }
    events
}

/// `0x…` → value (also accepts a `(0x…)`-wrapped instruction word).
fn hex(s: &str) -> Option<u64> {
    let s = s.trim_start_matches('(').trim_end_matches(')');
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}

/// Where and how the two implementations disagreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Divergence {
    /// We are about to execute a different instruction than Spike retired.
    Pc { spike: u64, ours: u64 },
    /// Same instruction, different result in the written register.
    Reg { rd: usize, spike: u64, ours: u64 },
    /// Spike retired the instruction but our step raised an exception.
    OurException(Exception),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffResult {
    /// Spike's whole log matched.
    Lockstep { instructions: u64 },
    /// Everything up to the test's final ecall matched (see module docs:
    /// the trap path diverges on mcause until phase 3 privilege modes).
    LockstepUntilEcall { instructions: u64 },
    /// First disagreement, as an index into the event slice.
    Diverged { index: usize, kind: Divergence },
}

/// Step our Cpu through Spike's events and compare.
pub fn run(cpu: &mut Cpu, events: &[Event]) -> DiffResult {
    let mut instructions = 0;
    for (index, event) in events.iter().enumerate() {
        if cpu.pc != event.pc() {
            let kind = Divergence::Pc { spike: event.pc(), ours: cpu.pc };
            return DiffResult::Diverged { index, kind };
        }
        match event {
            Event::Commit { raw, write, .. } => {
                // A store-conditional may fail spuriously: the spec lets
                // the environment drop a reservation at any time, and
                // Spike does so on its internal scheduling boundaries.
                // When the log shows a failed SC, drop our reservation
                // too, so both implementations take the same
                // architecturally-legal path.
                if let (Ok(Inst::ScW { .. } | Inst::ScD { .. }), Some((_, result))) =
                    (decode(*raw), write)
                {
                    if *result != 0 {
                        cpu.reservation = None;
                    }
                }
                if let Err(e) = cpu.step() {
                    let kind = Divergence::OurException(e);
                    return DiffResult::Diverged { index, kind };
                }
                instructions += 1;
                if let Some((rd, spike)) = *write {
                    let ours = cpu.regs[rd];
                    if ours != spike {
                        let kind = Divergence::Reg { rd, spike, ours };
                        return DiffResult::Diverged { index, kind };
                    }
                }
            }
            Event::Trap { cause, .. } => {
                if cause.contains("ecall") {
                    return DiffResult::LockstepUntilEcall { instructions };
                }
                match cpu.step() {
                    // Both trapped: follow Spike into the handler.
                    Err(e) => cpu.trap(&e),
                    // We executed what Spike refused (e.g. a CSR we model
                    // too permissively). The startup's mtvec-at-next-label
                    // pattern makes both paths converge; if they do not,
                    // the next event's pc check reports it.
                    Ok(()) => {}
                }
            }
        }
    }
    DiffResult::Lockstep { instructions }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::DRAM_BASE;

    /// Verbatim lines from `spike -l --log-commits rv64ui-p-add` (plus a
    /// store and a no-write jump), exercising every line shape at once.
    const SAMPLE: &str = "\
core   0: 0x0000000000001000 (0x00000297) auipc   t0, 0x0
core   0: 3 0x0000000000001000 (0x00000297) x5  0x0000000000001000
core   0: 3 0x000000000000100c (0x0182b283) x5  0x0000000080000000 mem 0x0000000000001018
core   0: 3 0x0000000080000010 (0x00028067)
core   0: 3 0x0000000080000040 (0xfc3f2223) mem 0x0000000080001000 0x00000001
core   0: 3 0x00000000800000dc (0x30529073) c773_mtvec 0x00000000800000e4
core   0: exception trap_illegal_instruction, epc 0x00000000800000e0
core   0:           tval 0x0000000074445073
core   0: >>>>  write_tohost
";

    #[test]
    fn parses_every_line_shape() {
        let events = parse_log(SAMPLE);
        assert_eq!(
            events,
            vec![
                // -l disassembly line skipped; commit with a write kept
                Event::Commit { pc: 0x1000, raw: 0x00000297, write: Some((5, 0x1000)) },
                // load: x-write kept, mem address ignored
                Event::Commit {
                    pc: 0x100c,
                    raw: 0x0182b283,
                    write: Some((5, DRAM_BASE)),
                },
                // jump: no writeback
                Event::Commit { pc: 0x8000_0010, raw: 0x00028067, write: None },
                // store: mem-only effect, no x-write
                Event::Commit { pc: 0x8000_0040, raw: 0xfc3f2223, write: None },
                // csr write: not an x-register, ignored
                Event::Commit { pc: 0x8000_00dc, raw: 0x30529073, write: None },
                // exception with its tval; the >>>> annotation is skipped
                Event::Trap {
                    epc: 0x8000_00e0,
                    cause: "trap_illegal_instruction".to_string(),
                    tval: Some(0x74445073),
                },
            ]
        );
    }

    fn cpu_with_program(words: &[u32]) -> Cpu {
        let mut cpu = Cpu::new();
        for (i, w) in words.iter().enumerate() {
            cpu.bus.store32(DRAM_BASE + 4 * i as u64, *w).unwrap();
        }
        cpu.pc = DRAM_BASE;
        cpu
    }

    #[test]
    fn lockstep_when_we_agree() {
        // addi x1, x0, 5 then addi x1, x1, 2 — and a log that agrees.
        let mut cpu = cpu_with_program(&[0x00500093, 0x00208093]);
        let events = vec![
            Event::Commit { pc: DRAM_BASE, raw: 0x00500093, write: Some((1, 5)) },
            Event::Commit { pc: DRAM_BASE + 4, raw: 0x00208093, write: Some((1, 7)) },
        ];
        assert_eq!(run(&mut cpu, &events), DiffResult::Lockstep { instructions: 2 });
    }

    #[test]
    fn reports_a_register_divergence() {
        let mut cpu = cpu_with_program(&[0x00500093]);
        // Claim Spike computed 6 where we compute 5.
        let events = vec![Event::Commit {
            pc: DRAM_BASE,
            raw: 0x00500093,
            write: Some((1, 6)),
        }];
        assert_eq!(
            run(&mut cpu, &events),
            DiffResult::Diverged {
                index: 0,
                kind: Divergence::Reg { rd: 1, spike: 6, ours: 5 },
            }
        );
    }

    #[test]
    fn reports_a_pc_divergence() {
        // Our jal lands at +8; the log claims Spike retired +4 next.
        let mut cpu = cpu_with_program(&[0x0080006f]);
        let events = vec![
            Event::Commit { pc: DRAM_BASE, raw: 0x0080006f, write: None },
            Event::Commit { pc: DRAM_BASE + 4, raw: 0, write: None },
        ];
        assert_eq!(
            run(&mut cpu, &events),
            DiffResult::Diverged {
                index: 1,
                kind: Divergence::Pc { spike: DRAM_BASE + 4, ours: DRAM_BASE + 8 },
            }
        );
    }

    #[test]
    fn stops_successfully_at_the_final_ecall() {
        let mut cpu = cpu_with_program(&[0x00500093, 0x00000073]);
        let events = vec![
            Event::Commit { pc: DRAM_BASE, raw: 0x00500093, write: Some((1, 5)) },
            Event::Trap {
                epc: DRAM_BASE + 4,
                cause: "trap_user_ecall".to_string(),
                tval: None,
            },
        ];
        assert_eq!(
            run(&mut cpu, &events),
            DiffResult::LockstepUntilEcall { instructions: 1 }
        );
    }

    #[test]
    fn follows_spike_through_a_spurious_sc_failure() {
        // lr.w a4, (a0) then sc.w a4, a5, (a0): our sc would succeed, but
        // the log says Spike's failed (a legal spurious failure). The
        // runner drops our reservation so both fail identically, and the
        // store must not happen.
        let mut cpu = cpu_with_program(&[0x1005272f, 0x18f5272f]);
        cpu.regs[10] = DRAM_BASE + 0x100;
        cpu.regs[15] = 99;
        let events = vec![
            Event::Commit { pc: DRAM_BASE, raw: 0x1005272f, write: Some((14, 0)) },
            Event::Commit { pc: DRAM_BASE + 4, raw: 0x18f5272f, write: Some((14, 1)) },
        ];
        assert_eq!(run(&mut cpu, &events), DiffResult::Lockstep { instructions: 2 });
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 0, "no store");
    }

    #[test]
    fn follows_spike_through_a_shared_trap() {
        // Both sides fault: an all-zero word does not decode. Our trap()
        // must redirect to mtvec so the next pc check still matches.
        let mut cpu = cpu_with_program(&[0x00000000]);
        cpu.csrs[0x305] = DRAM_BASE + 0x40; // mtvec
        cpu.bus.store32(DRAM_BASE + 0x40, 0x00500093).unwrap();
        let events = vec![
            Event::Trap {
                epc: DRAM_BASE,
                cause: "trap_illegal_instruction".to_string(),
                tval: Some(0),
            },
            Event::Commit {
                pc: DRAM_BASE + 0x40,
                raw: 0x00500093,
                write: Some((1, 5)),
            },
        ];
        assert_eq!(run(&mut cpu, &events), DiffResult::Lockstep { instructions: 1 });
    }
}
