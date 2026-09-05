//! riscv-tests pass/fail harness.
//!
//! The rv64ui-p tests report their result through `tohost`, a 64-bit variable
//! whose address is published in the ELF symbol table. The pass epilogue writes
//! 1; a failing test case writes (testnum << 1) | 1. Either way the test then
//! spins forever, so the harness polls tohost after every step and stops on the
//! first non-zero value.
//!
//! (Spike's HTIF multiplexes console and syscalls over tohost as well; the
//! bare-metal `-p` tests only ever use the exit code, so that is all we read.)

use crate::cpu::Cpu;

/// How a test run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// tohost = 1: every test case passed.
    Pass,
    /// tohost = (testnum << 1) | 1: `testnum` is the failing test case.
    Fail { testnum: u64 },
    /// The step budget ran out before tohost was written (livelock guard,
    /// which also catches a guest spinning through an unhandled trap).
    StepLimit,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Pass => write!(f, "PASS"),
            Outcome::Fail { testnum } => write!(f, "FAIL (test case {testnum})"),
            Outcome::StepLimit => write!(f, "step limit exhausted before tohost was written"),
        }
    }
}

/// Poll tohost: Some(outcome) once the test has reported, None while it is
/// still running. A read fault (tohost outside DRAM) reads as "not yet".
pub fn check_tohost(cpu: &Cpu, tohost: u64) -> Option<Outcome> {
    match cpu.bus.load64(tohost) {
        Ok(0) | Err(_) => None,
        Ok(1) => Some(Outcome::Pass),
        Ok(v) => Some(Outcome::Fail { testnum: v >> 1 }),
    }
}

/// Run the interpreter loop until the test reports through tohost or
/// `step_limit` steps elapse.
pub fn run(cpu: &mut Cpu, tohost: u64, step_limit: u64) -> Outcome {
    for _ in 0..step_limit {
        if let Err(e) = cpu.step() {
            // Hardware never "stops on an error": every exception becomes
            // an architectural trap into the guest's handler. The tests
            // rely on it (probing optional CSRs under a prepared mtvec),
            // and an unhandled trap just spins into StepLimit.
            cpu.trap(&e);
        }
        if let Some(outcome) = check_tohost(cpu, tohost) {
            return outcome;
        }
    }
    Outcome::StepLimit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::DRAM_BASE;

    /// tohost location used by the hand-written programs below (any in-DRAM
    /// address away from the code works).
    const TOHOST: u64 = DRAM_BASE + 0x100;

    fn cpu_with_program(words: &[u32]) -> Cpu {
        let mut cpu = Cpu::new();
        for (i, w) in words.iter().enumerate() {
            cpu.bus.store32(DRAM_BASE + 4 * i as u64, *w).unwrap();
        }
        cpu.pc = DRAM_BASE;
        cpu
    }

    /// The write_tohost idiom in miniature: build the address, store the
    /// result, spin. auipc (not lui) because lui 0x80000 would sign-extend
    /// to 0xffff_ffff_8000_0000 on RV64.
    fn tohost_writer(result: u32) -> Vec<u32> {
        vec![
            0x00000117,                  // auipc x2, 0      -> x2 = DRAM_BASE
            0x00000093 | (result << 20), // addi  x1, x0, result
            0x10113023,                  // sd    x1, 0x100(x2)
            0x0000006f,                  // jal   x0, 0      (spin)
        ]
    }

    #[test]
    fn pass_is_tohost_one() {
        let mut cpu = cpu_with_program(&tohost_writer(1));
        assert_eq!(run(&mut cpu, TOHOST, 10), Outcome::Pass);
    }

    #[test]
    fn fail_decodes_the_test_number() {
        // tohost = 5 = (2 << 1) | 1: test case 2 failed.
        let mut cpu = cpu_with_program(&tohost_writer(5));
        assert_eq!(run(&mut cpu, TOHOST, 10), Outcome::Fail { testnum: 2 });
    }

    #[test]
    fn exceptions_trap_into_the_guest_handler() {
        // An illegal word (all zeros) with mtvec aimed at a handler that
        // reports failure through tohost: the harness must trap and keep
        // going, like hardware, not stop and report an emulator error.
        let mut cpu = cpu_with_program(&[0x00000000]);
        let handler = DRAM_BASE + 0x40;
        cpu.csrs[0x305] = handler; // mtvec
        // Handler, reporting 3 = (1 << 1) | 1 to tohost. Like tohost_writer
        // but pc-relative from the handler: auipc x2, 0 lands at +0x40, so
        // the store reaches TOHOST at 0xc0(x2).
        let words = [
            0x00000117, // auipc x2, 0      -> x2 = handler
            0x00300093, // addi  x1, x0, 3
            0x0c113023, // sd    x1, 0xc0(x2)
            0x0000006f, // jal   x0, 0      (spin)
        ];
        for (i, w) in words.iter().enumerate() {
            cpu.bus.store32(handler + 4 * i as u64, *w).unwrap();
        }
        assert_eq!(run(&mut cpu, TOHOST, 10), Outcome::Fail { testnum: 1 });
        assert_eq!(cpu.csrs[0x342], 2, "mcause: illegal instruction");
    }

    #[test]
    fn an_unhandled_trap_spins_to_the_step_limit() {
        // mtvec = 0 (outside DRAM): the trap loop can never make progress,
        // and the step limit is the net that catches it.
        let mut cpu = cpu_with_program(&[0x00000000]);
        assert_eq!(run(&mut cpu, TOHOST, 10), Outcome::StepLimit);
    }

    #[test]
    fn step_limit_stops_a_livelock() {
        // jal x0, 0: a legal infinite loop that never touches tohost.
        let mut cpu = cpu_with_program(&[0x0000006f]);
        assert_eq!(run(&mut cpu, TOHOST, 3), Outcome::StepLimit);
    }
}
