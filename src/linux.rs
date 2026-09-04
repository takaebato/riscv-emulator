//! Linux user-mode emulation (qemu-user style): run a statically linked
//! Linux ELF by having the emulator itself act as the operating system.
//!
//! The run loop here intercepts ECALL before execute() sees it, so the same
//! instruction that harness.rs lets trap into a guest handler (riscv-tests)
//! is answered directly by the host in this environment. That is the whole
//! point of ECALL's design: the instruction only says "call the environment";
//! each environment gives it meaning. The Cpu stays a pure machine.
//!
//! Syscall ABI (Linux riscv64): number in a7, arguments in a0..a5, return
//! value in a0 — a negated errno on failure, just like the real kernel ABI.

use crate::cpu::Cpu;
use crate::exception::Exception;
use crate::inst::Inst;

// Linux riscv64 syscall numbers (the generic asm-generic table), as needed.
const SYS_WRITE: u64 = 64;
const SYS_EXIT: u64 = 93;
const SYS_EXIT_GROUP: u64 = 94;

// Negated errno values, returned to the guest in a0.
const EBADF: i64 = -9;
const EFAULT: i64 = -14;
const EIO: i64 = -5;

// Argument/return registers of the calling convention.
const A0: usize = 10;
const A1: usize = 11;
const A2: usize = 12;
const A7: usize = 17;

/// How a Linux-mode run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The guest called exit/exit_group with this code.
    Exited(i64),
    /// An exception escaped (unimplemented instruction, access fault, ...);
    /// `pc` is the offending instruction.
    Exception { pc: u64, e: Exception },
    /// The guest used a syscall this environment does not provide yet.
    UnknownSyscall { pc: u64, a7: u64 },
    /// The step budget ran out (livelock guard).
    StepLimit,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Exited(code) => write!(f, "exited with code {code}"),
            Outcome::Exception { pc, e } => write!(f, "exception at {pc:#x}: {e}"),
            Outcome::UnknownSyscall { pc, a7 } => {
                write!(f, "unknown syscall {a7} at {pc:#x}")
            }
            Outcome::StepLimit => write!(f, "step limit exhausted"),
        }
    }
}

/// Run the loop until the guest exits, faults, or exhausts `step_limit`.
/// Guest writes to stdout/stderr both land in `out`.
pub fn run(cpu: &mut Cpu, out: &mut dyn std::io::Write, step_limit: u64) -> Outcome {
    for _ in 0..step_limit {
        let (inst, len) = match cpu.fetch_decode() {
            Ok((inst, len, _)) => (inst, len),
            Err(e) => return Outcome::Exception { pc: cpu.pc, e },
        };
        if let Inst::Ecall = inst {
            match cpu.regs[A7] {
                SYS_WRITE => {
                    let ret = sys_write(cpu, out);
                    cpu.regs[A0] = ret as u64;
                }
                // exit and exit_group only differ for multi-threaded guests.
                SYS_EXIT | SYS_EXIT_GROUP => {
                    return Outcome::Exited(cpu.regs[A0] as i64);
                }
                a7 => return Outcome::UnknownSyscall { pc: cpu.pc, a7 },
            }
            // The syscall is the ecall's execution: retire it like any
            // other non-jump instruction.
            cpu.pc = cpu.pc.wrapping_add(len);
        } else if let Err(e) = cpu.execute(inst, len) {
            return Outcome::Exception { pc: cpu.pc, e };
        }
    }
    Outcome::StepLimit
}

/// write(fd, buf, len): copy `len` guest bytes at `buf` into `out`.
/// Returns the byte count, or a negated errno exactly like the kernel:
/// EBADF for a fd we did not hand out, EFAULT for an unmapped buffer.
fn sys_write(cpu: &Cpu, out: &mut dyn std::io::Write) -> i64 {
    let (fd, buf, len) = (cpu.regs[A0], cpu.regs[A1], cpu.regs[A2]);
    if fd != 1 && fd != 2 {
        return EBADF;
    }
    let mut bytes = Vec::with_capacity(len as usize);
    for i in 0..len {
        match cpu.bus.load8(buf.wrapping_add(i)) {
            Ok(b) => bytes.push(b),
            Err(_) => return EFAULT,
        }
    }
    match out.write_all(&bytes) {
        Ok(()) => len as i64,
        Err(_) => EIO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::DRAM_BASE;

    /// Guest buffer location for the write tests (clear of the code).
    const MSG: u64 = DRAM_BASE + 0x100;

    fn cpu_with_program(words: &[u32]) -> Cpu {
        let mut cpu = Cpu::new();
        for (i, w) in words.iter().enumerate() {
            cpu.bus.store32(DRAM_BASE + 4 * i as u64, *w).unwrap();
        }
        cpu.pc = DRAM_BASE;
        cpu
    }

    /// ecall; li a7, 93; ecall — write(...), then exit(a0). Because exit's
    /// code is whatever write left in a0, the exit code observes write's
    /// return value.
    const WRITE_THEN_EXIT: &[u32] = &[0x00000073, 0x05d00893, 0x00000073];

    fn write_syscall_cpu(fd: u64, buf: u64, len: u64) -> Cpu {
        let mut cpu = cpu_with_program(WRITE_THEN_EXIT);
        cpu.regs[A7] = SYS_WRITE;
        cpu.regs[A0] = fd;
        cpu.regs[A1] = buf;
        cpu.regs[A2] = len;
        cpu
    }

    #[test]
    fn write_reaches_the_host_and_returns_the_byte_count() {
        let mut cpu = write_syscall_cpu(1, MSG, 5);
        for (i, b) in b"Hello".iter().enumerate() {
            cpu.bus.store8(MSG + i as u64, *b).unwrap();
        }
        let mut out = Vec::new();
        assert_eq!(run(&mut cpu, &mut out, 10), Outcome::Exited(5));
        assert_eq!(out, b"Hello");
    }

    #[test]
    fn write_to_a_bad_pointer_returns_efault() {
        // buf = 0 is outside DRAM: the guest gets -EFAULT back, not a crash.
        let mut cpu = write_syscall_cpu(1, 0, 1);
        let mut out = Vec::new();
        assert_eq!(run(&mut cpu, &mut out, 10), Outcome::Exited(EFAULT));
        assert!(out.is_empty());
    }

    #[test]
    fn write_to_an_unopened_fd_returns_ebadf() {
        let mut cpu = write_syscall_cpu(7, MSG, 1);
        let mut out = Vec::new();
        assert_eq!(run(&mut cpu, &mut out, 10), Outcome::Exited(EBADF));
    }

    #[test]
    fn unknown_syscall_stops_the_run() {
        let mut cpu = cpu_with_program(&[0x00000073]);
        cpu.regs[A7] = 222; // mmap: real, but not provided here
        let mut out = Vec::new();
        assert_eq!(
            run(&mut cpu, &mut out, 10),
            Outcome::UnknownSyscall { pc: DRAM_BASE, a7: 222 }
        );
    }
}