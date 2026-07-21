//! Architectural exceptions raised by the guest.
//!
//! They propagate as `Result<T, Exception>` up to the fetch→decode→execute loop.
//! Later (phase 3) the loop will perform architectural trap handling via the CSRs.
//! Variant names follow the mcause table in the Privileged spec (added as needed).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    /// EBREAK executed (debugger breakpoint)
    Breakpoint,
    /// The instruction word does not decode to anything we implement (yet)
    IllegalInstruction(u32),
    /// Load from outside the implemented memory range
    LoadAccessFault(u64),
    /// Store to outside the implemented memory range (shares its cause number with AMO)
    StoreAmoAccessFault(u64),
    /// ECALL executed in machine mode (the only mode so far)
    EnvironmentCallFromMMode,
}

impl Exception {
    /// The mcause exception code. Interrupts would set bit 63 on top; every
    /// variant here is a synchronous exception, so the bit stays clear.
    pub fn cause(&self) -> u64 {
        match self {
            Exception::IllegalInstruction(_) => 2,
            Exception::Breakpoint => 3,
            Exception::LoadAccessFault(_) => 5,
            Exception::StoreAmoAccessFault(_) => 7,
            Exception::EnvironmentCallFromMMode => 11,
        }
    }

    /// The mtval payload: the faulting address / instruction, or 0 where the
    /// spec has nothing useful to report. Breakpoint's mtval (the pc of the
    /// EBREAK) is filled in by the trap dispatcher, which knows the pc.
    pub fn tval(&self) -> u64 {
        match self {
            Exception::IllegalInstruction(raw) => *raw as u64,
            Exception::LoadAccessFault(addr) => *addr,
            Exception::StoreAmoAccessFault(addr) => *addr,
            Exception::Breakpoint | Exception::EnvironmentCallFromMMode => 0,
        }
    }
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exception::Breakpoint => write!(f, "breakpoint"),
            Exception::IllegalInstruction(raw) => {
                write!(f, "illegal instruction {raw:#010x}")
            }
            Exception::LoadAccessFault(addr) => write!(f, "load access fault at {addr:#x}"),
            Exception::StoreAmoAccessFault(addr) => {
                write!(f, "store/AMO access fault at {addr:#x}")
            }
            Exception::EnvironmentCallFromMMode => write!(f, "environment call from M-mode"),
        }
    }
}

impl std::error::Error for Exception {}
