//! Architectural exceptions raised by the guest.
//!
//! They propagate as `Result<T, Exception>` up to the fetch→decode→execute loop.
//! Later (phase 3) the loop will perform architectural trap handling via the CSRs.
//! Variant names follow the mcause table in the Privileged spec (added as needed).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    /// The instruction word does not decode to anything we implement (yet)
    IllegalInstruction(u32),
    /// Load from outside the implemented memory range
    LoadAccessFault(u64),
    /// Store to outside the implemented memory range (shares its cause number with AMO)
    StoreAmoAccessFault(u64),
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exception::IllegalInstruction(raw) => {
                write!(f, "illegal instruction {raw:#010x}")
            }
            Exception::LoadAccessFault(addr) => write!(f, "load access fault at {addr:#x}"),
            Exception::StoreAmoAccessFault(addr) => {
                write!(f, "store/AMO access fault at {addr:#x}")
            }
        }
    }
}

impl std::error::Error for Exception {}
