//! ELF loader.
//!
//! A pure parser that extracts the LOAD segments (target address + bytes), the entry
//! point, and the symbol table from an ELF file. It does not write anything into
//! memory — how segments get placed is left to the Cpu/Bus side.

use std::collections::HashMap;

use object::{Object, ObjectSegment, ObjectSymbol};

/// Contents of a parsed ELF.
#[derive(Debug)]
pub struct LoadedElf {
    /// Entry point (the initial value of pc)
    pub entry: u64,
    /// LOAD segments (only PT_LOAD program headers)
    pub segments: Vec<Segment>,
    symbols: HashMap<String, u64>,
}

/// One region to be placed in memory.
#[derive(Debug)]
pub struct Segment {
    /// Target address
    pub addr: u64,
    /// Bytes present in the file (FileSiz worth)
    pub data: Vec<u8>,
    /// Size occupied in memory (MemSiz). If larger than data.len(), the excess is
    /// .bss-like and must be zero-filled.
    pub mem_size: u64,
}

impl LoadedElf {
    /// Look up a symbol address (used e.g. to find tohost in riscv-tests).
    pub fn symbol(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }
}

#[derive(Debug)]
pub enum LoadError {
    /// Not parseable as an ELF
    Parse(object::read::Error),
    /// Not an ELF for RISC-V (64-bit)
    UnsupportedArch(object::Architecture),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse(e) => write!(f, "cannot parse as ELF: {e}"),
            LoadError::UnsupportedArch(arch) => {
                write!(f, "not a 64-bit RISC-V ELF (architecture: {arch:?})")
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<object::read::Error> for LoadError {
    fn from(e: object::read::Error) -> Self {
        LoadError::Parse(e)
    }
}

/// Parse an ELF byte stream into LOAD segments, entry point, and symbol table.
pub fn load(bytes: &[u8]) -> Result<LoadedElf, LoadError> {
    let file = object::File::parse(bytes)?;

    if file.architecture() != object::Architecture::Riscv64 {
        return Err(LoadError::UnsupportedArch(file.architecture()));
    }

    let mut segments = Vec::new();
    for seg in file.segments() {
        segments.push(Segment {
            addr: seg.address(),
            data: seg.data()?.to_vec(),
            mem_size: seg.size(),
        });
    }

    let symbols = file
        .symbols()
        .filter_map(|sym| Some((sym.name().ok()?.to_string(), sym.address())))
        .collect();

    Ok(LoadedElf {
        entry: file.entry(),
        segments,
        symbols,
    })
}
