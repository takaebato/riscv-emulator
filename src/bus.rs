//! Bus: owner of the address space as seen from the CPU.
//!
//! For now it is just a single DRAM. From phase 3 onwards, MMIO devices (UART, CLINT,
//! ...) will be added as address-range branches inside load/store; the overall shape
//! stays the same.

use crate::exception::Exception;

/// Start address of DRAM. RISC-V convention (same as Spike / QEMU virt).
/// Addresses below this are reserved territory for ROM and MMIO.
pub const DRAM_BASE: u64 = 0x8000_0000;

/// DRAM size (128 MiB). Enough for now, even with xv6/Linux in sight.
pub const DRAM_SIZE: u64 = 128 * 1024 * 1024;

/// CLINT (Core-Local INTerruptor) base address, per the QEMU virt layout.
pub const CLINT_BASE: u64 = 0x200_0000;
const CLINT_SIZE: u64 = 0x10000;
// Register offsets within the CLINT (hart 0).
const CLINT_MSIP: u64 = 0x0;
const CLINT_MTIMECMP: u64 = 0x4000;
const CLINT_MTIME: u64 = 0xbff8;

/// The machine's clock and software-interrupt doorbell. mtime counts time
/// (one tick per executed instruction, this machine's definition of time);
/// MTIP is asserted while mtime >= mtimecmp — so "interrupt me at time T"
/// is written as `mtimecmp = T`, and writing mtimecmp is also how the
/// handler acknowledges the interrupt. msip is a 1-bit doorbell wired
/// straight to the MSIP pending bit (harts ring each other with it).
pub struct Clint {
    pub mtime: u64,
    pub mtimecmp: u64,
    pub msip: u32,
}

impl Clint {
    /// The 64-bit register containing byte `off`, as (register base, value).
    fn reg_at(&self, off: u64) -> Option<(u64, u64)> {
        match off {
            CLINT_MSIP..=0x3 => Some((CLINT_MSIP, self.msip as u64)),
            CLINT_MTIMECMP..=0x4007 => Some((CLINT_MTIMECMP, self.mtimecmp)),
            CLINT_MTIME..=0xbfff => Some((CLINT_MTIME, self.mtime)),
            _ => None,
        }
    }
}

pub struct Bus {
    dram: Vec<u8>,
    pub clint: Clint,
}

impl Bus {
    pub fn new() -> Self {
        // Allocate the whole DRAM zero-filled. load_elf relies on this to satisfy
        // the zeroing of ELF .bss (zero-initialized regions) for free.
        Self {
            dram: vec![0; DRAM_SIZE as usize],
            clint: Clint { mtime: 0, mtimecmp: 0, msip: 0 },
        }
    }

    /// If [addr, addr+len) fits in DRAM, return the offset into `dram`.
    fn dram_offset(&self, addr: u64, len: u64) -> Option<usize> {
        let end = addr.checked_add(len)?;
        if addr >= DRAM_BASE && end <= DRAM_BASE + DRAM_SIZE {
            Some((addr - DRAM_BASE) as usize)
        } else {
            None
        }
    }

    fn load_bytes<const N: usize>(&self, addr: u64) -> Result<[u8; N], Exception> {
        // MMIO: the CLINT pretends to be memory. Reads are served from the
        // containing 64-bit register's little-endian bytes.
        if (CLINT_BASE..CLINT_BASE + CLINT_SIZE).contains(&addr) {
            let (base, value) = self
                .clint
                .reg_at(addr - CLINT_BASE)
                .filter(|(base, _)| addr - CLINT_BASE - base + N as u64 <= 8)
                .ok_or(Exception::LoadAccessFault(addr))?;
            let lo = (addr - CLINT_BASE - base) as usize;
            return Ok(value.to_le_bytes()[lo..lo + N].try_into().unwrap());
        }
        let off = self
            .dram_offset(addr, N as u64)
            .ok_or(Exception::LoadAccessFault(addr))?;
        Ok(self.dram[off..off + N].try_into().unwrap())
    }

    fn store_bytes(&mut self, addr: u64, bytes: &[u8]) -> Result<(), Exception> {
        // MMIO: a store lands in the containing CLINT register
        // (read-modify-write of its 8 little-endian bytes).
        if (CLINT_BASE..CLINT_BASE + CLINT_SIZE).contains(&addr) {
            let off = addr - CLINT_BASE;
            let (base, value) = self
                .clint
                .reg_at(off)
                .filter(|(base, _)| off - base + bytes.len() as u64 <= 8)
                .ok_or(Exception::StoreAmoAccessFault(addr))?;
            let mut b = value.to_le_bytes();
            let lo = (off - base) as usize;
            b[lo..lo + bytes.len()].copy_from_slice(bytes);
            let v = u64::from_le_bytes(b);
            match base {
                CLINT_MSIP => self.clint.msip = v as u32 & 1,
                CLINT_MTIMECMP => self.clint.mtimecmp = v,
                _ => self.clint.mtime = v,
            }
            return Ok(());
        }
        let off = self
            .dram_offset(addr, bytes.len() as u64)
            .ok_or(Exception::StoreAmoAccessFault(addr))?;
        self.dram[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    // RISC-V is little-endian, so always convert via from_le_bytes/to_le_bytes.

    pub fn load8(&self, addr: u64) -> Result<u8, Exception> {
        Ok(u8::from_le_bytes(self.load_bytes(addr)?))
    }

    pub fn load16(&self, addr: u64) -> Result<u16, Exception> {
        Ok(u16::from_le_bytes(self.load_bytes(addr)?))
    }

    pub fn load32(&self, addr: u64) -> Result<u32, Exception> {
        Ok(u32::from_le_bytes(self.load_bytes(addr)?))
    }

    pub fn load64(&self, addr: u64) -> Result<u64, Exception> {
        Ok(u64::from_le_bytes(self.load_bytes(addr)?))
    }

    pub fn store8(&mut self, addr: u64, value: u8) -> Result<(), Exception> {
        self.store_bytes(addr, &value.to_le_bytes())
    }

    pub fn store16(&mut self, addr: u64, value: u16) -> Result<(), Exception> {
        self.store_bytes(addr, &value.to_le_bytes())
    }

    pub fn store32(&mut self, addr: u64, value: u32) -> Result<(), Exception> {
        self.store_bytes(addr, &value.to_le_bytes())
    }

    pub fn store64(&mut self, addr: u64, value: u64) -> Result<(), Exception> {
        self.store_bytes(addr, &value.to_le_bytes())
    }

    /// Bulk write, e.g. for loading ELF segments.
    pub fn write_bytes(&mut self, addr: u64, data: &[u8]) -> Result<(), Exception> {
        self.store_bytes(addr, data)
    }
}

impl Default for Bus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_load_store() {
        let mut bus = Bus::new();
        bus.store64(DRAM_BASE, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(bus.load64(DRAM_BASE).unwrap(), 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn little_endian_layout() {
        let mut bus = Bus::new();
        bus.store32(DRAM_BASE, 0x1234_5678).unwrap();
        // Little-endian: the least significant byte lives at the lowest address.
        assert_eq!(bus.load8(DRAM_BASE).unwrap(), 0x78);
        assert_eq!(bus.load8(DRAM_BASE + 3).unwrap(), 0x12);
    }

    #[test]
    fn clint_registers_are_memory_mapped() {
        let mut bus = Bus::new();
        // mtimecmp: plain 64-bit read/write through load/store
        bus.store64(CLINT_BASE + 0x4000, 0xdead).unwrap();
        assert_eq!(bus.load64(CLINT_BASE + 0x4000).unwrap(), 0xdead);
        // mtime: reads see the device's live state, not RAM
        bus.clint.mtime = 42;
        assert_eq!(bus.load64(CLINT_BASE + 0xbff8).unwrap(), 42);
        // msip is a 1-bit doorbell: writes are masked down
        bus.store32(CLINT_BASE, 0xffff_ffff).unwrap();
        assert_eq!(bus.load32(CLINT_BASE).unwrap(), 1);
        // unmapped offsets inside the CLINT window still fault
        assert!(bus.load32(CLINT_BASE + 0x8000).is_err());
    }

    #[test]
    fn out_of_range_access_fault() {
        let mut bus = Bus::new();
        // Below DRAM
        assert_eq!(
            bus.load32(0x1000),
            Err(Exception::LoadAccessFault(0x1000))
        );
        // Exactly at the end is OK; one byte past it faults (boundary condition).
        let last = DRAM_BASE + DRAM_SIZE - 8;
        assert!(bus.store64(last, 1).is_ok());
        assert_eq!(
            bus.store64(last + 1, 1),
            Err(Exception::StoreAmoAccessFault(last + 1))
        );
    }
}
