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

pub struct Bus {
    dram: Vec<u8>,
}

impl Bus {
    pub fn new() -> Self {
        // Allocate the whole DRAM zero-filled. load_elf relies on this to satisfy
        // the zeroing of ELF .bss (zero-initialized regions) for free.
        Self {
            dram: vec![0; DRAM_SIZE as usize],
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
        let off = self
            .dram_offset(addr, N as u64)
            .ok_or(Exception::LoadAccessFault(addr))?;
        Ok(self.dram[off..off + N].try_into().unwrap())
    }

    fn store_bytes(&mut self, addr: u64, bytes: &[u8]) -> Result<(), Exception> {
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
