//! Cpu: owner of the architectural state (integer registers and pc), plus the
//! execute stage of the fetch → decode → execute loop.

use crate::bus::Bus;
use crate::exception::Exception;
use crate::inst::{Inst, decode};
use crate::loader::LoadedElf;

pub struct Cpu {
    /// Integer registers x0..=x31. x0 is always 0 (enforced at the end of execute).
    pub regs: [u64; 32],
    /// Program counter
    pub pc: u64,
    /// Control and status registers, indexed by their 12-bit address (Zicsr).
    ///
    /// Flat array of all 4096 addresses for now: every CSR reads as 0 until
    /// something writes it, which happens to be the architecturally correct reset
    /// value for the ones the riscv-tests startup reads (e.g. mhartid = hart 0).
    /// Deliberately simplified: unimplemented CSRs should raise illegal-instruction
    /// (needed by rv64mi-p-csr, phase 3) but here they all read as 0.
    pub csrs: [u64; 4096],
    pub bus: Bus,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            regs: [0; 32],
            pc: 0,
            csrs: [0; 4096],
            bus: Bus::new(),
        }
    }

    /// Load an ELF into memory and point pc at the entry (the power-on step).
    pub fn load_elf(&mut self, elf: &LoadedElf) -> Result<(), Exception> {
        for seg in &elf.segments {
            self.bus.write_bytes(seg.addr, &seg.data)?;
            // The tail where mem_size > data.len() (i.e. .bss) needs no explicit
            // zeroing: Bus::new() allocates all of DRAM zero-filled.
        }
        self.pc = elf.entry;
        Ok(())
    }

    /// Fetch the 32-bit instruction word at pc.
    /// Will change to 16-bit parcel fetching when the C extension lands (phase 2).
    pub fn fetch(&self) -> Result<u32, Exception> {
        self.bus.load32(self.pc)
    }

    /// One turn of the interpreter loop: fetch → decode → execute.
    pub fn step(&mut self) -> Result<(), Exception> {
        let raw = self.fetch()?;
        let inst = decode(raw)?;
        self.execute(inst)
    }

    /// Execute one decoded instruction: update registers and advance pc.
    ///
    /// During execution `self.pc` still points at the current instruction (jumps and
    /// AUIPC need it); the new pc is committed at the very end.
    pub fn execute(&mut self, inst: Inst) -> Result<(), Exception> {
        // Straight-line default; jump/branch instructions overwrite this.
        let mut next_pc = self.pc.wrapping_add(4);

        match inst {
            Inst::Addi { rd, rs1, imm } => {
                // Guest integer overflow is architecturally defined to wrap, so all
                // arithmetic uses the wrapping_* family (avoids debug-build panics).
                self.regs[rd] = self.regs[rs1].wrapping_add(imm as u64);
            }
            Inst::Auipc { rd, imm } => {
                self.regs[rd] = self.pc.wrapping_add(imm as u64);
            }
            Inst::Jal { rd, offset } => {
                self.regs[rd] = self.pc.wrapping_add(4); // link: return address
                next_pc = self.pc.wrapping_add(offset as u64);
            }
            // Branches: on a taken branch the offset replaces the straight-line
            // next_pc. Signed conditions compare the same 64 bits reinterpreted
            // as i64 — the register file itself has no notion of signedness.
            Inst::Beq { rs1, rs2, offset } => {
                if self.regs[rs1] == self.regs[rs2] {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            Inst::Bne { rs1, rs2, offset } => {
                if self.regs[rs1] != self.regs[rs2] {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            Inst::Blt { rs1, rs2, offset } => {
                if (self.regs[rs1] as i64) < (self.regs[rs2] as i64) {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            Inst::Bge { rs1, rs2, offset } => {
                if (self.regs[rs1] as i64) >= (self.regs[rs2] as i64) {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            Inst::Bltu { rs1, rs2, offset } => {
                if self.regs[rs1] < self.regs[rs2] {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            Inst::Bgeu { rs1, rs2, offset } => {
                if self.regs[rs1] >= self.regs[rs2] {
                    next_pc = self.pc.wrapping_add(offset as u64);
                }
            }
            // Zicsr: read the old value first, so rd == rs1 still gets the swap
            // semantics right. The "skip the write when the mask source is zero"
            // rule was already enforced at decode time (it is static), but the
            // set/clear arms still guard on it so a plain read never dirties a CSR.
            Inst::Csrrw { rd, rs1, csr } => {
                let old = self.csrs[csr];
                self.csrs[csr] = self.regs[rs1];
                self.regs[rd] = old;
            }
            Inst::Csrrs { rd, rs1, csr } => {
                let old = self.csrs[csr];
                if rs1 != 0 {
                    self.csrs[csr] = old | self.regs[rs1];
                }
                self.regs[rd] = old;
            }
            Inst::Csrrc { rd, rs1, csr } => {
                let old = self.csrs[csr];
                if rs1 != 0 {
                    self.csrs[csr] = old & !self.regs[rs1];
                }
                self.regs[rd] = old;
            }
            Inst::Csrrwi { rd, uimm, csr } => {
                self.regs[rd] = self.csrs[csr];
                self.csrs[csr] = uimm;
            }
            Inst::Csrrsi { rd, uimm, csr } => {
                let old = self.csrs[csr];
                if uimm != 0 {
                    self.csrs[csr] = old | uimm;
                }
                self.regs[rd] = old;
            }
            Inst::Csrrci { rd, uimm, csr } => {
                let old = self.csrs[csr];
                if uimm != 0 {
                    self.csrs[csr] = old & !uimm;
                }
                self.regs[rd] = old;
            }
        }

        // x0 is hardwired to zero: writes are allowed but never stick. Clearing it
        // once per step is cheaper than branching on rd == 0 in every instruction.
        self.regs[0] = 0;
        self.pc = next_pc;
        Ok(())
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::DRAM_BASE;

    /// Write instruction words at DRAM_BASE and point pc there.
    fn cpu_with_program(insts: &[u32]) -> Cpu {
        let mut cpu = Cpu::new();
        for (i, &raw) in insts.iter().enumerate() {
            cpu.bus.store32(DRAM_BASE + 4 * i as u64, raw).unwrap();
        }
        cpu.pc = DRAM_BASE;
        cpu
    }

    #[test]
    fn addi_adds_and_advances_pc() {
        // addi x1, x0, 5  = 0x00500093
        let mut cpu = cpu_with_program(&[0x00500093]);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 5);
        assert_eq!(cpu.pc, DRAM_BASE + 4);
    }

    #[test]
    fn addi_negative_immediate_wraps() {
        // addi x1, x0, -1 → x1 = 0xffff_ffff_ffff_ffff (two's complement on 64 bits)
        // 0xfff00093 = addi x1, x0, -1
        let mut cpu = cpu_with_program(&[0xfff00093]);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], u64::MAX);
    }

    #[test]
    fn writes_to_x0_do_not_stick() {
        // addi x0, x0, 5 = 0x00500013 — architecturally a no-op (this encoding with
        // rd=x0 is also what the `nop` mnemonic assembles to... with imm=0)
        let mut cpu = cpu_with_program(&[0x00500013]);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[0], 0);
    }

    #[test]
    fn auipc_is_pc_relative() {
        // auipc x5, 0x1 = 0x00001297 → x5 = pc + 0x1000
        let mut cpu = cpu_with_program(&[0x00001297]);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[5], DRAM_BASE + 0x1000);
    }

    #[test]
    fn jal_links_and_jumps() {
        // jal x1, +8 = 0x008000ef
        let mut cpu = cpu_with_program(&[0x008000ef]);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], DRAM_BASE + 4, "link register holds pc+4");
        assert_eq!(cpu.pc, DRAM_BASE + 8, "pc jumped by the offset");
    }

    #[test]
    fn branch_taken_and_not_taken() {
        // beq x1, x2, +8 = 0x00208463
        let mut cpu = cpu_with_program(&[0x00208463]);
        cpu.regs[1] = 7;
        cpu.regs[2] = 7;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 8, "equal → taken");

        let mut cpu = cpu_with_program(&[0x00208463]);
        cpu.regs[1] = 7;
        cpu.regs[2] = 8;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 4, "not equal → fall through");
    }

    #[test]
    fn blt_is_signed_bltu_is_unsigned() {
        // Same register contents, opposite outcomes:
        // x1 = -1 as u64 (0xffff...ffff), x2 = 1.
        // blt x1, x2, +8 = 0x0020c463 — signed: -1 < 1 → taken
        let mut cpu = cpu_with_program(&[0x0020c463]);
        cpu.regs[1] = (-1i64) as u64;
        cpu.regs[2] = 1;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 8, "signed: -1 < 1");

        // bltu x1, x2, +8 = 0x0020e463 — unsigned: u64::MAX < 1 is false
        let mut cpu = cpu_with_program(&[0x0020e463]);
        cpu.regs[1] = (-1i64) as u64;
        cpu.regs[2] = 1;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 4, "unsigned: max < 1 is false");
    }

    /// Hand-assemble a SYSTEM/CSR instruction (funct3 picks the variant).
    fn csr_inst(funct3: u32, csr: u32, rs1_or_uimm: u32, rd: u32) -> u32 {
        (csr << 20) | (rs1_or_uimm << 15) | (funct3 << 12) | (rd << 7) | 0x73
    }

    const MSCRATCH: u32 = 0x340; // a read-write CSR with no special semantics
    const MHARTID: u32 = 0xf14; // read-only: hart (hardware thread) id

    #[test]
    fn csrrw_swaps_register_and_csr() {
        // csrrw x1, mscratch, x2
        let mut cpu = cpu_with_program(&[csr_inst(0x1, MSCRATCH, 2, 1)]);
        cpu.regs[2] = 0xdead;
        cpu.csrs[MSCRATCH as usize] = 0xbeef;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0xbeef, "rd got the old CSR value");
        assert_eq!(cpu.csrs[MSCRATCH as usize], 0xdead, "CSR got rs1");
    }

    #[test]
    fn csrr_reads_mhartid_as_zero() {
        // csrrs x10, mhartid, x0 — the `csrr a0, mhartid` from the riscv-tests
        // startup. rs1=x0 means no write, so the read-only CSR must not trap.
        let mut cpu = cpu_with_program(&[csr_inst(0x2, MHARTID, 0, 10)]);
        cpu.regs[10] = 0x1234; // stale value, must be overwritten by the read
        cpu.step().unwrap();
        assert_eq!(cpu.regs[10], 0, "single-hart machine: hart id 0");
    }

    #[test]
    fn csr_set_and_clear_bits() {
        let mut cpu = cpu_with_program(&[
            csr_inst(0x6, MSCRATCH, 0b101, 0), // csrrsi x0, mscratch, 0b101
            csr_inst(0x7, MSCRATCH, 0b001, 1), // csrrci x1, mscratch, 0b001
        ]);
        cpu.step().unwrap();
        assert_eq!(cpu.csrs[MSCRATCH as usize], 0b101, "set bits 0 and 2");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0b101, "rd sees the value before clearing");
        assert_eq!(cpu.csrs[MSCRATCH as usize], 0b100, "bit 0 cleared");
    }

    #[test]
    fn step_reports_illegal_instruction() {
        let mut cpu = cpu_with_program(&[0xffff_ffff]);
        assert_eq!(
            cpu.step(),
            Err(Exception::IllegalInstruction(0xffff_ffff))
        );
    }
}
