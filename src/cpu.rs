//! Cpu: owner of the architectural state (integer registers and pc), plus the
//! execute stage of the fetch → decode → execute loop.

use crate::bus::Bus;
use crate::exception::Exception;
use crate::inst::{Inst, decode};
use crate::loader::LoadedElf;

/// CSR address of mtvec (Machine Trap VECtor): where traps jump to.
const MTVEC: usize = 0x305;
/// CSR address of mepc (Machine Exception Program Counter): where a trap saves
/// the interrupted pc, and where MRET returns to.
const MEPC: usize = 0x341;
/// CSR address of mcause: why the last trap was taken.
const MCAUSE: usize = 0x342;
/// CSR address of mtval (Machine Trap VALue): extra evidence about the trap
/// (faulting address, offending instruction word, ...).
const MTVAL: usize = 0x343;

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
    /// The LR/SC reservation: the address the last LR registered, if no SC
    /// has consumed it yet. Single-hart, so no other-hart invalidation
    /// exists; interrupts (phase 3) land between instructions and leave it
    /// alone, matching the spec's allowance.
    pub reservation: Option<u64>,
    pub bus: Bus,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            regs: [0; 32],
            pc: 0,
            csrs: [0; 4096],
            reservation: None,
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

    /// Take a trap: record what happened in the CSRs and redirect pc to the
    /// handler. `self.pc` must still point at the instruction that trapped
    /// (execute guarantees this: pc is only committed on success).
    ///
    /// Still missing for phase 3: the mstatus MIE/MPIE/MPP shuffle (interrupt
    /// masking and privilege tracking) that MRET will then undo.
    pub fn trap(&mut self, e: &Exception) {
        self.csrs[MEPC] = self.pc;
        self.csrs[MCAUSE] = e.cause();
        self.csrs[MTVAL] = match e {
            // The spec puts the address of the EBREAK itself in mtval; only
            // the dispatcher knows the pc, so it is filled in here.
            Exception::Breakpoint => self.pc,
            _ => e.tval(),
        };
        // The low 2 bits of mtvec select direct vs vectored mode. Vectored
        // only affects interrupts (pc = base + 4 * cause); exceptions always
        // enter at base, so masking the mode bits off is correct here.
        self.pc = self.csrs[MTVEC] & !0b11;
    }

    /// One 32-bit atomic read-modify-write: rd = sign-extended old value,
    /// memory = op(old, rs2 low 32). A faulting address propagates before
    /// rd is written or pc commits, like any load/store.
    fn amo_w(
        &mut self,
        rd: usize,
        rs1: usize,
        rs2: usize,
        op: impl Fn(u32, u32) -> u32,
    ) -> Result<(), Exception> {
        let addr = self.regs[rs1];
        let old = self.bus.load32(addr)?;
        self.bus.store32(addr, op(old, self.regs[rs2] as u32))?;
        self.regs[rd] = old as i32 as u64;
        Ok(())
    }

    /// The 64-bit sibling of [`Self::amo_w`].
    fn amo_d(
        &mut self,
        rd: usize,
        rs1: usize,
        rs2: usize,
        op: impl Fn(u64, u64) -> u64,
    ) -> Result<(), Exception> {
        let addr = self.regs[rs1];
        let old = self.bus.load64(addr)?;
        self.bus.store64(addr, op(old, self.regs[rs2]))?;
        self.regs[rd] = old;
        Ok(())
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
            Inst::Slti { rd, rs1, imm } => {
                self.regs[rd] = ((self.regs[rs1] as i64) < imm) as u64;
            }
            Inst::Sltiu { rd, rs1, imm } => {
                // The immediate was sign-extended by decode; only the compare
                // is unsigned. So imm = -1 means "less than 0xffff...ffff".
                self.regs[rd] = (self.regs[rs1] < imm as u64) as u64;
            }
            Inst::Xori { rd, rs1, imm } => {
                self.regs[rd] = self.regs[rs1] ^ imm as u64;
            }
            Inst::Ori { rd, rs1, imm } => {
                self.regs[rd] = self.regs[rs1] | imm as u64;
            }
            Inst::Andi { rd, rs1, imm } => {
                self.regs[rd] = self.regs[rs1] & imm as u64;
            }
            Inst::Lui { rd, imm } => {
                self.regs[rd] = imm as u64;
            }
            Inst::Auipc { rd, imm } => {
                self.regs[rd] = self.pc.wrapping_add(imm as u64);
            }
            // Shifts: in Rust (as in the ISA) >> is logical on unsigned and
            // arithmetic on signed, so the u64/i64 cast picks the semantics.
            Inst::Slli { rd, rs1, shamt } => {
                self.regs[rd] = self.regs[rs1] << shamt;
            }
            Inst::Srli { rd, rs1, shamt } => {
                self.regs[rd] = self.regs[rs1] >> shamt;
            }
            Inst::Srai { rd, rs1, shamt } => {
                self.regs[rd] = ((self.regs[rs1] as i64) >> shamt) as u64;
            }
            // W family: compute in 32 bits, then the `as i32 as u64` pair
            // sign-extends the 32-bit result into the full register. Every
            // W instruction ends with this same idiom.
            Inst::Addiw { rd, rs1, imm } => {
                let result = (self.regs[rs1] as u32).wrapping_add(imm as u32);
                self.regs[rd] = result as i32 as u64;
            }
            Inst::Slliw { rd, rs1, shamt } => {
                self.regs[rd] = ((self.regs[rs1] as u32) << shamt) as i32 as u64;
            }
            Inst::Srliw { rd, rs1, shamt } => {
                self.regs[rd] = ((self.regs[rs1] as u32) >> shamt) as i32 as u64;
            }
            Inst::Sraiw { rd, rs1, shamt } => {
                self.regs[rd] = ((self.regs[rs1] as i32) >> shamt) as u64;
            }
            // OP: same operations as OP-IMM but the second operand comes from a
            // register. Register shifts read their amount from the low 6 bits of
            // rs2 (5 for the W forms); the upper bits are ignored, not an error.
            Inst::Add { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1].wrapping_add(self.regs[rs2]);
            }
            Inst::Sub { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1].wrapping_sub(self.regs[rs2]);
            }
            Inst::Sll { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1] << (self.regs[rs2] & 0x3f);
            }
            Inst::Slt { rd, rs1, rs2 } => {
                self.regs[rd] = ((self.regs[rs1] as i64) < (self.regs[rs2] as i64)) as u64;
            }
            Inst::Sltu { rd, rs1, rs2 } => {
                self.regs[rd] = (self.regs[rs1] < self.regs[rs2]) as u64;
            }
            Inst::Xor { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1] ^ self.regs[rs2];
            }
            Inst::Srl { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1] >> (self.regs[rs2] & 0x3f);
            }
            Inst::Sra { rd, rs1, rs2 } => {
                self.regs[rd] = ((self.regs[rs1] as i64) >> (self.regs[rs2] & 0x3f)) as u64;
            }
            Inst::Or { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1] | self.regs[rs2];
            }
            Inst::And { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1] & self.regs[rs2];
            }
            Inst::Addw { rd, rs1, rs2 } => {
                let result = (self.regs[rs1] as u32).wrapping_add(self.regs[rs2] as u32);
                self.regs[rd] = result as i32 as u64;
            }
            Inst::Subw { rd, rs1, rs2 } => {
                let result = (self.regs[rs1] as u32).wrapping_sub(self.regs[rs2] as u32);
                self.regs[rd] = result as i32 as u64;
            }
            Inst::Sllw { rd, rs1, rs2 } => {
                let shamt = self.regs[rs2] & 0x1f;
                self.regs[rd] = ((self.regs[rs1] as u32) << shamt) as i32 as u64;
            }
            Inst::Srlw { rd, rs1, rs2 } => {
                let shamt = self.regs[rs2] & 0x1f;
                self.regs[rd] = ((self.regs[rs1] as u32) >> shamt) as i32 as u64;
            }
            Inst::Sraw { rd, rs1, rs2 } => {
                let shamt = self.regs[rs2] & 0x1f;
                self.regs[rd] = ((self.regs[rs1] as i32) >> shamt) as u64;
            }
            // M multiplies: the full 64x64 product is 128 bits, so compute in
            // i128/u128 and pick a half. The casts encode the operand
            // signedness: `as i64 as i128` sign-extends, `as u128`/`as i128`
            // on a u64 zero-extends.
            Inst::Mul { rd, rs1, rs2 } => {
                self.regs[rd] = self.regs[rs1].wrapping_mul(self.regs[rs2]);
            }
            Inst::Mulh { rd, rs1, rs2 } => {
                let product = (self.regs[rs1] as i64 as i128) * (self.regs[rs2] as i64 as i128);
                self.regs[rd] = (product >> 64) as u64;
            }
            Inst::Mulhsu { rd, rs1, rs2 } => {
                let product = (self.regs[rs1] as i64 as i128) * (self.regs[rs2] as i128);
                self.regs[rd] = (product >> 64) as u64;
            }
            Inst::Mulhu { rd, rs1, rs2 } => {
                let product = (self.regs[rs1] as u128) * (self.regs[rs2] as u128);
                self.regs[rd] = (product >> 64) as u64;
            }
            Inst::Mulw { rd, rs1, rs2 } => {
                let result = (self.regs[rs1] as u32).wrapping_mul(self.regs[rs2] as u32);
                self.regs[rd] = result as i32 as u64;
            }
            // M divides: no traps, the error cases have defined values.
            // Division by zero: quotient all ones, remainder = the dividend.
            // Signed overflow (MIN / -1, the one two's-complement quotient
            // that does not fit): quotient MIN, remainder 0 — which is what
            // Rust's wrapping_div/wrapping_rem compute, so only the zero
            // divisor needs an explicit branch (a bare `/` would panic on
            // both cases).
            Inst::Div { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as i64, self.regs[rs2] as i64);
                self.regs[rd] = if b == 0 { u64::MAX } else { a.wrapping_div(b) as u64 };
            }
            Inst::Divu { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1], self.regs[rs2]);
                self.regs[rd] = if b == 0 { u64::MAX } else { a / b };
            }
            Inst::Rem { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as i64, self.regs[rs2] as i64);
                self.regs[rd] = if b == 0 { a as u64 } else { a.wrapping_rem(b) as u64 };
            }
            Inst::Remu { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1], self.regs[rs2]);
                self.regs[rd] = if b == 0 { a } else { a % b };
            }
            // The W variants divide the low 32 bits and, like every W
            // instruction, sign-extend the 32-bit result — including the
            // unsigned ones: a quotient with bit 31 set comes back negative.
            Inst::Divw { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as i32, self.regs[rs2] as i32);
                self.regs[rd] = if b == 0 { u64::MAX } else { a.wrapping_div(b) as i64 as u64 };
            }
            Inst::Divuw { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as u32, self.regs[rs2] as u32);
                self.regs[rd] = if b == 0 { u64::MAX } else { (a / b) as i32 as u64 };
            }
            Inst::Remw { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as i32, self.regs[rs2] as i32);
                self.regs[rd] = if b == 0 {
                    a as i64 as u64
                } else {
                    a.wrapping_rem(b) as i64 as u64
                };
            }
            Inst::Remuw { rd, rs1, rs2 } => {
                let (a, b) = (self.regs[rs1] as u32, self.regs[rs2] as u32);
                self.regs[rd] = if b == 0 {
                    a as i32 as u64
                } else {
                    (a % b) as i32 as u64
                };
            }
            // AMOs: each is the shared read-modify-write with its own
            // combining function; the signed min/max go through i32/i64.
            Inst::AmoaddW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a.wrapping_add(b))?;
            }
            Inst::AmoaddD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a.wrapping_add(b))?;
            }
            Inst::AmoswapW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |_, b| b)?;
            }
            Inst::AmoswapD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |_, b| b)?;
            }
            Inst::AmoxorW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a ^ b)?;
            }
            Inst::AmoxorD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a ^ b)?;
            }
            Inst::AmoorW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a | b)?;
            }
            Inst::AmoorD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a | b)?;
            }
            Inst::AmoandW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a & b)?;
            }
            Inst::AmoandD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a & b)?;
            }
            Inst::AmominW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| (a as i32).min(b as i32) as u32)?;
            }
            Inst::AmominD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| (a as i64).min(b as i64) as u64)?;
            }
            Inst::AmomaxW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| (a as i32).max(b as i32) as u32)?;
            }
            Inst::AmomaxD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| (a as i64).max(b as i64) as u64)?;
            }
            Inst::AmominuW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a.min(b))?;
            }
            Inst::AmominuD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a.min(b))?;
            }
            Inst::AmomaxuW { rd, rs1, rs2 } => {
                self.amo_w(rd, rs1, rs2, |a, b| a.max(b))?;
            }
            Inst::AmomaxuD { rd, rs1, rs2 } => {
                self.amo_d(rd, rs1, rs2, |a, b| a.max(b))?;
            }
            // LR/SC: the reservation is only taken after the load clears
            // its fault check, and SC drops it no matter which way it goes
            // (a failed SC must not leave a live reservation behind).
            Inst::LrW { rd, rs1 } => {
                let addr = self.regs[rs1];
                self.regs[rd] = self.bus.load32(addr)? as i32 as u64;
                self.reservation = Some(addr);
            }
            Inst::LrD { rd, rs1 } => {
                let addr = self.regs[rs1];
                self.regs[rd] = self.bus.load64(addr)?;
                self.reservation = Some(addr);
            }
            Inst::ScW { rd, rs1, rs2 } => {
                let addr = self.regs[rs1];
                if self.reservation.take() == Some(addr) {
                    self.bus.store32(addr, self.regs[rs2] as u32)?;
                    self.regs[rd] = 0;
                } else {
                    self.regs[rd] = 1;
                }
            }
            Inst::ScD { rd, rs1, rs2 } => {
                let addr = self.regs[rs1];
                if self.reservation.take() == Some(addr) {
                    self.bus.store64(addr, self.regs[rs2])?;
                    self.regs[rd] = 0;
                } else {
                    self.regs[rd] = 1;
                }
            }
            Inst::Jal { rd, offset } => {
                self.regs[rd] = self.pc.wrapping_add(4); // link: return address
                next_pc = self.pc.wrapping_add(offset as u64);
            }
            Inst::Jalr { rd, rs1, offset } => {
                // Read rs1 before writing rd: rd may be the same register
                // (rv64ui-p-jalr exercises jalr t0, t0). The spec requires
                // clearing bit 0 of the computed target (JAL cannot even
                // encode an odd offset, but a register sum can be odd).
                let target = self.regs[rs1].wrapping_add(offset as u64) & !1;
                self.regs[rd] = self.pc.wrapping_add(4); // link: return address
                next_pc = target;
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
            // Loads: address = rs1 + offset, then widen to 64 bits. The `as`
            // casts pick sign- vs zero-extension exactly as in the W family.
            // The `?` propagates access faults out of execute; pc is not
            // committed in that case, which trap handling will rely on.
            Inst::Lb { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load8(addr)? as i8 as u64;
            }
            Inst::Lh { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load16(addr)? as i16 as u64;
            }
            Inst::Lw { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load32(addr)? as i32 as u64;
            }
            Inst::Lbu { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load8(addr)? as u64;
            }
            Inst::Lhu { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load16(addr)? as u64;
            }
            Inst::Lwu { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load32(addr)? as u64;
            }
            Inst::Ld { rd, rs1, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.regs[rd] = self.bus.load64(addr)?;
            }
            // Stores: truncate rs2 to the access width.
            Inst::Sb { rs1, rs2, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.bus.store8(addr, self.regs[rs2] as u8)?;
            }
            Inst::Sh { rs1, rs2, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.bus.store16(addr, self.regs[rs2] as u16)?;
            }
            Inst::Sw { rs1, rs2, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.bus.store32(addr, self.regs[rs2] as u32)?;
            }
            Inst::Sd { rs1, rs2, offset } => {
                let addr = self.regs[rs1].wrapping_add(offset as u64);
                self.bus.store64(addr, self.regs[rs2])?;
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
            // ECALL/EBREAK do not compute: raising the exception is their whole
            // semantics. trap() sets pc itself, so return before the commit
            // at the bottom would overwrite it with next_pc.
            Inst::Ecall => {
                self.trap(&Exception::EnvironmentCallFromMMode);
                return Ok(());
            }
            Inst::Ebreak => {
                self.trap(&Exception::Breakpoint);
                return Ok(());
            }
            // No-op on this in-order single-hart interpreter (see the enum doc).
            Inst::Fence => {}
            // Fetch always reads DRAM directly (no icache to flush): see the
            // enum doc. The self-modifying-code test below proves it holds.
            Inst::FenceI => {}
            // Minimal MRET: just the jump back to mepc. Restoring the privilege
            // level and interrupt-enable state comes with phase 3, along with
            // clearing the low bits of mepc (guaranteed aligned in practice here).
            Inst::Mret => {
                next_pc = self.csrs[MEPC];
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
    fn mul_and_mulh_split_the_128_bit_product() {
        // (-1) * (-1) = 1: mul gives the low half (1), mulh the high (0).
        // mul a4, a1, a2 = 0x02c58733 / mulh a4, a1, a2 = 0x02c59733
        let mut cpu = cpu_with_program(&[0x02c58733, 0x02c59733]);
        cpu.regs[11] = (-1i64) as u64;
        cpu.regs[12] = (-1i64) as u64;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 1, "low half");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "high half of +1 is 0");
    }

    #[test]
    fn mulh_variants_disagree_on_the_same_bits() {
        // Same operands (all-ones, all-ones), three readings:
        //   signed x signed:     (-1) * (-1)      -> high 64 = 0
        //   unsigned x unsigned: (2^64-1)^2       -> high 64 = 2^64 - 2
        //   signed x unsigned:   (-1) * (2^64-1)  -> high 64 = 2^64 - 1
        // mulh = 0x02c59733 / mulhu = 0x02c5b733 / mulhsu = 0x02c5a733
        let mut cpu = cpu_with_program(&[0x02c59733, 0x02c5b733, 0x02c5a733]);
        cpu.regs[11] = u64::MAX;
        cpu.regs[12] = u64::MAX;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "mulh: signed");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], u64::MAX - 1, "mulhu: unsigned");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], u64::MAX, "mulhsu: mixed");
    }

    #[test]
    fn mulw_wraps_at_32_bits_and_sign_extends() {
        // 0x7fffffff * 2 = 0xfffffffe: the product carries into bit 31, so
        // the 32-bit result is negative and sign-extends.
        // mulw a4, a1, a2 = 0x02c5873b
        let mut cpu = cpu_with_program(&[0x02c5873b]);
        cpu.regs[11] = 0x7fff_ffff;
        cpu.regs[12] = 2;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0xffff_ffff_ffff_fffe);
    }

    #[test]
    fn division_by_zero_has_defined_results_and_no_trap() {
        // div, divu, rem, remu a4, a1, a2 with a2 = 0: the quotients read
        // all ones, the remainders return the dividend. Four instructions,
        // zero traps.
        let mut cpu =
            cpu_with_program(&[0x02c5c733, 0x02c5d733, 0x02c5e733, 0x02c5f733]);
        cpu.regs[11] = 42;
        cpu.regs[12] = 0;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], u64::MAX, "div by zero: all ones");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], u64::MAX, "divu by zero: all ones");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 42, "rem by zero: the dividend");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 42, "remu by zero: the dividend");
    }

    #[test]
    fn signed_overflow_min_divided_by_minus_one() {
        // The one quotient two's complement cannot represent: -MIN = 2^63.
        // The spec defines div -> MIN and rem -> 0 instead of trapping.
        let mut cpu = cpu_with_program(&[0x02c5c733, 0x02c5e733]);
        cpu.regs[11] = i64::MIN as u64;
        cpu.regs[12] = (-1i64) as u64;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], i64::MIN as u64, "quotient wraps to MIN");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "remainder is 0");
    }

    #[test]
    fn rem_sign_follows_the_dividend() {
        // -7 / 2: rounding toward zero gives -3 remainder -1 (not +1),
        // so div * divisor + rem == dividend holds.
        let mut cpu = cpu_with_program(&[0x02c5c733, 0x02c5e733]);
        cpu.regs[11] = (-7i64) as u64;
        cpu.regs[12] = 2;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], (-3i64) as u64);
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], (-1i64) as u64);
    }

    #[test]
    fn divuw_sign_extends_its_unsigned_result() {
        // 0x80000000 / 1: an unsigned 32-bit quotient whose bit 31 is set
        // still sign-extends, like every W instruction (the invariant that
        // upper halves always mirror bit 31 beats "unsigned" here).
        let mut cpu = cpu_with_program(&[0x02c5d73b]);
        cpu.regs[11] = 0x8000_0000;
        cpu.regs[12] = 1;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0xffff_ffff_8000_0000);
    }

    #[test]
    fn amoadd_returns_the_old_value_and_updates_memory() {
        // amoadd.w a4, a1, (a3) = 0x00b6a72f: one instruction does
        // rd = mem, mem += rs2.
        let mut cpu = cpu_with_program(&[0x00b6a72f]);
        cpu.regs[13] = DRAM_BASE + 0x100;
        cpu.regs[11] = 3;
        cpu.bus.store32(DRAM_BASE + 0x100, 5).unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 5, "rd holds the value before the add");
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 8);
    }

    #[test]
    fn amo_w_sign_extends_the_old_value() {
        // amoswap.w a4, a1, (a3) = 0x08b6a72f with old = 0x8000_0000:
        // like LW, the 32-bit old value sign-extends into rd.
        let mut cpu = cpu_with_program(&[0x08b6a72f]);
        cpu.regs[13] = DRAM_BASE + 0x100;
        cpu.regs[11] = 7;
        cpu.bus.store32(DRAM_BASE + 0x100, 0x8000_0000).unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0xffff_ffff_8000_0000);
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 7, "swapped in");
    }

    #[test]
    fn amomax_signed_vs_unsigned_on_the_same_bits() {
        // Memory holds 0xffffffff: -1 to amomax.w (loses to 1), the
        // largest u32 to amomaxu.w (beats 1).
        // amomax.w = 0xa0b6a72f / amomaxu.w = 0xe0b6a72f
        for (raw, expected) in [(0xa0b6a72fu32, 1u32), (0xe0b6a72f, 0xffff_ffff)] {
            let mut cpu = cpu_with_program(&[raw]);
            cpu.regs[13] = DRAM_BASE + 0x100;
            cpu.regs[11] = 1;
            cpu.bus.store32(DRAM_BASE + 0x100, 0xffff_ffff).unwrap();
            cpu.step().unwrap();
            assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), expected);
        }
    }

    #[test]
    fn lr_sc_pair_succeeds_and_stores() {
        // lr.w a4, (a0) = 0x1005272f then sc.w a4, a5, (a0) = 0x18f5272f:
        // the reservation from lr lets sc through, rd reads 0 (success).
        let mut cpu = cpu_with_program(&[0x1005272f, 0x18f5272f]);
        cpu.regs[10] = DRAM_BASE + 0x100;
        cpu.regs[15] = 99;
        cpu.bus.store32(DRAM_BASE + 0x100, 5).unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 5, "lr loaded the old value");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "sc reports success");
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 99);
    }

    #[test]
    fn sc_without_reservation_fails_and_does_not_store() {
        let mut cpu = cpu_with_program(&[0x18f5272f]);
        cpu.regs[10] = DRAM_BASE + 0x100;
        cpu.regs[15] = 99;
        cpu.bus.store32(DRAM_BASE + 0x100, 5).unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 1, "sc reports failure");
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 5, "untouched");
    }

    #[test]
    fn sc_to_a_different_address_fails_and_consumes_the_reservation() {
        // lr on one address, sc on another: the sc fails, and because any
        // sc drops the reservation, a second sc back on the reserved
        // address fails too.
        let mut cpu = cpu_with_program(&[0x1005272f, 0x18f5272f, 0x18f5272f]);
        cpu.regs[10] = DRAM_BASE + 0x100;
        cpu.regs[15] = 99;
        cpu.step().unwrap(); // lr.w at +0x100
        cpu.regs[10] = DRAM_BASE + 0x200;
        cpu.step().unwrap(); // sc.w at +0x200: wrong address
        assert_eq!(cpu.regs[14], 1, "address mismatch fails");
        cpu.regs[10] = DRAM_BASE + 0x100;
        cpu.step().unwrap(); // sc.w back at +0x100: reservation is gone
        assert_eq!(cpu.regs[14], 1, "the failed sc consumed the reservation");
        assert_eq!(cpu.bus.load32(DRAM_BASE + 0x100).unwrap(), 0, "never stored");
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
    fn jalr_jumps_to_rs1_and_links() {
        // jalr t0, 0(t1) = 0x000302e7 — an absolute jump: the target comes
        // from a register, not from pc like jal/branches.
        let mut cpu = cpu_with_program(&[0x000302e7]);
        cpu.regs[6] = DRAM_BASE + 0x100;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 0x100, "jumped to rs1 + offset");
        assert_eq!(cpu.regs[5], DRAM_BASE + 4, "link register holds pc+4");
    }

    #[test]
    fn jalr_with_rd_equal_rs1_reads_before_writing() {
        // jalr t0, 0(t0) = 0x000282e7 (the rv64ui-p-jalr trap): the target
        // must come from the old t0, not from the freshly written link.
        let mut cpu = cpu_with_program(&[0x000282e7]);
        cpu.regs[5] = DRAM_BASE + 0x80;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 0x80, "target read before the link write");
        assert_eq!(cpu.regs[5], DRAM_BASE + 4);
    }

    #[test]
    fn jalr_clears_bit_0_of_the_target() {
        // ret = jalr x0, 0(x1) = 0x00008067, here with an odd address in ra.
        let mut cpu = cpu_with_program(&[0x00008067]);
        cpu.regs[1] = DRAM_BASE + 0x101;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 0x100, "bit 0 masked off by spec");
        assert_eq!(cpu.regs[0], 0, "ret discards the link into x0");
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

    #[test]
    fn srli_is_logical_srai_is_arithmetic() {
        // x2 = -16: srli sees a huge unsigned number, srai divides by 4.
        // srli x1, x2, 2 = 0x00215093 / srai x1, x2, 2 = 0x40215093
        let mut cpu = cpu_with_program(&[0x00215093]);
        cpu.regs[2] = (-16i64) as u64;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], (u64::MAX - 15) >> 2, "zeros shifted in");

        let mut cpu = cpu_with_program(&[0x40215093]);
        cpu.regs[2] = (-16i64) as u64;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], (-4i64) as u64, "sign bits shifted in: -16/4");
    }

    #[test]
    fn li_idiom_builds_a_wide_constant() {
        // The exact pair from rv64ui-p-add's constant setup:
        //   addiw t0, zero, 1   (0x0010029b)
        //   slli  t0, t0, 0x35  (0x03529293)  → t0 = 1 << 53
        let mut cpu = cpu_with_program(&[0x0010029b, 0x03529293]);
        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[5], 1u64 << 53);
    }

    #[test]
    fn addiw_wraps_at_32_bits_and_sign_extends() {
        // x1 = 0x7fff_ffff (i32::MAX); addiw x1, x1, 1 (0x0010809b) overflows the
        // 32-bit world to i32::MIN, whose sign extension fills the upper half.
        // A 64-bit addi would have produced 0x0000_0000_8000_0000 instead.
        let mut cpu = cpu_with_program(&[0x0010809b]);
        cpu.regs[1] = 0x7fff_ffff;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0xffff_ffff_8000_0000);
    }

    #[test]
    fn sraiw_uses_bit_31_as_sign() {
        // x2 = 0x0000_0000_8000_0000: positive as a 64-bit value, but its low
        // 32 bits are i32::MIN. sraiw x1, x2, 4 (0x4041509b) must see the latter.
        let mut cpu = cpu_with_program(&[0x4041509b]);
        cpu.regs[2] = 0x8000_0000;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0xffff_ffff_f800_0000);
    }

    #[test]
    fn slt_materializes_comparisons() {
        // x2 = -1, x3 = 1: signed says less, unsigned says greater.
        // slt x1, x2, x3 = 0x003120b3 / sltu x1, x2, x3 = 0x003130b3
        let mut cpu = cpu_with_program(&[0x003120b3, 0x003130b3]);
        cpu.regs[2] = (-1i64) as u64;
        cpu.regs[3] = 1;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 1, "signed: -1 < 1");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0, "unsigned: u64::MAX < 1 is false");
    }

    #[test]
    fn slti_vs_sltiu_on_negative_operand() {
        // x13 = -1: signed says "below zero", unsigned says "the maximum".
        // slti a4, a3, 0 = 0x0006a713 / sltiu a4, a3, 0 = 0x0006b713
        let mut cpu = cpu_with_program(&[0x0006a713, 0x0006b713]);
        cpu.regs[13] = (-1i64) as u64;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 1, "signed: -1 < 0");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "unsigned: u64::MAX < 0 is false");
    }

    #[test]
    fn sltiu_with_imm_1_is_seqz() {
        // sltiu a4, a3, 1 = 0x0016b713: rd = (rs1 == 0), the seqz pseudo
        // (0 is the only value unsigned-below 1).
        let mut cpu = cpu_with_program(&[0x0016b713, 0x0016b713]);
        cpu.regs[13] = 0;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 1, "zero → 1");
        cpu.regs[13] = 7;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0, "non-zero → 0");
    }

    #[test]
    fn xori_with_all_ones_is_not() {
        // xori a4, a3, -1 = 0xfff6c713: rd = !rs1, the `not` pseudo. Works
        // on all 64 bits because the 12-bit immediate sign-extends to all ones.
        let mut cpu = cpu_with_program(&[0xfff6c713]);
        cpu.regs[13] = 0x0f0f_1234_abcd_5678;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], !0x0f0f_1234_abcd_5678u64);
    }

    #[test]
    fn andi_immediate_sign_extends_across_64_bits() {
        // andi a4, a3, -241 = 0xf0f6f713 (the rv64ui-p-andi encoding): the
        // "12-bit" mask really is the 64-bit 0xffff_ffff_ffff_ff0f, so the
        // upper half of rs1 survives the AND.
        let mut cpu = cpu_with_program(&[0xf0f6f713]);
        cpu.regs[13] = 0xdead_beef_0000_00ff;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[14], 0xdead_beef_0000_000f);
    }

    #[test]
    fn register_shift_amount_uses_low_6_bits() {
        // sll x1, x2, x3 = 0x003110b3 with x3 = 65: only 65 & 0x3f = 1 counts.
        let mut cpu = cpu_with_program(&[0x003110b3]);
        cpu.regs[2] = 0x10;
        cpu.regs[3] = 65;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0x20, "shifted by 1, not by 65");
    }

    #[test]
    fn subw_wraps_and_sign_extends() {
        // subw x1, x2, x3 = 0x403100bb with 0 - 1: the 32-bit result 0xffffffff
        // sign-extends to a full 64-bit -1.
        let mut cpu = cpu_with_program(&[0x403100bb]);
        cpu.regs[2] = 0;
        cpu.regs[3] = 1;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], u64::MAX);
    }

    #[test]
    fn load_sign_vs_zero_extension() {
        // Memory byte 0x80: lb reads it as -128, lbu as +128.
        // lb x1, 0(x2) = 0x00010083 / lbu x1, 0(x2) = 0x00014083
        let mut cpu = cpu_with_program(&[0x00010083, 0x00014083]);
        cpu.bus.store8(DRAM_BASE + 0x100, 0x80).unwrap();
        cpu.regs[2] = DRAM_BASE + 0x100;
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], (-128i64) as u64, "lb sign-extends");
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 128, "lbu zero-extends");
    }

    #[test]
    fn narrow_store_truncates() {
        // sb x3, 0(x2) = 0x00310023, then ld x1, 0(x2) = 0x00013083:
        // only the low byte of x3 must reach memory.
        let mut cpu = cpu_with_program(&[0x00310023, 0x00013083]);
        cpu.regs[2] = DRAM_BASE + 0x200;
        cpu.regs[3] = 0xaabb_ccdd_1122_3344;
        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.regs[1], 0x44);
    }

    #[test]
    fn load_fault_leaves_pc_uncommitted() {
        // ld x1, 0(x2) with x2 = 0: LoadAccessFault, and pc must still point
        // at the faulting instruction (trap handling depends on this).
        let mut cpu = cpu_with_program(&[0x00013083]);
        cpu.regs[2] = 0;
        assert_eq!(cpu.step(), Err(Exception::LoadAccessFault(0)));
        assert_eq!(cpu.pc, DRAM_BASE);
    }

    #[test]
    fn fence_i_orders_self_modifying_code() {
        // The rv64ui-p-fence_i pattern in miniature: patch the word two slots
        // ahead, fence.i, then run it. Slot 2 starts as an illegal word and
        // becomes addi x5, x0, 42 (0x02a00293) — fetch must see the store.
        // sw x6, 8(x7) = 0x0063a423 / fence.i = 0x0000100f
        let mut cpu = cpu_with_program(&[0x0063a423, 0x0000100f, 0x00000000]);
        cpu.regs[6] = 0x02a00293;
        cpu.regs[7] = DRAM_BASE;
        cpu.step().unwrap(); // sw: patch slot 2
        cpu.step().unwrap(); // fence.i: nothing to flush here, by design
        cpu.step().unwrap(); // the patched addi runs, not the illegal word
        assert_eq!(cpu.regs[5], 42);
    }

    #[test]
    fn ecall_traps_to_mtvec() {
        let mut cpu = cpu_with_program(&[0x00000073]);
        cpu.csrs[MTVEC] = DRAM_BASE + 0x40;
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 0x40, "entered the handler");
        assert_eq!(cpu.csrs[MEPC], DRAM_BASE, "mepc points at the ecall itself");
        assert_eq!(cpu.csrs[MCAUSE], 11, "environment call from M-mode");
    }

    #[test]
    fn ebreak_records_its_pc_in_mtval() {
        let mut cpu = cpu_with_program(&[0x00100073]);
        cpu.csrs[MTVEC] = DRAM_BASE + 0x40;
        cpu.step().unwrap();
        assert_eq!(cpu.csrs[MCAUSE], 3, "breakpoint");
        assert_eq!(cpu.csrs[MTVAL], DRAM_BASE, "mtval holds the ebreak's address");
    }

    #[test]
    fn mret_jumps_to_mepc() {
        // The riscv-tests startup idiom: write the test body's address into
        // mepc, then mret into it. csrw mepc, t0 = 0x34129073.
        let mut cpu = cpu_with_program(&[0x34129073, 0x30200073]);
        cpu.regs[5] = DRAM_BASE + 0x100;
        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(cpu.pc, DRAM_BASE + 0x100);
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
