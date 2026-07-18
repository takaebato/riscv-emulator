//! Instruction representation and the decoder (stage 1 of the two-stage design).
//!
//! `decode` is a pure function from a 32-bit instruction word to an `Inst`; nothing
//! here touches CPU state. Executing an `Inst` is cpu.rs's job. The enum grows
//! variant by variant, one-to-one with the instruction listing in the Unprivileged
//! spec (chapter "RV32/64I Base Integer Instruction Set").

use crate::exception::Exception;

/// A decoded instruction.
///
/// Register fields (rd/rs1/rs2) are kept as `usize` so they can index `Cpu::regs`
/// directly. Immediates are stored already sign-extended as `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    // --- RV64I (in progress) ---
    /// ADD Immediate: `rd = rs1 + imm`. Also the workhorse behind the `li` (load
    /// immediate, rs1=x0), `mv` (rd = rs1, imm=0) and `nop` pseudo-instructions.
    Addi { rd: usize, rs1: usize, imm: i64 },
    /// Add Upper Immediate to PC: `rd = pc + (imm << 12)`. Builds pc-relative
    /// addresses (the upper 20 bits; a following addi/load supplies the low 12).
    Auipc { rd: usize, imm: i64 },
    /// Jump And Link: `rd = pc + 4; pc += offset`. The saved return address makes
    /// it a function call; with rd=x0 the link is discarded and it is a plain jump.
    Jal { rd: usize, offset: i64 },

    // --- RV64I branches (B-type) ---
    // Compare rs1 with rs2 and, if the condition holds, jump pc-relative.
    // No condition-code register in RISC-V: every branch does its own compare.
    //
    /// Branch if EQual: `if rs1 == rs2 { pc += offset }`.
    Beq { rs1: usize, rs2: usize, offset: i64 },
    /// Branch if Not Equal.
    Bne { rs1: usize, rs2: usize, offset: i64 },
    /// Branch if Less Than (signed compare).
    Blt { rs1: usize, rs2: usize, offset: i64 },
    /// Branch if Greater or Equal (signed compare).
    Bge { rs1: usize, rs2: usize, offset: i64 },
    /// Branch if Less Than, Unsigned.
    Bltu { rs1: usize, rs2: usize, offset: i64 },
    /// Branch if Greater or Equal, Unsigned.
    Bgeu { rs1: usize, rs2: usize, offset: i64 },

    // --- Zicsr ---
    // All six atomically read the old CSR value into rd and combine a new value in.
    // The set/clear forms skip the write entirely when rs1/uimm is zero, so e.g.
    // `csrr rd, csr` (= csrrs rd, csr, x0) is a pure read.
    //
    /// CSR Read & Write: `rd = csr; csr = rs1`.
    Csrrw { rd: usize, rs1: usize, csr: usize },
    /// CSR Read & Set bits: `rd = csr; csr |= rs1` (rs1 is a bitmask of bits to set).
    Csrrs { rd: usize, rs1: usize, csr: usize },
    /// CSR Read & Clear bits: `rd = csr; csr &= !rs1`.
    Csrrc { rd: usize, rs1: usize, csr: usize },
    /// CSRRW with a 5-bit zero-extended immediate (the rs1 field) instead of a register.
    Csrrwi { rd: usize, uimm: u64, csr: usize },
    /// CSRRS with a 5-bit immediate bitmask.
    Csrrsi { rd: usize, uimm: u64, csr: usize },
    /// CSRRC with a 5-bit immediate bitmask.
    Csrrci { rd: usize, uimm: u64, csr: usize },
}

impl std::fmt::Display for Inst {
    /// Renders in objdump-like assembly syntax — a free disassembler, used by the
    /// trace output for Spike diff testing later.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Inst::Addi { rd, rs1, imm } => write!(f, "addi x{rd}, x{rs1}, {imm}"),
            // objdump prints the raw upper-20-bit field, not the shifted value
            Inst::Auipc { rd, imm } => write!(f, "auipc x{rd}, {:#x}", (imm >> 12) & 0xfffff),
            Inst::Jal { rd, offset } => write!(f, "jal x{rd}, {offset}"),
            Inst::Beq { rs1, rs2, offset } => write!(f, "beq x{rs1}, x{rs2}, {offset}"),
            Inst::Bne { rs1, rs2, offset } => write!(f, "bne x{rs1}, x{rs2}, {offset}"),
            Inst::Blt { rs1, rs2, offset } => write!(f, "blt x{rs1}, x{rs2}, {offset}"),
            Inst::Bge { rs1, rs2, offset } => write!(f, "bge x{rs1}, x{rs2}, {offset}"),
            Inst::Bltu { rs1, rs2, offset } => write!(f, "bltu x{rs1}, x{rs2}, {offset}"),
            Inst::Bgeu { rs1, rs2, offset } => write!(f, "bgeu x{rs1}, x{rs2}, {offset}"),
            // CSRs are printed by number for now; a name table (mhartid, ...) can
            // come later when the Spike-diff tooling needs it.
            Inst::Csrrw { rd, rs1, csr } => write!(f, "csrrw x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrs { rd, rs1, csr } => write!(f, "csrrs x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrc { rd, rs1, csr } => write!(f, "csrrc x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrwi { rd, uimm, csr } => write!(f, "csrrwi x{rd}, {csr:#x}, {uimm}"),
            Inst::Csrrsi { rd, uimm, csr } => write!(f, "csrrsi x{rd}, {csr:#x}, {uimm}"),
            Inst::Csrrci { rd, uimm, csr } => write!(f, "csrrci x{rd}, {csr:#x}, {uimm}"),
        }
    }
}

/// Decode one 32-bit instruction word.
pub fn decode(raw: u32) -> Result<Inst, Exception> {
    // Field positions are fixed across formats (the spec's big win for decoders):
    // opcode[6:0], rd[11:7], funct3[14:12], rs1[19:15], rs2[24:20], funct7[31:25]
    let opcode = raw & 0x7f;
    let rd = ((raw >> 7) & 0x1f) as usize;
    let rs1 = ((raw >> 15) & 0x1f) as usize;
    let rs2 = ((raw >> 20) & 0x1f) as usize;
    let funct3 = (raw >> 12) & 0x7;

    match opcode {
        // OP-IMM: register-immediate arithmetic (I-type)
        0x13 => match funct3 {
            0x0 => Ok(Inst::Addi { rd, rs1, imm: imm_i(raw) }),
            _ => Err(Exception::IllegalInstruction(raw)),
        },
        // AUIPC (U-type)
        0x17 => Ok(Inst::Auipc { rd, imm: imm_u(raw) }),
        // BRANCH (B-type): funct3 selects the condition
        0x63 => {
            let offset = imm_b(raw);
            match funct3 {
                0x0 => Ok(Inst::Beq { rs1, rs2, offset }),
                0x1 => Ok(Inst::Bne { rs1, rs2, offset }),
                0x4 => Ok(Inst::Blt { rs1, rs2, offset }),
                0x5 => Ok(Inst::Bge { rs1, rs2, offset }),
                0x6 => Ok(Inst::Bltu { rs1, rs2, offset }),
                0x7 => Ok(Inst::Bgeu { rs1, rs2, offset }),
                // funct3 2 and 3 are unused in the BRANCH opcode
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // JAL (J-type)
        0x6f => Ok(Inst::Jal { rd, offset: imm_j(raw) }),
        // SYSTEM: the CSR instructions (Zicsr). funct3=0 hosts ECALL/EBREAK/MRET,
        // which come later with trap handling.
        0x73 => {
            // The CSR address lives in the I-type immediate field, zero-extended.
            let csr = (raw >> 20) as usize;
            let uimm = rs1 as u64; // immediate forms reuse the rs1 field as a 5-bit constant
            // CSR addresses encode accessibility: bits [11:10] == 11 marks the CSR
            // read-only, and writing one raises illegal-instruction. Whether an
            // instruction writes is fully determined by its encoding (CSRRW always
            // writes; set/clear forms only when rs1/uimm != 0), so the check can
            // live here in the decoder. The privilege-level check (bits [9:8])
            // is dynamic and will move to execute in phase 3.
            let read_only = csr >> 10 == 0b11;
            let check = |writes: bool, inst: Inst| {
                if writes && read_only {
                    Err(Exception::IllegalInstruction(raw))
                } else {
                    Ok(inst)
                }
            };
            match funct3 {
                0x1 => check(true, Inst::Csrrw { rd, rs1, csr }),
                0x2 => check(rs1 != 0, Inst::Csrrs { rd, rs1, csr }),
                0x3 => check(rs1 != 0, Inst::Csrrc { rd, rs1, csr }),
                0x5 => check(true, Inst::Csrrwi { rd, uimm, csr }),
                0x6 => check(uimm != 0, Inst::Csrrsi { rd, uimm, csr }),
                0x7 => check(uimm != 0, Inst::Csrrci { rd, uimm, csr }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        _ => Err(Exception::IllegalInstruction(raw)),
    }
}

/// I-type immediate: inst[31:20], sign-extended.
/// Casting to i32 and arithmetic-shifting right by 20 does the sign extension.
fn imm_i(raw: u32) -> i64 {
    ((raw as i32) >> 20) as i64
}

/// U-type immediate: inst[31:12] << 12 (lower 12 bits zero), sign-extended.
fn imm_u(raw: u32) -> i64 {
    (raw & 0xffff_f000) as i32 as i64
}

/// B-type immediate: 13 bits scattered as inst[31]=imm[12], inst[30:25]=imm[10:5],
/// inst[11:8]=imm[4:1], inst[7]=imm[11]. Bit 0 is always zero, like J-type. This is
/// the S-type (store) layout with the sign bit and bit 11 swapped in, so branches
/// and stores share almost all their immediate wiring.
fn imm_b(raw: u32) -> i64 {
    let imm12 = ((raw >> 31) & 0x1) as u64;
    let imm10_5 = ((raw >> 25) & 0x3f) as u64;
    let imm4_1 = ((raw >> 8) & 0xf) as u64;
    let imm11 = ((raw >> 7) & 0x1) as u64;
    let imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
    // Sign-extend from bit 12.
    ((imm << 51) as i64) >> 51
}

/// J-type immediate: 21 bits scattered as inst[31]=imm[20], inst[30:21]=imm[10:1],
/// inst[20]=imm[11], inst[19:12]=imm[19:12]. Bit 0 is always zero (targets are
/// 2-byte aligned). This scattering exists so that source register fields stay at
/// fixed positions across formats — the cost is paid here, once, in the decoder.
fn imm_j(raw: u32) -> i64 {
    let imm20 = ((raw >> 31) & 0x1) as u64;
    let imm10_1 = ((raw >> 21) & 0x3ff) as u64;
    let imm11 = ((raw >> 20) & 0x1) as u64;
    let imm19_12 = ((raw >> 12) & 0xff) as u64;
    let imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    // Sign-extend from bit 20 (shift the sign bit up to bit 63, then back down).
    ((imm << 43) as i64) >> 43
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values below come from real riscv-tests binaries
    // (cross-checked against the objdump .dump listings).

    #[test]
    fn decodes_addi() {
        // 0x02028593 = addi a1, t0, 32  (a1 = x11, t0 = x5)
        assert_eq!(
            decode(0x02028593).unwrap(),
            Inst::Addi { rd: 11, rs1: 5, imm: 32 }
        );
        // 0xfff08093 = addi x1, x1, -1 (negative immediate → sign extension)
        assert_eq!(
            decode(0xfff08093).unwrap(),
            Inst::Addi { rd: 1, rs1: 1, imm: -1 }
        );
    }

    #[test]
    fn decodes_auipc() {
        // 0x00000297 = auipc t0, 0x0
        assert_eq!(decode(0x00000297).unwrap(), Inst::Auipc { rd: 5, imm: 0 });
    }

    #[test]
    fn decodes_jal() {
        // 0x0500006f = jal x0, +0x50 (the very first instruction of rv64ui-p-add)
        assert_eq!(
            decode(0x0500006f).unwrap(),
            Inst::Jal { rd: 0, offset: 0x50 }
        );
        // 0xffdff06f = jal x0, -4 (negative offset → J-type sign extension)
        assert_eq!(
            decode(0xffdff06f).unwrap(),
            Inst::Jal { rd: 0, offset: -4 }
        );
    }

    #[test]
    fn decodes_branches() {
        // 0x03ff0863 = beq t5, t6, +0x30 (the tohost check loop in the test env)
        assert_eq!(
            decode(0x03ff0863).unwrap(),
            Inst::Beq { rs1: 30, rs2: 31, offset: 0x30 }
        );
        // 0x4e771063 = bne a4, t2, +0x4e0 (jump to <fail>)
        assert_eq!(
            decode(0x4e771063).unwrap(),
            Inst::Bne { rs1: 14, rs2: 7, offset: 0x4e0 }
        );
        // beq x1, x2, -8 (hand-assembled: backward branch → B-type sign extension)
        assert_eq!(
            decode(0xfe208ce3).unwrap(),
            Inst::Beq { rs1: 1, rs2: 2, offset: -8 }
        );
        // funct3=2 is a hole in the BRANCH opcode
        assert_eq!(
            decode(0x00002063),
            Err(Exception::IllegalInstruction(0x00002063))
        );
    }

    #[test]
    fn decodes_csr_instructions() {
        // 0xf1402573 = csrr a0, mhartid — the instruction that stopped the demo.
        // csrr is a pseudo-instruction for csrrs with rs1=x0 (read, set nothing).
        assert_eq!(
            decode(0xf1402573).unwrap(),
            Inst::Csrrs { rd: 10, rs1: 0, csr: 0xf14 }
        );
        // 0x30529073 = csrw mtvec, t0 — csrrw with rd=x0 (write, discard old value)
        assert_eq!(
            decode(0x30529073).unwrap(),
            Inst::Csrrw { rd: 0, rs1: 5, csr: 0x305 }
        );
    }

    #[test]
    fn rejects_write_to_read_only_csr() {
        // csrrw x0, mhartid, x0: mhartid (0xf14) has address bits [11:10] = 11 →
        // read-only, and CSRRW always writes.
        assert_eq!(
            decode(0xf1401073),
            Err(Exception::IllegalInstruction(0xf1401073))
        );
        // ...but the pure read above (csrrs with rs1=x0) is fine, and so is
        // csrrsi with uimm=0.
        assert!(decode(0xf1402573).is_ok());
    }

    #[test]
    fn rejects_unknown_opcode() {
        assert_eq!(
            decode(0xffff_ffff),
            Err(Exception::IllegalInstruction(0xffff_ffff))
        );
        // All-zero word is deliberately not a valid RISC-V instruction.
        assert_eq!(decode(0), Err(Exception::IllegalInstruction(0)));
    }

    #[test]
    fn displays_as_assembly() {
        assert_eq!(decode(0x02028593).unwrap().to_string(), "addi x11, x5, 32");
        assert_eq!(decode(0x0500006f).unwrap().to_string(), "jal x0, 80");
    }
}
