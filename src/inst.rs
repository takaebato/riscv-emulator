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
    /// Set Less Than Immediate: `rd = ((rs1 as i64) < imm) ? 1 : 0` (signed).
    Slti { rd: usize, rs1: usize, imm: i64 },
    /// Set Less Than Immediate Unsigned. The immediate is still sign-extended
    /// first; only the comparison is unsigned ("unsigned compare", not
    /// "unsigned immediate"). `sltiu rd, rs1, 1` is the `seqz` (set if zero)
    /// pseudo-instruction: only 0 is unsigned-less-than 1.
    Sltiu { rd: usize, rs1: usize, imm: i64 },
    /// XOR Immediate. `xori rd, rs1, -1` (all ones) is the `not` pseudo-instruction.
    Xori { rd: usize, rs1: usize, imm: i64 },
    /// OR Immediate.
    Ori { rd: usize, rs1: usize, imm: i64 },
    /// AND Immediate. With small masks (e.g. `andi rd, rs1, 0xff`) the go-to
    /// low-bit extractor.
    Andi { rd: usize, rs1: usize, imm: i64 },
    /// Load Upper Immediate: `rd = imm << 12`. Places the upper 20 bits of a
    /// constant; a following I-type instruction supplies the (signed) low 12,
    /// so the pair covers any sign-extended 32-bit value.
    Lui { rd: usize, imm: i64 },
    /// Add Upper Immediate to PC: `rd = pc + (imm << 12)`. Builds pc-relative
    /// addresses (the upper 20 bits; a following addi/load supplies the low 12).
    Auipc { rd: usize, imm: i64 },
    /// Jump And Link: `rd = pc + 4; pc += offset`. The saved return address makes
    /// it a function call; with rd=x0 the link is discarded and it is a plain jump.
    Jal { rd: usize, offset: i64 },
    /// Jump And Link Register: `rd = pc + 4; pc = (rs1 + offset) & !1`. The
    /// computed-target jump: function returns (`ret` = `jalr x0, 0(x1)`),
    /// indirect calls, and the second half of the auipc+jalr far-call pair.
    /// Unlike JAL it can produce an odd address, so bit 0 is cleared by spec.
    Jalr { rd: usize, rs1: usize, offset: i64 },

    // --- RV64I OP-IMM shifts ---
    // I-type with the immediate field repurposed: the low 6 bits are the shift
    // amount (RV64; RV32 uses 5) and the leftover high bits act as funct6,
    // telling the logical and arithmetic right shifts apart.
    //
    /// Shift Left Logical Immediate: `rd = rs1 << shamt`. Zeros shift in.
    Slli { rd: usize, rs1: usize, shamt: u32 },
    /// Shift Right Logical Immediate: `rd = rs1 >> shamt`. Zeros shift in.
    Srli { rd: usize, rs1: usize, shamt: u32 },
    /// Shift Right Arithmetic Immediate: `rd = (rs1 as i64) >> shamt`. Copies of
    /// the sign bit shift in: division by 2^shamt rounding toward -infinity.
    Srai { rd: usize, rs1: usize, shamt: u32 },

    // --- RV64I OP-IMM-32 (the "W" family) ---
    // Operate on the low 32 bits and sign-extend the 32-bit result to 64. This is
    // what C's `int` arithmetic compiles to on RV64: a plain 64-bit add would leak
    // carries into the upper half instead of wrapping at 32 bits.
    //
    /// ADD Immediate Word: 32-bit addi, result sign-extended to 64 bits.
    Addiw { rd: usize, rs1: usize, imm: i64 },
    /// Shift Left Logical Immediate Word (shamt is back to 5 bits).
    Slliw { rd: usize, rs1: usize, shamt: u32 },
    /// Shift Right Logical Immediate Word: zeros shift into bit 31.
    Srliw { rd: usize, rs1: usize, shamt: u32 },
    /// Shift Right Arithmetic Immediate Word: bit 31 (not 63) is the sign.
    Sraiw { rd: usize, rs1: usize, shamt: u32 },

    // --- RV64I OP (R-type): register-register arithmetic ---
    // funct3 picks one of eight operations; bit 30 of funct7 flips ADD to SUB
    // and SRL to SRA, the same "alternate operation" bit as in the shifts above.
    // Register shifts take their amount from the low 6 bits of rs2 (5 for W).
    //
    /// `rd = rs1 + rs2`.
    Add { rd: usize, rs1: usize, rs2: usize },
    /// `rd = rs1 - rs2`.
    Sub { rd: usize, rs1: usize, rs2: usize },
    /// Shift Left Logical: `rd = rs1 << rs2`.
    Sll { rd: usize, rs1: usize, rs2: usize },
    /// Set Less Than (signed): `rd = if rs1 < rs2 { 1 } else { 0 }`. Materializes
    /// a comparison as a value, complementing the fused compare-and-branch.
    Slt { rd: usize, rs1: usize, rs2: usize },
    /// Set Less Than Unsigned.
    Sltu { rd: usize, rs1: usize, rs2: usize },
    /// `rd = rs1 ^ rs2`.
    Xor { rd: usize, rs1: usize, rs2: usize },
    /// Shift Right Logical: `rd = rs1 >> rs2`.
    Srl { rd: usize, rs1: usize, rs2: usize },
    /// Shift Right Arithmetic: `rd = (rs1 as i64) >> rs2`.
    Sra { rd: usize, rs1: usize, rs2: usize },
    /// `rd = rs1 | rs2`.
    Or { rd: usize, rs1: usize, rs2: usize },
    /// `rd = rs1 & rs2`.
    And { rd: usize, rs1: usize, rs2: usize },

    // --- RV64I OP-32: the register-register W family ---
    //
    /// 32-bit add, result sign-extended to 64 bits.
    Addw { rd: usize, rs1: usize, rs2: usize },
    /// 32-bit subtract, result sign-extended.
    Subw { rd: usize, rs1: usize, rs2: usize },
    /// 32-bit shift left (amount from rs2's low 5 bits).
    Sllw { rd: usize, rs1: usize, rs2: usize },
    /// 32-bit logical shift right.
    Srlw { rd: usize, rs1: usize, rs2: usize },
    /// 32-bit arithmetic shift right (bit 31 is the sign).
    Sraw { rd: usize, rs1: usize, rs2: usize },

    // --- M extension: multiplication (funct7=0000001 on OP / OP-32) ---
    // A 64×64 product is 128 bits; one instruction returns one half.
    //
    /// MULtiply: `rd = low 64 bits of rs1 * rs2`. The low half is identical
    /// whether the operands are read as signed or unsigned (two's-complement
    /// property), so one instruction covers both.
    Mul { rd: usize, rs1: usize, rs2: usize },
    /// MULtiply High: high 64 bits of the signed × signed product.
    Mulh { rd: usize, rs1: usize, rs2: usize },
    /// MULtiply High Signed×Unsigned: rs1 signed, rs2 unsigned. The fourth
    /// combination (unsigned × signed) needs no opcode: swap the operands.
    /// Exists to build multi-word signed multiplication.
    Mulhsu { rd: usize, rs1: usize, rs2: usize },
    /// MULtiply High Unsigned: high 64 bits of the unsigned × unsigned product.
    Mulhu { rd: usize, rs1: usize, rs2: usize },
    /// MULtiply Word (OP-32): low 32 × low 32, low 32 bits of the result
    /// sign-extended to 64. No MULHW: fetch the high half of a 32-bit
    /// product with a plain MUL on sign-extended operands and a shift.
    Mulw { rd: usize, rs1: usize, rs2: usize },

    // --- M extension: division (funct7=0000001, funct3 4-7) ---
    // No arithmetic traps in RISC-V: the two error cases have defined
    // result values instead. Division by zero returns all ones (DIV*) or
    // the dividend (REM*); signed overflow (MIN / -1) returns MIN / 0.
    //
    /// DIVide (signed): `rd = rs1 / rs2`, rounding toward zero.
    Div { rd: usize, rs1: usize, rs2: usize },
    /// DIVide Unsigned.
    Divu { rd: usize, rs1: usize, rs2: usize },
    /// REMainder (signed): the sign follows the dividend (rs1), pairing
    /// with DIV so that `div*rs2 + rem == rs1` always holds.
    Rem { rd: usize, rs1: usize, rs2: usize },
    /// REMainder Unsigned.
    Remu { rd: usize, rs1: usize, rs2: usize },
    /// DIVide Word: low 32 / low 32 as signed, result sign-extended.
    Divw { rd: usize, rs1: usize, rs2: usize },
    /// DIVide Unsigned Word. The 32-bit result is still sign-extended:
    /// a quotient with bit 31 set comes back with all upper bits ones.
    Divuw { rd: usize, rs1: usize, rs2: usize },
    /// REMainder Word (signed).
    Remw { rd: usize, rs1: usize, rs2: usize },
    /// REMainder Unsigned Word (sign-extended like DIVUW).
    Remuw { rd: usize, rs1: usize, rs2: usize },

    // --- A extension: atomic memory operations (opcode 0x2f) ---
    // Read-modify-write on mem[rs1] as one indivisible step: rd receives
    // the old memory value (sign-extended for .W), memory receives
    // op(old, rs2). On a single-hart in-order interpreter every load+store
    // pair is already indivisible, so atomicity costs nothing here; the
    // aq/rl ordering-hint bits are likewise no-ops and are not stored.
    // The .W forms compute on (and store) 32 bits.
    //
    /// AMO ADD word: mem += rs2.
    AmoaddW { rd: usize, rs1: usize, rs2: usize },
    /// AMO ADD doubleword.
    AmoaddD { rd: usize, rs1: usize, rs2: usize },
    /// AMO SWAP word: mem = rs2 (the old value lands in rd — an exchange).
    AmoswapW { rd: usize, rs1: usize, rs2: usize },
    /// AMO SWAP doubleword.
    AmoswapD { rd: usize, rs1: usize, rs2: usize },
    /// AMO XOR word.
    AmoxorW { rd: usize, rs1: usize, rs2: usize },
    /// AMO XOR doubleword.
    AmoxorD { rd: usize, rs1: usize, rs2: usize },
    /// AMO OR word.
    AmoorW { rd: usize, rs1: usize, rs2: usize },
    /// AMO OR doubleword.
    AmoorD { rd: usize, rs1: usize, rs2: usize },
    /// AMO AND word.
    AmoandW { rd: usize, rs1: usize, rs2: usize },
    /// AMO AND doubleword.
    AmoandD { rd: usize, rs1: usize, rs2: usize },
    /// AMO MINimum word (signed compare).
    AmominW { rd: usize, rs1: usize, rs2: usize },
    /// AMO MINimum doubleword (signed).
    AmominD { rd: usize, rs1: usize, rs2: usize },
    /// AMO MAXimum word (signed).
    AmomaxW { rd: usize, rs1: usize, rs2: usize },
    /// AMO MAXimum doubleword (signed).
    AmomaxD { rd: usize, rs1: usize, rs2: usize },
    /// AMO MINimum Unsigned word.
    AmominuW { rd: usize, rs1: usize, rs2: usize },
    /// AMO MINimum Unsigned doubleword.
    AmominuD { rd: usize, rs1: usize, rs2: usize },
    /// AMO MAXimum Unsigned word.
    AmomaxuW { rd: usize, rs1: usize, rs2: usize },
    /// AMO MAXimum Unsigned doubleword.
    AmomaxuD { rd: usize, rs1: usize, rs2: usize },

    // --- A extension: load-reserved / store-conditional ---
    // The build-your-own-atomic pair: LR loads and registers a reservation
    // on the address; SC stores only if the reservation still stands,
    // reporting success (0) or failure (1) in rd. Software retries the
    // whole LR..SC sequence on failure, so any read-modify-write can be
    // made atomic — this is how compare-and-swap is built on RISC-V.
    //
    /// Load-Reserved Word: `rd = sign-extended mem[rs1]`, reserve the address.
    LrW { rd: usize, rs1: usize },
    /// Load-Reserved Doubleword.
    LrD { rd: usize, rs1: usize },
    /// Store-Conditional Word: if reserved, `mem[rs1] = rs2 low 32; rd = 0`,
    /// else `rd = 1`. Always drops the reservation.
    ScW { rd: usize, rs1: usize, rs2: usize },
    /// Store-Conditional Doubleword.
    ScD { rd: usize, rs1: usize, rs2: usize },

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

    // --- RV64I loads (I-type) ---
    // `rd = mem[rs1 + offset]`, with the loaded value widened to 64 bits.
    // The plain forms sign-extend (a C `int8_t`/`int16_t`/`int32_t` load);
    // the U forms zero-extend (unsigned types). LD needs no LDU: 64 bits is
    // already the full register.
    //
    /// Load Byte (sign-extended).
    Lb { rd: usize, rs1: usize, offset: i64 },
    /// Load Halfword (16 bits, sign-extended).
    Lh { rd: usize, rs1: usize, offset: i64 },
    /// Load Word (32 bits, sign-extended).
    Lw { rd: usize, rs1: usize, offset: i64 },
    /// Load Byte Unsigned (zero-extended).
    Lbu { rd: usize, rs1: usize, offset: i64 },
    /// Load Halfword Unsigned.
    Lhu { rd: usize, rs1: usize, offset: i64 },
    /// Load Word Unsigned.
    Lwu { rd: usize, rs1: usize, offset: i64 },
    /// Load Doubleword (64 bits).
    Ld { rd: usize, rs1: usize, offset: i64 },

    // --- RV64I stores (S-type) ---
    // `mem[rs1 + offset] = low bits of rs2`. Narrow stores just truncate;
    // there is no sign/zero-extension question on the way out.
    //
    /// Store Byte.
    Sb { rs1: usize, rs2: usize, offset: i64 },
    /// Store Halfword.
    Sh { rs1: usize, rs2: usize, offset: i64 },
    /// Store Word.
    Sw { rs1: usize, rs2: usize, offset: i64 },
    /// Store Doubleword.
    Sd { rs1: usize, rs2: usize, offset: i64 },

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

    // --- Environment (SYSTEM, funct3=0) ---
    //
    /// Environment CALL: request a service from the execution environment by
    /// raising an exception (cause 11 from M-mode). What the "service" is
    /// belongs to the handler: an OS sees a syscall, riscv-tests sees "test
    /// finished, result is in gp".
    Ecall,
    /// Environment BREAK: raise a breakpoint exception (cause 3). Debuggers
    /// plant this word to regain control at a chosen spot.
    Ebreak,

    // --- Memory ordering ---
    //
    /// FENCE: order memory accesses as seen by other harts/devices. The pred and
    /// succ masks (which access kinds to order) are not stored: this emulator
    /// executes in program order on a single hart, so every fence is a no-op.
    Fence,
    /// FENCE.I (Zifencei extension): make instruction fetches see all stores
    /// this hart has already performed — the barrier self-modifying code runs
    /// after patching itself. Real hardware flushes the instruction cache /
    /// pipeline here; this interpreter fetches every instruction straight
    /// from memory, so fetches are always coherent and the fence is a no-op.
    FenceI,

    // --- Privileged (minimal, ahead of phase 3) ---
    //
    /// Machine-mode trap RETurn: `pc = mepc`. The full semantics also restore
    /// the privilege level (from mstatus.MPP) and the interrupt-enable bit
    /// (MIE from MPIE); those wait until phase 3 introduces privilege modes.
    /// riscv-tests uses it at the end of startup to jump to the test body.
    Mret,
}

impl std::fmt::Display for Inst {
    /// Renders in objdump-like assembly syntax — a free disassembler, used by the
    /// trace output for Spike diff testing later.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Inst::Addi { rd, rs1, imm } => write!(f, "addi x{rd}, x{rs1}, {imm}"),
            Inst::Slti { rd, rs1, imm } => write!(f, "slti x{rd}, x{rs1}, {imm}"),
            Inst::Sltiu { rd, rs1, imm } => write!(f, "sltiu x{rd}, x{rs1}, {imm}"),
            Inst::Xori { rd, rs1, imm } => write!(f, "xori x{rd}, x{rs1}, {imm}"),
            Inst::Ori { rd, rs1, imm } => write!(f, "ori x{rd}, x{rs1}, {imm}"),
            Inst::Andi { rd, rs1, imm } => write!(f, "andi x{rd}, x{rs1}, {imm}"),
            // objdump prints the raw upper-20-bit field, not the shifted value
            Inst::Lui { rd, imm } => write!(f, "lui x{rd}, {:#x}", (imm >> 12) & 0xfffff),
            Inst::Auipc { rd, imm } => write!(f, "auipc x{rd}, {:#x}", (imm >> 12) & 0xfffff),
            Inst::Jal { rd, offset } => write!(f, "jal x{rd}, {offset}"),
            Inst::Jalr { rd, rs1, offset } => write!(f, "jalr x{rd}, {offset}(x{rs1})"),
            // objdump prints shift amounts in hex
            Inst::Slli { rd, rs1, shamt } => write!(f, "slli x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Srli { rd, rs1, shamt } => write!(f, "srli x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Srai { rd, rs1, shamt } => write!(f, "srai x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Addiw { rd, rs1, imm } => write!(f, "addiw x{rd}, x{rs1}, {imm}"),
            Inst::Slliw { rd, rs1, shamt } => write!(f, "slliw x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Srliw { rd, rs1, shamt } => write!(f, "srliw x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Sraiw { rd, rs1, shamt } => write!(f, "sraiw x{rd}, x{rs1}, {shamt:#x}"),
            Inst::Add { rd, rs1, rs2 } => write!(f, "add x{rd}, x{rs1}, x{rs2}"),
            Inst::Sub { rd, rs1, rs2 } => write!(f, "sub x{rd}, x{rs1}, x{rs2}"),
            Inst::Sll { rd, rs1, rs2 } => write!(f, "sll x{rd}, x{rs1}, x{rs2}"),
            Inst::Slt { rd, rs1, rs2 } => write!(f, "slt x{rd}, x{rs1}, x{rs2}"),
            Inst::Sltu { rd, rs1, rs2 } => write!(f, "sltu x{rd}, x{rs1}, x{rs2}"),
            Inst::Xor { rd, rs1, rs2 } => write!(f, "xor x{rd}, x{rs1}, x{rs2}"),
            Inst::Srl { rd, rs1, rs2 } => write!(f, "srl x{rd}, x{rs1}, x{rs2}"),
            Inst::Sra { rd, rs1, rs2 } => write!(f, "sra x{rd}, x{rs1}, x{rs2}"),
            Inst::Or { rd, rs1, rs2 } => write!(f, "or x{rd}, x{rs1}, x{rs2}"),
            Inst::And { rd, rs1, rs2 } => write!(f, "and x{rd}, x{rs1}, x{rs2}"),
            Inst::Addw { rd, rs1, rs2 } => write!(f, "addw x{rd}, x{rs1}, x{rs2}"),
            Inst::Subw { rd, rs1, rs2 } => write!(f, "subw x{rd}, x{rs1}, x{rs2}"),
            Inst::Sllw { rd, rs1, rs2 } => write!(f, "sllw x{rd}, x{rs1}, x{rs2}"),
            Inst::Srlw { rd, rs1, rs2 } => write!(f, "srlw x{rd}, x{rs1}, x{rs2}"),
            Inst::Sraw { rd, rs1, rs2 } => write!(f, "sraw x{rd}, x{rs1}, x{rs2}"),
            Inst::Mul { rd, rs1, rs2 } => write!(f, "mul x{rd}, x{rs1}, x{rs2}"),
            Inst::Mulh { rd, rs1, rs2 } => write!(f, "mulh x{rd}, x{rs1}, x{rs2}"),
            Inst::Mulhsu { rd, rs1, rs2 } => write!(f, "mulhsu x{rd}, x{rs1}, x{rs2}"),
            Inst::Mulhu { rd, rs1, rs2 } => write!(f, "mulhu x{rd}, x{rs1}, x{rs2}"),
            Inst::Mulw { rd, rs1, rs2 } => write!(f, "mulw x{rd}, x{rs1}, x{rs2}"),
            Inst::Div { rd, rs1, rs2 } => write!(f, "div x{rd}, x{rs1}, x{rs2}"),
            Inst::Divu { rd, rs1, rs2 } => write!(f, "divu x{rd}, x{rs1}, x{rs2}"),
            Inst::Rem { rd, rs1, rs2 } => write!(f, "rem x{rd}, x{rs1}, x{rs2}"),
            Inst::Remu { rd, rs1, rs2 } => write!(f, "remu x{rd}, x{rs1}, x{rs2}"),
            Inst::Divw { rd, rs1, rs2 } => write!(f, "divw x{rd}, x{rs1}, x{rs2}"),
            Inst::Divuw { rd, rs1, rs2 } => write!(f, "divuw x{rd}, x{rs1}, x{rs2}"),
            Inst::Remw { rd, rs1, rs2 } => write!(f, "remw x{rd}, x{rs1}, x{rs2}"),
            Inst::Remuw { rd, rs1, rs2 } => write!(f, "remuw x{rd}, x{rs1}, x{rs2}"),
            Inst::AmoaddW { rd, rs1, rs2 } => write!(f, "amoadd.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoaddD { rd, rs1, rs2 } => write!(f, "amoadd.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoswapW { rd, rs1, rs2 } => write!(f, "amoswap.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoswapD { rd, rs1, rs2 } => write!(f, "amoswap.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoxorW { rd, rs1, rs2 } => write!(f, "amoxor.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoxorD { rd, rs1, rs2 } => write!(f, "amoxor.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoorW { rd, rs1, rs2 } => write!(f, "amoor.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoorD { rd, rs1, rs2 } => write!(f, "amoor.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoandW { rd, rs1, rs2 } => write!(f, "amoand.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmoandD { rd, rs1, rs2 } => write!(f, "amoand.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmominW { rd, rs1, rs2 } => write!(f, "amomin.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmominD { rd, rs1, rs2 } => write!(f, "amomin.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmomaxW { rd, rs1, rs2 } => write!(f, "amomax.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmomaxD { rd, rs1, rs2 } => write!(f, "amomax.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmominuW { rd, rs1, rs2 } => write!(f, "amominu.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmominuD { rd, rs1, rs2 } => write!(f, "amominu.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmomaxuW { rd, rs1, rs2 } => write!(f, "amomaxu.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::AmomaxuD { rd, rs1, rs2 } => write!(f, "amomaxu.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::LrW { rd, rs1 } => write!(f, "lr.w x{rd}, (x{rs1})"),
            Inst::LrD { rd, rs1 } => write!(f, "lr.d x{rd}, (x{rs1})"),
            Inst::ScW { rd, rs1, rs2 } => write!(f, "sc.w x{rd}, x{rs2}, (x{rs1})"),
            Inst::ScD { rd, rs1, rs2 } => write!(f, "sc.d x{rd}, x{rs2}, (x{rs1})"),
            Inst::Beq { rs1, rs2, offset } => write!(f, "beq x{rs1}, x{rs2}, {offset}"),
            Inst::Bne { rs1, rs2, offset } => write!(f, "bne x{rs1}, x{rs2}, {offset}"),
            Inst::Blt { rs1, rs2, offset } => write!(f, "blt x{rs1}, x{rs2}, {offset}"),
            Inst::Bge { rs1, rs2, offset } => write!(f, "bge x{rs1}, x{rs2}, {offset}"),
            Inst::Bltu { rs1, rs2, offset } => write!(f, "bltu x{rs1}, x{rs2}, {offset}"),
            Inst::Bgeu { rs1, rs2, offset } => write!(f, "bgeu x{rs1}, x{rs2}, {offset}"),
            Inst::Lb { rd, rs1, offset } => write!(f, "lb x{rd}, {offset}(x{rs1})"),
            Inst::Lh { rd, rs1, offset } => write!(f, "lh x{rd}, {offset}(x{rs1})"),
            Inst::Lw { rd, rs1, offset } => write!(f, "lw x{rd}, {offset}(x{rs1})"),
            Inst::Lbu { rd, rs1, offset } => write!(f, "lbu x{rd}, {offset}(x{rs1})"),
            Inst::Lhu { rd, rs1, offset } => write!(f, "lhu x{rd}, {offset}(x{rs1})"),
            Inst::Lwu { rd, rs1, offset } => write!(f, "lwu x{rd}, {offset}(x{rs1})"),
            Inst::Ld { rd, rs1, offset } => write!(f, "ld x{rd}, {offset}(x{rs1})"),
            Inst::Sb { rs1, rs2, offset } => write!(f, "sb x{rs2}, {offset}(x{rs1})"),
            Inst::Sh { rs1, rs2, offset } => write!(f, "sh x{rs2}, {offset}(x{rs1})"),
            Inst::Sw { rs1, rs2, offset } => write!(f, "sw x{rs2}, {offset}(x{rs1})"),
            Inst::Sd { rs1, rs2, offset } => write!(f, "sd x{rs2}, {offset}(x{rs1})"),
            // CSRs are printed by number for now; a name table (mhartid, ...) can
            // come later when the Spike-diff tooling needs it.
            Inst::Csrrw { rd, rs1, csr } => write!(f, "csrrw x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrs { rd, rs1, csr } => write!(f, "csrrs x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrc { rd, rs1, csr } => write!(f, "csrrc x{rd}, {csr:#x}, x{rs1}"),
            Inst::Csrrwi { rd, uimm, csr } => write!(f, "csrrwi x{rd}, {csr:#x}, {uimm}"),
            Inst::Csrrsi { rd, uimm, csr } => write!(f, "csrrsi x{rd}, {csr:#x}, {uimm}"),
            Inst::Csrrci { rd, uimm, csr } => write!(f, "csrrci x{rd}, {csr:#x}, {uimm}"),
            Inst::Ecall => write!(f, "ecall"),
            Inst::Ebreak => write!(f, "ebreak"),
            Inst::Fence => write!(f, "fence"),
            Inst::FenceI => write!(f, "fence.i"),
            Inst::Mret => write!(f, "mret"),
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
        // LOAD (I-type): funct3's low 2 bits are the width's log2, bit 2 selects
        // zero-extension. (0x7 would be a 128-bit load; illegal on RV64.)
        0x03 => {
            let offset = imm_i(raw);
            match funct3 {
                0x0 => Ok(Inst::Lb { rd, rs1, offset }),
                0x1 => Ok(Inst::Lh { rd, rs1, offset }),
                0x2 => Ok(Inst::Lw { rd, rs1, offset }),
                0x3 => Ok(Inst::Ld { rd, rs1, offset }),
                0x4 => Ok(Inst::Lbu { rd, rs1, offset }),
                0x5 => Ok(Inst::Lhu { rd, rs1, offset }),
                0x6 => Ok(Inst::Lwu { rd, rs1, offset }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // STORE (S-type)
        0x23 => {
            let offset = imm_s(raw);
            match funct3 {
                0x0 => Ok(Inst::Sb { rs1, rs2, offset }),
                0x1 => Ok(Inst::Sh { rs1, rs2, offset }),
                0x2 => Ok(Inst::Sw { rs1, rs2, offset }),
                0x3 => Ok(Inst::Sd { rs1, rs2, offset }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // OP-IMM: register-immediate arithmetic (I-type)
        0x13 => {
            // Shift encodings: shamt in inst[25:20] (6 bits on RV64), the
            // remaining inst[31:26] is funct6. Only 000000 and 010000 (SRAI,
            // reusing the bit-30 "alternate operation" convention) are valid.
            let shamt = (raw >> 20) & 0x3f;
            let funct6 = raw >> 26;
            match (funct3, funct6) {
                (0x0, _) => Ok(Inst::Addi { rd, rs1, imm: imm_i(raw) }),
                (0x1, 0b000000) => Ok(Inst::Slli { rd, rs1, shamt }),
                (0x2, _) => Ok(Inst::Slti { rd, rs1, imm: imm_i(raw) }),
                (0x3, _) => Ok(Inst::Sltiu { rd, rs1, imm: imm_i(raw) }),
                (0x4, _) => Ok(Inst::Xori { rd, rs1, imm: imm_i(raw) }),
                (0x5, 0b000000) => Ok(Inst::Srli { rd, rs1, shamt }),
                (0x5, 0b010000) => Ok(Inst::Srai { rd, rs1, shamt }),
                (0x6, _) => Ok(Inst::Ori { rd, rs1, imm: imm_i(raw) }),
                (0x7, _) => Ok(Inst::Andi { rd, rs1, imm: imm_i(raw) }),
                // Only reachable for funct3=1/5 with a bad funct6 now: every
                // funct3 value carries an instruction.
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // AUIPC (U-type)
        0x17 => Ok(Inst::Auipc { rd, imm: imm_u(raw) }),
        // LUI (U-type)
        0x37 => Ok(Inst::Lui { rd, imm: imm_u(raw) }),
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
        // AMO (A extension): R-type with funct7 split into funct5 | aq | rl.
        // The aq/rl ordering hints are meaningless on this in-order
        // single-hart interpreter, so any combination is accepted and
        // dropped. funct3 selects the width: 2 = .W, 3 = .D.
        0x2f => {
            let funct5 = raw >> 27;
            match (funct3, funct5) {
                (0x2, 0b00000) => Ok(Inst::AmoaddW { rd, rs1, rs2 }),
                (0x3, 0b00000) => Ok(Inst::AmoaddD { rd, rs1, rs2 }),
                (0x2, 0b00001) => Ok(Inst::AmoswapW { rd, rs1, rs2 }),
                (0x3, 0b00001) => Ok(Inst::AmoswapD { rd, rs1, rs2 }),
                (0x2, 0b00100) => Ok(Inst::AmoxorW { rd, rs1, rs2 }),
                (0x3, 0b00100) => Ok(Inst::AmoxorD { rd, rs1, rs2 }),
                (0x2, 0b01000) => Ok(Inst::AmoorW { rd, rs1, rs2 }),
                (0x3, 0b01000) => Ok(Inst::AmoorD { rd, rs1, rs2 }),
                (0x2, 0b01100) => Ok(Inst::AmoandW { rd, rs1, rs2 }),
                (0x3, 0b01100) => Ok(Inst::AmoandD { rd, rs1, rs2 }),
                (0x2, 0b10000) => Ok(Inst::AmominW { rd, rs1, rs2 }),
                (0x3, 0b10000) => Ok(Inst::AmominD { rd, rs1, rs2 }),
                (0x2, 0b10100) => Ok(Inst::AmomaxW { rd, rs1, rs2 }),
                (0x3, 0b10100) => Ok(Inst::AmomaxD { rd, rs1, rs2 }),
                (0x2, 0b11000) => Ok(Inst::AmominuW { rd, rs1, rs2 }),
                (0x3, 0b11000) => Ok(Inst::AmominuD { rd, rs1, rs2 }),
                (0x2, 0b11100) => Ok(Inst::AmomaxuW { rd, rs1, rs2 }),
                (0x3, 0b11100) => Ok(Inst::AmomaxuD { rd, rs1, rs2 }),
                // LR carries no source operand: rs2 is hardwired zero and
                // anything else is a reserved encoding.
                (0x2, 0b00010) if rs2 == 0 => Ok(Inst::LrW { rd, rs1 }),
                (0x3, 0b00010) if rs2 == 0 => Ok(Inst::LrD { rd, rs1 }),
                (0x2, 0b00011) => Ok(Inst::ScW { rd, rs1, rs2 }),
                (0x3, 0b00011) => Ok(Inst::ScD { rd, rs1, rs2 }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // OP: register-register arithmetic (R-type)
        0x33 => {
            let funct7 = raw >> 25;
            match (funct3, funct7) {
                (0x0, 0b0000000) => Ok(Inst::Add { rd, rs1, rs2 }),
                (0x0, 0b0100000) => Ok(Inst::Sub { rd, rs1, rs2 }),
                (0x1, 0b0000000) => Ok(Inst::Sll { rd, rs1, rs2 }),
                (0x2, 0b0000000) => Ok(Inst::Slt { rd, rs1, rs2 }),
                (0x3, 0b0000000) => Ok(Inst::Sltu { rd, rs1, rs2 }),
                (0x4, 0b0000000) => Ok(Inst::Xor { rd, rs1, rs2 }),
                (0x5, 0b0000000) => Ok(Inst::Srl { rd, rs1, rs2 }),
                (0x5, 0b0100000) => Ok(Inst::Sra { rd, rs1, rs2 }),
                (0x6, 0b0000000) => Ok(Inst::Or { rd, rs1, rs2 }),
                (0x7, 0b0000000) => Ok(Inst::And { rd, rs1, rs2 }),
                // funct7 = 0000001: the M extension. funct3 bit 2 splits it
                // into multiplies (0-3) and divides (4-7).
                (0x0, 0b0000001) => Ok(Inst::Mul { rd, rs1, rs2 }),
                (0x1, 0b0000001) => Ok(Inst::Mulh { rd, rs1, rs2 }),
                (0x2, 0b0000001) => Ok(Inst::Mulhsu { rd, rs1, rs2 }),
                (0x3, 0b0000001) => Ok(Inst::Mulhu { rd, rs1, rs2 }),
                (0x4, 0b0000001) => Ok(Inst::Div { rd, rs1, rs2 }),
                (0x5, 0b0000001) => Ok(Inst::Divu { rd, rs1, rs2 }),
                (0x6, 0b0000001) => Ok(Inst::Rem { rd, rs1, rs2 }),
                (0x7, 0b0000001) => Ok(Inst::Remu { rd, rs1, rs2 }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // OP-32: register-register W family (R-type)
        0x3b => {
            let funct7 = raw >> 25;
            match (funct3, funct7) {
                (0x0, 0b0000000) => Ok(Inst::Addw { rd, rs1, rs2 }),
                (0x0, 0b0100000) => Ok(Inst::Subw { rd, rs1, rs2 }),
                (0x1, 0b0000000) => Ok(Inst::Sllw { rd, rs1, rs2 }),
                (0x5, 0b0000000) => Ok(Inst::Srlw { rd, rs1, rs2 }),
                (0x5, 0b0100000) => Ok(Inst::Sraw { rd, rs1, rs2 }),
                // M extension, word width. Only MULW here (no MULHW).
                (0x0, 0b0000001) => Ok(Inst::Mulw { rd, rs1, rs2 }),
                (0x4, 0b0000001) => Ok(Inst::Divw { rd, rs1, rs2 }),
                (0x5, 0b0000001) => Ok(Inst::Divuw { rd, rs1, rs2 }),
                (0x6, 0b0000001) => Ok(Inst::Remw { rd, rs1, rs2 }),
                (0x7, 0b0000001) => Ok(Inst::Remuw { rd, rs1, rs2 }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // OP-IMM-32: W-family register-immediate arithmetic (I-type)
        0x1b => {
            // W shifts are back to a 5-bit shamt, so the discriminator is a full
            // funct7 again; a set bit 25 (shamt >= 32) is an illegal instruction.
            let shamt = (raw >> 20) & 0x1f;
            let funct7 = raw >> 25;
            match (funct3, funct7) {
                (0x0, _) => Ok(Inst::Addiw { rd, rs1, imm: imm_i(raw) }),
                (0x1, 0b0000000) => Ok(Inst::Slliw { rd, rs1, shamt }),
                (0x5, 0b0000000) => Ok(Inst::Srliw { rd, rs1, shamt }),
                (0x5, 0b0100000) => Ok(Inst::Sraiw { rd, rs1, shamt }),
                _ => Err(Exception::IllegalInstruction(raw)),
            }
        }
        // MISC-MEM: FENCE (funct3=0) and FENCE.I (funct3=1, Zifencei). Their
        // rd/rs1/imm fields are reserved for finer-grained fences; the spec
        // tells implementations to ignore them, so only funct3 is inspected.
        0x0f => match funct3 {
            0x0 => Ok(Inst::Fence),
            0x1 => Ok(Inst::FenceI),
            _ => Err(Exception::IllegalInstruction(raw)),
        },
        // JALR (I-type): the only instruction on its opcode, so funct3 must be 0
        0x67 => match funct3 {
            0x0 => Ok(Inst::Jalr { rd, rs1, offset: imm_i(raw) }),
            _ => Err(Exception::IllegalInstruction(raw)),
        },
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
                // funct3=0 holds the zero-operand instructions, told apart by the
                // whole word: every other field is a fixed constant.
                0x0 if raw == 0x00000073 => Ok(Inst::Ecall),
                0x0 if raw == 0x00100073 => Ok(Inst::Ebreak),
                0x0 if raw == 0x30200073 => Ok(Inst::Mret),
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

/// S-type immediate: 12 bits split as inst[31:25]=imm[11:5] (the funct7 slot)
/// and inst[11:7]=imm[4:0] (the rd slot, unused by stores). Same value range as
/// I-type; only the placement differs, to keep rs2 at its fixed position.
fn imm_s(raw: u32) -> i64 {
    let imm11_5 = ((raw as i32) >> 25) as i64; // arithmetic shift: sign-extends
    let imm4_0 = ((raw >> 7) & 0x1f) as i64;
    (imm11_5 << 5) | imm4_0
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

/// Decode a 16-bit compressed parcel (C extension) into the same Inst its
/// 32-bit expansion would decode to. The spec defines every compressed
/// instruction as an alias of a full-width one, so execute never needs to
/// know compression exists: only fetch (parcel size) and the pc advance
/// (len = 2) see the difference.
///
/// Layout: op = parcel[1:0] picks the quadrant (11 would be a full-width
/// instruction and never reaches here), funct3 = parcel[15:13] the row.
/// Growing quadrant by quadrant, like decode() grew opcode by opcode.
pub fn decode_compressed(parcel: u16) -> Result<Inst, Exception> {
    let op = parcel & 0b11;
    let funct3 = parcel >> 13;
    // Full-width register fields: rd/rs1 at [11:7], rs2 at [6:2]. The
    // primed (') 3-bit fields at [9:7]/[4:2] address only x8-x15 — the
    // eight registers compilers use most, per the C extension's statistics.
    let rd_full = ((parcel >> 7) & 0x1f) as usize;
    let rs2_full = ((parcel >> 2) & 0x1f) as usize;
    let rs1_c = 8 + ((parcel >> 7) & 0b111) as usize;
    let rd_c = 8 + ((parcel >> 2) & 0b111) as usize; // doubles as rs2'
    let illegal = Err(Exception::IllegalInstruction(parcel as u32));
    match (op, funct3) {
        // --- Quadrant 0: memory access through the primed registers ---
        //
        // C.ADDI4SPN: addi rd', sp, nzuimm — materialize the address of a
        // stack slot. nzuimm = 0 makes the all-zero parcel land here, and
        // the spec defines it illegal on purpose: jumping into zeroed
        // memory should trap, not silently no-op.
        (0b00, 0b000) => {
            let uimm = imm_ciw(parcel);
            if uimm == 0 {
                return illegal;
            }
            Ok(Inst::Addi { rd: rd_c, rs1: 2, imm: uimm })
        }
        // C.LW/C.LD and C.SW/C.SD: the workhorse field accesses. The
        // offset is unsigned and scaled (fields are multiples of the
        // access size — no bits wasted encoding misalignment).
        (0b00, 0b010) => Ok(Inst::Lw { rd: rd_c, rs1: rs1_c, offset: imm_c_mem_w(parcel) }),
        (0b00, 0b011) => Ok(Inst::Ld { rd: rd_c, rs1: rs1_c, offset: imm_c_mem_d(parcel) }),
        (0b00, 0b110) => Ok(Inst::Sw { rs1: rs1_c, rs2: rd_c, offset: imm_c_mem_w(parcel) }),
        (0b00, 0b111) => Ok(Inst::Sd { rs1: rs1_c, rs2: rd_c, offset: imm_c_mem_d(parcel) }),
        // (001/101 are C.FLD/C.FSD: illegal until the D extension.)

        // --- Quadrant 1: immediates, arithmetic, and control flow ---
        //
        // C.ADDI: addi rd, rd, imm6. rd=0 imm=0 is the canonical C.NOP;
        // other rd=0 forms are HINTs, which execute fine as addi x0.
        (0b01, 0b000) => Ok(Inst::Addi {
            rd: rd_full,
            rs1: rd_full,
            imm: imm_ci(parcel),
        }),
        // C.ADDIW (RV64): rd = 0 is reserved (RV32 uses this slot for C.JAL).
        (0b01, 0b001) if rd_full != 0 => Ok(Inst::Addiw {
            rd: rd_full,
            rs1: rd_full,
            imm: imm_ci(parcel),
        }),
        // C.LI: addi rd, x0, imm6 — load a small constant.
        (0b01, 0b010) => Ok(Inst::Addi { rd: rd_full, rs1: 0, imm: imm_ci(parcel) }),
        // funct3=011 splits on rd: x2 means C.ADDI16SP (grow/shrink the
        // stack frame, imm scaled by 16), anything else C.LUI (imm6
        // placed at bits 17:12). Both reserve the all-zero immediate.
        (0b01, 0b011) => {
            if rd_full == 2 {
                let imm = imm_addi16sp(parcel);
                if imm == 0 {
                    return illegal;
                }
                Ok(Inst::Addi { rd: 2, rs1: 2, imm })
            } else {
                let imm = imm_ci(parcel);
                if imm == 0 {
                    return illegal;
                }
                Ok(Inst::Lui { rd: rd_full, imm: imm << 12 })
            }
        }
        // funct3=100: the arithmetic block on primed registers, split by
        // bits 11:10 and, for register-register forms, bit 12 + bits 6:5.
        (0b01, 0b100) => {
            let rd = rs1_c; // rd' sits in the rs1' position here
            let rs2 = rd_c;
            match (parcel >> 10) & 0b11 {
                0b00 => Ok(Inst::Srli { rd, rs1: rd, shamt: shamt_ci(parcel) }),
                0b01 => Ok(Inst::Srai { rd, rs1: rd, shamt: shamt_ci(parcel) }),
                0b10 => Ok(Inst::Andi { rd, rs1: rd, imm: imm_ci(parcel) }),
                _ => match ((parcel >> 12) & 1, (parcel >> 5) & 0b11) {
                    (0, 0b00) => Ok(Inst::Sub { rd, rs1: rd, rs2 }),
                    (0, 0b01) => Ok(Inst::Xor { rd, rs1: rd, rs2 }),
                    (0, 0b10) => Ok(Inst::Or { rd, rs1: rd, rs2 }),
                    (0, 0b11) => Ok(Inst::And { rd, rs1: rd, rs2 }),
                    (1, 0b00) => Ok(Inst::Subw { rd, rs1: rd, rs2 }),
                    (1, 0b01) => Ok(Inst::Addw { rd, rs1: rd, rs2 }),
                    _ => illegal,
                },
            }
        }
        // C.J: jal x0 — the compressed unconditional jump (±2 KiB).
        (0b01, 0b101) => Ok(Inst::Jal { rd: 0, offset: imm_cj(parcel) }),
        // C.BEQZ/C.BNEZ: compare against x0 only — the most common
        // branch by far, per the statistics C is built on (±256 B).
        (0b01, 0b110) => Ok(Inst::Beq { rs1: rs1_c, rs2: 0, offset: imm_cb(parcel) }),
        (0b01, 0b111) => Ok(Inst::Bne { rs1: rs1_c, rs2: 0, offset: imm_cb(parcel) }),

        // --- Quadrant 2: stack-pointer-relative + full-register ops ---
        //
        // C.SLLI: slli rd, rd, shamt (6-bit shamt like the full form).
        (0b10, 0b000) => Ok(Inst::Slli { rd: rd_full, rs1: rd_full, shamt: shamt_ci(parcel) }),
        // C.LWSP/C.LDSP: load from the stack frame; rd = x0 is reserved.
        (0b10, 0b010) if rd_full != 0 => {
            Ok(Inst::Lw { rd: rd_full, rs1: 2, offset: imm_lwsp(parcel) })
        }
        (0b10, 0b011) if rd_full != 0 => {
            Ok(Inst::Ld { rd: rd_full, rs1: 2, offset: imm_ldsp(parcel) })
        }
        // funct3=100 packs five instructions, split by bit 12 and the
        // zero-ness of the register fields.
        (0b10, 0b100) => {
            let bit12 = (parcel >> 12) & 1;
            match (bit12, rd_full, rs2_full) {
                // C.JR: jalr x0, 0(rs1) — `ret` is c.jr x1. rs1=0 reserved.
                (0, rs1, 0) if rs1 != 0 => Ok(Inst::Jalr { rd: 0, rs1, offset: 0 }),
                (0, 0, 0) => illegal,
                // C.MV: add rd, x0, rs2 (a copy through the adder).
                (0, rd, rs2) => Ok(Inst::Add { rd, rs1: 0, rs2 }),
                // C.EBREAK: the 16-bit breakpoint (debuggers need it to
                // patch compressed code without growing it).
                (1, 0, 0) => Ok(Inst::Ebreak),
                // C.JALR: jalr x1, 0(rs1) — links pc+2 via execute's len.
                (1, rs1, 0) => Ok(Inst::Jalr { rd: 1, rs1, offset: 0 }),
                // C.ADD: add rd, rd, rs2.
                (_, rd, rs2) => Ok(Inst::Add { rd, rs1: rd, rs2 }),
            }
        }
        // C.SWSP/C.SDSP: store to the stack frame.
        (0b10, 0b110) => Ok(Inst::Sw { rs1: 2, rs2: rs2_full, offset: imm_swsp(parcel) }),
        (0b10, 0b111) => Ok(Inst::Sd { rs1: 2, rs2: rs2_full, offset: imm_sdsp(parcel) }),
        _ => illegal,
    }
}

/// CI-format immediate: parcel[12] = imm[5] (the sign), parcel[6:2] =
/// imm[4:0]. Six bits, sign-extended.
fn imm_ci(parcel: u16) -> i64 {
    let imm = ((((parcel >> 12) & 0x1) << 5) | ((parcel >> 2) & 0x1f)) as u64;
    ((imm << 58) as i64) >> 58
}

// The compressed memory offsets are unsigned (stack slots and struct
// fields sit at positive offsets) and scaled to the access size, so no
// bit encodes anything below the alignment. Each format scatters its
// bits differently — the price of squeezing fields into 16 bits.

/// CIW (C.ADDI4SPN): [12:11]=uimm[5:4], [10:7]=uimm[9:6], [6]=uimm[2], [5]=uimm[3].
fn imm_ciw(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 11) & 0x3) << 4)
        | (((p >> 7) & 0xf) << 6)
        | (((p >> 6) & 0x1) << 2)
        | (((p >> 5) & 0x1) << 3)) as i64
}

/// CL/CS word offset (C.LW/C.SW): [12:10]=uimm[5:3], [6]=uimm[2], [5]=uimm[6].
fn imm_c_mem_w(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 10) & 0x7) << 3) | (((p >> 6) & 0x1) << 2) | (((p >> 5) & 0x1) << 6)) as i64
}

/// CL/CS doubleword offset (C.LD/C.SD): [12:10]=uimm[5:3], [6:5]=uimm[7:6].
fn imm_c_mem_d(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 10) & 0x7) << 3) | (((p >> 5) & 0x3) << 6)) as i64
}

/// C.LWSP offset: [12]=uimm[5], [6:4]=uimm[4:2], [3:2]=uimm[7:6].
fn imm_lwsp(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 12) & 0x1) << 5) | (((p >> 4) & 0x7) << 2) | (((p >> 2) & 0x3) << 6)) as i64
}

/// C.LDSP offset: [12]=uimm[5], [6:5]=uimm[4:3], [4:2]=uimm[8:6].
fn imm_ldsp(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 12) & 0x1) << 5) | (((p >> 5) & 0x3) << 3) | (((p >> 2) & 0x7) << 6)) as i64
}

/// C.SWSP offset: [12:9]=uimm[5:2], [8:7]=uimm[7:6].
fn imm_swsp(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 9) & 0xf) << 2) | (((p >> 7) & 0x3) << 6)) as i64
}

/// C.SDSP offset: [12:10]=uimm[5:3], [9:7]=uimm[8:6].
fn imm_sdsp(parcel: u16) -> i64 {
    let p = parcel as u64;
    ((((p >> 10) & 0x7) << 3) | (((p >> 7) & 0x7) << 6)) as i64
}

/// CI shift amount (C.SLLI/C.SRLI/C.SRAI): [12]=shamt[5], [6:2]=shamt[4:0].
fn shamt_ci(parcel: u16) -> u32 {
    ((((parcel >> 12) & 0x1) << 5) | ((parcel >> 2) & 0x1f)) as u32
}

/// C.ADDI16SP immediate, scaled by 16: [12]=imm[9] (sign), [6]=imm[4],
/// [5]=imm[6], [4:3]=imm[8:7], [2]=imm[5].
fn imm_addi16sp(parcel: u16) -> i64 {
    let p = parcel as u64;
    let imm = (((p >> 12) & 0x1) << 9)
        | (((p >> 6) & 0x1) << 4)
        | (((p >> 5) & 0x1) << 6)
        | (((p >> 3) & 0x3) << 7)
        | (((p >> 2) & 0x1) << 5);
    ((imm << 54) as i64) >> 54
}

/// CJ offset (C.J), the most scattered of them all: [12]=imm[11] (sign),
/// [11]=imm[4], [10:9]=imm[9:8], [8]=imm[10], [7]=imm[6], [6]=imm[7],
/// [5:3]=imm[3:1], [2]=imm[5].
fn imm_cj(parcel: u16) -> i64 {
    let p = parcel as u64;
    let imm = (((p >> 12) & 0x1) << 11)
        | (((p >> 11) & 0x1) << 4)
        | (((p >> 9) & 0x3) << 8)
        | (((p >> 8) & 0x1) << 10)
        | (((p >> 7) & 0x1) << 6)
        | (((p >> 6) & 0x1) << 7)
        | (((p >> 3) & 0x7) << 1)
        | (((p >> 2) & 0x1) << 5);
    ((imm << 52) as i64) >> 52
}

/// CB offset (C.BEQZ/C.BNEZ): [12]=imm[8] (sign), [11:10]=imm[4:3],
/// [6:5]=imm[7:6], [4:3]=imm[2:1], [2]=imm[5].
fn imm_cb(parcel: u16) -> i64 {
    let p = parcel as u64;
    let imm = (((p >> 12) & 0x1) << 8)
        | (((p >> 10) & 0x3) << 3)
        | (((p >> 5) & 0x3) << 6)
        | (((p >> 3) & 0x3) << 1)
        | (((p >> 2) & 0x1) << 5);
    ((imm << 55) as i64) >> 55
}

#[cfg(test)]
mod tests {
    use super::*;

    // Encodings below are written in binary with underscores at the field
    // boundaries, so the decomposition decode performs is visible in the
    // literal itself. Each is cross-checked against the same word in hex,
    // as it appears in the objdump .dump listings of real riscv-tests
    // binaries (or as hand-assembled): getting a field split wrong fails
    // the equality before decode even runs.

    /// Decode a hand-split binary word after checking it against the objdump
    /// form: the binary shows the structure, the hex proves the provenance.
    fn decode_checked(binary: u32, hex: u32) -> Result<Inst, Exception> {
        assert_eq!(binary, hex, "field split does not match the objdump word");
        decode(binary)
    }

    #[test]
    fn decodes_addi() {
        // I-type: imm[11:0]_rs1_funct3_rd_opcode
        // addi a1, t0, 32  (a1 = x11, t0 = x5)
        assert_eq!(
            decode_checked(0b000000100000_00101_000_01011_0010011, 0x02028593).unwrap(),
            Inst::Addi { rd: 11, rs1: 5, imm: 32 }
        );
        // addi x1, x1, -1  (negative immediate → sign extension)
        assert_eq!(
            decode_checked(0b111111111111_00001_000_00001_0010011, 0xfff08093).unwrap(),
            Inst::Addi { rd: 1, rs1: 1, imm: -1 }
        );
    }

    #[test]
    fn decodes_op_imm_comparisons_and_logic() {
        // The five OP-IMM funct3 gaps (2/3/4/6/7), encodings straight from
        // the rv64ui-p tests that used to fail on them (a3 = x13, a4 = x14).
        // I-type: imm[11:0]_rs1_funct3_rd_opcode
        // slti a4, a3, 0
        assert_eq!(
            decode_checked(0b000000000000_01101_010_01110_0010011, 0x0006a713).unwrap(),
            Inst::Slti { rd: 14, rs1: 13, imm: 0 }
        );
        // sltiu a4, a3, 0
        assert_eq!(
            decode_checked(0b000000000000_01101_011_01110_0010011, 0x0006b713).unwrap(),
            Inst::Sltiu { rd: 14, rs1: 13, imm: 0 }
        );
        // xori a4, a3, -241  (0xf0f sign-extends)
        assert_eq!(
            decode_checked(0b111100001111_01101_100_01110_0010011, 0xf0f6c713).unwrap(),
            Inst::Xori { rd: 14, rs1: 13, imm: -241 }
        );
        // ori a4, a3, -241
        assert_eq!(
            decode_checked(0b111100001111_01101_110_01110_0010011, 0xf0f6e713).unwrap(),
            Inst::Ori { rd: 14, rs1: 13, imm: -241 }
        );
        // andi a4, a3, -241
        assert_eq!(
            decode_checked(0b111100001111_01101_111_01110_0010011, 0xf0f6f713).unwrap(),
            Inst::Andi { rd: 14, rs1: 13, imm: -241 }
        );
    }

    #[test]
    fn decodes_auipc() {
        // U-type: imm[31:12]_rd_opcode
        // auipc t0, 0x0
        assert_eq!(
            decode_checked(0b00000000000000000000_00101_0010111, 0x00000297).unwrap(),
            Inst::Auipc { rd: 5, imm: 0 }
        );
    }

    #[test]
    fn decodes_lui() {
        // lui a2, 0xffff8 — on RV64 the 32-bit value 0xffff8000 is further
        // sign-extended to 64 bits.
        assert_eq!(
            decode_checked(0b11111111111111111000_01100_0110111, 0xffff8637).unwrap(),
            Inst::Lui { rd: 12, imm: 0xffff_8000u32 as i32 as i64 }
        );
    }

    #[test]
    fn decodes_jal() {
        // J-type: imm[20]_imm[10:1]_imm[11]_imm[19:12]_rd_opcode
        // jal x0, +0x50  (the very first instruction of rv64ui-p-add;
        // 0x50 = 80 → imm[10:1] carries 80/2 = 40 = 0b0000101000)
        assert_eq!(
            decode_checked(0b0_0000101000_0_00000000_00000_1101111, 0x0500006f).unwrap(),
            Inst::Jal { rd: 0, offset: 0x50 }
        );
        // jal x0, -4  (negative offset → all-ones upper immediate bits)
        assert_eq!(
            decode_checked(0b1_1111111110_1_11111111_00000_1101111, 0xffdff06f).unwrap(),
            Inst::Jal { rd: 0, offset: -4 }
        );
    }

    #[test]
    fn decodes_jalr() {
        // I-type: imm[11:0]_rs1_funct3_rd_opcode
        // jalr t0, 0(t1) — where rv64ui-p-jalr used to stop (t0=x5, t1=x6)
        assert_eq!(
            decode_checked(0b000000000000_00110_000_00101_1100111, 0x000302e7).unwrap(),
            Inst::Jalr { rd: 5, rs1: 6, offset: 0 }
        );
        // jalr x0, 0(x1) — the `ret` pseudo-instruction
        assert_eq!(
            decode_checked(0b000000000000_00001_000_00000_1100111, 0x00008067).unwrap(),
            Inst::Jalr { rd: 0, rs1: 1, offset: 0 }
        );
        // JALR owns its whole opcode: funct3 != 0 does not decode
        assert_eq!(
            decode_checked(0b000000000000_00001_001_00000_1100111, 0x00009067),
            Err(Exception::IllegalInstruction(0x00009067))
        );
    }

    #[test]
    fn decodes_shifts() {
        // Shift-immediate: funct6_shamt_rs1_funct3_rd_opcode
        // slli t0, t0, 0x35 — shamt 53 > 31 needs the 6-bit RV64 shamt
        // field (this same encoding is illegal on RV32).
        assert_eq!(
            decode_checked(0b000000_110101_00101_001_00101_0010011, 0x03529293).unwrap(),
            Inst::Slli { rd: 5, rs1: 5, shamt: 0x35 }
        );
        // Hand-assembled: srli x1, x2, 4 / srai x1, x2, 4 (bit 30 apart)
        assert_eq!(
            decode_checked(0b000000_000100_00010_101_00001_0010011, 0x00415093).unwrap(),
            Inst::Srli { rd: 1, rs1: 2, shamt: 4 }
        );
        assert_eq!(
            decode_checked(0b010000_000100_00010_101_00001_0010011, 0x40415093).unwrap(),
            Inst::Srai { rd: 1, rs1: 2, shamt: 4 }
        );
        // SLLI with the SRAI funct6 pattern (bit 30 set) is not a thing
        assert_eq!(
            decode_checked(0b010000_000100_00010_001_00001_0010011, 0x40411093),
            Err(Exception::IllegalInstruction(0x40411093))
        );
    }

    #[test]
    fn decodes_w_family() {
        // I-type: imm[11:0]_rs1_funct3_rd_opcode
        // addiw t0, zero, 1 — the instruction that once stopped the demo
        assert_eq!(
            decode_checked(0b000000000001_00000_000_00101_0011011, 0x0010029b).unwrap(),
            Inst::Addiw { rd: 5, rs1: 0, imm: 1 }
        );
        // addiw t2, t2, -1  (sign-extended negative immediate)
        assert_eq!(
            decode_checked(0b111111111111_00111_000_00111_0011011, 0xfff3839b).unwrap(),
            Inst::Addiw { rd: 7, rs1: 7, imm: -1 }
        );
        // W shifts: funct7_shamt_rs1_funct3_rd_opcode
        // Hand-assembled: slliw x1, x2, 3 / sraiw x1, x2, 3
        assert_eq!(
            decode_checked(0b0000000_00011_00010_001_00001_0011011, 0x0031109b).unwrap(),
            Inst::Slliw { rd: 1, rs1: 2, shamt: 3 }
        );
        assert_eq!(
            decode_checked(0b0100000_00011_00010_101_00001_0011011, 0x4031509b).unwrap(),
            Inst::Sraiw { rd: 1, rs1: 2, shamt: 3 }
        );
        // W shamt is 5 bits: bit 25 (the low funct7 bit) set must not decode
        assert_eq!(
            decode_checked(0b0000001_00011_00010_001_00001_0011011, 0x0231109b),
            Err(Exception::IllegalInstruction(0x0231109b))
        );
    }

    #[test]
    fn decodes_op() {
        // R-type: funct7_rs2_rs1_funct3_rd_opcode
        // add a4, a1, a2 — where the demo once stopped
        assert_eq!(
            decode_checked(0b0000000_01100_01011_000_01110_0110011, 0x00c58733).unwrap(),
            Inst::Add { rd: 14, rs1: 11, rs2: 12 }
        );
        // sub is add with bit 30 set
        assert_eq!(
            decode_checked(0b0100000_01100_01011_000_01110_0110011, 0x40c58733).unwrap(),
            Inst::Sub { rd: 14, rs1: 11, rs2: 12 }
        );
        // Hand-assembled: sltu x1, x2, x3 / sra x1, x2, x3
        assert_eq!(
            decode_checked(0b0000000_00011_00010_011_00001_0110011, 0x003130b3).unwrap(),
            Inst::Sltu { rd: 1, rs1: 2, rs2: 3 }
        );
        assert_eq!(
            decode_checked(0b0100000_00011_00010_101_00001_0110011, 0x403150b3).unwrap(),
            Inst::Sra { rd: 1, rs1: 2, rs2: 3 }
        );
        // An unassigned funct7 (0000011) must not decode: only 0000000,
        // 0100000 and the M extension's 0000001 mean anything on OP.
        assert_eq!(
            decode_checked(0b0000011_00011_00010_000_00001_0110011, 0x063100b3),
            Err(Exception::IllegalInstruction(0x063100b3))
        );
    }

    #[test]
    fn decodes_op_32() {
        // R-type: funct7_rs2_rs1_funct3_rd_opcode
        // Hand-assembled: addw/subw a4, a1, a2
        assert_eq!(
            decode_checked(0b0000000_01100_01011_000_01110_0111011, 0x00c5873b).unwrap(),
            Inst::Addw { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0100000_01100_01011_000_01110_0111011, 0x40c5873b).unwrap(),
            Inst::Subw { rd: 14, rs1: 11, rs2: 12 }
        );
        // OP-32 has no logic/compare ops: AND's funct3 slot is a hole here
        assert_eq!(
            decode_checked(0b0000000_01100_01011_111_01110_0111011, 0x00c5f73b),
            Err(Exception::IllegalInstruction(0x00c5f73b))
        );
    }

    #[test]
    fn decodes_m_multiplies() {
        // R-type: funct7_rs2_rs1_funct3_rd_opcode
        // The multiply half of the M extension: funct7=0000001, funct3=0-3.
        // Encodings from the rv64um-p dumps (a4, a1, a2 = x14, x11, x12).
        // mul a4, a1, a2 — where rv64um-p-mul used to stop
        assert_eq!(
            decode_checked(0b0000001_01100_01011_000_01110_0110011, 0x02c58733).unwrap(),
            Inst::Mul { rd: 14, rs1: 11, rs2: 12 }
        );
        // mulh a4, a1, a2
        assert_eq!(
            decode_checked(0b0000001_01100_01011_001_01110_0110011, 0x02c59733).unwrap(),
            Inst::Mulh { rd: 14, rs1: 11, rs2: 12 }
        );
        // mulhsu a4, a1, a2
        assert_eq!(
            decode_checked(0b0000001_01100_01011_010_01110_0110011, 0x02c5a733).unwrap(),
            Inst::Mulhsu { rd: 14, rs1: 11, rs2: 12 }
        );
        // mulhu a4, a1, a2
        assert_eq!(
            decode_checked(0b0000001_01100_01011_011_01110_0110011, 0x02c5b733).unwrap(),
            Inst::Mulhu { rd: 14, rs1: 11, rs2: 12 }
        );
        // mulw a4, a1, a2 (OP-32)
        assert_eq!(
            decode_checked(0b0000001_01100_01011_000_01110_0111011, 0x02c5873b).unwrap(),
            Inst::Mulw { rd: 14, rs1: 11, rs2: 12 }
        );
        // OP-32 has no MULHW: funct3=1 with the M funct7 stays illegal
        assert_eq!(
            decode_checked(0b0000001_01100_01011_001_01110_0111011, 0x02c5973b),
            Err(Exception::IllegalInstruction(0x02c5973b))
        );
    }

    #[test]
    fn decodes_m_divides() {
        // R-type: funct7_rs2_rs1_funct3_rd_opcode
        // The divide half of M: funct3 4-7 = div/divu/rem/remu, and the
        // same four on OP-32 as the W variants. Encodings from the
        // rv64um-p dumps (a4, a1, a2 = x14, x11, x12).
        assert_eq!(
            decode_checked(0b0000001_01100_01011_100_01110_0110011, 0x02c5c733).unwrap(),
            Inst::Div { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_101_01110_0110011, 0x02c5d733).unwrap(),
            Inst::Divu { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_110_01110_0110011, 0x02c5e733).unwrap(),
            Inst::Rem { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_111_01110_0110011, 0x02c5f733).unwrap(),
            Inst::Remu { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_100_01110_0111011, 0x02c5c73b).unwrap(),
            Inst::Divw { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_101_01110_0111011, 0x02c5d73b).unwrap(),
            Inst::Divuw { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_110_01110_0111011, 0x02c5e73b).unwrap(),
            Inst::Remw { rd: 14, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_checked(0b0000001_01100_01011_111_01110_0111011, 0x02c5f73b).unwrap(),
            Inst::Remuw { rd: 14, rs1: 11, rs2: 12 }
        );
    }

    #[test]
    fn decodes_amos() {
        // AMO: funct5_aq_rl_rs2_rs1_funct3_rd_opcode
        // funct3 picks the width (010 = .w, 011 = .d); aq/rl stay 0 in the
        // rv64ua-p encodings. All operands a4, a1, (a3) = x14, x11, (x13).
        let cases: &[(u32, u32, fn(usize, usize, usize) -> Inst)] = &[
            (0b00000_0_0_01011_01101_010_01110_0101111, 0x00b6a72f, |rd, rs1, rs2| {
                Inst::AmoaddW { rd, rs1, rs2 }
            }),
            (0b00000_0_0_01011_01101_011_01110_0101111, 0x00b6b72f, |rd, rs1, rs2| {
                Inst::AmoaddD { rd, rs1, rs2 }
            }),
            (0b00001_0_0_01011_01101_010_01110_0101111, 0x08b6a72f, |rd, rs1, rs2| {
                Inst::AmoswapW { rd, rs1, rs2 }
            }),
            (0b00001_0_0_01011_01101_011_01110_0101111, 0x08b6b72f, |rd, rs1, rs2| {
                Inst::AmoswapD { rd, rs1, rs2 }
            }),
            (0b00100_0_0_01011_01101_010_01110_0101111, 0x20b6a72f, |rd, rs1, rs2| {
                Inst::AmoxorW { rd, rs1, rs2 }
            }),
            (0b00100_0_0_01011_01101_011_01110_0101111, 0x20b6b72f, |rd, rs1, rs2| {
                Inst::AmoxorD { rd, rs1, rs2 }
            }),
            (0b01000_0_0_01011_01101_010_01110_0101111, 0x40b6a72f, |rd, rs1, rs2| {
                Inst::AmoorW { rd, rs1, rs2 }
            }),
            (0b01000_0_0_01011_01101_011_01110_0101111, 0x40b6b72f, |rd, rs1, rs2| {
                Inst::AmoorD { rd, rs1, rs2 }
            }),
            (0b01100_0_0_01011_01101_010_01110_0101111, 0x60b6a72f, |rd, rs1, rs2| {
                Inst::AmoandW { rd, rs1, rs2 }
            }),
            (0b01100_0_0_01011_01101_011_01110_0101111, 0x60b6b72f, |rd, rs1, rs2| {
                Inst::AmoandD { rd, rs1, rs2 }
            }),
            (0b10000_0_0_01011_01101_010_01110_0101111, 0x80b6a72f, |rd, rs1, rs2| {
                Inst::AmominW { rd, rs1, rs2 }
            }),
            (0b10000_0_0_01011_01101_011_01110_0101111, 0x80b6b72f, |rd, rs1, rs2| {
                Inst::AmominD { rd, rs1, rs2 }
            }),
            (0b10100_0_0_01011_01101_010_01110_0101111, 0xa0b6a72f, |rd, rs1, rs2| {
                Inst::AmomaxW { rd, rs1, rs2 }
            }),
            (0b10100_0_0_01011_01101_011_01110_0101111, 0xa0b6b72f, |rd, rs1, rs2| {
                Inst::AmomaxD { rd, rs1, rs2 }
            }),
            (0b11000_0_0_01011_01101_010_01110_0101111, 0xc0b6a72f, |rd, rs1, rs2| {
                Inst::AmominuW { rd, rs1, rs2 }
            }),
            (0b11000_0_0_01011_01101_011_01110_0101111, 0xc0b6b72f, |rd, rs1, rs2| {
                Inst::AmominuD { rd, rs1, rs2 }
            }),
            (0b11100_0_0_01011_01101_010_01110_0101111, 0xe0b6a72f, |rd, rs1, rs2| {
                Inst::AmomaxuW { rd, rs1, rs2 }
            }),
            (0b11100_0_0_01011_01101_011_01110_0101111, 0xe0b6b72f, |rd, rs1, rs2| {
                Inst::AmomaxuD { rd, rs1, rs2 }
            }),
        ];
        for (binary, hex, expect) in cases {
            assert_eq!(decode_checked(*binary, *hex).unwrap(), expect(14, 13, 11));
        }
        // The aq/rl hint bits decode fine (amoswap.w.aq.rl per objdump).
        assert_eq!(
            decode_checked(0b00001_1_1_01011_01101_010_01110_0101111, 0x0eb6a72f).unwrap(),
            Inst::AmoswapW { rd: 14, rs1: 13, rs2: 11 }
        );
    }

    #[test]
    fn decodes_lr_sc() {
        // AMO: funct5_aq_rl_rs2_rs1_funct3_rd_opcode
        // lr.w a4, (a0) / sc.w a4, a5, (a0) straight from rv64ua-p-lrsc;
        // the .d forms are hand-assembled (funct3 011).
        assert_eq!(
            decode_checked(0b00010_0_0_00000_01010_010_01110_0101111, 0x1005272f).unwrap(),
            Inst::LrW { rd: 14, rs1: 10 }
        );
        assert_eq!(
            decode_checked(0b00011_0_0_01111_01010_010_01110_0101111, 0x18f5272f).unwrap(),
            Inst::ScW { rd: 14, rs1: 10, rs2: 15 }
        );
        assert_eq!(
            decode_checked(0b00010_0_0_00000_01010_011_01110_0101111, 0x1005372f).unwrap(),
            Inst::LrD { rd: 14, rs1: 10 }
        );
        assert_eq!(
            decode_checked(0b00011_0_0_01111_01010_011_01110_0101111, 0x18f5372f).unwrap(),
            Inst::ScD { rd: 14, rs1: 10, rs2: 15 }
        );
        // LR's rs2 field is hardwired zero: anything else is reserved.
        assert_eq!(
            decode_checked(0b00010_0_0_00001_01010_010_01110_0101111, 0x1015272f),
            Err(Exception::IllegalInstruction(0x1015272f))
        );
    }

    #[test]
    fn decodes_branches() {
        // B-type: imm[12]_imm[10:5]_rs2_rs1_funct3_imm[4:1]_imm[11]_opcode
        // beq t5, t6, +0x30  (the tohost check loop; t5/t6 = x30/x31;
        // 0x30 = 0b0110000 → [10:5] = 000001, [4:1] = 1000)
        assert_eq!(
            decode_checked(0b0_000001_11111_11110_000_1000_0_1100011, 0x03ff0863).unwrap(),
            Inst::Beq { rs1: 30, rs2: 31, offset: 0x30 }
        );
        // bne a4, t2, +0x4e0  (jump to <fail>; 0x4e0 → [10:5] = 100111)
        assert_eq!(
            decode_checked(0b0_100111_00111_01110_001_0000_0_1100011, 0x4e771063).unwrap(),
            Inst::Bne { rs1: 14, rs2: 7, offset: 0x4e0 }
        );
        // beq x1, x2, -8  (hand-assembled: backward branch → the sign bit
        // inst[31] and the upper immediate bits are all ones)
        assert_eq!(
            decode_checked(0b1_111111_00010_00001_000_1100_1_1100011, 0xfe208ce3).unwrap(),
            Inst::Beq { rs1: 1, rs2: 2, offset: -8 }
        );
        // funct3=2 is a hole in the BRANCH opcode
        assert_eq!(
            decode_checked(0b0_000000_00000_00000_010_0000_0_1100011, 0x00002063),
            Err(Exception::IllegalInstruction(0x00002063))
        );
    }

    #[test]
    fn decodes_loads() {
        // I-type: imm[11:0]_rs1_funct3_rd_opcode
        // Hand-assembled: ld x1, 8(x2) / lbu x1, 0(x2) / lw x1, -4(x2).
        // funct3 = width log2 (011 = 8 bytes), bit 2 = zero-extend (100 = lbu).
        assert_eq!(
            decode_checked(0b000000001000_00010_011_00001_0000011, 0x00813083).unwrap(),
            Inst::Ld { rd: 1, rs1: 2, offset: 8 }
        );
        assert_eq!(
            decode_checked(0b000000000000_00010_100_00001_0000011, 0x00014083).unwrap(),
            Inst::Lbu { rd: 1, rs1: 2, offset: 0 }
        );
        assert_eq!(
            decode_checked(0b111111111100_00010_010_00001_0000011, 0xffc12083).unwrap(),
            Inst::Lw { rd: 1, rs1: 2, offset: -4 }
        );
        // funct3=7 would be a 128-bit load: illegal on RV64
        assert_eq!(
            decode_checked(0b000000000000_00010_111_00001_0000011, 0x00017083),
            Err(Exception::IllegalInstruction(0x00017083))
        );
    }

    #[test]
    fn decodes_stores() {
        // S-type: imm[11:5]_rs2_rs1_funct3_imm[4:0]_opcode
        // sw gp, -60(t5) — the riscv-tests tohost result write (gp = x3,
        // t5 = x30; -60 = 0b111111000100 splits into 1111110 / 00100)
        assert_eq!(
            decode_checked(0b1111110_00011_11110_010_00100_0100011, 0xfc3f2223).unwrap(),
            Inst::Sw { rs1: 30, rs2: 3, offset: -60 }
        );
        // Hand-assembled: sd x3, 16(x4)  (16 = 0000000_10000: the split is
        // invisible for small offsets — imm[11:5] is simply zero)
        assert_eq!(
            decode_checked(0b0000000_00011_00100_011_10000_0100011, 0x00323823).unwrap(),
            Inst::Sd { rs1: 4, rs2: 3, offset: 16 }
        );
    }

    #[test]
    fn decodes_fences() {
        // MISC-MEM: fm_pred_succ_rs1_funct3_rd_opcode
        // fence iorw, iorw — as emitted by riscv-tests: pred and succ both
        // name all four access kinds (i/o/r/w), fm = 0000.
        assert_eq!(
            decode_checked(0b0000_1111_1111_00000_000_00000_0001111, 0x0ff0000f).unwrap(),
            Inst::Fence
        );
        // fence.i (Zifencei): funct3=1, every other field zero (reserved).
        assert_eq!(
            decode_checked(0b0000_0000_0000_00000_001_00000_0001111, 0x0000100f).unwrap(),
            Inst::FenceI
        );
        // funct3=2 has no fence assigned
        assert_eq!(
            decode_checked(0b0000_0000_0000_00000_010_00000_0001111, 0x0000200f),
            Err(Exception::IllegalInstruction(0x0000200f))
        );
    }

    #[test]
    fn decodes_csr_instructions() {
        // Zicsr (I-type): csr_rs1_funct3_rd_opcode
        // csrr a0, mhartid — csrrs with rs1=x0 (read, set nothing);
        // csr address 0xf14 sits in the immediate field.
        assert_eq!(
            decode_checked(0b111100010100_00000_010_01010_1110011, 0xf1402573).unwrap(),
            Inst::Csrrs { rd: 10, rs1: 0, csr: 0xf14 }
        );
        // csrw mtvec, t0 — csrrw with rd=x0 (write, discard old value)
        assert_eq!(
            decode_checked(0b001100000101_00101_001_00000_1110011, 0x30529073).unwrap(),
            Inst::Csrrw { rd: 0, rs1: 5, csr: 0x305 }
        );
    }

    #[test]
    fn decodes_ecall_and_ebreak() {
        // SYSTEM funct3=0: funct12_rs1_funct3_rd_opcode
        // Everything but funct12 is hardwired zero; ecall/ebreak differ
        // in the single bit 20.
        assert_eq!(
            decode_checked(0b000000000000_00000_000_00000_1110011, 0x00000073).unwrap(),
            Inst::Ecall
        );
        assert_eq!(
            decode_checked(0b000000000001_00000_000_00000_1110011, 0x00100073).unwrap(),
            Inst::Ebreak
        );
    }

    #[test]
    fn decodes_mret() {
        // mret: funct12 = 0011000_00010 (MRET is funct7=0011000, rs2=00010)
        assert_eq!(
            decode_checked(0b0011000_00010_00000_000_00000_1110011, 0x30200073).unwrap(),
            Inst::Mret
        );
        // Same shape, different funct7: SRET (0001000) and WFI (rs2=00101)
        // are not implemented yet and must stay illegal, not alias to MRET.
        assert_eq!(
            decode_checked(0b0001000_00010_00000_000_00000_1110011, 0x10200073),
            Err(Exception::IllegalInstruction(0x10200073))
        );
        assert_eq!(
            decode_checked(0b0001000_00101_00000_000_00000_1110011, 0x10500073),
            Err(Exception::IllegalInstruction(0x10500073))
        );
    }

    #[test]
    fn rejects_write_to_read_only_csr() {
        // csrrw x0, mhartid, x0: mhartid (0xf14) has address bits [11:10]
        // = 11 (visible as the leading two bits of the csr field) →
        // read-only, and CSRRW always writes.
        assert_eq!(
            decode_checked(0b111100010100_00000_001_00000_1110011, 0xf1401073),
            Err(Exception::IllegalInstruction(0xf1401073))
        );
        // ...but the pure read (csrrs with rs1=x0, f3=010) is fine.
        assert!(decode(0xf1402573).is_ok());
    }

    /// The compressed sibling of decode_checked: the binary shows the
    /// field split, the hex is the word binutils assembled.
    fn decode_compressed_checked(binary: u16, hex: u16) -> Result<Inst, Exception> {
        assert_eq!(binary, hex, "field split does not match the assembled word");
        decode_compressed(binary)
    }

    #[test]
    fn decodes_compressed_quadrant0() {
        // CIW: funct3_uimm[5:4]_uimm[9:6]_uimm[2]_uimm[3]_rd'_op
        // c.addi4spn a0, sp, 40 (all hex words in this test assembled by
        // riscv64-elf-gcc from the c.* mnemonics)
        assert_eq!(
            decode_compressed_checked(0b000_10_0000_0_1_010_00, 0x1028).unwrap(),
            Inst::Addi { rd: 10, rs1: 2, imm: 40 }
        );
        // The all-zero parcel is nzuimm=0 here and defined illegal:
        // jumping into zeroed memory must trap.
        assert_eq!(decode_compressed(0x0000), Err(Exception::IllegalInstruction(0)));
        // CL: funct3_uimm[5:3]_rs1'_uimm[2]_uimm[6]_rd'_op
        // c.lw a1, 4(a0)
        assert_eq!(
            decode_compressed_checked(0b010_000_010_1_0_011_00, 0x414c).unwrap(),
            Inst::Lw { rd: 11, rs1: 10, offset: 4 }
        );
        // CL doubleword: funct3_uimm[5:3]_rs1'_uimm[7:6]_rd'_op
        // c.ld a2, 8(a0)
        assert_eq!(
            decode_compressed_checked(0b011_001_010_00_100_00, 0x6510).unwrap(),
            Inst::Ld { rd: 12, rs1: 10, offset: 8 }
        );
        // CS: same layouts with rd' read as rs2'
        // c.sw a1, 12(a0)
        assert_eq!(
            decode_compressed_checked(0b110_001_010_1_0_011_00, 0xc54c).unwrap(),
            Inst::Sw { rs1: 10, rs2: 11, offset: 12 }
        );
        // c.sd a2, 16(a0)
        assert_eq!(
            decode_compressed_checked(0b111_010_010_00_100_00, 0xe910).unwrap(),
            Inst::Sd { rs1: 10, rs2: 12, offset: 16 }
        );
    }

    #[test]
    fn decodes_compressed_quadrant2() {
        // CI: funct3_shamt[5]_rd_shamt[4:0]_op
        // c.slli a0, 3
        assert_eq!(
            decode_compressed_checked(0b000_0_01010_00011_10, 0x050e).unwrap(),
            Inst::Slli { rd: 10, rs1: 10, shamt: 3 }
        );
        // c.lwsp a1, 8(sp): funct3_uimm[5]_rd_uimm[4:2]_uimm[7:6]_op
        assert_eq!(
            decode_compressed_checked(0b010_0_01011_010_00_10, 0x45a2).unwrap(),
            Inst::Lw { rd: 11, rs1: 2, offset: 8 }
        );
        // c.ldsp a2, 16(sp): funct3_uimm[5]_rd_uimm[4:3]_uimm[8:6]_op
        assert_eq!(
            decode_compressed_checked(0b011_0_01100_10_000_10, 0x6642).unwrap(),
            Inst::Ld { rd: 12, rs1: 2, offset: 16 }
        );
        // c.lwsp to x0 is reserved
        assert_eq!(
            decode_compressed_checked(0b010_0_00000_010_00_10, 0x4022),
            Err(Exception::IllegalInstruction(0x4022))
        );
        // CSS: funct3_uimm[5:2]_uimm[7:6]_rs2_op
        // c.swsp a1, 24(sp)
        assert_eq!(
            decode_compressed_checked(0b110_0110_00_01011_10, 0xcc2e).unwrap(),
            Inst::Sw { rs1: 2, rs2: 11, offset: 24 }
        );
        // c.sdsp a2, 32(sp): funct3_uimm[5:3]_uimm[8:6]_rs2_op
        assert_eq!(
            decode_compressed_checked(0b111_100_000_01100_10, 0xf032).unwrap(),
            Inst::Sd { rs1: 2, rs2: 12, offset: 32 }
        );
    }

    #[test]
    fn decodes_compressed_jumps_and_register_ops() {
        // funct3=100 of quadrant 2: bit 12 and the zero-ness of the two
        // register fields split it five ways.
        // CR: funct3_bit12_rs1/rd_rs2_op
        // c.jr a0 = jalr x0, 0(a0)
        assert_eq!(
            decode_compressed_checked(0b100_0_01010_00000_10, 0x8502).unwrap(),
            Inst::Jalr { rd: 0, rs1: 10, offset: 0 }
        );
        // c.jalr a0 = jalr x1, 0(a0) — the link is pc+2, handled by len
        assert_eq!(
            decode_compressed_checked(0b100_1_01010_00000_10, 0x9502).unwrap(),
            Inst::Jalr { rd: 1, rs1: 10, offset: 0 }
        );
        // c.mv a1, a2 = add a1, x0, a2
        assert_eq!(
            decode_compressed_checked(0b100_0_01011_01100_10, 0x85b2).unwrap(),
            Inst::Add { rd: 11, rs1: 0, rs2: 12 }
        );
        // c.add a1, a2 = add a1, a1, a2
        assert_eq!(
            decode_compressed_checked(0b100_1_01011_01100_10, 0x95b2).unwrap(),
            Inst::Add { rd: 11, rs1: 11, rs2: 12 }
        );
        // c.ebreak
        assert_eq!(
            decode_compressed_checked(0b100_1_00000_00000_10, 0x9002).unwrap(),
            Inst::Ebreak
        );
        // c.jr x0 is reserved
        assert_eq!(
            decode_compressed_checked(0b100_0_00000_00000_10, 0x8002),
            Err(Exception::IllegalInstruction(0x8002))
        );
    }

    #[test]
    fn decodes_compressed_quadrant1_immediates() {
        // CI: funct3_imm[5]_rd_imm[4:0]_op (hex from binutils)
        // c.addiw a0, -1
        assert_eq!(
            decode_compressed_checked(0b001_1_01010_11111_01, 0x357d).unwrap(),
            Inst::Addiw { rd: 10, rs1: 10, imm: -1 }
        );
        // c.li a0, 21 = addi a0, x0, 21
        assert_eq!(
            decode_compressed_checked(0b010_0_01010_10101_01, 0x4555).unwrap(),
            Inst::Addi { rd: 10, rs1: 0, imm: 21 }
        );
        // c.lui a1, 5 — the 6-bit immediate lands at bits 17:12
        assert_eq!(
            decode_compressed_checked(0b011_0_01011_00101_01, 0x6595).unwrap(),
            Inst::Lui { rd: 11, imm: 0x5000 }
        );
        // c.addi16sp sp, -64 (rd=2 turns C.LUI's row into the sp adjuster;
        // imm scatter [9]_[4]_[6]_[8:7]_[5], scaled by 16)
        assert_eq!(
            decode_compressed_checked(0b011_1_00010_0111_0_01, 0x7139).unwrap(),
            Inst::Addi { rd: 2, rs1: 2, imm: -64 }
        );
        // c.addiw with rd=0 is reserved (RV32 would put C.JAL here)
        assert_eq!(
            decode_compressed_checked(0b001_1_00000_11111_01, 0x307d),
            Err(Exception::IllegalInstruction(0x307d))
        );
    }

    #[test]
    fn decodes_compressed_quadrant1_arithmetic() {
        // CB-form arithmetic: funct3_shamt[5]_funct2_rs1'_shamt[4:0]_op
        // c.srli a1, 4 / c.srai a1, 4 / c.andi a1, -9
        assert_eq!(
            decode_compressed_checked(0b100_0_00_011_00100_01, 0x8191).unwrap(),
            Inst::Srli { rd: 11, rs1: 11, shamt: 4 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_0_01_011_00100_01, 0x8591).unwrap(),
            Inst::Srai { rd: 11, rs1: 11, shamt: 4 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_1_10_011_10111_01, 0x99dd).unwrap(),
            Inst::Andi { rd: 11, rs1: 11, imm: -9 }
        );
        // CA-form: funct3_bit12_11_rd'_funct2_rs2'_op — all rd op= rs2
        // c.sub / c.xor / c.or / c.and a1, a2
        assert_eq!(
            decode_compressed_checked(0b100_0_11_011_00_100_01, 0x8d91).unwrap(),
            Inst::Sub { rd: 11, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_0_11_011_01_100_01, 0x8db1).unwrap(),
            Inst::Xor { rd: 11, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_0_11_011_10_100_01, 0x8dd1).unwrap(),
            Inst::Or { rd: 11, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_0_11_011_11_100_01, 0x8df1).unwrap(),
            Inst::And { rd: 11, rs1: 11, rs2: 12 }
        );
        // bit 12 = 1 selects the W forms: c.subw / c.addw a1, a2
        assert_eq!(
            decode_compressed_checked(0b100_1_11_011_00_100_01, 0x9d91).unwrap(),
            Inst::Subw { rd: 11, rs1: 11, rs2: 12 }
        );
        assert_eq!(
            decode_compressed_checked(0b100_1_11_011_01_100_01, 0x9db1).unwrap(),
            Inst::Addw { rd: 11, rs1: 11, rs2: 12 }
        );
        // (1, 10) and (1, 11) are reserved
        assert_eq!(
            decode_compressed_checked(0b100_1_11_011_10_100_01, 0x9dd1),
            Err(Exception::IllegalInstruction(0x9dd1))
        );
    }

    #[test]
    fn decodes_compressed_quadrant1_control_flow() {
        // CJ: funct3_[11]_[4]_[9:8]_[10]_[6]_[7]_[3:1]_[5]_op
        // c.j . (offset 0: every immediate bit clear)
        assert_eq!(
            decode_compressed_checked(0b101_0_0_00_0_0_0_000_0_01, 0xa001).unwrap(),
            Inst::Jal { rd: 0, offset: 0 }
        );
        // CB: funct3_[8]_[4:3]_rs1'_[7:6]_[2:1]_[5]_op
        // c.beqz a1, -2 / c.bnez a1, -4 (backward: sign bit set)
        assert_eq!(
            decode_compressed_checked(0b110_1_11_011_11_11_1_01, 0xddfd).unwrap(),
            Inst::Beq { rs1: 11, rs2: 0, offset: -2 }
        );
        assert_eq!(
            decode_compressed_checked(0b111_1_11_011_11_10_1_01, 0xfdf5).unwrap(),
            Inst::Bne { rs1: 11, rs2: 0, offset: -4 }
        );
    }

    #[test]
    fn decodes_compressed_addi() {
        // CI format: funct3_imm[5]_rd_imm[4:0]_op
        // c.nop = c.addi x0, 0 — the canonical 16-bit no-op.
        assert_eq!(
            decode_compressed(0b000_0_00000_00000_01).unwrap(),
            Inst::Addi { rd: 0, rs1: 0, imm: 0 }
        );
        // c.addi a0, -3 (rd doubles as rs1; imm6 sign-extends: 111101 = -3)
        assert_eq!(
            decode_compressed(0b000_1_01010_11101_01).unwrap(),
            Inst::Addi { rd: 10, rs1: 10, imm: -3 }
        );
        // c.fld's slot stays illegal until the D extension arrives.
        assert_eq!(
            decode_compressed(0b001_0_00000_00000_00),
            Err(Exception::IllegalInstruction(0b001_0_00000_00000_00))
        );
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
