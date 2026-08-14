# RISC-V Emulator (written in Rust, for learning)

## 1. Project positioning (most important)

- This is a **hobby project for learning and savoring low-level programming, free from any obligation to be practical**. It is acceptable from the outset that it "ends as mere study."
- The goal is learning the low layers: dialogue with the ISA specification, decoder design with enums and exhaustive `match`, verification against deterministic oracles (official tests and reference implementations), and understanding the mechanisms an OS requires from hardware.
- **Stopping at any milestone counts as "complete."** Getting off early is not a failure. Proposals that undermine this premise (pressure to ship faster at the cost of understanding) are unnecessary.
- The repository is **public**, and the plan is to eventually **publish the crate to crates.io**. The `publish = false` guard in Cargo.toml stays until the first release is actually cut; before that release, add a license (Rust-conventional MIT OR Apache-2.0 dual) and crate metadata. Publishing is a byproduct, not the goal — learning pace still comes first.
- Being public, **never include security-sensitive content**: credentials, API keys, tokens, personal information, local environment details, etc. must not appear in code, documentation, or commit messages.

## 2. What we are building

A RISC-V emulator **from scratch in Rust**. No emulation frameworks. The core is `loop { fetch → decode → execute }` with exhaustive `match`. Start as an interpreter (JIT is a future optional boss).

The decoder uses a **two-stage design**: `enum Inst` (one-to-one with the instruction list in the spec) + a pure function `decode(raw: u32) -> Result<Inst, Exception>` + `execute(cpu, inst)`. This buys a unit-testable decoder, a free disassembler via `Display`, and `match` exhaustiveness checking.

### Milestone staircase

| # | Content | Boss fight (completion criteria) |
|---|---------|----------------------------------|
| 1 | RV64I core + syscall emulation (statically linked ELF) | hello world runs |
| 2 | M / A / C extensions + Zicsr (CSR) | realistic compiler output runs |
| 3 | Privileged modes (M/S), traps, CLINT (timer), UART | runs interrupt-driven |
| 4 | Sv39 MMU (page table walk) | xv6-riscv boots to the shell |
| 5 | virtio-blk, PLIC, device tree | Linux boots |

## 3. Verification policy

- Run **riscv-tests** locally / in CI from the very beginning (stage 1).
- **Diff testing against Spike (the reference simulator)**: run the same ELF on both, compare architectural state (PC, registers, CSRs) instruction by instruction, and pinpoint the first diverging instruction. This is the primary debugging weapon.
- Mid-project onwards: conformance testing with RISCOF + riscv-arch-test.
- Mid-project reward: integrate the **gdbstub** crate so GDB breakpoints work on guest programs.

## 4. Environment & tools (macOS / Apple Silicon)

- Host: macOS. The emulator itself is platform-independent pure Rust.
- Cross compiler: `riscv64-elf-gcc` (homebrew-core, no newlib; used to build riscv-tests).
- Reference: Spike (`riscv-isa-sim` from the `riscv-software-src/riscv` tap).
- Rust targets: `riscv64gc-unknown-none-elf` (bare metal), `riscv64gc-unknown-linux-musl` (static Linux binaries — primary candidate for hello world).
- Source repos: riscv-tests, riscv-opcodes (for checking the decoder), xv6-riscv.
- Candidate crates: `object` (or `goblin`) for ELF loading, `gdbstub` for debugging, `vm-fdt` for the Linux boot stage.

### Rules for adding dependencies

- crates.io dependencies are **restricted to versions published at least 14 days ago** (supply-chain mitigation: keeps a freshly published, possibly compromised release out of Cargo.lock until it has had time to be vetted or yanked). Configured via `min-publish-age` in `.cargo/config.toml` (checked into the repository).
- After adding or updating a dependency, regenerate the lockfile with the cooldown applied: `cargo +nightly generate-lockfile`. The feature only works on nightly (>= 2026-06-21); stable cargo silently ignores it (harmless).

## 5. How to work with Claude Code (must follow)

- **Claude Code drives the implementation in fine-grained steps.** Each step stays small enough to read and understand, and comes with an explanation of what was built and why.
- **The user asks questions about anything unclear.** This question-and-explanation loop is the primary learning path; Claude Code answers thoroughly, down to the relevant spec sections and design background.
- At major design forks (shape of trap handling, MMU structure, ...), briefly present the options and a recommendation before proceeding.
- Prefer "a pace understanding can keep up with" over "the fastest path to something working." The learning purpose stays unchanged.

## 6. Language

- Documentation, code comments, and commit messages are written in **English**.

## 7. Act two (optional, after the emulator)

- Long-term goal: run binaries produced by the homemade backend on the homemade emulator.
- Further optional boss: JIT compilation (cranelift / dynasm-rs).

## 8. Current status

- **Phase 0 (environment setup) done** (2026-07-07). See `docs/setup.md` for the procedure and known limitations.
  - riscv64-elf-gcc 16.1.0 / Spike 1.1.1-dev / two Rust riscv targets installed
  - 199 `-p` variant tests of riscv-tests (`vendor/riscv-tests`) built and smoke-tested with Spike
  - Limitation: Homebrew's gcc has no newlib, so `-v` variant tests cannot be built (not needed until phase 3)
- **Phase 1 (milestone 1) done** (2026-08-11): cargo skeleton, ELF loader (`loader.rs`), exceptions (`exception.rs`), Bus (`bus.rs`, 128 MiB DRAM @ 0x8000_0000), Cpu (`cpu.rs`), and the two-stage decoder (`inst.rs`, with a `Display` disassembler). Implemented so far: ADDI, AUIPC, JAL, the six branches (B-type), the six Zicsr instructions (real semantics, `csrs: [u64; 4096]` storage; static read-only-CSR write check in decode), OP-IMM shifts (SLLI/SRLI/SRAI, 6-bit shamt), the OP-IMM-32 W-family (ADDIW/SLLIW/SRLIW/SRAIW), a minimal MRET (pc = mepc only; privilege/interrupt state waits for phase 3), the OP group (R-type: ADD/SUB/SLL/SLT/SLTU/XOR/SRL/SRA/OR/AND), OP-32 (ADDW/SUBW/SLLW/SRLW/SRAW), LUI, FENCE (correctly a no-op for an in-order single-hart interpreter), the eleven loads/stores (LB/LH/LW/LBU/LHU/LWU/LD, SB/SH/SW/SD; S-type immediate; access faults propagate before pc commits), the OP-IMM comparisons/logic (SLTI/SLTIU/XORI/ORI/ANDI, filling the funct3 gaps), JALR (bit 0 of the target cleared; rd written after reading rs1), and FENCE.I (Zifencei; a no-op like FENCE since fetch always reads DRAM — verified by a self-modifying-code test). The RV64I base integer instruction set is complete. ECALL/EBREAK with a shared trap dispatcher (`Cpu::trap`: mepc/mcause/mtval + jump to mtvec; mstatus shuffle waits for phase 3). The tohost pass/fail harness (`harness.rs`, task #9) polls the tohost symbol's address after every step and reports Pass / Fail{testnum} / escaped Exception / StepLimit; `tests/riscv_tests.rs` sweeps the whole official rv64ui-p directory via cargo test: all 54 tests pass. Linux user-mode emulation (`linux.rs`, qemu-user style): the run loop intercepts ECALL before execute and answers write (with EBADF/EFAULT/EIO errno semantics) / exit / exit_group itself; unknown syscalls stop the run. main.rs picks the environment by ELF shape (tohost symbol → riscv-tests trace mode, otherwise → Linux mode). The boss fight is won: `guest/hello.s` (raw syscalls, RV64I only, linked at DRAM_BASE; build command in its header; binary gitignored) prints "Hello, world!" and exits 0, verified by `tests/hello_world.rs`. Spike diff testing (task #10, `difftest.rs` + `src/bin/difftest.rs`): parses `spike --misaligned -l --log-commits` into Commit/Trap events (--misaligned because our Bus handles misaligned accesses in hardware, an implementation-defined choice), skips Spike's boot ROM, then steps our Cpu in lockstep comparing pc and register writebacks; first divergence is reported with disassembled context. Comparison stops (successfully) at the final ecall because Spike drops rv64ui tests to U-mode, so mcause reads 8 vs our M-mode-only 11 — resolving that waits for phase 3 privilege modes. All 54 rv64ui-p tests run in lockstep with Spike.
- **Phase 2 in progress**: the M-extension multiplies (MUL/MULH/MULHSU/MULHU on OP funct7=0000001, MULW on OP-32; 128-bit products via i128/u128, one half per instruction). rv64um-p-mul* all pass and run in lockstep with Spike.
- Next: the divide half of M (DIV/DIVU/REM/REMU + DIVW/DIVUW/REMW/REMUW, funct3 4-7; div-by-zero returns -1/all-ones and overflow MIN/-1 returns MIN per spec — no traps), then widen the rv64um sweep to the whole directory. After M: A, then C (which real riscv64gc Linux binaries need), plus the rest of Zicsr. Commit at each step boundary.
