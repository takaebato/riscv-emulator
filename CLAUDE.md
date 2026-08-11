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
- **Phase 1 in progress**: cargo skeleton, ELF loader (`loader.rs`), exceptions (`exception.rs`), Bus (`bus.rs`, 128 MiB DRAM @ 0x8000_0000), Cpu (`cpu.rs`), and the two-stage decoder (`inst.rs`, with a `Display` disassembler). Implemented so far: ADDI, AUIPC, JAL, the six branches (B-type), the six Zicsr instructions (real semantics, `csrs: [u64; 4096]` storage; static read-only-CSR write check in decode), OP-IMM shifts (SLLI/SRLI/SRAI, 6-bit shamt), the OP-IMM-32 W-family (ADDIW/SLLIW/SRLIW/SRAIW), a minimal MRET (pc = mepc only; privilege/interrupt state waits for phase 3), the OP group (R-type: ADD/SUB/SLL/SLT/SLTU/XOR/SRL/SRA/OR/AND), OP-32 (ADDW/SUBW/SLLW/SRLW/SRAW), LUI, FENCE (correctly a no-op for an in-order single-hart interpreter), the eleven loads/stores (LB/LH/LW/LBU/LHU/LWU/LD, SB/SH/SW/SD; S-type immediate; access faults propagate before pc commits), and the OP-IMM comparisons/logic (SLTI/SLTIU/XORI/ORI/ANDI, filling the funct3 gaps). ECALL/EBREAK with a shared trap dispatcher (`Cpu::trap`: mepc/mcause/mtval + jump to mtvec; mstatus shuffle waits for phase 3). The tohost pass/fail harness (`harness.rs`, task #9) polls the tohost symbol's address after every step and reports Pass / Fail{testnum} / escaped Exception / StepLimit; `tests/riscv_tests.rs` runs the official rv64ui-p suite through it via cargo test: 52 of 54 tests pass.
- Next: the 2 remaining rv64ui-p failures, both illegal-instruction on missing decodes: JALR (opcode 0x67, the last RV64I instruction) and FENCE.I (Zifencei, no-op for this interpreter). Add each to decode/execute and extend the PASSING list until it is the whole directory. Commit at each step boundary.
