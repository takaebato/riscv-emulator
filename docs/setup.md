# Environment setup (macOS / Apple Silicon)

Record of phase 0 (2026-07-07): reproducible steps and troubleshooting notes.

## Installed components

| Tool | Version | How |
|------|---------|-----|
| riscv64-elf-gcc | 16.1.0 | `brew install riscv64-elf-gcc` (pulls in riscv64-elf-binutils 2.46.1) |
| Spike (riscv-isa-sim) | 1.1.1-dev | `riscv-software-src/riscv` tap (bottled; no source build was needed) |
| Rust targets | — | `riscv64gc-unknown-none-elf`, `riscv64gc-unknown-linux-musl` |
| riscv-tests | submodule | `vendor/riscv-tests` (includes the env submodule) |

## Steps

```sh
# 1. Cross compiler (bare metal, no newlib)
brew install riscv64-elf-gcc

# 2. Spike (reference simulator)
brew tap riscv-software-src/riscv
brew trust --formula riscv-software-src/riscv/riscv-isa-sim  # Homebrew 6.x tap-trust
brew install riscv-software-src/riscv/riscv-isa-sim

# 3. Rust riscv targets
rustup target add riscv64gc-unknown-none-elf riscv64gc-unknown-linux-musl

# 4. riscv-tests (skip if the submodule is already checked out)
git submodule update --init --recursive

# 5. Build the ISA tests (rv64 only; override the toolchain prefix for Homebrew)
make -C vendor/riscv-tests/isa -k -j8 RISCV_PREFIX=riscv64-elf-
```

## Smoke test

```sh
cd vendor/riscv-tests/isa
spike rv64ui-p-add && echo PASS   # exit 0 = PASS (1 was written to tohost)

# Instruction commit log (the main weapon for diff testing); works out of the box
spike -l --log-commits rv64ui-p-add
```

Verified (2026-07-07): rv64ui-p-add / rv64ui-p-beq / rv64um-p-mul / rv64ua-p-amoadd_d /
rv64uc-p-rvc all PASS. 199 `-p` variant test binaries built.

## Known limitations & troubleshooting

### `-v` variant tests cannot be built (fine for now)

Homebrew's `riscv64-elf-gcc` is **plain gcc without newlib (libc)**, so the `-v` variants
(tests that run in a virtual-memory environment; `env/v/*.c` includes `string.h` etc.)
do not compile. We build only the pure-assembly `-p` variants, using `make -k` to skip
the failures.

- Phases 1–3 only need the `-p` variants (physical memory, M-mode), so there is no impact.
- If the `-v` variants become desirable (e.g. as an aid for phase-4 MMU verification),
  install a newlib-enabled toolchain — `brew install riscv-software-src/riscv/riscv-gnu-toolchain`
  (slow source build) or xPack's `riscv-none-elf-gcc` — and rebuild with a matching
  `RISCV_PREFIX`.

### Homebrew tap-trust

Since Homebrew 6.x, formulae from third-party taps are blocked until trusted.
We trusted the single formula (`brew trust --formula ...`) rather than the whole tap.

### Makefile prefix

riscv-tests assumes the `riscv64-unknown-elf-` toolchain prefix; Homebrew's toolchain
uses `riscv64-elf-`, so override it with `make RISCV_PREFIX=riscv64-elf-`.

## Spike notes (used in phase 1)

- Passing a test ELF directly to `spike` runs it with HTIF pass/fail detection; PASS means
  exit code 0.
- Spike starts execution in a boot ROM at 0x1000 (5 instructions: auipc → ld → jump) and
  then jumps to 0x8000_0000 (start of DRAM, where the ELF is loaded). Our emulator starts
  directly at the ELF entry point, so **diff testing must skip this preamble**.
- `--log-commits` output format: pairs of a disassembly line and a commit line carrying
  the register/memory writes.

```
core   0: 0x0000000000001000 (0x00000297) auipc   t0, 0x0
core   0: 3 0x0000000000001000 (0x00000297) x5  0x0000000000001000
```
