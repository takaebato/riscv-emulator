# Minimal "hello world" for the emulator's Linux user-mode emulation:
# raw Linux syscalls, no libc, RV64I instructions only (the C extension
# is not implemented until phase 2, hence -march=rv64i).
#
# Build (Homebrew riscv64-elf-gcc):
#   riscv64-elf-gcc -march=rv64i -mabi=lp64 -nostdlib -static \
#       -Wl,-Ttext-segment=0x80000000 -o guest/hello guest/hello.s
#
# Linked at 0x80000000 because the emulator's DRAM starts there. (Real
# Linux binaries link near 0x10000; mapping those comes later, with the
# C extension and a proper stack/auxv setup.)

    .global _start
    .text
_start:
    li      a7, 64              # write(fd, buf, len)
    li      a0, 1               #   fd  = 1 (stdout)
    la      a1, msg             #   buf   (assembles to auipc+addi)
    li      a2, 14              #   len of msg (li takes plain constants only)
    ecall

    li      a7, 93              # exit(code)
    li      a0, 0               #   code = 0 (success)
    ecall

    .section .rodata
msg:
    .ascii  "Hello, world!\n"
msg_end:
