# FizzBuzz 1..=30 for the emulator's Linux user-mode emulation.
# Side quest after implementing the M extension: REMU picks the branch,
# DIVU/REMU convert the counter to decimal digits. RV64IM only.
#
# Build (Homebrew riscv64-elf-gcc):
#   riscv64-elf-gcc -march=rv64im -mabi=lp64 -nostdlib -static \
#       -Wl,-Ttext-segment=0x80000000 -o guest/fizzbuzz guest/fizzbuzz.s
#
# Register plan: s1 = counter i, s2 = limit; t-registers are scratch;
# a-registers carry the write(2) syscall arguments.

    .global _start
    .text
_start:
    li      s1, 1               # i = 1
    li      s2, 30              # limit

loop:
    li      t0, 15
    remu    t1, s1, t0          # i % 15 == 0 -> "FizzBuzz"
    beqz    t1, fizzbuzz
    li      t0, 3
    remu    t1, s1, t0          # i % 3 == 0  -> "Fizz"
    beqz    t1, fizz
    li      t0, 5
    remu    t1, s1, t0          # i % 5 == 0  -> "Buzz"
    beqz    t1, buzz

    # Otherwise print i in decimal: peel digits off the low end with
    # remu/divu by 10, storing them right-to-left in front of a newline.
    la      t0, buf
    addi    t0, t0, 15          # t0 = one past the last digit slot
    li      t1, 10              # '\n'
    sb      t1, 0(t0)
    addi    t2, s1, 0           # t2 = value being converted
    li      t3, 10              # divisor
itoa:
    addi    t0, t0, -1
    remu    t1, t2, t3          # digit = t2 % 10
    addi    t1, t1, 48          # + '0'
    sb      t1, 0(t0)
    divu    t2, t2, t3          # t2 /= 10
    bnez    t2, itoa
    addi    a1, t0, 0           # buf = first digit
    la      t4, buf
    addi    t4, t4, 16
    sub     a2, t4, t0          # len = digits + newline
    j       print

fizz:
    la      a1, fizz_msg
    li      a2, 5
    j       print
buzz:
    la      a1, buzz_msg
    li      a2, 5
    j       print
fizzbuzz:
    la      a1, fizzbuzz_msg
    li      a2, 9

print:
    li      a7, 64              # write(1, a1, a2)
    li      a0, 1
    ecall

    addi    s1, s1, 1           # i += 1
    ble     s1, s2, loop

    li      a7, 93              # exit(0)
    li      a0, 0
    ecall

    .section .rodata
fizz_msg:
    .ascii  "Fizz\n"
buzz_msg:
    .ascii  "Buzz\n"
fizzbuzz_msg:
    .ascii  "FizzBuzz\n"

    .section .bss
buf:
    .space  16
