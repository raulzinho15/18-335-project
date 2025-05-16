
# Prepares all the matrix sizes to be used

all_sizes = []
sizes = [2**i for i in range(1,10)] + [980, 990, 1000, 1010, 1024]

for m in sizes:
    all_sizes.append((m, m, m))

sizes = [8, 48, 128]
for m in sizes:
    for n in sizes[sizes.index(m):]:
        for p in sizes[sizes.index(n):]:
            if m == n == p == 8 or m == n == p == 128:
                continue
            all_sizes.append((m,n,p))
        

for m,n,p in all_sizes:

    # Naive Matrix Multiplication
    f_add_naive = m*p*(n-1)
    f_sub_naive = 0
    f_mul_naive = m*p*n
    f_write_naive = 0
    f_read_naive = 0
    # print(f"{m} & {n} & {p} & {f_add_naive+f_sub_naive} & {f_mul_naive} & {f_read_naive} & {f_write_naive} \\\\")


    # Strassen's Algorithm
    x = (max(m,n,p)-1).bit_length()
    f_add_strass = 12 * (7**x - 1) // 6
    f_sub_strass = 6 * (7**x - 1) // 6
    f_mul_strass = 7**x
    f_write_strass = 5 * (7**x - 1) // 6
    f_read_strass = 10 * (7**x - 1) // 6
    # print(f"{m} & {n} & {p} & {x} & {f_add_strass+f_sub_strass} & {f_mul_strass} & {f_read_strass} & {f_write_strass} \\\\")


    # Winograd's Variant
    f_add_wino = 7 * (7**x - 1) // 6
    f_sub_wino = 8 * (7**x - 1) // 6
    f_mul_wino = 7**x
    f_write_wino = 8 * (7**x - 1) // 6
    f_read_wino = 16 * (7**x - 1) // 6
    # print(f"{m} & {n} & {p} & {x} & {f_add_wino+f_sub_wino} & {f_mul_wino} & {f_read_wino} & {f_write_wino} \\\\")


    # Multiplication Table
    # print(f"{m} & {n} & {p} & {f_mul_naive} & {f_mul_strass} & {f_mul_wino} \\\\")


    # Addition/Subtraction Table
    # print(f"{m} & {n} & {p} & {f_add_naive+f_sub_naive} & {f_add_strass+f_sub_strass} & {f_add_wino+f_sub_wino} \\\\")


    # Memory Table
    # print(f"{m} & {n} & {p} & {f_read_naive+f_write_naive} & {f_read_strass+f_write_strass} & {f_read_wino+f_write_wino} \\\\")