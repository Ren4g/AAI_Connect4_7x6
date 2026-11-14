COLS = 7
ROWS = 6
STRIDE = 7  # 6 ô thật + 1 bit đệm ở đỉnh mỗi cột

def debug_bits(bitboard: int, title: str = "") -> None:
    print(f"--- {title} (Giá trị số: {bitboard}) ---")
    for r in range(STRIDE - 1, -1, -1):  # In từ hàng đệm (6) xuống đáy (0)
        row_str = []
        for c in range(COLS):
            bit_index = c * STRIDE + r

            is_on = (bitboard >> bit_index) & 1
            row_str.append(str(is_on))
        label = "Đệm" if r == 6 else f"H{5 - r} "
        print(f"{label}: " + " ".join(row_str))

test_board = (1 << 0) | (1 << 14)
debug_bits(test_board, "Thử nghiệm bật 2 ô đáy")