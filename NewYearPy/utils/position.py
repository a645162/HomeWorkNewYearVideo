def get_start_by_center(center: (int, int), width: int, height: int):
    center_x, center_y = center
    start_x = center_x - width // 2
    start_y = center_y - height // 2
    return start_x, start_y
