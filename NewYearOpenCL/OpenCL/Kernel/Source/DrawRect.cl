__kernel void drawRectangle(__global uchar *d_image, const int width,
                            const int height, const int x_1, const int y_1,
                            const int x_2, const int y_2, const int thickness,
                            uchar3 board_color, uchar3 fill_color,
                            const unsigned int channels, const int fill,
                            const int sine_waves_board, const float frequency) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int x1 = min(x_1, x_2);
    const int y1 = min(y_1, y_2);

    const int x2 = max(x_1, x_2);
    const int y2 = max(y_1, y_2);

    const int index = (y * width + x) * channels;

    if ((x1 <= x && x <= x2) && (y1 <= y && y <= y2)) {
        // Draw Area

        // Fill the rectangle
        if (thickness <= 0) {
            if (fill != 0) {
                d_image[index + 0] = fill_color.x;
                d_image[index + 1] = fill_color.y;
                d_image[index + 2] = fill_color.z;
            }
            return;
        }

        const int fill_x1 = x1 + thickness;
        const int fill_y1 = y1 + thickness;

        const int fill_x2 = x2 - thickness;
        const int fill_y2 = y2 - thickness;

        if ((fill_x1 <= x && x <= fill_x2) && (fill_y1 <= y && y <= fill_y2)) {
            // Fill Area
            if (fill != 0) {
                d_image[index + 0] = fill_color.x;
                d_image[index + 1] = fill_color.y;
                d_image[index + 2] = fill_color.z;
            }
        } else {
            // Bord Area
            uchar3 current_board_color = board_color;

            if (sine_waves_board != 0) {
                // Draw sine waves along the four edges of the rectangle
                float sineValue = 0.0;

                // X Fixed Vertical Line
                if ((x1 <= x && x <= x1 + thickness) ||
                    (x2 - thickness <= x && x <= x2)) {
                    sineValue = sin((float)y * frequency);
                }

                // Y Fixed Horizontal Line
                if ((y1 <= y && y <= y1 + thickness) ||
                    (y2 - thickness <= y && y <= y2)) {
                    sineValue = sin((float)x * frequency);
                }

                // Normalize the sine value to the range [0, 1]
                sineValue += 1.0;
                sineValue /= 2.0;

                const int color_x =
                    (int)(sineValue * (float)current_board_color.x);
                const int color_y =
                    (int)(sineValue * (float)current_board_color.y);
                const int color_z =
                    (int)(sineValue * (float)current_board_color.z);

                current_board_color = (uchar3)(color_x, color_y, color_z);
            }

            d_image[index + 0] = current_board_color.x;
            d_image[index + 1] = current_board_color.y;
            d_image[index + 2] = current_board_color.z;
        }
    }
}
