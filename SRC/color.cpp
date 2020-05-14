//
// Created by 유승우 on 2020/05/15.
//

#include <iostream>
#include "../Include/color.h"

void write_color(std::ostream &out, color pixel_color)
{
    // Write the translated [0, 255] value of each color component.
    out << static_cast<int>(255.999 * pixel_color.getX()) << ' '
        << static_cast<int>(255.999 * pixel_color.getY()) << ' '
        << static_cast<int>(255.999 * pixel_color.getZ()) << '\n';
}