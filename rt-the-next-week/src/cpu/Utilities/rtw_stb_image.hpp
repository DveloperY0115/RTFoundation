//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_RTW_STB_IMAGE_HPP
#define RTFOUNDATION_RTW_STB_IMAGE_HPP

// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Restore warning levels.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (pop)
#endif

#endif //RTFOUNDATION_RTW_STB_IMAGE_HPP
