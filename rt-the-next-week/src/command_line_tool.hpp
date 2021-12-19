//
// Created by 유승우 on 2020/12/30.
//

#ifndef RTFOUNDATION_COMMAND_LINE_TOOL_HPP
#define RTFOUNDATION_COMMAND_LINE_TOOL_HPP

#include <time.h>

class loading_bar {

public:
    loading_bar(
            unsigned int cp, unsigned int width, unsigned int total_task
            ) {
        checkpoint = cp;
        cmd_width = width;
        tasks = total_task;
    }

    void draw(time_t took_time, unsigned int processed_lines) {
        interval = took_time != 0 ? took_time : interval;
        time_t estimated_remaining = interval * (tasks - processed_lines);

        if (static_cast<int>(processed_lines % checkpoint) == 0) {
            int cnt = 0;
            std::cerr << "\r[" << static_cast<int>((static_cast<float>(processed_lines) / tasks) * 100) << "%] |";
            for (int i = 0; i < static_cast<int>(processed_lines / checkpoint); i++) {
                std::cerr << "=";
                cnt++;
            }

            std::cerr << ">" << std::flush;

            for (int i = 0; i < cmd_width - cnt; i++) {
                std::cerr << " ";
            }

            std::cerr << "[ETA: " << estimated_remaining << " sec]" << std::flush;
        }
    }


private:
    unsigned int tasks;
    unsigned int checkpoint;
    unsigned int cmd_width;
    time_t interval;
};

#endif //FIRSTRAYTRACER_COMMAND_LINE_TOOL_HPP
