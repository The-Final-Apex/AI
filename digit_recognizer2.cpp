#include <SDL2/SDL.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <deque>
#include <chrono>
#include <cstdlib>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int WINDOW_WIDTH = 280;
const int WINDOW_HEIGHT = 280;
const int CANVAS_WIDTH = 28;
const int CANVAS_HEIGHT = 28;
const int HIDDEN_SIZE = 128;
const int OUTPUT_SIZE = 10;

class NeuralNetwork {
private:
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<double> bias_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_output;

    double relu(double x) { return std::max(0.0, x); }

    void softmax(std::vector<double>& z) {
        double max_z = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        for (auto& val : z) { val = std::exp(val - max_z); sum += val; }
        for (auto& val : z) val /= sum;
    }

public:
    NeuralNetwork() {
        weights_input_hidden.resize(HIDDEN_SIZE, std::vector<double>(CANVAS_WIDTH * CANVAS_HEIGHT));
        bias_hidden.resize(HIDDEN_SIZE);
        weights_hidden_output.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        bias_output.resize(OUTPUT_SIZE);
    }

    bool load_weights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        for (auto& row : weights_input_hidden) file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(bias_hidden.data()), bias_hidden.size() * sizeof(double));
        for (auto& row : weights_hidden_output) file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(bias_output.data()), bias_output.size() * sizeof(double));
        return !file.fail();
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> norm = input;
        double max_val = *std::max_element(norm.begin(), norm.end());
        if (max_val > 0) for (auto& val : norm) val /= max_val;

        std::vector<double> hidden(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            hidden[i] = std::inner_product(norm.begin(), norm.end(), weights_input_hidden[i].begin(), 0.0) + bias_hidden[i];
            hidden[i] = relu(hidden[i]);
        }

        std::vector<double> output(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            output[i] = std::inner_product(hidden.begin(), hidden.end(), weights_hidden_output[i].begin(), 0.0) + bias_output[i];
        }

        softmax(output);
        return output;
    }

    int predict(const std::vector<double>& input, bool rage = false) {
        auto output = forward(input);
        if (rage) return (std::distance(output.begin(), std::max_element(output.begin(), output.end())) + rand() % 9 + 1) % 10;
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }
};

class DigitDrawer {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    std::vector<double> canvas;
    NeuralNetwork nn;
    bool drawing = false;
    bool soft_brush = true;
    bool white_brush = true;
    bool random_brush = false;
    bool rage_mode = false;
    bool freeze_mode = false;
    bool rainbow_mode = false;
    bool game_mode = false;
    bool sound_mode = false;
    double hue = 0.0;
    std::deque<int> prediction_history;
    std::chrono::time_point<std::chrono::steady_clock> game_start;

    void clear_canvas() {
        std::fill(canvas.begin(), canvas.end(), 0.0);
    }

    void update_prediction_history(int pred) {
        prediction_history.push_back(pred);
        if (prediction_history.size() > 10) prediction_history.pop_front();
    }

    void draw_brush(int x, int y, double intensity = 1.0) {
        int radius = soft_brush ? (random_brush ? rand() % 4 + 2 : 3) : 1;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < WINDOW_WIDTH && ny >= 0 && ny < WINDOW_HEIGHT) {
                    double dist = std::sqrt(dx * dx + dy * dy);
                    double value = intensity * (soft_brush ? std::max(0.0, 1.0 - dist / (radius + 1)) : 1.0);
                    if (random_brush) value = (rand() % 100) / 100.0;
                    int idx = ny * WINDOW_WIDTH + nx;
                    canvas[idx] = std::max(canvas[idx], white_brush ? value : 1.0 - value);
                }
            }
        }
        if (sound_mode) std::cout << "\a";  // beep!
    }

    std::vector<double> downsample_image() {
        std::vector<double> small(CANVAS_WIDTH * CANVAS_HEIGHT, 0.0);
        int sx = WINDOW_WIDTH / CANVAS_WIDTH;
        int sy = WINDOW_HEIGHT / CANVAS_HEIGHT;

        for (int y = 0; y < CANVAS_HEIGHT; ++y)
            for (int x = 0; x < CANVAS_WIDTH; ++x) {
                double sum = 0.0;
                for (int dy = 0; dy < sy; ++dy)
                    for (int dx = 0; dx < sx; ++dx)
                        sum += canvas[(y * sy + dy) * WINDOW_WIDTH + (x * sx + dx)];
                small[y * CANVAS_WIDTH + x] = sum / (sx * sy);
            }
        return small;
    }

    void print_ascii_preview() {
        auto img = downsample_image();
        const char* chars = "@#*+=-:. ";
        for (int y = 0; y < CANVAS_HEIGHT; ++y) {
            for (int x = 0; x < CANVAS_WIDTH; ++x) {
                double v = img[y * CANVAS_WIDTH + x];
                char c = chars[(int)(v * 7)];
                std::cout << c;
            }
            std::cout << "\n";
        }
    }

public:
    DigitDrawer() {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Digit Recognizer Chaos",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        canvas.resize(WINDOW_WIDTH * WINDOW_HEIGHT);
        clear_canvas();
        srand(time(0));
        nn.load_weights("mnist_weights.bin");
    }

    ~DigitDrawer() {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void run() {
        bool running = true;
        SDL_Event e;

        while (running) {
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_QUIT) running = false;
                if (e.type == SDL_MOUSEBUTTONDOWN) drawing = true;
                if (e.type == SDL_MOUSEBUTTONUP) drawing = false;
                if (e.type == SDL_MOUSEMOTION && drawing)
                    draw_brush(e.motion.x, e.motion.y);

                if (e.type == SDL_KEYDOWN) {
                    switch (e.key.keysym.sym) {
                        case SDLK_c: clear_canvas(); break;
                        case SDLK_r: random_brush = !random_brush; break;
                        case SDLK_b: white_brush = !white_brush; break;
                        case SDLK_t: soft_brush = !soft_brush; break;
                        case SDLK_p: print_ascii_preview(); break;
                        case SDLK_x: rage_mode = !rage_mode; break;
                        case SDLK_f: freeze_mode = !freeze_mode; break;
                        case SDLK_l: rainbow_mode = !rainbow_mode; break;
                        case SDLK_m: sound_mode = !sound_mode; break;
                        case SDLK_g: game_mode = true; game_start = std::chrono::steady_clock::now(); break;
                    }
                }
            }

            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);

            for (int y = 0; y < WINDOW_HEIGHT; ++y) {
                for (int x = 0; x < WINDOW_WIDTH; ++x) {
                    double val = canvas[y * WINDOW_WIDTH + x];
                    if (val > 0.0) {
                        Uint8 c = static_cast<Uint8>(std::clamp(val, 0.0, 1.0) * 255);
                        SDL_SetRenderDrawColor(renderer, c, c, c, 255);
                        SDL_RenderDrawPoint(renderer, x, y);
                        if (freeze_mode) canvas[y * WINDOW_WIDTH + x] *= 0.999; // slow fade
                    }
                }
            }

            SDL_RenderPresent(renderer);

            auto image = downsample_image();
            int pred = nn.predict(image, rage_mode);
            update_prediction_history(pred);
            std::cout << "\rPrediction: " << pred << " | History: ";
            for (auto p : prediction_history) std::cout << p << " ";
            std::cout << std::flush;

            if (game_mode) {
                auto now = std::chrono::steady_clock::now();
                int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - game_start).count();
                if (elapsed >= 10) {
                    std::cout << "\n⏱ Game Over! Final prediction: " << pred << "\n";
                    game_mode = false;
                }
            }

            if (pred == 6 || pred == 9) {
                auto p2 = nn.forward(image);
                if (p2[6] > 0.4 && p2[9] > 0.4)
                    std::cout << "\n🎁 Secret Unlocked: Nice. (69 detected)\n";
            }

            SDL_Delay(16); // ~60fps
        }
    }
};

int main() {
    DigitDrawer app;
    app.run();
    return 0;
}




