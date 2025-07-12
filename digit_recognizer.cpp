#include <SDL2/SDL.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

// Constants
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

    double relu(double x) {
        return std::max(0.0, x);
    }

    void softmax(std::vector<double>& z) {
        double max_z = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        
        for (auto& val : z) {
            val = std::exp(val - max_z);
            sum += val;
        }
        
        for (auto& val : z) {
            val /= sum;
        }
    }

public:
    NeuralNetwork() {
        // Initialize with correct sizes (matches Python model)
        weights_input_hidden.resize(HIDDEN_SIZE, std::vector<double>(CANVAS_WIDTH * CANVAS_HEIGHT));
        bias_hidden.resize(HIDDEN_SIZE);
        weights_hidden_output.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        bias_output.resize(OUTPUT_SIZE);
    }

    bool load_weights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening weights file" << std::endl;
            return false;
        }

        // Read weights in exact order saved from Python
        for (auto& row : weights_input_hidden) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        }
        file.read(reinterpret_cast<char*>(bias_hidden.data()), bias_hidden.size() * sizeof(double));
        
        for (auto& row : weights_hidden_output) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        }
        file.read(reinterpret_cast<char*>(bias_output.data()), bias_output.size() * sizeof(double));

        return !file.fail();
    }

    std::vector<double> forward(const std::vector<double>& input) {
        // Normalize input to [0,1] like Python did
        std::vector<double> normalized = input;
        double max_val = *std::max_element(normalized.begin(), normalized.end());
        if (max_val > 0) {
            for (auto& val : normalized) {
                val /= max_val;
            }
        }

        // Hidden layer
        std::vector<double> hidden(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < normalized.size(); ++j) {
                hidden[i] += weights_input_hidden[i][j] * normalized[j];
            }
            hidden[i] += bias_hidden[i];
            hidden[i] = relu(hidden[i]);
        }

        // Output layer
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                output[i] += weights_hidden_output[i][j] * hidden[j];
            }
            output[i] += bias_output[i];
        }

        softmax(output);
        return output;
    }

    int predict(const std::vector<double>& input) {
        auto output = forward(input);
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }
};

class DigitDrawer {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    std::vector<std::vector<double>> canvas; // Using double for intensity
    NeuralNetwork nn;

    void clear_canvas() {
        canvas.assign(WINDOW_HEIGHT, std::vector<double>(WINDOW_WIDTH, 0.0));
    }

    void draw_brush(int x, int y, double intensity = 1.0) {
        // Draw with soft edges (better for recognition)
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < WINDOW_WIDTH && ny >= 0 && ny < WINDOW_HEIGHT) {
                    double distance = sqrt(dx*dx + dy*dy);
                    double new_intensity = intensity * std::max(0.0, 1.0 - distance/4.0);
                    canvas[ny][nx] = std::max(canvas[ny][nx], new_intensity);
                }
            }
        }
    }

    std::vector<double> downsample_image() {
        std::vector<double> small_image(CANVAS_WIDTH * CANVAS_HEIGHT, 0.0);
        
        double scale_x = static_cast<double>(WINDOW_WIDTH) / CANVAS_WIDTH;
        double scale_y = static_cast<double>(WINDOW_HEIGHT) / CANVAS_HEIGHT;
        
        for (int y = 0; y < CANVAS_HEIGHT; ++y) {
            for (int x = 0; x < CANVAS_WIDTH; ++x) {
                int start_x = static_cast<int>(x * scale_x);
                int start_y = static_cast<int>(y * scale_y);
                int end_x = static_cast<int>((x + 1) * scale_x);
                int end_y = static_cast<int>((y + 1) * scale_y);
                
                double sum = 0.0;
                for (int sy = start_y; sy < end_y; ++sy) {
                    for (int sx = start_x; sx < end_x; ++sx) {
                        sum += canvas[sy][sx];
                    }
                }
                
                small_image[y * CANVAS_WIDTH + x] = sum / ((end_x - start_x) * (end_y - start_y));
            }
        }
        
        return small_image;
    }

public:
    DigitDrawer() {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Digit Recognizer", 
                                SDL_WINDOWPOS_CENTERED, 
                                SDL_WINDOWPOS_CENTERED,
                                WINDOW_WIDTH, WINDOW_HEIGHT, 
                                SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        clear_canvas();
        
        if (!nn.load_weights("mnist_weights.bin")) {
            std::cerr << "Failed to load weights. Using random weights instead." << std::endl;
        }
    }

    ~DigitDrawer() {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void run() {
        bool running = true;
        bool drawing = false;
        SDL_Event event;
        
        while (running) {
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        running = false;
                        break;
                        
                    case SDL_MOUSEBUTTONDOWN:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            drawing = true;
                            draw_brush(event.button.x, event.button.y);
                        }
                        break;
                        
                    case SDL_MOUSEBUTTONUP:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            drawing = false;
                            auto image = downsample_image();
                            int prediction = nn.predict(image);
                            std::cout << "Predicted: " << prediction << std::endl;
                            
                            // Display probabilities
                            auto probs = nn.forward(image);
                            for (int i = 0; i < probs.size(); ++i) {
                                printf("%d: %.1f%%  ", i, probs[i]*100);
                            }
                            std::cout << "\n";
                        }
                        break;
                        
                    case SDL_MOUSEMOTION:
                        if (drawing) {
                            draw_brush(event.motion.x, event.motion.y);
                        }
                        break;
                        
                    case SDL_KEYDOWN:
                        if (event.key.keysym.sym == SDLK_c) {
                            clear_canvas();
                        }
                        break;
                }
            }
            
            // Render
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            for (int y = 0; y < WINDOW_HEIGHT; ++y) {
                for (int x = 0; x < WINDOW_WIDTH; ++x) {
                    if (canvas[y][x] > 0) {
                        Uint8 intensity = static_cast<Uint8>(canvas[y][x] * 255);
                        SDL_SetRenderDrawColor(renderer, intensity, intensity, intensity, 255);
                        SDL_RenderDrawPoint(renderer, x, y);
                    }
                }
            }
            
            SDL_RenderPresent(renderer);
        }
    }
};

int main() {
    DigitDrawer app;
    app.run();
    return 0;
}
