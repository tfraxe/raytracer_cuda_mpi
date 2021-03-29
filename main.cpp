#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cassert>

float* run_raytracer(int, int, float*);
float clamp(float, float, float);

int main(int argc, char* argv[]) {
    int rank, number_of_processes, error;

    error = MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    const int image_height = 400;
    const int image_width = 600;
    const int number_of_samples = 5;
    const int frame_buffer_size = 3 * image_height * image_width;

    float *sub_frame_buffer = (float*)malloc((frame_buffer_size / number_of_processes) * sizeof(float));

    run_raytracer(rank + 1, number_of_processes, sub_frame_buffer);
    assert(sub_frame_buffer != NULL);
    std::cout << "Process " << rank << " received frame buffer " << '\n';

    float* frame_buffer = NULL;
    if(rank == 0) {
        frame_buffer = (float*)malloc(frame_buffer_size  * sizeof(float));
        assert(frame_buffer != NULL);
    }

    std::cout << "Process " << rank << " is gathering " << '\n';
    MPI_Gather(sub_frame_buffer, frame_buffer_size / number_of_processes, 
            MPI_FLOAT, frame_buffer, frame_buffer_size / number_of_processes, 
            MPI_FLOAT, 0, MPI_COMM_WORLD); 
    std::cout << "Process " << rank << " is done gathering " << '\n';

    if (rank == 0) {
        std::cout << "P3\n" << image_width << ' ' << image_height << '\n' << "255\n";
        const float scale = 1.0f/number_of_samples;
        for (int j = image_height - 1; j >= 0; j--)
            for (int i = 0; i < image_width; i++) {

                int current_index = j * image_width + i; 
                float red = sqrt( scale * frame_buffer[3 * current_index + 0]);
                float green = sqrt( scale * frame_buffer[3 * current_index + 1]);
                float blue = sqrt( scale * frame_buffer[3 * current_index + 2]);
                std::cout << static_cast<int>(256 * clamp(red, 0.0f, 0.999f)) << ' '
                    << static_cast<int>(256 * clamp(green, 0.0f, 0.999f)) << ' '
                    << static_cast<int>(256 * clamp(blue, 0.0f, 0.999f)) << '\n';


            }
    }

    error = MPI_Finalize();

    return 0;
}
