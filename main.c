#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include <nnpack.h>

static void* malloc_with_alignment(size_t size, size_t alignment) {
	void* memory_block = NULL;
	if (posix_memalign(&memory_block, alignment, size) != 0) {
		return NULL;
	}

	return memory_block;
}

void benchmark_convolution(
	enum nnp_convolution_algorithm algorithm,
	enum nnp_convolution_transform_strategy transform_strategy,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_subsampling,
	float input[],
	float kernel[],
	const float bias[],
	float output[],
	pthreadpool_t threadpool)
{
	enum nnp_status status = nnp_status_success;
	void* memory_block = NULL;
	void* transformed_kernel = NULL;
	size_t memory_size = 0, transformed_kernel_size = 0;

	if (transform_strategy == nnp_convolution_transform_strategy_precompute) {
		status = nnp_convolution_inference(
			algorithm, transform_strategy,
			input_channels, output_channels,
			input_size, input_padding, kernel_size, output_subsampling,
			NULL, NULL, NULL, NULL, NULL, &transformed_kernel_size,
			nnp_activation_identity, NULL,
			threadpool, NULL);
		switch (status) {
			case nnp_status_success:
				break;
			case nnp_status_invalid_algorithm:
			case nnp_status_unsupported_algorithm:
				break;
			case nnp_status_unsupported_transform_strategy:
				/* Fall back to compute strategy */
				transform_strategy = nnp_convolution_transform_strategy_compute;
				break;
			default:
				fprintf(stderr, "Error: failed to detect transformed kernel size: status %d\n", status);
				exit(EXIT_FAILURE);
		}
	}
	if (transform_strategy == nnp_convolution_transform_strategy_precompute) {
		transformed_kernel = malloc_with_alignment(transformed_kernel_size, 64);
		if (transformed_kernel == NULL) {
			fprintf(stderr, "Error: failed to allocate %zu bytes for transformed kernel\n", memory_size);
			exit(EXIT_FAILURE);
		}

		status = nnp_convolution_inference(
			algorithm, transform_strategy,
			input_channels, output_channels,
			input_size, input_padding, kernel_size, output_subsampling,
			NULL, kernel, NULL, NULL, transformed_kernel, &transformed_kernel_size,
			nnp_activation_identity, NULL,
			threadpool, NULL);
		if (status != nnp_status_success) {
			fprintf(stderr, "Error: failed to pre-compute kernel transform: status %d\n", status);
			exit(EXIT_FAILURE);
		}
		transform_strategy = nnp_convolution_transform_strategy_reuse;
	}

	status = nnp_convolution_inference(
		algorithm, transform_strategy,
		input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		NULL, NULL, NULL, NULL, NULL, &memory_size,
		nnp_activation_identity, NULL,
		threadpool, NULL);

	switch (status) {
		case nnp_status_success:
			break;
		case nnp_status_invalid_algorithm:
		case nnp_status_unsupported_algorithm:
			return;
			break;
		default:
			fprintf(stderr, "Error: failed to detect workspace memory size: status %d\n", status);
			exit(EXIT_FAILURE);
	}
	if (memory_size != 0) {
		memory_block = malloc_with_alignment(memory_size, 64);
		if (memory_block == NULL) {
			fprintf(stderr, "Error: failed to allocate %zu bytes for workspace\n", memory_size);
			exit(EXIT_FAILURE);
		}
	}

	nnp_convolution_inference(
		algorithm, transform_strategy,
		input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		input, transformed_kernel == NULL ? kernel : transformed_kernel, bias, output,
		memory_block, memory_size == 0 ? NULL : &memory_size,
		nnp_activation_identity, NULL,
		threadpool,
		NULL);

	free(memory_block);

	return;
}

struct options {
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_subsampling;
	enum nnp_convolution_algorithm algorithm;
	enum nnp_convolution_transform_strategy transform_strategy;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -ic  --input-channels     The number of input channels\n"
"  -oc  --output-channels    The number of output channels\n"
"  -is  --input-size         Input height and width\n"
"  -ks  --kernel-size        Kernel height and width\n"
"Optional parameters:\n"
"  -a   --algorithm          The algorithm (auto, ft8x8, ft16x16, wt8x8, implicit-gemm, or direct) for computing convolution (default: auto)\n"
"  -ts  --transform-strategy The transformation strategy (compute, or precompute) for kernel transformation (default: compute)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -s   --output-subsampling The size of a output subsampling region, AKA stride (default: 1x1)\n"
"  -ip  --input-padding      Implicit input padding (default: 0)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.batch_size = 1,
		.input_channels = 0,
		.output_channels = 0,
		.input_size = { 0, 0 },
		.input_padding = 0,
		.kernel_size = { 0, 0 },
		.output_subsampling = { 1, 1 },
		.algorithm = nnp_convolution_algorithm_auto,
		.transform_strategy = nnp_convolution_transform_strategy_compute,
	};

	return options;
}

int main(int argc, char** argv) {
	enum nnp_status init_status = nnp_initialize();
	if (init_status != nnp_status_success) {
		fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
		exit(EXIT_FAILURE);
	}

	const struct options options = parse_options(argc, argv);

	const size_t batch_size = options.batch_size;
	const size_t input_channels = options.input_channels;
	const size_t output_channels = options.output_channels;
	const struct nnp_padding input_padding = { options.input_padding, options.input_padding, options.input_padding, options.input_padding };
	const struct nnp_size input_size = options.input_size;
	const struct nnp_size kernel_size = options.kernel_size;
	const struct nnp_size output_subsampling = options.output_subsampling;
	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};
	struct nnp_size tile_size;

	printf("Batch size: %zu\n", batch_size);
	printf("Input channels: %zu\n", input_channels);
	printf("Output channels: %zu\n", output_channels);
	printf("Input: %zux%zu with implicit padding %zu\n", input_size.height, input_size.width, options.input_padding);
	printf("Kernel: %zux%zu\n", kernel_size.height, kernel_size.width);
	printf("Subsampling: %zux%zu\n", output_subsampling.height, output_subsampling.width);

	const struct nnp_size output_tile_size = {
		.height = tile_size.height - kernel_size.height + 1,
		.width = tile_size.width - kernel_size.width + 1
	};
	const size_t tile_count =
		(output_size.height / output_tile_size.height + !!(output_size.height % output_tile_size.height)) *
		(output_size.width / output_tile_size.width + !!(output_size.width % output_tile_size.width));

	void* input = malloc(batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	void* kernel = malloc(input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	void* output = malloc(batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	void* bias = malloc(output_channels * sizeof(float));

	memset(input, 0, batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	memset(kernel, 0, input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	memset(output, 0, batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	memset(bias, 0, output_channels * sizeof(float));

	pthreadpool_t threadpool = NULL;

	benchmark_convolution(
		options.algorithm,
		options.transform_strategy,
		batch_size, input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		input, kernel, bias, output,
		threadpool);

	return EXIT_SUCCESS;
}
