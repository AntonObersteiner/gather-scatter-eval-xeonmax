#include "common.cpp"
#include "gather/simd_variants/avx512/agg_avx512_32BitVariants.h"

constexpr bool multi_threaded = true;
constexpr bool avx512 = true;

using ResultT = uint32_t;

// 64 bits? else 32 bit integers
constexpr bool bits64 = std::is_same<ResultT, uint64_t>::value;

int main(int argc, const char** argv) {
	int data_size_log2 = 0;
	int numa_node = 0;
	int result = read_cmdline_arguments(
		argc, argv,
		data_size_log2,
		numa_node
	);
	if (result != SUCCESS)
		return result;

	const vector<aggregator_t<ResultT>> aggregators	{
		{ aggregate_scalar,					"scalar",	false },
		{ aggregate_linear_avx512,			"linear",	false },
		{ aggregate_strided_gather_avx512,	"gather",	true },
		{ aggregate_strided_set_avx512,		"seti",		true },
	};
	return main_multi_threaded<ResultT>(
		aggregators,
		data_size_log2,	// log2 of number of integers
		multi_threaded,
		avx512,
		bits64,
		numa_node
	);
}
