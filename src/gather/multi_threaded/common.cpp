#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<random>
#include<chrono>
#include "immintrin.h"
#include<fstream>
#include <string.h>
#include <math.h>
#include <functional>
#include <future>
#include <thread>
#include <vector>
#include <algorithm>

#include "gather/simd_variants/avx/agg_avx_64BitVariants.h"
#include "error_codes.h"
#include "cmdline_args.h"

// ITERATIONS and MAX_CORES
#include "parameters.h"

using namespace std;

#include "allocate.cpp"
#include "aggregation_type.h"
#include "measures.h"
#include "make_label.cpp"
multithreaded_measures scalar, linear, gather, seti;

#include "create_thread.cpp"
#include "log_multithreaded_results.cpp"
#include "generate_random_values.cpp"

int find_next_best_cpu (
	uint64_t &cpu_id,
	cpu_set_t* in_use,
	const uint64_t cpu_numa_node,
	uint64_t &nonspecificity,
	const uint64_t max_nonspecificity = 3
) {
	// the 2**nonspec numa nodes in alignment around cpu_numa_node are considered.
	while (true) {
		const uint64_t nonspecificity_mask = 0ul - (1ul << nonspecificity);
		const uint64_t numa_node = numa_node_of_cpu(cpu_id);
		if (
			!CPU_ISSET(cpu_id, in_use) &&
			(numa_node & nonspecificity_mask) ==
			(cpu_numa_node & nonspecificity_mask)
		) {
			// cpu is free and on the accepted numa nodes
			CPU_SET(cpu_id, in_use);
			return SUCCESS;
		}
		// look at next cpu
		cpu_id++;
		if (cpu_id > numa_num_configured_cpus()) {
			// we went through all cpus registered in the system,
			// let's be less specific in our request for numa node adherence
			nonspecificity++;
			if (nonspecificity > max_nonspecificity) {
				std::cerr << "could not find a cpu not used "
					"by own threads, even accepting "
					<< (1 << (nonspecificity - 1)) << " NUMA nodes!"
					<< std::endl;
				std::cerr << "in_use cpus: " << CPU_COUNT(in_use) << std::endl;
				std::cerr << "cpu_id: " << cpu_id << std::endl;
				std::cerr << "nonspecificity: " << nonspecificity << std::endl;
				fprintf(stderr, "nonspecificity_mask: %16b\n", nonspecificity_mask);
				return NOT_ENOUGH_CPUS;
			}
			cpu_id = 0;
		}
	}
}

template <class ResultT>
int benchmark(
	multithreaded_measures* res,
	const uint64_t cpu_numa_node,
	uint64_t correct_result,
	const ResultT* values,
	uint64_t n,
	const uint32_t stride,
	double GB,
	aggregation_function_t<ResultT> func
) {
    for ( size_t core_cnt = 1; core_cnt <= MAX_CORES; core_cnt *= 2 ) { /* Run with 1, 2, 4, ... MAX_CORES cores */
        std::vector< std::thread* > pool;

        ResultT* tmp_res = (ResultT*) aligned_alloc( 8 * sizeof(ResultT), core_cnt * sizeof( ResultT ) );
        double* tmp_dur   = (double*)   aligned_alloc( 64, core_cnt * sizeof( double )  );
        bool* ready_vec = (bool*) malloc( core_cnt * sizeof( bool ) );

        auto magic = [core_cnt, values, n, stride] ( const uint64_t tid, ResultT* local_result, double* local_duration, bool* local_ready, std::shared_future< void >* sync_barrier, aggregation_function_t<ResultT> local_func ) {
            // flush all caches and TLB
            // clean start setting
            void flush_cache_all(void);
            void flush_tlb_all(void);
            local_ready[ tid ] = true;
            const uint64_t my_value_count = n / core_cnt; /* Should be always divisible by 2, 4 or 8 */
			// is uint32_t in some benchmarks, wich is hopefully irrelevant
            const uint64_t my_offset = tid * my_value_count;
            sync_barrier->wait();

            auto begin = chrono::high_resolution_clock::now();
            local_result[ tid ] = local_func(values + my_offset, my_value_count, stride);
            auto end = std::chrono::high_resolution_clock::now();

            local_duration[ tid ] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        };

        double averaged_duration = 0.0;
        for (int i=0; i<ITERATIONS; i++) {
            std::promise< void > p;
		    std::shared_future< void > ready_future( p.get_future( ) );

            memset( tmp_res, 0, core_cnt * sizeof( ResultT ) );
            memset( tmp_dur, 0, core_cnt * sizeof( double ) );
            memset( ready_vec, 0, core_cnt * sizeof( bool ) );

			// cpu_id is restricted to the cpus of cpu_numa_node (if possible)
			uint64_t cpu_id = 0;
			uint64_t nonspecificity = 0; // how far away the cpu may be from cpu_numa_node
			cpu_set_t cpus_in_use;
			CPU_ZERO(&cpus_in_use);

            for ( size_t tid = 0; tid < core_cnt; ++tid, ++cpu_id ) {
				if (find_next_best_cpu(cpu_id, &cpus_in_use, cpu_numa_node, nonspecificity) != SUCCESS) {
					std::cerr
						<< "could not find " << tid + 1
						<< " cpus in / around NUMA node " << cpu_numa_node
						<< ", quitting!"
						<< std::endl;
					return NOT_ENOUGH_CPUS;
				}
				// std::cout << "create_thread( tid: " << tid << ", cpu: " << cpu_id << ", ...);" << std::endl;
                pool.emplace_back( create_thread( tid, cpu_id, tmp_res, tmp_dur, ready_vec, &ready_future, magic, func ) );
            }
            bool all_ready = false;
            while ( !all_ready ) {
                /* Wait until all threads are ready to go before we pull the trigger */
                using namespace std::chrono_literals;
                std::this_thread::sleep_for( 1ms );
                all_ready = true;
                for ( size_t i = 0; i < core_cnt; ++i ) {
                    all_ready &= ready_vec[ i ];
                }
            }
            p.set_value(); /* Start execution by notifying on the void promise */
            std::for_each( pool.begin(), pool.end(),
                []( std::thread* t ) {
                     t->join();
                     delete t; }
            ); /* Join and delete threads as soon as they are finished */
            pool.clear();
            double iteration_duration = 0.0;
            for ( size_t i = 0; i < core_cnt; ++i ) {
                iteration_duration += tmp_dur[ i ];
            }
            averaged_duration += iteration_duration / static_cast< double >( core_cnt );
        }

        /* Beware, this is an average of averages. We can also do average of max(thread_runtimes) */
        const double cur_dur = static_cast< double >( averaged_duration ) / static_cast< double >( ITERATIONS );
        /* Integer in Millions / time * 10^9 (becausue nanoseconds) */
        const double cur_mis = ( static_cast<double>( n ) / 1000000.0 ) / ( cur_dur * 1e-9 );
        const double cur_tput = GB / ( cur_dur * 1e-9 );
        uint64_t cur_res = 0;
        for ( size_t i = 0; i < core_cnt; ++i ) {
            cur_res += tmp_res[ i ];
        }

        const struct measures tmp_measures = { cur_res, cur_dur, cur_tput, cur_mis };
        (*res)[ core_cnt ] = tmp_measures;

        free( ready_vec );
        free( tmp_dur );
        free( tmp_res );
    }

    for ( size_t core_cnt = 1; core_cnt <= MAX_CORES; core_cnt *= 2 ) {
		if ((*res)[ core_cnt ].result != correct_result) {
			std::cerr
				<< "the correct result is " << correct_result
				<< " but with " << core_cnt
				<< " cores, we got " << (*res)[core_cnt].result << " instead :/"
				<< std::endl;
			return RESULT_INCORRECT;
		}
    }

    return SUCCESS;
}

template <class ResultT>
int main_multi_threaded(
	const vector<aggregator_t<ResultT>> aggregators,
	uint64_t data_size_log2,
	bool multi_threaded,
	bool avx512,
	bool bits64,
	const uint64_t numa_node,
	const uint64_t cpu_numa_node
) {
	// run this thread and it's children on the given numa node
	// struct bitmask cpu_mask; // conversion to cpu mask not needed?
	// numa_node_to_cpus(cpu_numa_node, &cpu_mask);
	int result = numa_run_on_node(cpu_numa_node);
	if (result != 0) {
		std::cerr << "could not run on NUMA node " << cpu_numa_node << "! ";
		perror("numa_run_on_node");
	}

    // define number of values
    // 27 --> 134 million integers --> 8GB
    // 26 --> 67 million integers --> 4GB
    uint64_t number_of_values = pow(2, data_size_log2);
	cerr << "number_of_values: " << number_of_values << endl;


    // define max stride size (power of 2)
    size_t max_stride = 15;
	cerr << "max_stride: " << max_stride << ", 2**max_stride: " << (1 << max_stride) << endl;

	if (max_stride + 1 < data_size_log2) {
		/* this stride is fine */
	} else {
		cerr
			<< "Data Size is 2**" << data_size_log2 << " == " << (1<<data_size_log2)
			<< " which does not allow the hardcoded maximum stride of "
			<< "2**" << max_stride << " == " << (1<<max_stride) << "!"
		<< endl;
		return DATA_SIZE_TOO_LOW;
	}

    //compute GB for number of values
    double GB = (((double)number_of_values*sizeof(ResultT)/(double)1024)/(double)1024)/(double)1024;


    /**
     * allocate memory and fill with random numbers
     */
    ResultT* array = allocate<ResultT>(number_of_values, numa_node);
    if (array != NULL) {
        cout << "Memory allocated on NUMA node " << numa_node << " - " << number_of_values << " values" << endl;
    } else {
        cout << "Memory not allocated on NUMA node " << numa_node << endl;
		exit(NO_MEMORY);
    }
    generate_random_values(array, number_of_values);
    uint64_t correct = aggregate_scalar(array, number_of_values);
    cout <<"Generation done."<<endl;

    /**
     * run several benchmarks on generated data
     */

	// measurement result structs
	vector<multithreaded_measures> measurements;
	measurements.assign(aggregators.size(), multithreaded_measures());

    // open files to store runtime measurements
	string label = make_label(data_size_log2, multi_threaded, avx512, bits64, numa_node, cpu_numa_node);
	string result_filename_base = "./data/gather/" + label;

	/*
	if (result_file.good()) {
		cout << "writing data to '" << result_filename << "'." << endl;
	} else {
		cerr << "writing data to '" << result_filename << "' failed!" << endl;
		return RESULT_FILE_NOT_OPENED;
	}
	*/


	// note: the stride is the outer loop for the benefit of the output file,
	// non-strided aggregation methods will still run only once.
    bool first_run = true;
	const int min_stride_pow = 1;
	for (int stride_pow = min_stride_pow; stride_pow <= max_stride; stride_pow++) {
		uint64_t stride_size = pow(2, stride_pow);

		for (int a = 0; a < aggregators.size(); a++) {
			const aggregation_function_t<ResultT>& function = aggregators[a].function;
			const string& label = aggregators[a].label;
			const bool& strided = aggregators[a].strided;

			multithreaded_measures& measurement = measurements[a];

			if (!strided) {
				if (stride_pow == 1) {
					int error_code = benchmark(
						&measurement,
						cpu_numa_node,
						correct,
						array,
						number_of_values,
						0,
						GB,
						function
					);
					if (error_code == SUCCESS) {
						cout << label << " done" << endl;
					} else {
						cout << label << " failed with code " << error_code << endl;
						return error_code;
					}
				}
			} else {
				int error_code = benchmark(
					&measurement,
					cpu_numa_node,
					correct,
					array,
					number_of_values,
					stride_size,
					GB,
					function
				);
				if (error_code == SUCCESS) {
					cout << label << " done" << endl;
				} else {
					cout << label << " failed with code " << error_code << endl;
					return error_code;
				}
			}

		}

		/* Write all to one file */
		log_multithreaded_results_per_file(
			result_filename_base,
			stride_pow,
			measurements,
			first_run
		);

		if (first_run) {
			first_run = false;
		}
	}
	for (int a = 0; a < aggregators.size(); a++) {
	    print_multithreaded_results( cout, aggregators[a].label, measurements[a] );
	}

	cerr << "freeing array!" << endl;
    free(array);

	return SUCCESS;
}

