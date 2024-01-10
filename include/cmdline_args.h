#ifndef CMDLINE_ARGS_H
#define CMDLINE_ARGS_H

#include "error_codes.h"

/** reads the commandline arguments.
 * argv[0] is ignored.
 * argv[1] must exist and is read with atoi into data_size_log2.
 * if argv[1] is not given, this returns NO_DATA_SIZE_GIVEN.
 * argv[2] is optional and, if given, is read into numa_mode.
 * argv[3] is optional and, if given, is read into cpu_numa_mode.
 * more argv (detected as argc > 4) raises TOO_MANY_ARGUMENTS.
 * successful call returns error_codes.h's SUCCESS
 */
inline gather_error_codes read_cmdline_arguments (
	const int argc,
	const char** argv,
	int &data_size_log2,
	int &numa_node,
	int &cpu_numa_node
) {
    if (argc < 2) {
        std::cerr << "Data Size as input expected (as log_2)!" << std::endl;
        return NO_DATA_SIZE_GIVEN;
    }

    data_size_log2 = atoi(argv[1]);

	// argv[2] is optional
	if (argc < 3)
		return SUCCESS;

	numa_node = atoi(argv[2]);

	if (argc < 4)
		return SUCCESS;

	cpu_numa_node = atoi(argv[3]);

	if (argc > 4) {
        std::cerr << "too many arguments (" << argc << ") to " << argv[0] << "!" << std::endl;
		return TOO_MANY_ARGUMENTS;
	}

	return SUCCESS;
}

#endif // include guard CMDLINE_ARGS_H
