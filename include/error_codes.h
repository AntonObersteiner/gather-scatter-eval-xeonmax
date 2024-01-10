#ifndef GATHER_ERROR_CODES_H
#define GATHER_ERROR_CODES_H

/** enum for errors that may occur in executing any part of this project.
 */
enum gather_error_codes {
	SUCCESS = 0,
	NO_DATA_SIZE_GIVEN = 1,
	DATA_SIZE_TOO_LOW = 2,
	RESULT_FILE_NOT_OPENED = 3,
	NO_MEMORY = 4,
	TOO_MANY_ARGUMENTS = 5,
	NOT_ENOUGH_CPUS = 6,
	RESULT_INCORRECT = 7,
	RUNNING_ON_WRONG_CPU_NUMA_NODE = 8,
};

#endif // include guard GATHER_ERROR_CODES_H
