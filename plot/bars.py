#!/usr/bin/python3
from matplotlib import pyplot as plt
import seaborn as sns
import pandas  as pd
import numpy   as np
import re
from collections import OrderedDict

filename_regex = re.compile(
	r'.*/?(\d\d)_(multi|single)_threaded_(avx256|avx512)'
	r'_(\d\d)bit'
	r'(_node(\d\d)_cpus(\d\d))?'
	r'(_(\d\d?)_cores|_results).dat'
)
def int_or_empty(default_value = None):
	""" this accepts valid arguments to integers and falsey values.
	the latter are mapped to the default_value,
	because when my file do not specify which numa nodes were used,
	there needs to be a way of marking ignorance.
	"""
	def converter(arg):
		if not arg:
			return default_value

		try:
			return int(arg)
		except:
			raise ValueError(
				f"The value '{arg}' was supposed to be falsey "
				"or convertable to an int, it’s neither!"
			)

	return converter

filename_labeling = OrderedDict([
	# a name for the property here, the index in the regex groups and the type
	("data_size_log2", (0, int)),
	("threaded", (1, str)),
	("avx", (2, str)),
	("bit", (3, int)),
	("mem_node", (5, int_or_empty(-1))),
	("cpu_node", (6, int_or_empty(-1))),
	("cores", (8, int_or_empty(0))),        #marking the single-core case
])
def read_label(filename):
	matches = filename_regex.fullmatch(filename)
	if matches is None:
		raise ValueError(
			f"The given filename '{filename}' does not match the required "
			"patterns!"
		)
	matches = matches.groups()
	result = OrderedDict([
		(name, typ(matches[i]))
		for name, (i, typ) in filename_labeling.items()
	])
	return result

def label_dict_to_tuple(label):
	return tuple(label.values())

def label_tuple_to_dict(label):
	return OrderedDict([
		(name, value)
		for name, value in zip(filename_labeling.keys(), label)
	])

def test_label_conversions():
	examples = {
		"30_single_threaded_avx512_32bit_results.dat":
		((30, "single", "avx512", 32, -1, -1, 0),
		{"data_size_log2": 30, "threaded": "single", "avx": "avx512",
		"bit": 32, "mem_node": -1, "cpu_node": -1, "cores": 0}),
		"30_single_threaded_avx512_32bit_node10_cpus02_results.dat":
		((30, "single", "avx512", 32, 10, 2, 0),
		{"data_size_log2": 30, "threaded": "single", "avx": "avx512",
		"bit": 32, "mem_node": 10, "cpu_node": 2, "cores": 0}),
		"26_multi_threaded_avx256_64bit_node02_cpus06_1_cores.dat":
		((26, "multi", "avx256", 64, 2, 6, 1),
		{"data_size_log2": 26, "threaded": "multi", "avx": "avx256",
		"bit": 64, "mem_node": 2, "cpu_node": 6, "cores": 1}),
		"29_multi_threaded_avx512_64bit_16_cores.dat":
		((29, "multi", "avx512", 64, -1, -1, 16),
		{"data_size_log2": 29, "threaded": "multi", "avx": "avx512",
		"bit": 64, "mem_node": -1, "cpu_node": -1, "cores": 16}),
	}
	for filename, (label_tuple, label_dict) in examples.items():
		read = read_label(filename)
		assert read == label_dict
		assert label_dict_to_tuple(read) == label_tuple
		assert label_tuple_to_dict(label_tuple) == label_dict

column_names_and_types = OrderedDict([
	("stride",       np.int32),
	("stride_bytes", np.int32),
	("mis-scalar",   np.float32),
	("thr-scalar",   np.float32),
	("mis-linear",   np.float32),
	("thr-linear",   np.float32),
	("mis-gather",   np.float32),
	("thr-gather",   np.float32),
	("mis-seti",     np.float32),
	("thr-seti",     np.float32),
])
drop_columns = [
	"stride_bytes",
] + [
	column
	for column in column_names_and_types.keys()
	if column.startswith("mis-")
]
def read_data(files):
	data = {
		file: pd.read_csv(
			file,
			sep = " ",
			names = list(column_names_and_types.keys()),
			converters = dict(column_names_and_types),
		)
		for file in files
	}
	for file, frame in data.items():
		# sometimes, the strides are stored as 2, 4, 8, …, (power style)
		# sometimes as the corresponding exponents: 1, 2, 3, … (log2 style)
		# this is automatically detected and converted to the log2 style
		strides = frame["stride"]
		if strides[6] - strides[5] == 1:
			pass
		elif strides[6] / strides[5] == 2:
			frame["stride"] = np.int32(np.log2(frame["stride"]))
		else:
			raise ValueError(
				f"strides usually are either "
				f"powers of two (but {strides[6]} / {strides[5]} ≠ 2) "
				f"or log2 consecutive integers (but {strides[6]} - {strides[5]} ≠ 1)."
			)

		# remove the columns listed above, they are redundant or uninteresting
		frame.drop(
			columns = drop_columns,
			inplace = True,
		)

	return data

def main(files):
	data = read_data(files)
	labels = { file: read_label(file) for file in files }
	suffixes = ["_hbm", "_ddr"]
	infixes = ["-scalar", "-linear", "-gather", "-seti"]
	mydata = data[files[0]]
	print(mydata)
	print(data[files[1]])
	mydata = mydata.merge(
		data[files[1]],
		left_on = "stride",
		right_on = "stride",
		suffixes = suffixes,
	)
	print(mydata)
	def plot(label):
		return sns.lineplot(
			data = mydata,
			x = "stride",
			y = label,
			label = label,
			legend = "brief",
			#hue = "technique",
		)
	ax = None
	for suffix in suffixes:
		for infix in infixes:
			ax = plot("thr" + infix + suffix)
	ax.legend()
	plt.show()

def test():
	test_label_conversions()

if __name__ == "__main__":
	test()
	from sys import argv
	main(argv[1:])

