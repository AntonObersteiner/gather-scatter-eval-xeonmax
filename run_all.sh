

# both should get 4 GiB (order 32)
# because the number i write after is the number of integers
for f in bin/*bench*_32; do
	./$f 30;
done
for f in bin/*bench*_64; do
	./$f 29;
done
