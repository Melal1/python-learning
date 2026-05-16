# Data Compression
import zlib

data = b"This is a simple text data that will be compressed."

compressed_data = zlib.compress(data)
decompressed_data = zlib.decompress(compressed_data)

# TODO: Print the size of data in (bytes) before and after compress
