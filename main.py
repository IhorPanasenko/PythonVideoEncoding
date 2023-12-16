from arithmetic_encoding_video import arithmetic_encoding_array_test
from thinging_compression import compress_tinging

if __name__ == '__main__':
    input_file_path = './Resources/WaterfallInitialQuality.mp4'
    compressed_file_path = './Resources/CompressedWaterfall.mp4'
    decompressed_file_path = './Resources/DecompressedWaterfall.mp4'
    compress_tinging(input_file_path, compressed_file_path, decompressed_file_path)
