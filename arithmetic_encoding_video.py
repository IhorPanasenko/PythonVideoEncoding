import cv2
import numpy as np
from bitarray import bitarray


def calculate_probabilities_from_frame(frame):
    pixels = flatten_frame(frame)

    # Count occurrences of each pixel value
    pixel_counts = {}
    total_pixels = len(pixels)

    for pixel in pixels:
        pixel_counts[pixel] = pixel_counts.get(pixel, 0) + 1

    # Normalize frequencies to obtain probabilities
    pixel_probabilities = [pixel_counts[pixel] / total_pixels for pixel in range(256)]

    return pixel_probabilities


def calculate_probabilities_from_array(data):
    # Count occurrences of each element
    element_counts = {}
    total_elements = len(data)

    for element in data:
        element_counts[element] = element_counts.get(element, 0) + 1

    # Normalize frequencies to obtain probabilities
    element_probabilities = [element_counts.get(element, 0) / total_elements for element in range(256)]

    return element_probabilities


def flatten_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Flatten the 2D array to a 1D array
    return gray_frame.flatten()


class ArithmeticCoding:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.low = 0.0
        self.high = 1.0
        self.range = 1.0
        self.precision = 32  # Number of bits in the mantissa

    def encode_symbol(self, symbol):
        low_range = self.low + self.range * sum(self.probabilities[:symbol])
        high_range = low_range + self.range * self.probabilities[symbol]

        self.low = low_range
        self.high = high_range
        self.range = high_range - low_range

    def encode_data(self, data):
        encoded_data = bitarray()

        for symbol in data:
            self.encode_symbol(symbol)

            # Update ranges for precision control
            while self.high - self.low < 2 ** -self.precision or self.low >= 0.5 or self.high < 0.5:
                if self.low >= 0.5:
                    encoded_data.append(False)
                    self.low = 2 * (self.low - 0.5)
                    self.high = 2 * (self.high - 0.5)
                elif self.high < 0.5:
                    encoded_data.append(True)
                    self.low = 2 * self.low
                    self.high = 2 * self.high

                    # Additional condition to break the loop
                if self.low == 0.0 and self.high == 0.0:
                    break

        return encoded_data

    def decode_data(self, encoded_data, data_length):
        decoded_data = []

        # Convert bitarray to integer
        encoded_value = int(encoded_data.to01(), 2)

        for _ in range(data_length):
            target_value = (encoded_value - self.low) / self.range

            # Find the symbol that corresponds to the current range
            cumulative_prob = 0.0
            for symbol, prob in enumerate(self.probabilities):
                cumulative_prob += prob
                if target_value < cumulative_prob:
                    decoded_data.append(symbol)
                    break

            # Update ranges for precision control
            while self.high - self.low < 2 ** -self.precision or self.low >= 0.5 or self.high < 0.5:
                if self.low >= 0.5:
                    self.low = 2 * (self.low - 0.5)
                    self.high = 2 * (self.high - 0.5)
                elif self.high < 0.5:
                    self.low = 2 * self.low
                    self.high = 2 * self.high

        return decoded_data


def arithmetic_compress_array(input_array, output_path):
    array_probabilities = calculate_probabilities_from_array(input_array)
    ac = ArithmeticCoding(array_probabilities)
    encoded_data = ac.encode_data(input_array)

    with open(output_path, 'wb') as output_file:
        encoded_data.tofile(output_file)

    return encoded_data


def arithmetic_decompress_array(encoded_data, array_probabilities, data_length):
    ac = ArithmeticCoding(array_probabilities)
    decoded_data = ac.decode_data(encoded_data, data_length)
    return np.array(decoded_data)


def arithmetic_compress_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frames_probabilities = []
    encoded_datas = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    i = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_probability = calculate_probabilities_from_frame(frame)
        frames_probabilities.append(frame_probability)

        flattened_frame = flatten_frame(frame)
        ac = ArithmeticCoding(frame_probability)
        encoded_data = ac.encode_data(flattened_frame)
        encoded_datas.append(encoded_data)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        i += 2

    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, 'wb') as output_file:
        for encoded_data in encoded_datas:
            encoded_data.tofile(output_file)

    return encoded_datas


def arithmetic_encoding_array_test():
    # Example usage
    input_array = np.random.randint(0, 256, 100)  # Generate a random array of size 100 with values between 0 and 255
    output_path = 'Resources/output_arithmetic.bin'

    # Compress
    encoded_data = arithmetic_compress_array(input_array, output_path)

    # Decompress
    decoded_array = arithmetic_decompress_array(encoded_data, calculate_probabilities_from_array(input_array),
                                                len(input_array))
    print("Original Array:", input_array)
    print("Decoded Array:", decoded_array)