import decimal
from decimal import Decimal
from decimal import getcontext

import cv2
import numpy as np


class ArithmeticEncoding:
    def __init__(self, frequency_table):
        self.probability_table = self.get_probability_table(frequency_table)

    def get_probability_table(self, frequency_table):
        total_frequency = sum(list(frequency_table.values()))

        probability_table = {}
        for key, value in frequency_table.items():
            probability_table[key] = value / total_frequency

        return probability_table

    def get_encoded_value(self, encoder):
        last_stage = list(encoder[-1].values())
        last_stage_values = []
        for sublist in last_stage:
            for element in sublist:
                last_stage_values.append(element)

        last_stage_min = min(last_stage_values)
        last_stage_max = max(last_stage_values)

        return (last_stage_min + last_stage_max) / 2

    def process_stage(self, probability_table, stage_min, stage_max):
        stage_probs = {}
        stage_domain = stage_max - stage_min
        for term_idx in range(len(probability_table.items())):
            term = list(probability_table.keys())[term_idx]
            term_prob = Decimal(probability_table[term])
            cum_prob = term_prob * stage_domain + stage_min
            stage_probs[term] = [stage_min, cum_prob]
            stage_min = cum_prob
        return stage_probs

    def encode(self, msg, probability_table):
        encoder = []

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for msg_term_idx in range(len(msg)):
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            msg_term = msg[msg_term_idx]
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            encoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        encoder.append(stage_probs)

        encoded_msg = self.get_encoded_value(encoder)

        return encoder, encoded_msg

    def decode(self, encoded_msg, msg_length, probability_table):
        decoder = []
        decoded_msg = ""

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for idx in range(msg_length):
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            for msg_term, value in stage_probs.items():
                if encoded_msg >= value[0] and encoded_msg <= value[1]:
                    break

            decoded_msg = decoded_msg + msg_term
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            decoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        decoder.append(stage_probs)

        return decoder, decoded_msg


def calculate_frequency(input_string):
    frequency_table = {}
    for symbol in input_string:
        if symbol in frequency_table:
            frequency_table[symbol] += 1
        else:
            frequency_table[symbol] = 1
    return frequency_table


def frame_to_array(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_array = frame.flatten()

    return frame_array


def array_to_frame(frame_array, shape):
    reconstructed_frame = frame_array.reshape(shape)

    if len(shape) == 2:
        reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_GRAY2BGR)

    return reconstructed_frame


def array_to_hex_string(frame_array):
    hex_string = ''.join(format(byte, '02X') for byte in frame_array)
    return hex_string


def hex_string_to_array(hex_string):
    reconstructed_frame_array = np.array([int(hex_string[i:i + 2], 16) for i in range(0, len(hex_string), 2)])
    return reconstructed_frame_array


def encode_video(file_path, output_file):
    cap = cv2.VideoCapture(file_path)
    frames_frequencies = []
    frames_length = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    open(output_file, 'w').close()

    with open(output_file, 'w') as file:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_array = frame_to_array(frame)
            hex_frame = array_to_hex_string(frame_array)
            frequency_table = calculate_frequency(hex_frame)
            frames_frequencies.append(frequency_table)
            frames_length.append(len(hex_frame))
            AE = ArithmeticEncoding(frequency_table)
            encoder, encoded_msg = AE.encode(msg=hex_frame,
                                             probability_table=AE.probability_table)

            file.write(encoded_msg)
            file.write('\n')

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return frames_frequencies, frames_length


def decode_video(encoded_file, output_file, frequency_tables, messages_length):
    with open(encoded_file, 'r') as file:
        encoded_messages = file.readlines()

    cap = cv2.VideoCapture(encoded_file)
    if not cap.isOpened():
        print("Error: Could not open encoded file.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter.fourcc(*'mp4v'), 30, (frame_width, frame_height))

    i = 0
    for encoded_msg in encoded_messages:
        encoded_msg = encoded_msg.strip()

        AE = ArithmeticEncoding(frequency_tables[i])
        decoder, decoded_msg = AE.decode(encoded_msg=encoded_msg,
                                         msg_length=messages_length[i],
                                         probability_table=AE.probability_table)

        reconstructed_frame_array = hex_string_to_array(decoded_msg)
        reconstructed_frame = array_to_frame(reconstructed_frame_array, (frame_height, frame_width))

        out.write(reconstructed_frame)
        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def array_test(array):
    print("Initial array: ", array)
    hex_array = array_to_hex_string(array)
    print("Hex string from array: ", hex_array)
    frequency_table = calculate_frequency(hex_array)
    AE = ArithmeticEncoding(frequency_table)
    encoder, encoded_msg = AE.encode(msg=hex_array,
                                     probability_table=AE.probability_table)

    print("Encoded Result: {msg}".format(msg=encoded_msg))

    decoder, decoded_msg = AE.decode(encoded_msg=encoded_msg,
                                     msg_length=len(hex_array),
                                     probability_table=AE.probability_table)

    print("Decoded hex string: {msg}".format(msg=decoded_msg))
    print("Message Decoded Successfully? {result}".format(result=hex_array == decoded_msg))
    decoded_array = hex_string_to_array(decoded_msg)
    print("Decoded array: ", decoded_array)


def array_test_with_file(output_file):
    open(output_file, 'w').close()
    with open(output_file, 'w') as file:
        for _ in range(9000):
            array = np.random.randint(0, 256, 100)

            hex_array = array_to_hex_string(array)
            frequency_table = calculate_frequency(hex_array)
            AE = ArithmeticEncoding(frequency_table)
            encoder, encoded_msg = AE.encode(msg=hex_array, probability_table=AE.probability_table)

            file.write(str(encoded_msg))
            file.write('\n')


if __name__ == "__main__":
    getcontext().prec = decimal.MAX_PREC
    input_file_path = './Resources/WaterfallInitialQuality.mp4'
    output_file_path = './Resources/output_arithmetic.txt'
    # encode_video(input_file_path, output_file_path)

    input_array = np.random.randint(0, 256, 100)
    array_test(input_array)
    #array_test_with_file(output_file_path)
