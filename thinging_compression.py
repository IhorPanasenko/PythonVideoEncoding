import cv2

def read_video(file_path, scale_factor=1.0):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    original_width = int(cap.get(3))
    original_height = int(cap.get(4))

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    cv2.namedWindow("Scaled Frame", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("Scaled Frame", resized_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def compress_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    original_width = int(cap.get(3))
    original_height = int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(5), (original_width, original_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)
        ret, frame = cap.read()

        if not ret:
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video file: ", input_path, "\nCompressed to file: ", output_path, "\n")


def decompress_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    original_width = int(cap.get(3))
    original_height = int(cap.get(4))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(5), (original_width, original_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)
        out.write(frame)

        if not ret:
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video file: ", input_path, "\nDecompressed to file: ", output_path, "\n")


def compress_tinging(input_file_path, compressed_file_path, decompressed_file_path):
    compress_video(input_file_path, compressed_file_path)
    decompress_video(compressed_file_path, decompressed_file_path)
