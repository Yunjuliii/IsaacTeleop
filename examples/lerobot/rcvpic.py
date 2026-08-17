import socket
import struct
from pathlib import Path

import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5000

# 存到 calib_data/pico_camera/，跟 collect_fisheye_calib.py 的 calib_data/ 布局对齐，
# 方便后面 solve_pico_to_head_extrinsics.py 直接从这里读。
OUT_DIR = Path(__file__).parent / "calib_data" / "pico_camera"


def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            raise ConnectionError("PICO disconnected")

        data.extend(packet)

    return bytes(data)


def receive_calibration(conn):
    # int width
    # int height
    # 4 doubles: fx fy cx cy
    # 7 floats: pos xyz + quat xyzw
    # 16 floats: left extrinsic matrix

    fmt = "<ii4d7f16f"
    size = struct.calcsize(fmt)

    raw = recv_exact(conn, size)

    values = struct.unpack(fmt, raw)

    width = values[0]
    height = values[1]

    fx = values[2]
    fy = values[3]
    cx = values[4]
    cy = values[5]

    pos = values[6:9]
    quat = values[9:13]

    matrix_values = values[13:29]

    matrix = np.array(
        matrix_values,
        dtype=np.float32
    ).reshape(4, 4)

    print("\n========== CAMERA CALIBRATION ==========")

    print("Resolution:")
    print(width, "x", height)

    print("\nIntrinsics:")
    print("fx =", fx)
    print("fy =", fy)
    print("cx =", cx)
    print("cy =", cy)

    print("\nLEFT camera position:")
    print(pos)

    print("\nLEFT camera quaternion (x y z w):")
    print(quat)

    print("\nLEFT extrinsic matrix:")
    print(matrix)

    print("========================================\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    out_path = OUT_DIR / "left_intrinsics.npz"
    np.savez(
        out_path,
        width=width, height=height,
        fx=fx, fy=fy, cx=cx, cy=cy, K=K,
        position=np.array(pos), quaternion_xyzw=np.array(quat),
        extrinsic_matrix=matrix,
    )
    print(f"内参已自动存到: {out_path}\n")


def receive_frame(conn):
    # uint64 timestamp
    # int width
    # int height
    # int image_size
    # 7 floats = head pose position + quaternion

    fmt = "<QIII7f"

    header_size = struct.calcsize(fmt)

    header = recv_exact(
        conn,
        header_size
    )

    values = struct.unpack(
        fmt,
        header
    )

    timestamp = values[0]

    width = values[1]
    height = values[2]
    image_size = values[3]

    head_pos = values[4:7]
    head_rot = values[7:11]

    image_bytes = recv_exact(
        conn,
        image_size
    )

    return (
        timestamp,
        width,
        height,
        head_pos,
        head_rot,
        image_bytes
    )


with socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
) as server:

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind(
        (HOST, PORT)
    )

    server.listen(1)

    print("================================")
    print("Waiting for PICO...")
    print("TCP port:", PORT)
    print("================================")

    conn, address = server.accept()

    print("PICO connected:", address)

    with conn:

        frame_number = 0

        try:

            while True:

                packet_type = recv_exact(
                    conn,
                    1
                )[0]

                # --------------------------
                # Calibration packet
                # --------------------------

                if packet_type == 1:

                    receive_calibration(
                        conn
                    )


                # --------------------------
                # Camera frame
                # --------------------------

                elif packet_type == 2:

                    (
                        timestamp,
                        width,
                        height,
                        head_pos,
                        head_rot,
                        image_bytes
                    ) = receive_frame(conn)

                    frame_number += 1

                    expected_size = (
                        width
                        * height
                        * 4
                    )

                    if len(image_bytes) != expected_size:

                        print(
                            "Unexpected image size:",
                            len(image_bytes),
                            "expected:",
                            expected_size
                        )

                        continue

                    image = np.frombuffer(
                        image_bytes,
                        dtype=np.uint8
                    )

                    image = image.reshape(
                        height,
                        width,
                        4
                    )

                    # 第一版先按 RGBA -> BGR 尝试
                    frame = cv2.cvtColor(
                        image,
                        cv2.COLOR_RGBA2BGR
                    )

                    if frame_number % 30 == 0:

                        print(
                            "frame:",
                            frame_number,
                            "timestamp:",
                            timestamp,
                            "head pos:",
                            head_pos
                        )

                    # 只截取第一帧存盘，不弹窗。PNG 无损，跟标定用途匹配。
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = OUT_DIR / "pico_frame.png"

                    cv2.imwrite(
                        str(out_path),
                        frame
                    )

                    print("已保存一帧到:", out_path)

                    break

                else:

                    print(
                        "Unknown packet type:",
                        packet_type
                    )

                    break

        except (
            ConnectionError,
            BrokenPipeError
        ) as e:

            print(
                "Connection closed:",
                e
            )