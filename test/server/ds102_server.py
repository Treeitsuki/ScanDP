#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DS102/DS112 Controller LAN Remote Control Server (Robust Version)
機能:
  - TCPソケット経由でのシリアル制御
  - Ctrl+C による安全な停止
  - 排他制御によるスレッドセーフ化
  - 詳細なログ出力
"""

import logging
import signal
import socket
import sys
import threading
import time

import serial

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DS102Server:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False

        # Serial Management
        self.ser = serial.Serial()
        self.serial_lock = threading.Lock()  # 排他制御用ロック

        # Signal Handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Ctrl+C 等のシグナルを受け取った時の処理"""
        logger.info("Shutdown signal received. Stopping server...")
        self.stop()

    def stop(self):
        """サーバー停止処理"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        self._serial_disconnect()
        logger.info("Server stopped.")
        sys.exit(0)

    # -------------------------
    # Serial Functions (Thread-Safe)
    # -------------------------
    def _serial_connect(self, port, baudrate):
        with self.serial_lock:
            if self.ser.is_open:
                self.ser.close()

            try:
                self.ser = serial.Serial(port, baudrate, timeout=2)
                logger.info(f"Serial connected: {port} @ {baudrate}")
                return "OK:CONNECTED"
            except Exception as e:
                logger.error(f"Serial connect failed: {e}")
                return f"ERR:CONNECT_FAILED:{str(e)}"

    def _serial_disconnect(self):
        with self.serial_lock:
            if self.ser.is_open:
                self.ser.close()
                logger.info("Serial disconnected")
            return "OK:DISCONNECTED"

    def _serial_write(self, cmd: str):
        with self.serial_lock:
            if not self.ser.is_open:
                return "ERR:NOT_CONNECTED"
            try:
                self.ser.write((cmd + "\r").encode("utf-8"))
                return "OK:SENT"
            except Exception as e:
                return f"ERR:WRITE_ERROR:{str(e)}"

    def _serial_write_read(self, cmd: str):
        with self.serial_lock:
            if not self.ser.is_open:
                return "ERR:NOT_CONNECTED"
            try:
                self.ser.write((cmd + "\r").encode("utf-8"))
                time.sleep(0.05)  # 短い待機
                data = self.ser.read_until(b"\r")
                return data.decode(errors="ignore").strip()
            except Exception as e:
                return f"ERR:IO_ERROR:{str(e)}"

    # -------------------------
    # TCP Client Handler
    # -------------------------
    def handle_client(self, conn, addr):
        logger.info(f"Client connected: {addr}")
        try:
            conn.send(b"DS102 Remote Server Ready (v2)\n")

            while self.running:
                data = conn.recv(1024)
                if not data:
                    break

                cmd_line = data.decode(errors='ignore').strip()
                if not cmd_line:
                    continue

                # 複数コマンドが同時に来た場合のために改行で分割対応しても良いが
                # ここでは1パケット1コマンドと仮定

                logger.debug(f"RX from {addr}: {cmd_line}")
                response = ""

                # ---- Command Parsing ----
                if cmd_line.upper() == "EXIT":
                    conn.send(b"BYE\n")
                    break

                elif cmd_line.startswith("CONNECT"):
                    parts = cmd_line.split()
                    if len(parts) == 3:
                        response = self._serial_connect(
                            parts[1], int(parts[2]))
                    else:
                        response = "ERR:FORMAT CONNECT <PORT> <BAUDRATE>"

                elif cmd_line == "DISCONNECT":
                    response = self._serial_disconnect()

                elif cmd_line.startswith("WRITE "):
                    real_cmd = cmd_line.replace("WRITE ", "", 1)
                    response = self._serial_write(real_cmd)

                elif cmd_line.startswith("WRR "):
                    real_cmd = cmd_line.replace("WRR ", "", 1)
                    response = self._serial_write_read(real_cmd)

                else:
                    # Default behavior: Write only (or Write+Read depending on usage)
                    # オリジナルに合わせて Write Only で実装、必要なら変更
                    response = self._serial_write(cmd_line)
                    # response = self._serial_write_read(cmd_line)

                # Send Response
                if response:
                    conn.send((response + "\n").encode())

        except ConnectionResetError:
            logger.warning(f"Connection reset by peer: {addr}")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            conn.close()
            logger.info(f"Client disconnected: {addr}")

    # -------------------------
    # Main Server Loop
    # -------------------------
    def run(self):
        while True:  # リトライループ（ポートが空くまで待つなど）
            try:
                self.server_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                # 1秒ごとにタイムアウトしてループを回す（Ctrl+C検知のため）
                self.server_socket.settimeout(1.0)

                self.running = True
                logger.info(
                    f"Listening on {self.host}:{self.port} ... (Press Ctrl+C to stop)")

                while self.running:
                    try:
                        conn, addr = self.server_socket.accept()
                        # クライアントごとにスレッド起動
                        t = threading.Thread(
                            target=self.handle_client, args=(conn, addr))
                        t.daemon = True
                        t.start()
                    except socket.timeout:
                        # タイムアウトは正常動作（ループ判定に戻るため）
                        continue
                    except OSError:
                        # ソケットが閉じられた場合など
                        break
                    except Exception as e:
                        logger.error(f"Accept error: {e}")

            except Exception as e:
                logger.error(f"Server start failed: {e}")
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
            finally:
                if not self.running:
                    break


if __name__ == "__main__":
    server = DS102Server()
    server.run()
