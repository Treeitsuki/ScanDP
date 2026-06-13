#!/usr/bin/python
# -*- coding: utf-8 -*-


#!/usr/bin/python
# -*- coding: utf-8 -*-

import threading
import signal
import sys
import time
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

# ロボットライブラリのインポート
from i611_MCS import *
from i611shm import *

# サーバー設定
HOST = '192.168.0.23'
PORT = 4416


class RobotRPCInterface:
    def __init__(self):
        # インスタンス変数として管理
        self.rb = i611Robot()
        self._BASE = Base()

        # 状態保持用のダミーオブジェクト（元のコードのCO1, P1の代替）
        self.temp_coord = Coordinate()
        self.temp_pos = Position()

        # 排他制御用のロック（同時に複数の移動命令が来ないようにする）
        self.lock = threading.Lock()

        # ロボット接続と初期設定
        print "Connecting to Robot..."
        self.rb.open()
        self.rb.asyncm(sw=2)
        # 初期モーションパラメータ設定
        self.rb.motionparam(jnt_speed=10, acctime=0.1,
                            dacctime=0.1, passm=1, overlap=500, zone=20)
        print "Robot Connected and Ready."

    def cleanup(self):
        """終了時のクリーンアップ"""
        print "Closing connection..."
        try:
            self.rb.close()
        except:
            pass

    # --- 基本関数 ---
    def Hello(self):
        return 'Hello World'

    def Add(self, x, y):
        return x + y

    # --- Coordinate Wrapper ---
    # XML-RPCはオブジェクトの状態変更を遠隔で行うのには不向きですが、
    # 元コードの仕様(CO1の書き換え)を再現します。
    def coord_replace(self, *args, **kwargs):
        self.temp_coord.replace(*args, **kwargs)
        return True  # 何か返り値を返す必要がある

    def coord_shift(self, *args, **kwargs):
        self.temp_coord.shift(*args, **kwargs)
        return True

    def coord_copy(self):
        # Copyして新しいオブジェクトを作るが、XML-RPCでオブジェクトは返せないのでリストを返す
        c = self.temp_coord.copy()
        return c.pos  # 内部リストを返す

    def coord_clear(self):
        self.temp_coord.clear()
        return True

    def coord_g2b(self, X, Y, Z, RZ, RY, RX):
        return self.temp_coord.g2b(X, Y, Z, RZ, RY, RX)

    def coord_b2g(self, x, y, z, rz, ry, rx):
        return self.temp_coord.b2g(x, y, z, rz, ry, rx)

    # --- Position Wrapper ---
    def pos_replace(self, *args, **kwargs):
        self.temp_pos.replace(*args, **kwargs)
        return True

    def pos_offset(self, *args, **kwargs):
        self.temp_pos.offset(*args, **kwargs)
        return True

    def pos_shift(self, *args, **kwargs):
        self.temp_pos.shift(*args, **kwargs)
        return True

    def pos_copy(self):
        return self.temp_pos.pos2list()

    def pos_clear(self):
        self.temp_pos.clear()
        return True

    def pos2list(self):
        return self.temp_pos.pos2list()

    def pos2dict(self):
        return self.temp_pos.pos2dict()

    def position(self):
        # 親座標系変換後のリストを返す
        return self.temp_pos.position()

    # --- Robot Wrapper ---
    # オブジェクトを返すメソッドは、必ず list/dict/基本型 に変換して返す

    def getjnt(self):
        """現在の関節角度をリストで返す"""
        try:
            j = self.rb.getjnt()
            return j.jnt2list()  # Jointオブジェクトではなくリストを返す
        except Exception as e:
            print "Error in getjnt:", e
            return []

    def getpos(self):
        """現在の位置をリストで返す"""
        try:
            p = self.rb.getpos()
            return p.pos2list()
        except Exception as e:
            print "Error in getpos:", e
            return []

    # スレッドを用いた移動処理
    def _threaded_move(self, target_obj):
        try:
            self.rb.move(target_obj)
        except Exception as e:
            print "Move Error:", e

    def move_joint(self, jnt_lis
                   msg.name=self.joint_names
                   msg.position=joint_degreest):
        """リストを受け取り、非同期で移動を開始する"""
        self.rb.abort()  # 安全のため停止
        try:
            # 入力がリストであることを想定
            j = Joint(jnt_list)
            t = threading.Thread(target=self._threaded_move, args=(j,))
            t.daemon = True  # メイン終了時に強制終了されるようにする
            t.start()
            return True
        except Exception as e:
            print "Error in move_joint:", e
            return False

    def move_position(self, pos_list):
        """リストを受け取り、Joint変換して非同期移動"""
        self.rb.abort()
        try:
            # Positionオブジェクト作成 (引数がリストか確認が必要)
            p = Position(pos_list)
            # 逆運動学計算
            j = self.rb.Position2Joint(p)
            t = threading.Thread(target=self._threaded_move, args=(j,))
            t.daemon = True
            t.start()
            return True
        except Exception as e:
            print "Error in move_position:", e
            return False

    def move_joint_nores(self, jnt_list):
        """レスポンスを待たずに即時実行(スレッドなし、ライブラリのasync依存)"""
        print 'move_joint_nores'
        try:
            j = Joint(jnt_list)
            self.rb.move(j)  # asyncm=1なので即戻るはずだが、念の為
            return True
        except Exception as e:
            print "Error:", e
            return False

    def get_buffer_num(self):
        # _syssts は内部関数のようですがそのまま使用
        return self.rb._syssts(5)

    def Joint2Position(self, jnt_list):
        j = Joint(jnt_list)
        p = self.rb.Joint2Position(j)
        return p.pos2list()  # Listで返す

    def Position2Joint(self, pos_list):
        p = Position(pos_list)
        j = self.rb.Position2Joint(p)
        return j.jnt2list()  # Listで返す

    # その他のRobotメソッドのパススルー
    # 必要に応じて引数の型変換を追加してください
    def version(self): return self.rb.version()
    def MCS_version(self): return self.rb.MCS_version()
    def open(self): return self.rb.open()
    def close(self): return self.rb.close()
    def exit(self): return self.rb.exit()
    def svoff(self): return self.rb.svoff()
    def svstat(self): return self.rb.svstat()
    def asyncm(self, *args, **kwargs): return self.rb.asyncm(*args, **kwargs)
    def join(self): return self.rb.join()
    def abort(self): return self.rb.abort()
    def home(self): return self.rb.home()

    # パラメータ設定系
    def motionparam(self, *args, **kwargs):
        # キーワード引数はXML-RPCでは辞書として渡されることが多い
        # クライアント側で kwargsを展開して送る必要があるが、
        # Pythonのxmlrpcなら辞書を一つ受け取る形に変更したほうが安全かもしれません。
        # ここではそのまま通します。
        self.rb.motionparam(*args, **kwargs)
        return True

    def getmotionparam(self):
        mp = self.rb.getmotionparam()
        return mp.mp2dict()  # 辞書に変換して返す

    def override(self, val): return self.rb.override(val)
    def settool(self, *args): return self.rb.settool(*args)
    def changetool(self, val): return self.rb.changetool(val)
    def set_mdo(self, *args): return self.rb.set_mdo(*args)
    def enable_mdo(self, *args): return self.rb.enable_mdo(*args)
    def disable_mdo(self, *args): return self.rb.disable_mdo(*args)
    def user_hook(self, *args): return self.rb.user_hook(*args)
    def set_behavior(self, *args): return self.rb.set_behavior(*args)
    def release_stopevent(self): return self.rb.release_stopevent()
    def cause_user_error(self, err): return self.rb.cause_user_error(err)
    def enable_interrupt(self, *args): return self.rb.enable_interrupt(*args)
    def check_ready(self): return self.rb.check_ready()

    # SHM
    def rpc_shm_read(self, *args):
        # グローバルのshm_readをラップ
        return shm_read(*args)


def main():
    # インターフェースの初期化
    robot_interface = RobotRPCInterface()

    # サーバーの設定
    server = SimpleXMLRPCServer(
        (HOST, PORT), logRequests=False, allow_none=True)
    print "Listening on %s:%d" % (HOST, PORT)

    # 終了シグナルの登録 (Ctrl+Cで安全に閉じる)
    def signal_handler(sig, frame):
        print '\nStopping Server...'
        robot_interface.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 関数の登録
    server.register_function(robot_interface.Hello, "Hello")
    server.register_function(robot_interface.Add, "Add")

    # Coordinate (インスタンスメソッドを直接登録せず、ラッパーを登録)
    server.register_function(robot_interface.coord_replace, "replace")
    server.register_function(robot_interface.coord_shift, "shift")
    server.register_function(robot_interface.coord_copy, "copy")
    server.register_function(robot_interface.coord_clear, "clear")
    server.register_function(robot_interface.coord_g2b, "g2b")
    server.register_function(robot_interface.coord_b2g, "b2g")

    # Position
    server.register_function(robot_interface.pos_replace, "pos_replace")
    server.register_function(robot_interface.pos_offset, "pos_offset")
    server.register_function(robot_interface.pos_shift, "pos_shift")
    server.register_function(robot_interface.pos_copy, "pos_copy")
    server.register_function(robot_interface.pos_clear, "pos_clear")
    server.register_function(robot_interface.pos2list, "pos2list")
    server.register_function(robot_interface.pos2dict, "pos2dict")
    server.register_function(robot_interface.position, "position")

    # Robot Functions
    # 基本的にラッパーを通すことで返り値の型変換を行う
    server.register_function(robot_interface.version, "version")
    server.register_function(robot_interface.MCS_version, "MCS_version")
    server.register_function(robot_interface.open, "open")
    server.register_function(robot_interface.close, "close")
    server.register_function(robot_interface.exit, "exit")
    server.register_function(robot_interface.svoff, "svoff")
    server.register_function(robot_interface.asyncm, "asyncm")
    server.register_function(robot_interface.join, "join")
    server.register_function(robot_interface.abort, "abort")
    server.register_function(robot_interface.home, "home")

    # Move系
    # オリジナルの move / line / toolmove は引数がオブジェクトだとエラーになるため
    # クライアント側でリストを投げる move_joint / move_position の使用を推奨しますが
    # 一応登録はしておきます（使用注意）
    # server.register_function(robot_interface.move, "move")

    server.register_function(robot_interface.move_joint, "move_joint")
    server.register_function(robot_interface.move_position, "move_position")
    server.register_function(
        robot_interface.move_joint_nores, "move_joint_nores")

    server.register_function(robot_interface.motionparam, "motionparam")
    server.register_function(robot_interface.getmotionparam, "getmotionparam")
    server.register_function(robot_interface.override, "override")
    server.register_function(robot_interface.settool, "settool")
    server.register_function(robot_interface.changetool, "changetool")

    server.register_function(robot_interface.set_mdo, "set_mdo")
    server.register_function(robot_interface.enable_mdo, "enable_mdo")
    server.register_function(robot_interface.disable_mdo, "disable_mdo")

    server.register_function(robot_interface.getpos, "getpos")
    server.register_function(robot_interface.getjnt, "getjnt")

    server.register_function(robot_interface.Joint2Position, "Joint2Position")
    server.register_function(robot_interface.Position2Joint, "Position2Joint")

    server.register_function(robot_interface.svstat, "svstat")
    server.register_function(robot_interface.user_hook, "user_hook")
    server.register_function(robot_interface.set_behavior, "set_behavior")
    server.register_function(
        robot_interface.release_stopevent, "release_stopevent")
    server.register_function(
        robot_interface.cause_user_error, "cause_user_error")
    server.register_function(
        robot_interface.enable_interrupt, "enable_interrupt")
    server.register_function(robot_interface.check_ready, "check_ready")
    server.register_function(robot_interface.get_buffer_num, "get_buffer_num")

    # SHM
    server.register_function(robot_interface.rpc_shm_read, 'shm_read')

    server.register_introspection_functions()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        robot_interface.cleanup()


if __name__ == '__main__':
    main()
