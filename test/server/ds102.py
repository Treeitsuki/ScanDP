"""
DS102/DS112コントローラ制御用サンプルプログラム

Copyright 2023 SURUGA SEIKI Co.,Ltd. All rights reserved.
"""

import serial
import serial.tools.list_ports
import time
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

axisNo = '1'            # 軸番号
direction = 'CCW'       # 駆動方向設定(-(CCW)、+(CW))
mode = 0                # 駆動方法(0: 連続駆動、1: ステップ駆動、2: 原点復帰)
ser = serial.Serial()


# 接続ボタンを押した時の処理
def connect_button_click(event):
    root.after(10, comm_port_open)


# 通信ポートを設定する
def comm_port_open():
    global ser

    if ser.is_open:
        ser.close()

    try:
        ser = serial.Serial(cmbCommPort.get(), cmbBaudrate.get(), timeout=2)
    except serial.SerialException:
        ser.close()
        lblState['text'] = '接続エラー発生'
        root.after(1, showerror('接続エラー発生'))
        return

    # ---------------------------------------------------------
    # ID要求
    # DS102 / DS112コントローラ接続確認
    # ---------------------------------------------------------
    r_data = serial_write_read(('*IDN?' + '\r').encode('utf-8'))

    if 'SURUGA,DS1' in str(r_data):
        # ---------------------------------------------------------
        # バージョン要求
        # ---------------------------------------------------------
        r_data = serial_write_read(('DS102VER?' + '\r').encode('utf-8'))
        lblFirmware['text'] = r_data

        # ---------------------------------------------------------
        # 制御軸数ステータス
        # ---------------------------------------------------------
        r_data = serial_write_read(('CONTA?' + '\r').encode('utf-8'))

        btnAxisX['state'] = 'disabled'
        btnAxisY['state'] = 'disabled'
        btnAxisZ['state'] = 'disabled'
        btnAxisU['state'] = 'disabled'
        btnAxisV['state'] = 'disabled'
        btnAxisW['state'] = 'disabled'

        # 取得した軸数 2(X、Y)
        if int(r_data) == 2:
            for axNo in range(2):
                serial_write(('AXI' + str(axNo + 1) + ':UNIT 0:SELSP 0' + '\r').encode('utf-8'))
                time.sleep(0.1)

            btnAxisX['state'] = 'normal'
            btnAxisY['state'] = 'normal'
        # 取得した軸数 4(X～U)
        elif int(r_data) == 4:
            for axNo in range(4):
                serial_write(('AXI' + str(axNo + 1) + ':UNIT 0:SELSP 0' + '\r').encode('utf-8'))
                time.sleep(0.1)

            btnAxisX['state'] = 'normal'
            btnAxisY['state'] = 'normal'
            btnAxisZ['state'] = 'normal'
            btnAxisU['state'] = 'normal'
        # 取得した軸数 6(X～W)
        else:
            for axNo in range(6):
                serial_write(('AXI' + str(axNo + 1) + ':UNIT 0:SELSP 0' + '\r').encode('utf-8'))
                time.sleep(0.1)

            btnAxisX['state'] = 'normal'
            btnAxisY['state'] = 'normal'
            btnAxisZ['state'] = 'normal'
            btnAxisU['state'] = 'normal'
            btnAxisV['state'] = 'normal'
            btnAxisW['state'] = 'normal'

        update_status()
    else:
        ser.close()
        lblState['text'] = '受信エラー発生'
        root.after(1, showerror(cmbCommPort.get() + 'にDS102/DS112コントローラが接続されていません'))


# 切断ボタンを押した時の処理
def disconnect_button_click(event):
    # Init Serial Port Setting
    global ser
    if ser.is_open:
        ser.close()

    btnAxisX['state'] = 'normal'
    btnAxisY['state'] = 'normal'
    btnAxisZ['state'] = 'normal'
    btnAxisU['state'] = 'normal'
    btnAxisV['state'] = 'normal'
    btnAxisW['state'] = 'normal'

    lblState['text'] = '未接続'


# X軸選択ボタンを押した時の処理
def axis_x_button_click():
    global axisNo
    axisNo = '1'
    btnAxisX.config(bg="LightBlue")
    btnAxisY.config(bg="SystemButtonFace")
    btnAxisZ.config(bg="SystemButtonFace")
    btnAxisU.config(bg="SystemButtonFace")
    btnAxisV.config(bg="SystemButtonFace")
    btnAxisW.config(bg="SystemButtonFace")

    update_status()


# Y軸選択ボタンを押した時の処理
def axis_y_button_click():
    global axisNo
    axisNo = '2'
    btnAxisX.config(bg="SystemButtonFace")
    btnAxisY.config(bg="LightBlue")
    btnAxisZ.config(bg="SystemButtonFace")
    btnAxisU.config(bg="SystemButtonFace")
    btnAxisV.config(bg="SystemButtonFace")
    btnAxisW.config(bg="SystemButtonFace")

    update_status()


# Z軸選択ボタンを押した時の処理
def axis_z_button_click():
    global axisNo
    axisNo = '3'
    btnAxisX.config(bg="SystemButtonFace")
    btnAxisY.config(bg="SystemButtonFace")
    btnAxisZ.config(bg="LightBlue")
    btnAxisU.config(bg="SystemButtonFace")
    btnAxisV.config(bg="SystemButtonFace")
    btnAxisW.config(bg="SystemButtonFace")

    update_status()


# U軸選択ボタンを押した時の処理
def axis_u_button_click():
    global axisNo
    axisNo = '4'
    btnAxisX.config(bg="SystemButtonFace")
    btnAxisY.config(bg="SystemButtonFace")
    btnAxisZ.config(bg="SystemButtonFace")
    btnAxisU.config(bg="LightBlue")
    btnAxisV.config(bg="SystemButtonFace")
    btnAxisW.config(bg="SystemButtonFace")

    update_status()


# V軸選択ボタンを押した時の処理
def axis_v_button_click():
    global axisNo
    axisNo = '5'
    btnAxisX.config(bg="SystemButtonFace")
    btnAxisY.config(bg="SystemButtonFace")
    btnAxisZ.config(bg="SystemButtonFace")
    btnAxisU.config(bg="SystemButtonFace")
    btnAxisV.config(bg="LightBlue")
    btnAxisW.config(bg="SystemButtonFace")

    update_status()


# W軸選択ボタンを押した時の処理
def axis_w_button_click():
    global axisNo
    axisNo = '6'
    btnAxisX.config(bg="SystemButtonFace")
    btnAxisY.config(bg="SystemButtonFace")
    btnAxisZ.config(bg="SystemButtonFace")
    btnAxisU.config(bg="SystemButtonFace")
    btnAxisV.config(bg="SystemButtonFace")
    btnAxisW.config(bg="LightBlue")

    update_status()


# 連続駆動を押した時の処理
def continue_mode():
    global mode
    btnCCW['text'] = '- (CCW)'
    btnCW['text'] = '+ (CW)'
    mode = var.get()


# ステップ駆動を押した時の処理
def step_mode():
    global mode
    btnCCW['text'] = '- (CCW)'
    btnCW['text'] = '+ (CW)'
    mode = var.get()


# 原点復帰を押した時の処理
def org_mode():
    global mode
    btnCCW['text'] = '原点復帰開始'
    btnCW['text'] = '原点復帰開始'
    mode = var.get()


# 停止ボタンを押した時の処理
def stop_button_click(event):
    # ---------------------------------------------------------
    # 停止
    # ---------------------------------------------------------
    serial_write(('STOP 0' + '\r').encode('utf-8'))


# CCWボタンを押した時の処理
def ccw_button_press(event):
    global direction
    direction = 'CCW'
    move_stage()


# CCWボタンを離した時の処理
def ccw_button_release(event):
    global mode
    if mode == 0:
        # ---------------------------------------------------------
        # 停止
        # ---------------------------------------------------------
        serial_write(('STOP 0' + '\r').encode('utf-8'))


# CWボタンを押した時の処理
def cw_button_press(event):
    global direction
    direction = 'CW'
    move_stage()


# CWボタンを離した時の処理
def cw_button_release(event):
    global mode
    if mode == 0:
        # ---------------------------------------------------------
        # 停止
        # ---------------------------------------------------------
        serial_write(('STOP 0' + '\r').encode('utf-8'))


# ステージを駆動する
def move_stage():
    global axisNo
    global direction
    global mode

    # 連続駆動
    if mode == 0:
        if direction == 'CW':
            # ---------------------------------------------------------
            # 連続駆動 CW方向
            # ---------------------------------------------------------
            serial_write(('AXI' + axisNo + ':L0 ' + txtLSpeed.get() +
                          ':R0 ' + txtRate.get() + ':S0 ' + txtSRate.get() +
                          ':F0 ' + txtSpeed.get() + ':GO CWJ' + '\r').encode('utf-8'))
        else:
            # ---------------------------------------------------------
            # 連続駆動 CCW方向
            # ---------------------------------------------------------
            serial_write(('AXI' + axisNo + ':L0 ' + txtLSpeed.get() +
                          ':R0 ' + txtRate.get() + ':S0 ' + txtSRate.get() +
                          ':F0 ' + txtSpeed.get() + ':GO CCWJ' + '\r').encode('utf-8'))
    # ステップ駆動
    elif mode == 1:
        # ---------------------------------------------------------
        # ステップ駆動
        # ---------------------------------------------------------
        serial_write(('AXI' + axisNo + ':L0 ' + txtLSpeed.get() +
                      ':R0 ' + txtRate.get() + ':S0 ' + txtSRate.get() +
                      ':F0 ' + txtSpeed.get() + ':PULS ' + txtStep.get() +
                      ':GO ' + direction + '\r').encode('utf-8'))
    # 原点復帰
    elif mode == 2:
        # 原点復帰方式を変更する
        # ---------------------------------------------------------
        # メモリスイッチ0設定
        # ---------------------------------------------------------
        serial_write(('AXI' + axisNo + ':MEMSW0 ' + str(cmbOrgMode.current()) + '\r').encode('utf-8'))
        time.sleep(0.1)
        # ---------------------------------------------------------
        # 原点復帰
        # ---------------------------------------------------------
        serial_write(('AXI' + axisNo + ':L0 ' + txtLSpeed.get() +
                      ':R0 ' + txtRate.get() + ':S0 ' + txtSRate.get() +
                      ':F0 ' + txtSpeed.get() + ':GO ORG' + '\r').encode('utf-8'))

    timer = threading.Timer(0.1, get_status)
    timer.start()


# 画面更新処理
def get_status():
    if update_status() == 'run':
        timer = threading.Timer(0.1, get_status)
        timer.start()


# ステータスを更新する
def update_status():
    # ---------------------------------------------------------
    # ステータス3要求
    # ---------------------------------------------------------
    r_data = serial_write_read(('AXI' + axisNo + ':SB3?' + '\r').encode('utf-8'))

    # 数値に変換できなければ終了する
    try:
        if r_data is not None:
            int(r_data)
        else:
            status = 'Stop'
            return status
    except ValueError:
        status = 'Stop'
        return status

    if not int(r_data) & 0x01 == 0x01:
        lblState['text'] = '軸選択不可能'
        status = 'Stop'
        return status
    else:
        # ---------------------------------------------------------
        # ステータス1要求
        # ---------------------------------------------------------
        r_data = serial_write_read(('AXI' + axisNo + ':SB1?' + '\r').encode('utf-8'))

        if int(r_data) & 0x40 == 0x40:
            lblState['text'] = '動作中'
            status = 'run'
        elif int(r_data) & 0x10 == 0x10:
            lblState['text'] = '原点検出'
            status = 'Stop'
        elif int(r_data) & 0x02 == 0x02 or int(r_data) & 0x04 == 0x04:
            # 機械リミット検出
            # ---------------------------------------------------------
            # ステータス2要求
            # ---------------------------------------------------------
            r_data = serial_write_read(('AXI' + axisNo + ':SB2?' + '\r').encode('utf-8'))
            if int(r_data) & 0x03 == 0x03:
                lblState['text'] = 'ステージ未接続'
            elif int(r_data) & 0x01 == 0x01:
                lblState['text'] = 'CWリミット検出'
            elif int(r_data) & 0x02 == 0x02:
                lblState['text'] = 'CCWリミット検出'
            elif int(r_data) & 0x04 == 0x04:
                lblState['text'] = 'CWソフトリミット検出'
            elif int(r_data) & 0x08 == 0x08:
                lblState['text'] = 'CCWソフトリミット検出'
            status = 'Stop'
        else:
            lblState['text'] = '停止'
            status = 'Stop'

        # ---------------------------------------------------------
        # 現在位置要求
        # ---------------------------------------------------------
        r_data = serial_write_read(('AXI' + axisNo + ':POS?' + '\r').encode('utf-8'))
        txtPosition.delete(0, tk.END)
        txtPosition.insert(tk.END, r_data)

        return status


# 送信
def serial_write(write_data):
    if ser.isOpen():
        ser.write(write_data)


# 送受信
def serial_write_read(write_data):
    if ser.isOpen():
        ser.write(write_data)
        time.sleep(0.1)
        read_data = ser.read_until(b'\r')

        return read_data


# ポジション設定ボタンを押した時の処理
def position_button_click(event):
    # ---------------------------------------------------------
    # 現在位置設定
    # ---------------------------------------------------------
    serial_write(('AXI' + axisNo + ':POS ' + txtPosition.get() + '\r').encode('utf-8'))


# 閉じるボタンを押した時の処理
def close_button_click(event):
    global ser
    if ser.isOpen():
        ser.close()

    root.destroy()


def showerror(msg):
    messagebox.showerror('エラー', msg)


# main
if __name__ == '__main__':
    # Window
    root = tk.Tk()
    root.title(u'DS102/DS112コントローラ・サンプルプログラム Ver.1.0.0')
    root.geometry('580x340')

    # 駆動軸
    btnAxisX = tk.Button(text=u'X', width=3, height=1, command=axis_x_button_click)
    btnAxisX.place(x=10, y=10)

    btnAxisY = tk.Button(text=u'Y', width=3, height=1, command=axis_y_button_click)
    btnAxisY.place(x=45, y=10)

    btnAxisZ = tk.Button(text=u'Z', width=3, height=1, command=axis_z_button_click)
    btnAxisZ.place(x=80, y=10)

    btnAxisU = tk.Button(text=u'U', width=3, height=1, command=axis_u_button_click)
    btnAxisU.place(x=115, y=10)

    btnAxisV = tk.Button(text=u'V', width=3, height=1, command=axis_v_button_click)
    btnAxisV.place(x=150, y=10)

    btnAxisW = tk.Button(text=u'W', width=3, height=1, command=axis_w_button_click)
    btnAxisW.place(x=185, y=10)

    lblFirmware = tk.Label(text=u'DS102')
    lblFirmware.place(x=310, y=15)

    btnClose = tk.Button(text=u'閉じる', width=10, height=1)
    btnClose.bind('<Button-1>', close_button_click)
    btnClose.place(x=476, y=10)

    # 駆動速度設定
    frameDriveSetting = tk.LabelFrame(root, text="駆動速度設定", width=270, height=145)
    frameDriveSetting.pack(fill=tk.BOTH, expand=True)
    frameDriveSetting.place(x=10, y=50)
    frameDriveSetting.propagate(False)

    frameLSpeed = tk.Frame(frameDriveSetting)
    frameLSpeed.pack(fill=tk.BOTH, expand=True)

    lblLSpeed = tk.Label(frameLSpeed, text=u'初速度(L)：', width=14, anchor='w')
    lblLSpeed.pack(side=tk.LEFT, padx=3, pady=0)

    txtLSpeed = tk.Entry(frameLSpeed, width=18)
    txtLSpeed.insert(tk.END, '100')
    txtLSpeed.pack(side=tk.LEFT)

    lblLSpeedUnit = tk.Label(frameLSpeed, text=u'pps')
    lblLSpeedUnit.pack(side=tk.LEFT, padx=9, pady=0)

    frameLRate = tk.Frame(frameDriveSetting)
    frameLRate.pack(fill=tk.BOTH, expand=True)

    lblRate = tk.Label(frameLRate, text=u'加減速レート(R)：', width=14, anchor='w')
    lblRate.pack(side=tk.LEFT, padx=3, pady=0)

    txtRate = tk.Entry(frameLRate, width=18)
    txtRate.insert(tk.END, '100')
    txtRate.pack(side=tk.LEFT)

    lblRateUnit = tk.Label(frameLRate, text=u'ms')
    lblRateUnit.pack(side=tk.LEFT, padx=9, pady=0)

    frameSRate = tk.Frame(frameDriveSetting)
    frameSRate.pack(fill=tk.BOTH, expand=True)

    lblSRate = tk.Label(frameSRate, text=u'S字レート(S)：', width=14, anchor='w')
    lblSRate.pack(side=tk.LEFT, padx=3, pady=0)

    txtSRate = tk.Entry(frameSRate, width=18)
    txtSRate.insert(tk.END, '100')
    txtSRate.pack(side=tk.LEFT)

    lblSRateUnit = tk.Label(frameSRate, text=u'%')
    lblSRateUnit.pack(side=tk.LEFT, padx=9, pady=0)

    frameSpeed = tk.Frame(frameDriveSetting)
    frameSpeed.pack(fill=tk.BOTH, expand=True)

    lblSpeed = tk.Label(frameSpeed, text=u'駆動速度(F)：', width=14, anchor='w')
    lblSpeed.pack(side=tk.LEFT, padx=3, pady=0)

    txtSpeed = tk.Entry(frameSpeed, width=18)
    txtSpeed.insert(tk.END, '1000')
    txtSpeed.pack(side=tk.LEFT)

    lblSpeedUnit = tk.Label(frameSpeed, text=u'pps')
    lblSpeedUnit.pack(side=tk.LEFT, padx=9, pady=0)

    # 駆動方法選択
    frameDriveSelect = tk.LabelFrame(root, text="駆動方法選択", width=270, height=110)
    frameDriveSelect.pack(fill=tk.BOTH, expand=True)
    frameDriveSelect.place(x=10, y=215)
    frameDriveSelect.propagate(False)

    frameContinueMode = tk.Frame(frameDriveSelect)
    frameContinueMode.pack(fill=tk.BOTH, expand=True)

    var = tk.IntVar()
    var.set(0)

    rbnContinueMode = tk.Radiobutton(frameContinueMode, text='連続駆動', value='0',
                                     variable=var, command=continue_mode, width=10, anchor='w')
    rbnContinueMode.pack(side=tk.LEFT, padx=3, pady=0)

    frameStepMode = tk.Frame(frameDriveSelect)
    frameStepMode.pack(fill=tk.BOTH, expand=True)

    rbnStepMode = tk.Radiobutton(frameStepMode, text='ステップ駆動', value='1',
                                 variable=var, command=step_mode, width=10, anchor='w')
    rbnStepMode.pack(side=tk.LEFT, padx=3, pady=0)

    txtStep = tk.Entry(frameStepMode, width=18)
    txtStep.insert(tk.END, '1000')
    txtStep.pack(side=tk.LEFT, padx=3, pady=0)

    lblStepUnit = tk.Label(frameStepMode, text=u'Pulse')
    lblStepUnit.pack(side=tk.LEFT, padx=9, pady=0)

    frameOrgMode = tk.Frame(frameDriveSelect)
    frameOrgMode.pack(fill=tk.BOTH, expand=True)

    rbnOrgMode = tk.Radiobutton(frameOrgMode, text='原点復帰', value='2',
                                variable=var, command=org_mode, width=10, anchor='w')
    rbnOrgMode.pack(side=tk.LEFT, padx=3, pady=0)

    org = ['ORG 0', 'ORG 1', 'ORG 2', 'ORG 3', 'ORG 4', 'ORG 5', 'ORG 6',
           'ORG 7', 'ORG 8', 'ORG 9', 'ORG 10', 'ORG 11', 'ORG 12']
    orgList = tk.StringVar()
    cmbOrgMode = ttk.Combobox(frameOrgMode, values=org, textvariable=orgList, width=15)
    cmbOrgMode.pack(side=tk.LEFT, padx=3, pady=0)
    cmbOrgMode.set(org[0])

    # 通信設定
    frameConnection = tk.LabelFrame(root, text="通信設定", width=270, height=100)
    frameConnection.pack(fill=tk.BOTH, expand=True)
    frameConnection.place(x=302, y=50)
    frameConnection.propagate(False)

    frameCommPort = tk.Frame(frameConnection)
    frameCommPort.pack(fill=tk.BOTH, expand=True)

    lblCommPort = tk.Label(frameCommPort, text=u'通信ポート：', width=10)
    lblCommPort.pack(side=tk.LEFT)

    port = ['COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7',
            'COM8', 'COM9', 'COM10', 'COM11', 'COM12', 'COM13', 'COM14',
            'COM15', 'COM16', 'COM17', 'COM18', 'COM19', 'COM20']
    portList = tk.StringVar()
    cmbCommPort = ttk.Combobox(frameCommPort, values=port, textvariable=portList, width=10)
    cmbCommPort.set(port[0])
    cmbCommPort.pack(side=tk.LEFT)

    btnConnect = tk.Button(frameCommPort, text=u'接続', width=20, height=1)
    btnConnect.bind('<Button-1>', connect_button_click)
    btnConnect.pack(side=tk.LEFT, padx=12, pady=0)

    frameBaudrate = tk.Frame(frameConnection)
    frameBaudrate.pack(fill=tk.BOTH, expand=True)

    lblBaudrate = tk.Label(frameBaudrate, text=u'ボーレート：', width=10, height=1)
    lblBaudrate.pack(side=tk.LEFT)

    baudrate = ['38400', '19200', '9600', '4800']
    baudrateList = tk.StringVar()
    cmbBaudrate = ttk.Combobox(frameBaudrate, values=baudrate, textvariable=baudrateList, width=10)
    cmbBaudrate.set(baudrate[0])
    cmbBaudrate.pack(side=tk.LEFT)

    btnDisconnect = tk.Button(frameBaudrate, text=u'切断', width=20, height=1)
    btnDisconnect.bind('<Button-1>', disconnect_button_click)
    btnDisconnect.pack(side=tk.LEFT, padx=12, pady=0)

    # 駆動
    frameDrive = tk.LabelFrame(root, text="駆動", width=270, height=155)
    frameDrive.pack(fill=tk.BOTH, expand=True)
    frameDrive.place(x=302, y=170)
    frameDrive.propagate(False)

    framePosition = tk.Frame(frameDrive)
    framePosition.pack(fill=tk.BOTH, expand=True)

    lblPosition = tk.Label(framePosition, text=u'ポジション：', width=10)
    lblPosition.pack(side=tk.LEFT)

    txtPosition = tk.Entry(framePosition, width=13)
    txtPosition.insert(tk.END, '0')
    txtPosition.pack(side=tk.LEFT)

    btnPosition = tk.Button(framePosition, text=u'ポジション設定', width=20, height=1)
    btnPosition.bind('<Button-1>', position_button_click)
    btnPosition.pack(side=tk.LEFT, padx=12, pady=0)

    frameStatus = tk.Frame(frameDrive)
    frameStatus.pack(fill=tk.BOTH, expand=True)

    lblStatus = tk.Label(frameStatus, text=u'ステータス：', width=10)
    lblStatus.pack(side=tk.LEFT)

    lblState = tk.Label(frameStatus, text=u'未接続')
    lblState.pack(side=tk.LEFT)

    frameRun = tk.Frame(frameDrive)
    frameRun.pack(fill=tk.BOTH, expand=True)

    btnCCW = tk.Button(frameRun, text=u'- (CCW)', width=10, height=2)
    btnCCW.bind('<ButtonPress-1>', ccw_button_press)
    btnCCW.bind('<ButtonRelease-1>', ccw_button_release)
    btnCCW.pack(side=tk.LEFT, padx=4, pady=0)

    btnStop = tk.Button(frameRun, text=u'停止', width=10, height=2)
    btnStop.bind('<Button-1>', stop_button_click)
    btnStop.pack(side=tk.LEFT, padx=4, pady=0)

    btnCW = tk.Button(frameRun, text=u'+ (CW)', width=10, height=2)
    btnCW.bind('<ButtonPress-1>', cw_button_press)
    btnCW.bind('<ButtonRelease-1>', cw_button_release)
    btnCW.pack(side=tk.LEFT, padx=4, pady=0)

    root.mainloop()
