#!/usr/bin/python
# -*- coding: utf-8 -*-

from i611_MCS import *
from i611shm import *
import SimpleXMLRPCServer as xmlrpc_server
import threading


def Hello():
    return 'Hello World'


def Add(x, y):
    return x+y


def main():
    CO1 = Coordinate()
    P1 = Position()

    rb1 = i611Robot()
    _BASE = Base()
    rb1.open()
    rb1.asyncm(sw=1)
    rb1.motionparam(jnt_speed=10, acctime=0.1, dacctime=0.1,
                    passm=1, overlap=500, zone=20)
    # rb1.motionparam(jnt_speed=10, acctime=0, dacctime=0, passm=1, overlap=500, zone=20)
    server = xmlrpc_server.SimpleXMLRPCServer(
        ('192.168.0.23', 4416), logRequests=False,  allow_none=True)

    server.register_function(Hello)
    server.register_function(Add)

    # class Coordinate
    server.register_function(CO1.replace, "replace")
    server.register_function(CO1.shift, "shift")
    server.register_function(CO1.copy, "copy")
    server.register_function(CO1.clear, "clear")
    server.register_function(CO1.g2b, "g2b")
    server.register_function(CO1.b2g, "b2g")

    # class Position
    server.register_function(P1.replace, "pos_replace")
    server.register_function(P1.offset, "pos_offset")
    server.register_function(P1.shift, "pos_shift")
    server.register_function(P1.copy, "pos_copy")
    server.register_function(P1.clear, "pos_clear")
    server.register_function(P1.pos2list, "pos2list")
    server.register_function(P1.pos2dict, "pos2dict")
    server.register_function(P1.position, "position")

    # class 611Robot
    server.register_function(rb1.version, "version")
    server.register_function(rb1.MCS_version, "MCS_version")
    server.register_function(rb1.open, "open")
    server.register_function(rb1.close, "close")
    server.register_function(rb1.exit, "exit")
    server.register_function(rb1.svoff, "svoff")
    server.register_function(rb1.asyncm, "asyncm")
    server.register_function(rb1.join, "join")
    server.register_function(rb1.abort, "abort")
    server.register_function(rb1.home, "home")
    server.register_function(rb1.move, "move")
    server.register_function(rb1.line, "line")
    server.register_function(rb1.toolmove, "toolmove")
    server.register_function(rb1.motionparam, "motionparam")
    server.register_function(rb1.getmotionparam, "getmotionparam")
    server.register_function(rb1.override, "override")
    server.register_function(rb1.settool, "settool")
    server.register_function(rb1.changetool, "changetool")
    server.register_function(rb1.set_mdo, "set_mdo")
    server.register_function(rb1.enable_mdo, "enable_mdo")
    server.register_function(rb1.disable_mdo, "disable_mdo")
    server.register_function(rb1.getpos, "getpos")
    server.register_function(rb1.getjnt, "getjnt")
    server.register_function(rb1.Joint2Position, "Joint2Position")
    server.register_function(rb1.Position2Joint, "Position2Joint")
    server.register_function(rb1.svstat, "svstat")
    # server.register_function(rb1.is_open, "is_open")
    server.register_function(rb1.user_hook, "user_hook")
    server.register_function(rb1.set_behavior, "set_behavior")
    server.register_function(rb1.release_stopevent, "release_stopevent")
    server.register_function(rb1.cause_user_error, "cause_user_error")
    server.register_function(rb1.enable_interrupt, "enable_interrupt")
    server.register_function(rb1.check_ready, "check_ready")

    def move_joint(jnt):
        rb1.abort()
        # print 'move_joint'
        # print jnt
        # rb1.asyncm(sw=1)
        # rb1.motionparam(acctime=0,dacctime=0,passm=1,overlap=0,zone=20)
        j = Joint(jnt)
        # rb1.move(j)
        thread_1 = threading.Thread(target=rb1.move, args=[j])
        thread_1.start()

    server.register_function(move_joint, "move_joint")

    def move_position(pos):
        rb1.abort()
        p = Position(pos)
        j = rb1.Position2Joint(p)
        thread_2 = threading.Thread(target=rb1.move, args=[j])
        thread_2.start()
    server.register_function(move_position, "move_position")

    def get_buffer_num():
        # print rb1._syssts(5)
        return rb1._syssts(5)
        # thread_2 = threading.Thread(target=rb1._syssts, args=[5])
        # return thread_2.start()

    server.register_function(get_buffer_num, "get_buffer_num")

    def get_joint():
        print 'get_joint'
        j = rb1.getjnt()
        return j

    server.register_function(get_joint, "get_joint")

    def move_joint_nores(jnt):
        print 'move_joint_nores'
        j = Joint(jnt)
        rb1.move(j)

    server.register_function(move_joint_nores, 'move_joint_nores')

    server.register_function(shm_read, 'shm_read')

    server.register_introspection_functions()
    print(get_joint())
    server.serve_forever()


if __name__ == '__main__':
    main()
